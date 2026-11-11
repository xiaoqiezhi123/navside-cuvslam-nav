"""Ordered goal picking on the reviewed planning map, before navigation starts."""
from pathlib import Path
from datetime import datetime
import json
import queue

import numpy as np

from task_nav_astar import TASK_POINT_Y_M, save_task_points, validate_task_points
from task_path_preview import PreviewWorker, pair_key


class TaskPointPicker:
    def __init__(self, planner, output, initial_points=None, *, preview_root=None):
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.colors import ListedColormap
        from matplotlib.widgets import Button
        fonts = [f.name for f in font_manager.fontManager.ttflist if 'CJK' in f.name or 'WenQuanYi' in f.name]
        if fonts:
            plt.rcParams['font.family'] = fonts[0]
        plt.rcParams['axes.unicode_minus'] = False
        # Keep picker shortcuts from also triggering toolbar pan/fullscreen/history.
        owned_keys = {'z', 'ctrl+z', 'backspace', 'c', 'h', 'enter', 'escape',
                      'left', 'right', 'tab', 'f', 'p'}
        for key in plt.rcParams:
            if key.startswith('keymap.'):
                plt.rcParams[key] = [v for v in plt.rcParams[key] if v not in owned_keys]
        self.planner, self.output = planner, Path(output)
        self.points = [] if not initial_points else validate_task_points(planner, initial_points)
        self.accepted = False
        self.artists = []
        self.route_artists = []
        self.preview_worker = PreviewWorker(planner)
        self.revision = 0
        self.previews = {}
        self.selected_pair = 0
        self.show_references = True
        self.closed = False
        self.preview_root = Path(preview_root) if preview_root else Path(__file__).resolve().parents[1]/'logs/task_point_previews'
        self.last_preview_directory = None
        self.pan = None
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title('分段导航：任务选点与 A* 预览')
        self.ax = self.fig.add_axes([.06, .33, .65, .54])
        self.ax.set_facecolor('#e5e7eb')
        self.table = self.fig.add_axes([.75, .30, .23, .57])
        self.table.axis('off')
        self.fig.suptitle('任务选点与 A* 参考路线预览', fontsize=18)
        self.fig.text(.06, .91, '橙色 T：任务目标    蓝线 / 蓝点：当前预览及参考点    灰蓝线：其他预览    虚线：当前原始 A*', fontsize=11)
        pm = planner.pm
        known = pm.occupancy != -1
        rows, cols = np.flatnonzero(known.any(1)), np.flatnonzero(known.any(0))
        if not len(rows):
            raise ValueError('地图没有已知区域')
        pad = int(np.ceil(.8/pm.meta['resolution_m']))
        lo = np.maximum([rows[0]-pad, cols[0]-pad], 0)
        hi = np.minimum([rows[-1]+pad+1, cols[-1]+pad+1], pm.occupancy.shape)
        region = np.s_[lo[0]:hi[0], lo[1]:hi[1]]
        occ = pm.occupancy[region]
        semantic = np.zeros(occ.shape, dtype=np.uint8)
        semantic[occ == 0] = 1
        semantic[occ == 100] = 2
        semantic[pm.traversable[region]] = 3
        x, z = pm.meta['origin_xz']; res = pm.meta['resolution_m']
        self.extent = [x+lo[1]*res, x+hi[1]*res, z+lo[0]*res, z+hi[0]*res]
        self.ax.imshow(semantic, origin='lower', extent=self.extent, interpolation='nearest',
                       vmin=0, vmax=3, cmap=ListedColormap(['#e5e7eb', '#d8edc4', '#212121', '#489967']))
        self.ax.set_xlabel('地图 X (m)'); self.ax.set_ylabel('地图 Z (m)')
        self.ax.set_aspect('equal')
        self.status = self.fig.text(.06, .255, '', fontsize=11)
        self.fig.text(.06, .195, '只选任务目标；至少两个目标时自动预览 T1→T2、T2→T3…；正式导航仍按每次真实重定位重新规划。\n'
                      '启动位置→T1 尚未定位，无法预览。深绿可选；左键加目标，中键拖动，滚轮缩放。参考点 Y=0。', fontsize=10)
        self.buttons = []
        rows = [(.12, [('上一段 [←]', self.previous_pair), ('下一段 [→]', self.next_pair),
                      ('目标/参考点 [Tab]', self.toggle_table), ('本段放大 [F]', self.focus), ('保存预览 [P]', self.save_preview)]),
                (.045, [('撤销末点 [Z]', self.undo), ('清空 [C]', self.clear),
                        ('全图 [H]', self.home), ('确认目标 [Enter]', self.confirm), ('取消 [Esc]', self.cancel)])]
        for y, actions in rows:
            for i, (name, callback) in enumerate(actions):
                button = Button(self.fig.add_axes([.06+i*.18, y, .16, .05]), name)
                button.on_clicked(callback)
                self.buttons.append(button)
        for name, callback in [('button_press_event', self.on_press), ('button_release_event', self.on_release),
                ('motion_notify_event', self.on_motion), ('scroll_event', self.on_scroll),
                ('key_press_event', self.on_key), ('close_event', self.on_close)]:
            self.fig.canvas.mpl_connect(name, callback)
        self.timer = self.fig.canvas.new_timer(interval=150)
        self.timer.add_callback(self.poll_previews)
        self.timer.start()
        self.home()
        self.request_previews()

    def pairs(self):
        return [pair_key(a, b) for a, b in zip(self.points, self.points[1:])]

    def current_preview(self):
        pairs = self.pairs()
        return self.previews.get(pairs[self.selected_pair]) if pairs else None

    def request_previews(self):
        pairs = self.pairs()
        self.selected_pair = min(self.selected_pair, max(0, len(pairs)-1))
        self.previews = {k:v for k,v in self.previews.items() if k in pairs and not isinstance(v, Exception)}
        self.last_preview_directory = None
        self.revision = self.preview_worker.request(self.points)
        self.refresh()

    def poll_previews(self):
        if self.closed:
            return
        changed = False
        while True:
            try:
                revision, key, outcome = self.preview_worker.results.get_nowait()
            except queue.Empty:
                break
            if revision == self.revision and key in self.pairs():
                self.previews[key] = outcome
                changed = True
        if changed:
            self.refresh()

    def refresh(self):
        import matplotlib.patheffects as pe
        for artist in self.artists+self.route_artists:
            artist.remove()
        self.artists.clear()
        self.route_artists.clear()
        pairs = self.pairs()
        for i, key in enumerate(pairs):
            result = self.previews.get(key)
            if result is None or isinstance(result, Exception):
                continue
            selected = i == self.selected_pair
            full = np.asarray(result['reference_polyline_xy'])
            refs = np.asarray(result['references_xy'])
            self.route_artists += self.ax.plot(*full.T, color='#1676e8' if selected else '#829fb5',
                                               linewidth=2.2 if selected else 1.4, zorder=3 if selected else 2)
            self.route_artists.append(self.ax.scatter(*refs.T, s=28 if selected else 14,
                c='#20a8ff' if selected else '#a4c3d8', edgecolors='white', linewidths=.6, zorder=4))
            if selected:
                raw = np.asarray(result['astar_path_xy'])
                self.route_artists += self.ax.plot(*raw.T, '--', color='#75b9f3', linewidth=.9, zorder=2)
                for n, p in enumerate(refs, 1):
                    # Put numbers on alternating sides of the route, not between two
                    # neighbouring points where labels overlap on vertical corridors.
                    tangent = refs[min(n, len(refs)-1)]-refs[max(0, n-2)]
                    side = 1 if n % 2 else -1
                    vertical = abs(tangent[1]) > abs(tangent[0])
                    offset = (side*10, 0) if vertical else (0, side*11)
                    ha = ('left' if side > 0 else 'right') if vertical else 'center'
                    va = 'center' if vertical else ('bottom' if side > 0 else 'top')
                    self.route_artists.append(self.ax.annotate(str(n), p, xytext=offset, ha=ha, va=va,
                        textcoords='offset points', fontsize=9, color='#084991', zorder=7,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]))
        self.table.clear(); self.table.axis('off')
        for i, (x, _, z) in enumerate(self.points, 1):
            self.artists.append(self.ax.scatter(x, z, c='#ef8a17', edgecolors='white', s=65, zorder=5))
            self.artists.append(self.ax.annotate(f'T{i}', (x, z), xytext=(-24, 12), textcoords='offset points',
                fontsize=11, weight='bold', zorder=8, color='#8d3a00',
                path_effects=[pe.withStroke(linewidth=3, foreground='white')]))
        result = self.current_preview()
        if pairs and self.show_references:
            self.table.text(0, 1, f'预览 T{self.selected_pair+1} → T{self.selected_pair+2}', va='top', fontsize=14)
            self.table.text(0, .93, '15 个参考点：X / Z (m)，Y=0', va='top', fontsize=10)
            if isinstance(result, Exception):
                import textwrap
                self.table.text(0, .83, '本段预览失败\n'+textwrap.fill(str(result), 20), va='top', color='#b42318', fontsize=11)
            elif result is None:
                self.table.text(0, .82, '正在后台规划…\n可以继续选点、缩放或撤销', va='top', fontsize=11)
            else:
                for row, (x, z) in enumerate(result['references_xy']):
                    self.table.text(0, .845-row*.043, f'{row+1:02d}  {x:9.3f}  {z:9.3f}',
                                    fontsize=10, fontfamily='DejaVu Sans Mono')
                self.table.text(0, .14, f"长度 {result['reference_length_m']:.2f} m\n最小地图净空 {result['min_clearance_m']:.2f} m", va='top', fontsize=10)
                self.table.text(0, .045, ('包含起点' if result['includes_start'] else '不含起点')+'；'+
                    ('包含终点' if result['includes_goal'] else '不含终点'), fontsize=10)
        else:
            self.table.text(0, 1, f'目标顺序（共 {len(self.points)} 点）', va='top', fontsize=13)
            self.table.text(0, .93, '编号       X          Z (m)\n           Y = 0.695', va='top', fontsize=10)
            first = max(0, len(self.points)-16)
            for row, i in enumerate(range(first, len(self.points))):
                x, _, z = self.points[i]
                self.table.text(0, .81-row*.046, f'T{i+1:02d} {x:9.3f} {z:9.3f}',
                                fontsize=10, fontfamily='DejaVu Sans Mono')
            if first:
                self.table.text(0, .02, '表格显示最后 16 点；保存包含全部目标', fontsize=9)
        good = sum(isinstance(self.previews.get(k), dict) for k in pairs)
        bad = sum(isinstance(self.previews.get(k), Exception) for k in pairs)
        pending = len(pairs)-good-bad
        if not pairs:
            self.message('至少选择两个任务目标，自动预览相邻目标之间的 A* 和 15 点；首段等待真实定位。')
        else:
            self.message(f'目标 {len(self.points)} 个｜相邻路线：成功 {good} / 失败 {bad} / 等待 {pending}｜'
                         f'当前 T{self.selected_pair+1}→T{self.selected_pair+2}；左右键切段，F 放大。', bad > 0)

    def message(self, text, error=False):
        self.status.set_text(text)
        self.status.set_color('#b42318' if error else '#263648')
        self.fig.canvas.draw_idle()

    def add(self, x, z):
        try:
            points = validate_task_points(self.planner, [*self.points, [x, TASK_POINT_Y_M, z]])
        except ValueError as exc:
            self.message(str(exc), True)
            return
        self.points = points
        self.selected_pair = max(0, len(points)-2)
        self.request_previews()

    def undo(self, _event=None):
        if self.points:
            self.points.pop()
        self.request_previews()

    def clear(self, _event=None):
        self.points = []
        self.request_previews()

    def previous_pair(self, _event=None):
        if self.pairs():
            self.selected_pair = (self.selected_pair-1) % len(self.pairs())
            self.refresh()

    def next_pair(self, _event=None):
        if self.pairs():
            self.selected_pair = (self.selected_pair+1) % len(self.pairs())
            self.refresh()

    def toggle_table(self, _event=None):
        self.show_references = not self.show_references
        self.refresh()

    def focus(self, _event=None):
        result = self.current_preview()
        if isinstance(result, dict):
            xy = np.asarray(result['reference_polyline_xy'])
        elif self.pairs():
            xy = np.asarray(self.pairs()[self.selected_pair])[:, [0, 2]]
        else:
            return
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        pad = max(.6, float((hi-lo).max())*.1)
        span = hi-lo+2*pad
        box = self.ax.get_position(original=True)
        width, height = self.fig.get_size_inches()
        ratio = box.width*width/(box.height*height)
        span[0] = max(span[0], span[1]*ratio)
        span[1] = max(span[1], span[0]/ratio)
        centre = (lo+hi)/2
        self.ax.set_xlim(centre[0]-span[0]/2, centre[0]+span[0]/2)
        self.ax.set_ylim(centre[1]-span[1]/2, centre[1]+span[1]/2)
        self.fig.canvas.draw_idle()

    def save_preview(self, _event=None):
        pairs = self.pairs()
        if not pairs or any(k not in self.previews for k in pairs):
            self.message('请先选至少两个目标，并等待本轮预览完成后保存。', True)
            return
        directory = self.preview_root/('preview_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        try:
            directory.mkdir(parents=True, exist_ok=False)
            routes = [dict(from_task=i+1, to_task=i+2,
                           error=str(self.previews[k])) if isinstance(self.previews[k], Exception)
                      else dict(from_task=i+1, to_task=i+2, route=self.previews[k])
                      for i,k in enumerate(pairs)]
            payload = dict(preview_only=True, note='相邻任务目标预览；不含启动首段，不用于下发。',
                           task_points_xyz=self.points, selected_pair=self.selected_pair+1,
                           map_id=self.planner.pm.meta['map_id'], grid_sha256=self.planner.pm.meta['grid_sha256'],
                           routes=routes)
            (directory/'preview.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            self.fig.savefig(directory/'preview.png', dpi=150)
        except (OSError, ValueError) as exc:
            self.message(f'预览保存失败：{exc}', True)
            return
        self.last_preview_directory = directory
        print(f'[PREVIEW] 选点预览 PNG / JSON：{directory}', flush=True)
        self.message('预览 PNG / JSON 已保存，完整路径见启动终端。此操作不保存/确认任务目标。')

    def confirm(self, _event=None):
        import matplotlib.pyplot as plt
        try:
            save_task_points(self.planner, self.points, self.output)
        except (ValueError, OSError) as exc:
            self.message(str(exc), True)
            return
        self.accepted = True
        self.on_close()
        plt.close(self.fig)

    def cancel(self, _event=None):
        import matplotlib.pyplot as plt
        self.accepted = False
        self.on_close()
        plt.close(self.fig)

    def on_close(self, _event=None):
        self.closed = True
        self.timer.stop()
        self.preview_worker.close()

    def home(self, _event=None):
        self.ax.set_xlim(*self.extent[:2]); self.ax.set_ylim(*self.extent[2:])
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar and toolbar.mode:
            return
        if event.button == 1:
            self.add(event.xdata, event.ydata)
        elif event.button == 2:
            self.pan = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

    def on_release(self, _event):
        self.pan = None

    def on_motion(self, event):
        if self.pan is None or event.x is None:
            return
        x, y, xlim, ylim = self.pan
        self.ax.set_xlim(np.asarray(xlim)-(event.x-x)*(xlim[1]-xlim[0])/self.ax.bbox.width)
        self.ax.set_ylim(np.asarray(ylim)-(event.y-y)*(ylim[1]-ylim[0])/self.ax.bbox.height)
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        scale = 1/1.25 if event.button == 'up' else 1.25
        xlim, ylim = np.asarray(self.ax.get_xlim()), np.asarray(self.ax.get_ylim())
        if .1 < (xlim[1]-xlim[0])*scale < 500:
            self.ax.set_xlim(event.xdata+(xlim-event.xdata)*scale)
            self.ax.set_ylim(event.ydata+(ylim-event.ydata)*scale)
            self.fig.canvas.draw_idle()

    def on_key(self, event):
        action = {'z': self.undo, 'ctrl+z': self.undo, 'backspace': self.undo, 'c': self.clear,
                  'h': self.home, 'enter': self.confirm, 'escape': self.cancel,
                  'left': self.previous_pair, 'right': self.next_pair, 'tab': self.toggle_table,
                  'f': self.focus, 'p': self.save_preview}.get(event.key)
        if action:
            action()


def pick_task_points(planner, output, initial_points=None):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    picker = TaskPointPicker(planner, output, initial_points)
    plt.show()
    return picker.points if picker.accepted else None
