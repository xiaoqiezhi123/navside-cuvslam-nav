"""Click two points, plan A*, display/export 15 safe 2D references."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import queue
import threading

import numpy as np

from map2d_data import load_planning_map, UNKNOWN, FREE, OBSTACLE
from map2d_astar import (PlanningError, check_endpoint, plan_reference_path,
                         export_coordinates)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT/'maps2d/viz_test_02/v6_reviewed'
DEFAULT_OUTPUT = ROOT/'maps2d/viz_test_02/astar_routes'


def configure_fonts():
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'SimHei']
    selected = next((name for name in preferred if name in names), 'DejaVu Sans')
    matplotlib.rcParams.update({'font.family': selected, 'axes.unicode_minus': False})


class PlannerWindow:
    def __init__(self, pm, output_root=DEFAULT_OUTPUT, include_goal=False, interactive=True):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        from matplotlib.widgets import Button, CheckButtons
        configure_fonts()
        self.pm, self.output_root = pm, Path(output_root)
        self.include_goal, self.interactive = include_goal, interactive
        self.start = self.goal = self.result = self.saved_dir = None
        self.mode = 'start'
        self.generation = 0
        self.cancel = threading.Event()
        self.mailbox = queue.Queue()
        self.closed = False
        self.artists = []
        self.pan = None
        self.fig = plt.figure(figsize=(16, 9), dpi=100)
        self.fig.canvas.manager.set_window_title('A* 地图选点 · 15 个参考点')
        self.ax = self.fig.add_axes([.05, .26, .66, .61])
        self.table_ax = self.fig.add_axes([.755, .26, .23, .61])
        self.table_ax.axis('off')
        self.fig.suptitle('A* 选点规划与 15 个参考点', x=.05, ha='left', fontsize=19)
        self.fig.text(.755, .955, f"地图：{pm.meta.get('map_version')}  |  分辨率 {pm.meta['resolution_m']:.2f} m", fontsize=10)
        known = pm.occupancy != UNKNOWN
        rows, cols = np.flatnonzero(known.any(1)), np.flatnonzero(known.any(0))
        if not len(rows):
            raise ValueError('地图没有已知区域')
        pad = int(np.ceil(.8/pm.meta['resolution_m']))
        lo = np.maximum([rows[0]-pad, cols[0]-pad], 0)
        hi = np.minimum([rows[-1]+pad+1, cols[-1]+pad+1], pm.occupancy.shape)
        region = np.s_[lo[0]:hi[0], lo[1]:hi[1]]
        occ = pm.occupancy[region]
        semantic = np.zeros(occ.shape, dtype=np.uint8)
        semantic[occ == FREE] = 1
        semantic[occ == OBSTACLE] = 2
        semantic[pm.traversable[region]] = 3
        res = pm.meta['resolution_m']
        x, y = pm.meta['origin_xz']
        self.extent = [x+lo[1]*res, x+hi[1]*res, y+lo[0]*res, y+hi[0]*res]
        colors = ['#e5e7eb', '#d8edc4', '#212121', '#489967']
        self.ax.imshow(semantic, origin='lower', extent=self.extent, interpolation='nearest',
                       cmap=ListedColormap(colors), vmin=0, vmax=3)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('x = 地图 X (m)')
        self.ax.set_ylabel('y = 地图 Z (m)')
        self.ax.grid(alpha=.13)
        self.home()
        self.fig.legend(handles=[Patch(color=c, label=t) for c, t in zip(colors,
            ['未知 / 不可通行', '自由区边缘 / 中心不可走', '墙线与障碍', '机器人中心可通行'])],
            loc='upper left', bbox_to_anchor=(.045, .925), ncol=4, frameon=False, fontsize=10)
        self.status = self.fig.text(.05, .145, '', fontsize=11, va='top', color='#263648')
        self.stats = self.fig.text(.05, .215, '', fontsize=10, va='top')
        self.fig.text(.755, .215, '坐标：全局地图 X、Z；单位：米\n表格保留 3 位小数，文件保留完整精度', fontsize=10, va='top')
        self.buttons = {}
        if interactive:
            actions = [('选起点 [S]', lambda e: self.set_mode('start')),
                       ('选终点 [G]', lambda e: self.set_mode('goal')),
                       ('交换 [X]', self.swap), ('重选 [R]', self.reset),
                       ('保存图 [P]', self.save), ('全图 [H]', self.home)]
            for i, (label, action) in enumerate(actions):
                button = Button(self.fig.add_axes([.05+i*.105, .065, .097, .045]), label)
                button.on_clicked(action)
                self.buttons[label] = button
            self.protocol = CheckButtons(self.fig.add_axes([.755, .065, .225, .065]),
                                         ['15 点包含终点'], [include_goal])
            self.protocol.on_clicked(self.change_protocol)
            self.fig.text(.05, .025, '左键依次选起点、终点；滚轮缩放；中键拖动。重选清空两点，交换会重新规划，保存图导出当前全图。', fontsize=10)
            for event, handler in [('button_press_event', self.on_press),
                                   ('button_release_event', self.on_release),
                                   ('motion_notify_event', self.on_motion),
                                   ('scroll_event', self.on_scroll),
                                   ('key_press_event', self.on_key),
                                   ('close_event', self.on_close)]:
                self.fig.canvas.mpl_connect(event, handler)
            self.timer = self.fig.canvas.new_timer(interval=100)
            self.timer.add_callback(self.poll)
            self.timer.start()
        else:
            self.fig.text(.05, .055, '蓝色虚线：A* 原始路径    橙色连线与编号：15 点参考路线    S：起点    G：终点\n所有参考点及相邻连线均已检查；浅绿、灰色和黑色区域均不可选。', fontsize=11)
        self.refresh()
        self.set_status('请在深绿色内部点击起点，再点击终点。默认 15 个中间点，不包含起终点。')

    def set_status(self, text, error=False):
        self.status.set_text(text)
        self.status.set_color('#b42318' if error else '#263648')
        self.fig.canvas.draw_idle()

    def invalidate(self):
        self.cancel.set()
        self.cancel = threading.Event()
        self.generation += 1
        self.result = self.saved_dir = None

    def refresh(self):
        import matplotlib.patheffects as pe
        for artist in self.artists:
            artist.remove()
        self.artists.clear()
        table = self.table_ax
        table.clear()
        table.axis('off')
        table.text(0, 1.0, '15 点坐标', fontsize=15, weight='bold', va='top')
        table.text(0, .93, '不含起点；' + ('第 15 点为终点' if self.include_goal else '不含终点'), fontsize=10)
        table.text(0, .87, '编号', fontsize=10)
        table.text(.29, .87, 'x / X (m)', fontsize=10)
        table.text(.66, .87, 'y / Z (m)', fontsize=10)
        if self.result:
            raw = np.asarray(self.result['astar_path_xy'])
            full = np.asarray(self.result['reference_polyline_xy'])
            refs = np.asarray(self.result['references_xy'])
            self.artists += self.ax.plot(*raw.T, '--', color='#1565d8', linewidth=1.4, zorder=3)
            self.artists += self.ax.plot(*full.T, color='#e68112', linewidth=1.8, zorder=4)
            self.artists.append(self.ax.scatter(*refs.T, s=38, c='#ffbc66', edgecolors='#733800', zorder=6))
            for i, point in enumerate(refs, 1):
                self.artists.append(self.ax.annotate(str(i), point, xytext=(4, 8 if i % 2 else -14),
                    textcoords='offset points', fontsize=9, weight='bold', zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]))
                yy = .815-(i-1)*.050
                for xx, val in [(0, f'{i:02d}'), (.29, f'{point[0]:.3f}'), (.66, f'{point[1]:.3f}')]:
                    table.text(xx, yy, val, fontsize=11, fontfamily='DejaVu Sans Mono')
            r = self.result
            self.stats.set_text(f"参考路线 {r['reference_length_m']:.2f} m  |  15 点及全部连线检查通过  |  最小保守净空 {r['min_clearance_m']:.2f} m\n"
                                f"规划 {r['planning_seconds']:.2f} s  |  蓝虚线：原始 A*；橙线：参考路线（保留转角，非全程等间距）")
        else:
            self.stats.set_text('深绿色为机器人中心可走区域；当前圆形包络半径 '
                                f"{self.pm.meta.get('robot_radius_m', 0):.2f} m")
            table.text(0, .77, '选好两点后自动规划', fontsize=11, color='#666666')
        for point, label, color, marker in [(self.start, 'S 起点', '#006a39', 'o'),
                                             (self.goal, 'G 终点', '#bb1555', 'X')]:
            if point is not None:
                self.artists.append(self.ax.scatter(*point, s=95, c=color, marker=marker,
                                                     edgecolors='white', linewidths=1.2, zorder=9))
                self.artists.append(self.ax.annotate(label, point, xytext=(7, -25),
                    textcoords='offset points', color=color, fontsize=10, weight='bold', zorder=10,
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')]))
        self.fig.canvas.draw_idle()

    def set_mode(self, mode):
        self.mode = mode
        self.set_status('请点击新的' + ('起点。' if mode == 'start' else '终点。'))

    def select(self, point):
        try:
            point = check_endpoint(self.pm, point, '起点' if self.mode == 'start' else '终点')
        except PlanningError as exc:
            self.set_status(str(exc), True)
            return
        self.invalidate()
        if self.mode == 'start':
            self.start = point
            self.mode = 'goal'
        else:
            self.goal = point
        self.refresh()
        if self.start is not None and self.goal is not None:
            self.request_plan()
        else:
            self.set_status('起点已选，请点击终点。' if self.start is not None else '终点已选，请点击“选起点”。')

    def request_plan(self):
        self.invalidate()
        self.refresh()
        generation, cancel = self.generation, self.cancel
        start, goal, include_goal = self.start, self.goal, self.include_goal
        self.set_status('正在后台规划 A* 与 15 点路线，可继续缩放或重新选点……')

        def work():
            try:
                result = plan_reference_path(self.pm, start, goal, include_goal=include_goal, cancel=cancel)
            except Exception as exc:
                result = exc
            self.mailbox.put((generation, result))
        threading.Thread(target=work, daemon=True, name='map2d-astar').start()

    def poll(self):
        if self.closed:
            return
        while True:
            try:
                generation, result = self.mailbox.get_nowait()
            except queue.Empty:
                break
            if generation != self.generation:
                continue
            if isinstance(result, Exception):
                self.set_status(f'规划失败：{result}', True)
                continue
            self.result = result
            self.refresh()
            print_coordinates(result)
            try:
                name = datetime.now().strftime('route_%Y%m%d_%H%M%S_%f')
                self.saved_dir = export_coordinates(result, self.output_root/name)
                self.set_status(f'规划完成，坐标已自动保存：{self.saved_dir.name}\nJSON / CSV：{self.output_root}；点击“保存图”输出 PNG。')
            except OSError as exc:
                self.set_status(f'规划完成，但坐标保存失败：{exc}', True)

    def change_protocol(self, _event):
        self.include_goal = bool(self.protocol.get_status()[0])
        if self.start is not None and self.goal is not None:
            self.request_plan()
        else:
            self.refresh()

    def reset(self, _event=None):
        self.invalidate()
        self.start = self.goal = None
        self.mode = 'start'
        self.refresh()
        self.set_status('已清空，请依次点击起点、终点。')

    def swap(self, _event=None):
        if self.start is None or self.goal is None:
            self.set_status('交换需要先选好起点和终点。', True)
            return
        self.start, self.goal = self.goal, self.start
        self.request_plan()

    def save(self, _event=None):
        if self.result is None:
            self.set_status('当前没有成功的规划结果，请先选好两点。', True)
            return
        try:
            if self.saved_dir is None:
                name = datetime.now().strftime('route_%Y%m%d_%H%M%S_%f')
                self.saved_dir = export_coordinates(self.result, self.output_root/name)
            self.save_figure(self.saved_dir/'route.png')
            self.set_status(f'PNG / CSV / JSON 已保存：{self.saved_dir.name}\n目录：{self.output_root}')
            print(f'路线文件：{self.saved_dir}', flush=True)
        except OSError as exc:
            self.set_status(f'保存失败：{exc}', True)

    def save_figure(self, path):
        # Render a clean full-map figure so zoom/pan never clips the exported route.
        import matplotlib.pyplot as plt
        figure = PlannerWindow(self.pm, include_goal=self.include_goal, interactive=False)
        figure.start, figure.goal, figure.result = self.start, self.goal, self.result
        figure.refresh()
        figure.set_status(f'起点 S = ({self.start[0]:.3f}, {self.start[1]:.3f})    '
                          f'终点 G = ({self.goal[0]:.3f}, {self.goal[1]:.3f})\n'
                          '精确坐标见 reference_points.csv；地图身份及完整路径见 route.json。')
        try:
            figure.fig.savefig(path, dpi=140)
        finally:
            plt.close(figure.fig)

    def home(self, _event=None):
        self.ax.set_xlim(*self.extent[:2])
        self.ax.set_ylim(*self.extent[2:])
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar and toolbar.mode:
            return
        if event.button == 1:
            self.select((event.xdata, event.ydata))
        elif event.button == 2:
            self.pan = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

    def on_release(self, _event):
        self.pan = None

    def on_motion(self, event):
        if self.pan is None or event.x is None:
            return
        x, y, xlim, ylim = self.pan
        dx = (event.x-x)*(xlim[1]-xlim[0])/self.ax.bbox.width
        dy = (event.y-y)*(ylim[1]-ylim[0])/self.ax.bbox.height
        self.ax.set_xlim(np.asarray(xlim)-dx)
        self.ax.set_ylim(np.asarray(ylim)-dy)
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        scale = 1/1.25 if event.button == 'up' else 1.25
        xlim, ylim = np.asarray(self.ax.get_xlim()), np.asarray(self.ax.get_ylim())
        if not .1 < (xlim[1]-xlim[0])*scale < 500:
            return
        self.ax.set_xlim(event.xdata+(xlim-event.xdata)*scale)
        self.ax.set_ylim(event.ydata+(ylim-event.ydata)*scale)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        actions = {'s': lambda: self.set_mode('start'), 'g': lambda: self.set_mode('goal'),
                   'x': self.swap, 'r': self.reset, 'escape': self.reset, 'h': self.home, 'p': self.save}
        if event.key in actions:
            actions[event.key]()

    def on_close(self, _event=None):
        self.closed = True
        self.cancel.set()
        self.generation += 1
        if hasattr(self, 'timer'):
            self.timer.stop()


def print_coordinates(result):
    print(f"\n规划成功：{result['reference_length_m']:.3f} m，{result['planning_seconds']:.3f} s")
    print('坐标 x=地图 X，y=地图 Z，单位 m；不含起点；'+('包含终点' if result['includes_goal'] else '不含终点'))
    print('编号           x           y')
    for i, (x, y) in enumerate(result['references_xy'], 1):
        print(f'{i:02d}    {x:11.6f} {y:11.6f}')
    print(f"S={result['start_xy']}  G={result['goal_xy']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, default=DEFAULT_MAP)
    parser.add_argument('--start', nargs=2, type=float, metavar=('X', 'Y'), help='二维 x=X地图、y=Z地图，单位米')
    parser.add_argument('--goal', nargs=2, type=float, metavar=('X', 'Y'))
    parser.add_argument('--include-goal', action='store_true', help='15 点包含终点；默认是 15 个中间点')
    parser.add_argument('--output', type=Path, help='无窗口模式为新结果目录；GUI 模式为结果父目录')
    parser.add_argument('--no-gui', action='store_true', help='输入坐标，保存 PNG / CSV / JSON，不打开窗口')
    args = parser.parse_args()
    if (args.start is None) != (args.goal is None) or (args.no_gui and args.start is None):
        parser.error('--start 与 --goal 必须同时提供；--no-gui 模式必须提供两点')
    import matplotlib
    matplotlib.use('Agg' if args.no_gui else 'TkAgg')
    import matplotlib.pyplot as plt
    try:
        pm = load_planning_map(args.map)
        ui = PlannerWindow(pm, output_root=args.output or DEFAULT_OUTPUT,
                           include_goal=args.include_goal, interactive=not args.no_gui)
        if args.start is not None:
            ui.start, ui.goal = args.start, args.goal
            if args.no_gui:
                result = plan_reference_path(pm, args.start, args.goal, include_goal=args.include_goal)
                ui.result = result
                output = args.output or DEFAULT_OUTPUT/datetime.now().strftime('route_%Y%m%d_%H%M%S_%f')
                export_coordinates(result, output)
                ui.save_figure(output/'route.png')
                print_coordinates(result)
                print(f'已保存：{output.resolve()}')
                plt.close(ui.fig)
                return 0
            ui.request_plan()
        if not args.no_gui:
            plt.show()
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'错误：{exc}')
        if not args.no_gui:
            print('交互窗口请在本机图形桌面终端启动；无显示器时可用 --no-gui --start X Y --goal X Y。')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
