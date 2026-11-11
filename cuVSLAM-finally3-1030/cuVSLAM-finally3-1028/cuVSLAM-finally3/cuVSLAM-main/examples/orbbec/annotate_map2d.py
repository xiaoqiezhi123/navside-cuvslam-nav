# -*- coding: utf-8 -*-
"""annotate_map2d.py —— 二维规划地图人工标注工具（阶段 3，纯鼠标操作）

性能设计：概览按屏幕像素聚合，局部放大显示原始点；橡皮擦使用 KDTree，
笔刷用 blit 局部刷新。仅显示聚合，保存/擦除/规划始终使用原始点和米制坐标。
可按全局地图 Y 做显示切片；被切片隐藏的点不参与擦除，原始点和规划参数不变。

操作：
  Free 模式：左键点矩形两个对角点，第二点自动完成；右键取消第一点
  左键 = 加顶点 / 右键 = 闭合多边形（Obstacle/Unknown/RemoveCand 模式）
  WallSeg 模式：左键点两点 = 一段墙线；右键取消起点
  Erase 模式：按住左键拖动擦除画刷内特征点；右键结束
  Select 模式：左键选中标注，Delete 删除；右键取消选择；重复点击轮换重叠标注
  中键拖动 = 平移；滚轮 = 缩放
  键盘：1–7 切模式，z/Backspace 撤销顶点或墙起点，x 删除选中项或本类末项，w 保存，q 退出，
        [ / ] 缩小/放大橡皮擦半径（0.05~3.0m），也可使用滑块/输入框/加减按钮
底部中英双语按钮、悬停说明、? 帮助；无中文字体时回退英文。顶点保存为米制 (x,z)。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from map2d_display import PointCloudIndex


class Annotator:
    MODES = [("Free", "free", "#2ca02c"), ("Obstacle", "obstacle", "#111111"),
             ("Unknown", "unknown", "#1f77b4"), ("RemoveCand", "candidate_remove", "#ff7f0e"),
             ("WallSeg", "wallseg", "#444444"), ("Erase", "erase", "#cc0000"),
             ("Select", "select", "#80deea")]
    MODE_KEYS = {"1": "free", "2": "obstacle", "3": "unknown",
                 "4": "candidate_remove", "5": "wallseg", "6": "erase", "7": "select"}
    COLORS = {mode: color for _, mode, color in MODES}
    POLYGON_KEYS = ("free", "obstacle", "unknown", "candidate_remove")
    ERASE_RADIUS_M = 0.3
    ERASE_RADIUS_MIN = 0.05
    ERASE_RADIUS_MAX = 3.0
    LABELS_ZH = {"free": "自由区", "obstacle": "障碍区", "unknown": "未知区",
                 "candidate_remove": "剔除候选", "wallseg": "墙线段", "erase": "擦除特征点",
                 "select": "选择标注", "DeleteSel": "删除选中",
                 "UndoPt": "撤销顶点", "DelPoly": "删除本类末项", "Undo": "撤销操作",
                 "Redo": "重做操作", "Home": "返回全图", "Save": "保存标注", "Help": "操作说明"}
    HELP_ZH = {
        "free": "Free：点两个对角点生成 X/Z 轴对齐矩形；第二点完成，移动鼠标预览，右键 / z 取消第一点。",
        "obstacle": "Obstacle：圈出墙体或固定障碍；左键加点、右键闭合。优先级最高。",
        "unknown": "Unknown：圈出未核验区域；左键加点、右键闭合。规划时禁止进入。",
        "candidate_remove": "RemoveCand：圈选剔除自动候选障碍的范围；不会删除人工障碍，也不会自动变成自由区。",
        "wallseg": "WallSeg：左键依次点两个端点，补一段墙；右键取消起点。",
        "erase": "Erase：拖动擦除当前高度层的特征点；隐藏高度的点保留。下方调半径，Ctrl+Z 撤销。",
        "select": "Select / 7：左键选中区域或墙线（青色高亮），Delete 删除；重复点击轮换重叠项，右键取消选择。",
        "DeleteSel": "Delete / 删除选中：删除青色高亮的标注；Ctrl+Z 恢复，Ctrl+Y 重做。可删除重新载入的标注。",
        "UndoPt": "UndoPt / z / Backspace：撤销多边形末顶点，或取消自由区第一点、墙线段起点。",
        "DelPoly": "DelLast：删除当前绘制类型的最后一项；x 优先删选中项，否则删本类末项。Ctrl+Z 恢复。",
        "Undo": "Undo / Ctrl+Z：撤销一次完整标注、删除或擦除操作；最多保留 100 次。",
        "Redo": "Redo / Ctrl+Y：恢复刚撤销的操作；开始新编辑后清空重做记录。",
        "Home": "Home / h：适配当前高度层和已完成标注的范围；不改变标注。",
        "Save": "Save / w：保存已完成的标注和擦除记录；尚未完成的自由区、多边形或墙线不保存。",
        "Help": "Help / ?：打开完整按钮说明和快捷键。中键拖动平移，滚轮缩放，Esc 取消当前绘制。",
        "radius": "擦除半径单位为米（不是直径）；范围 0.05–3.00m。拖动滑块、输入数字后回车，或点击 − / +。",
        "height": "高度切片：填写地图地面 Y、相对高度下限/上限，点 Apply 或回车。高度=地面 Y−点 Y；仅控制显示。",
        "candidates": "红点：开关候选障碍着色；灰色底图仍受高度切片限制，不改变最终地图的障碍判定。",
    }
    HELP_EN = {
        "free": "Free: click two opposite corners of an XZ-aligned rectangle. Move to preview; second click completes. Right / z cancels.",
        "obstacle": "Obstacle: mark walls or fixed obstacles. Left click adds vertices; right click closes. Highest priority.",
        "unknown": "Unknown: mark unverified regions. These remain blocked for planning.",
        "candidate_remove": "RemoveCand: remove automatic candidates in the polygon. Does not erase manual obstacles or declare free space.",
        "wallseg": "WallSeg: two left clicks add a wall segment; right click cancels the first endpoint.",
        "erase": "Erase: drag to erase points in the displayed height slice. Hidden heights are preserved. Ctrl+Z restores a stroke.",
        "select": "Select / 7: click an area or wall (cyan highlight), then Delete. Repeated clicks cycle overlaps; right click clears.",
        "DeleteSel": "Delete: remove the highlighted annotation, including loaded items. Ctrl+Z restores it; Ctrl+Y redoes it.",
        "UndoPt": "UndoPt / z / Backspace: remove an unfinished polygon vertex, free rectangle corner, or wall start.",
        "DelPoly": "DelLast: delete the last item of the active drawing type. x prefers the selection, otherwise this type. Undo restores it.",
        "Undo": "Undo / Ctrl+Z: undo one completed annotation, deletion or eraser stroke (up to 100 actions).",
        "Redo": "Redo / Ctrl+Y: redo an undone action; a new edit clears the redo history.",
        "Home": "Home / h: fit the current height slice and completed annotations without editing them.",
        "Save": "Save / w: save completed annotations and erased IDs. Unfinished rectangles, polygons and walls are not saved.",
        "Help": "Help / ?: show controls. Middle drag pans, wheel zooms, Esc cancels unfinished drawing.",
        "radius": "Eraser radius (not diameter), 0.05–3.00 metres. Drag slider, enter a number and press Enter, or use - / +.",
        "height": "Slice: enter map ground Y and relative height limits, then Apply / Enter. Height = ground Y - point Y. Display only.",
        "candidates": "Red: toggle candidate coloring within the display slice. Does not change the map builder's obstacle policy.",
    }

    def __init__(self, pts: np.ndarray, candidates: np.ndarray | None,
                 annotations: dict, out_path: Path, landmark_ids=None,
                 cand_ids=None, backend="TkAgg", display_mode="auto", erase_radius_m=0.3,
                 display_ground_y=None, display_height_min=-0.15, display_height_max=0.15,
                 show_candidates=True):
        import matplotlib
        matplotlib.use(backend)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider, TextBox, CheckButtons
        from matplotlib import font_manager

        cjk_fonts = [f.name for f in font_manager.fontManager.ttflist
                     if "CJK" in f.name or "WenQuanYi" in f.name]
        self._ui_font = cjk_fonts[0] if cjk_fonts else "DejaVu Sans"
        self._has_cjk = bool(cjk_fonts)
        self._help_fig = None
        self._hover_key = None
        self._syncing_radius = False
        self._syncing_height = False
        self.display_ground_y = display_ground_y
        self.display_height_min = display_height_min
        self.display_height_max = display_height_max
        self.height_filter_enabled = display_ground_y is not None
        self.show_candidates = show_candidates
        if not np.isfinite(erase_radius_m) or not self.ERASE_RADIUS_MIN <= erase_radius_m <= self.ERASE_RADIUS_MAX:
            raise ValueError("擦除半径必须在 0.05–3.00m 之间")
        self.ERASE_RADIUS_M = round(float(erase_radius_m), 2)

        matplotlib.rcParams["keymap.fullscreen"] = ""
        matplotlib.rcParams["keymap.save"] = ""
        matplotlib.rcParams["keymap.quit"] = ""

        self.pts = pts
        self.candidates = candidates
        self.annotations = annotations
        for key in (*self.POLYGON_KEYS, "obstacle_segments"):
            self.annotations.setdefault(key, [])
        self.out_path = out_path
        self.mode = "free"
        self.current: list = []
        self.current_line = None
        self.free_preview = None
        self.vertex_labels: list = []
        self.poly_artists = []
        self.selected = None
        self.selection_artist = None
        self._selection_hits = []
        self._selection_click_px = None
        self.wall_start = None
        self.wall_start_marker = None
        self._pan = None
        self._pan_image = None
        self._pan_shift = (0., 0.)
        self.landmark_ids = (landmark_ids if landmark_ids is not None
                             else np.arange(len(pts)))
        self.cand_ids = (cand_ids if cand_ids is not None
                         else np.arange(len(candidates))
                         if candidates is not None else np.array([], dtype=np.int64))
        self.erased = set(annotations.get("erased_point_ids", []))
        self._erasing = False
        self.brush = None
        self._last_redraw = 0.0
        self.display_mode = display_mode
        self.index = PointCloudIndex(pts, self.landmark_ids, self.cand_ids)
        self.index.set_height_filter(display_ground_y, display_height_min, display_height_max)
        self.index.set_erased(self.erased)
        self._background = None
        self._frame_background = None
        self._capturing_frame = False
        self._updating_view = False
        self._view_pending = False
        self._history, self._redo_history = [], []
        self._erase_before = None
        self._stroke_last = None

        self.fig, self.ax = plt.subplots(figsize=(13, 11))
        plt.subplots_adjust(bottom=0.36)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Z (m)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_autoscale_on(False)
        self._home_limits = self._height_home_limits()
        self.ax.set_xlim(self._home_limits[0]); self.ax.set_ylim(self._home_limits[1])
        self.sc_pts = self.ax.scatter([], [], s=1, c="0.4", linewidths=0, zorder=1)
        self.sc_cand = self.ax.scatter([], [], s=3, c="tab:red", linewidths=0, zorder=2)
        self.pixel_pts, = self.ax.plot([], [], linestyle="", marker=",", color="0.35", zorder=1)
        self.pixel_cand, = self.ax.plot([], [], linestyle="", marker=",", color="tab:red", zorder=2)
        self.point_image = self.ax.imshow(np.ones((2,2,3)), origin="lower",
                                         interpolation="nearest", zorder=1,
                                         extent=[*self._home_limits[0], *self._home_limits[1]])
        self._view_timer = self.fig.canvas.new_timer(interval=33)
        self._view_timer.single_shot = True
        self._view_timer.add_callback(self._flush_view)
        self.ax.callbacks.connect("xlim_changed", self._queue_view)
        self.ax.callbacks.connect("ylim_changed", self._queue_view)
        self.fig.canvas.mpl_connect("resize_event", self._queue_view)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self._rebuild_points()
        self._redraw_existing()
        self._update_title()

        # ---- Two rows of labelled actions, a radius row, and contextual help ----
        self.buttons = {}
        self._control_help = {}
        # Display-only controls are deliberately separate from obstacle thresholds.
        slice_ax = self.fig.add_axes([0.02, 0.258, 0.17, 0.047])
        self.height_check = CheckButtons(slice_ax, ["Slice / 高度切片" if self._has_cjk else "Height slice"],
                                         [self.height_filter_enabled])
        self.height_check.labels[0].set_fontfamily(self._ui_font)
        self.height_check.labels[0].set_fontsize(10)
        self.height_check.on_clicked(self._toggle_height_filter)
        self._control_help[slice_ax] = "height"
        self.height_boxes = {}
        for key, x, title, value in (
                ("ground", .26, "地面 Y (m)" if self._has_cjk else "Ground Y (m)", display_ground_y),
                ("min", .43, "高度下限 (m)" if self._has_cjk else "Min height (m)", display_height_min),
                ("max", .60, "高度上限 (m)" if self._has_cjk else "Max height (m)", display_height_max)):
            self.fig.text(x, .307, title, fontsize=10, fontfamily=self._ui_font)
            bx = self.fig.add_axes([x, .26, .095, .038])
            box = TextBox(bx, "", initial="" if value is None else f"{value:g}", textalignment="center")
            box.drawon = False
            box.on_submit(lambda _text: self._apply_height_controls())
            self.height_boxes[key] = box
            self._control_help[bx] = "height"
        apply_ax = self.fig.add_axes([.735, .26, .07, .038])
        self.buttons["HeightApply"] = Button(apply_ax, "Apply\n应用" if self._has_cjk else "Apply")
        self.buttons["HeightApply"].label.set_fontfamily(self._ui_font)
        self.buttons["HeightApply"].label.set_fontsize(10)
        self.buttons["HeightApply"].on_clicked(lambda _event: self._apply_height_controls())
        self._control_help[apply_ax] = "height"
        red_ax = self.fig.add_axes([.83, .258, .15, .047])
        self.candidate_check = CheckButtons(red_ax, ["Red / 红点" if self._has_cjk else "Red points"],
                                            [show_candidates])
        self.candidate_check.labels[0].set_fontfamily(self._ui_font)
        self.candidate_check.labels[0].set_fontsize(10)
        self.candidate_check.on_clicked(self._toggle_candidates)
        self._control_help[red_ax] = "candidates"
        util_actions = [("UndoPt", self._undo_point), ("DeleteSel", self._delete_selected), ("DelPoly", self._del_poly),
                        ("Undo", self._undo), ("Redo", self._redo),
                        ("Home", self._home), ("Save", self._save), ("Help", self._show_help)]
        w = 0.96 / len(self.MODES)
        for i, (name, mode, color) in enumerate(self.MODES):
            bx = self.fig.add_axes([0.02 + i * w, 0.20, w - 0.008, 0.045])
            label = f"{name} ({i+1})" + (f"\n{self.LABELS_ZH[mode]}" if self._has_cjk else "")
            self.buttons[mode] = Button(bx, label,
                                        color=color, hovercolor="#cccccc")
            self.buttons[mode].label.set_fontsize(10)
            self.buttons[mode].label.set_fontfamily(self._ui_font)
            if mode in ("obstacle", "wallseg"):
                self.buttons[mode].label.set_color("white")
            self.buttons[mode].on_clicked(lambda _e, m=mode: self._set_mode(m))
            self._control_help[bx] = mode
        w = 0.96 / len(util_actions)
        for j, (label, fn) in enumerate(util_actions):
            bx = self.fig.add_axes([0.02 + j * w, 0.14, w - 0.008, 0.045])
            name = {"DeleteSel": "Delete", "DelPoly": "DelLast"}.get(label, label)
            caption = name + (f"\n{self.LABELS_ZH[label]}" if self._has_cjk else "")
            self.buttons[label] = Button(bx, caption, color="#eeeeee",
                                         hovercolor="#cccccc")
            self.buttons[label].label.set_fontfamily(self._ui_font)
            self.buttons[label].label.set_fontsize(10)
            self.buttons[label].on_clicked(lambda _e, f=fn: f())
            self._control_help[bx] = label

        slider_ax = self.fig.add_axes([0.17, 0.087, 0.44, 0.022])
        self.radius_slider = Slider(slider_ax, "Radius (m)",
                                    self.ERASE_RADIUS_MIN, self.ERASE_RADIUS_MAX,
                                    valinit=self.ERASE_RADIUS_M, valstep=0.01, valfmt="%.2f")
        self.radius_slider.label.set_fontfamily(self._ui_font)
        if self._has_cjk:
            self.radius_slider.label.set_text("擦除半径 (m)")
        self.radius_slider.drawon = False
        self.radius_slider.on_changed(self._set_erase_radius)
        box_ax = self.fig.add_axes([0.79, 0.077, 0.085, 0.038])
        self.radius_box = TextBox(box_ax, "", initial=f"{self.ERASE_RADIUS_M:.2f}", textalignment="center")
        self.radius_box.drawon = False
        self.radius_box.on_submit(self._submit_radius)
        for label, x, factor in (("Radius-", 0.69, 1/1.5), ("Radius+", 0.91, 1.5)):
            bx = self.fig.add_axes([x, 0.077, 0.055, 0.038])
            self.buttons[label] = Button(bx, "−" if factor < 1 else "+", color="#eeeeee")
            self.buttons[label].on_clicked(lambda _e, f=factor: self._scale_erase_radius(f))
            self._control_help[bx] = "radius"
        self._control_help[slider_ax] = self._control_help[box_ax] = "radius"
        self.help_text = self.fig.text(0.025, 0.035, "", va="center", fontsize=10,
                                       fontfamily=self._ui_font, color="#333333")
        self._set_help_text(self.mode)
        self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_control_hover)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        print("操作: Free左键两次画矩形; 其他多边形左键加点/右键闭合; 中键平移, 滚轮缩放; Erase左键擦除,"
              " [/] 调半径; 1-7切模式 z撤销顶点/起点 Select选中后Delete删除 Ctrl+Z/Y撤销/重做 w保存 q退出")

    def _apply_height_controls(self):
        if self._syncing_height:
            return
        try:
            ground_y, low, high = (float(self.height_boxes[k].text) for k in ("ground", "min", "max"))
            if not np.all(np.isfinite([ground_y, low, high])) or low > high:
                raise ValueError
        except (TypeError, ValueError):
            self._sync_height_check()
            self._set_help_text(message=("高度输入无效：请填写有限的地面 Y，且高度下限 ≤ 上限。" if self._has_cjk else
                                         "Invalid height: enter finite ground Y and min <= max."))
            return
        self._finish_erase()
        self.display_ground_y, self.display_height_min, self.display_height_max = ground_y, low, high
        self.height_filter_enabled = True
        self.index.set_height_filter(ground_y, low, high)
        self._sync_height_check()
        self._flush_view()
        self._update_title()
        count = np.count_nonzero(self.index.alive & self.index.display_mask)
        self._set_help_text(message=(f"当前高度层共 {count} 点；Home 适配范围。隐藏点保留；生成地图的障碍参数不变。" if self._has_cjk else
                                     f"Slice: {count} points. Home fits the slice. Hidden points and builder settings are preserved."))

    def _sync_height_check(self):
        self._syncing_height = True
        try:
            if self.height_check.get_status()[0] != self.height_filter_enabled:
                self.height_check.set_active(0)
        finally:
            self._syncing_height = False

    def _toggle_height_filter(self, _label):
        if self._syncing_height:
            return
        if self.height_check.get_status()[0]:
            self._apply_height_controls()
        else:
            self._finish_erase()
            self.height_filter_enabled = False
            self.index.set_height_filter()
            self._flush_view()
            self._update_title()
            self._set_help_text("height")

    def _toggle_candidates(self, _label):
        self.show_candidates = self.candidate_check.get_status()[0]
        self._flush_view()
        self._set_help_text("candidates")

    def _set_help_text(self, key=None, message=None):
        if not hasattr(self, "help_text"):
            return
        help_map = self.HELP_ZH if self._has_cjk else self.HELP_EN
        text = message or help_map.get(key, help_map[self.mode])
        if self.help_text.get_text() != text:
            self.help_text.set_text(text)
            self.fig.canvas.draw_idle()

    def _on_control_hover(self, event):
        key = self._control_help.get(event.inaxes, self.mode)
        if key != self._hover_key:
            self._hover_key = key
            self._set_help_text(key)

    def _set_erase_radius(self, value):
        if self._syncing_radius:
            return
        try:
            radius = float(value)
            if not np.isfinite(radius) or not self.ERASE_RADIUS_MIN <= radius <= self.ERASE_RADIUS_MAX:
                raise ValueError
        except (TypeError, ValueError):
            self._sync_radius_controls()
            self._set_help_text(message=("半径输入无效，请填写 0.05–3.00 米。" if self._has_cjk else
                                         "Invalid radius: enter 0.05–3.00 metres."))
            return
        self.ERASE_RADIUS_M = round(radius, 2)
        self._stroke_last = None
        if self.brush is not None:
            self.brush.set_radius(self.ERASE_RADIUS_M)
        self._sync_radius_controls()
        self._set_help_text("radius")
        self._update_title()

    def _sync_radius_controls(self):
        self._syncing_radius = True
        try:
            if hasattr(self, "radius_slider"):
                self.radius_slider.set_val(self.ERASE_RADIUS_M)
            if hasattr(self, "radius_box"):
                self.radius_box.set_val(f"{self.ERASE_RADIUS_M:.2f}")
        finally:
            self._syncing_radius = False

    def _submit_radius(self, text):
        self._set_erase_radius(text)

    def _scale_erase_radius(self, factor):
        self._set_erase_radius(np.clip(self.ERASE_RADIUS_M * factor,
                                      self.ERASE_RADIUS_MIN, self.ERASE_RADIUS_MAX))

    def _show_help(self):
        import matplotlib.pyplot as plt
        if self._help_fig is None or not plt.fignum_exists(self._help_fig.number):
            self._help_fig = plt.figure(figsize=(12, 9))
            self._help_fig.canvas.manager.set_window_title("Map2D controls")
            title = "二维地图标注：按钮与操作说明" if self._has_cjk else "Map2D controls"
            self._help_fig.text(.035, .95, title, fontsize=16, fontfamily=self._ui_font)
            help_map = self.HELP_ZH if self._has_cjk else self.HELP_EN
            for i, key in enumerate([m[1] for m in self.MODES] +
                                    ["UndoPt", "DeleteSel", "DelPoly", "Undo", "Redo", "Home", "Save", "Help", "radius", "height", "candidates"]):
                self._help_fig.text(.035, .89-i*.043, help_map[key], fontsize=10,
                                    va="top", fontfamily=self._ui_font, wrap=True)
            note = ("灰点＝当前显示高度层的特征；红点＝候选障碍。切片不识别黄色线；空白不代表自由。\n"
                    "自由区点两个对角点；其他多边形右键闭合。中键平移，滚轮缩放；w 保存，q 保存并退出。" if self._has_cjk else
                    "Grey = features in the display slice; red = candidates. No yellow-line recognition. Blank is not free.\n"
                    "Free: two corners. Other polygons: right click closes. Middle pans, wheel zooms; w saves, q exits.")
            self._help_fig.text(.035,.06,note,fontsize=10,fontfamily=self._ui_font,color="#555555")
        self._help_fig.canvas.draw_idle()
        if "agg" != str(plt.get_backend()).lower():
            self._help_fig.show()

    # ------------------------------------------------------------------ 底图
    def _rebuild_points(self):
        """Update erasures without resetting the user's zoom/pan."""
        self.index.set_erased(self.erased)
        self._refresh_view()

    def _queue_view(self, *events):
        if self._updating_view:
            return
        if events and getattr(events[0], "name", None) == "resize_event":
            self._frame_background = None
        self._background = None
        if not self._view_pending:
            self._view_pending = True
            self._view_timer.start()

    def _flush_view(self):
        self._view_timer.stop()
        self._view_pending = False
        self._refresh_view()

    def _refresh_view(self):
        self._updating_view = True
        try:
            xl, zl = self.ax.get_xlim(), self.ax.get_ylim()
            idx = self.index.visible(xl, zl)
            raster = self.display_mode == "density"
            self.display_aggregated = self.display_mode == "auto" and len(idx) > 12000
            self.point_image.set_visible(raster)
            self.sc_pts.set_visible(not raster and not self.display_aggregated)
            self.sc_cand.set_visible(not raster and not self.display_aggregated)
            self.pixel_pts.set_visible(self.display_aggregated)
            self.pixel_cand.set_visible(self.display_aggregated)
            width = max(64, min(1200, int(self.ax.bbox.width)))
            height = max(64, min(1200, int(self.ax.bbox.height)))
            if raster:
                self.point_image.set_data(self.index.raster(idx, xl, zl, width, height, self.show_candidates))
                self.point_image.set_extent([*xl, *zl])
            elif self.display_aggregated:
                grey, red = self.index.screen_indices(idx, xl, zl, width, height, self.show_candidates)
                self.pixel_pts.set_data(self.index.xz[grey].T)
                self.pixel_cand.set_data(self.index.xz[red].T)
            else:
                self.sc_pts.set_offsets(self.index.xz[idx])
                red = idx[self.index.candidate[idx]] if self.show_candidates else np.array([], dtype=int)
                self.sc_cand.set_offsets(self.index.xz[red])
            self._background = None
            self._draw_view()
        finally:
            self._updating_view = False

    def _on_draw(self, event):
        if self._capturing_frame:
            return
        self._frame_background = None
        if self.fig.canvas.supports_blit:
            self._background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self._draw_transient_artists()

    def _draw_view(self):
        """Redraw axes/ticks from a cached frame without repainting all buttons."""
        canvas = self.fig.canvas
        if not canvas.supports_blit:
            canvas.draw_idle()
            return
        if self._frame_background is None:
            self._capturing_frame = True
            self.ax.set_visible(False)
            try:
                canvas.draw()
                self._frame_background = canvas.copy_from_bbox(self.fig.bbox)
            finally:
                self.ax.set_visible(True)
                self._capturing_frame = False
        canvas.restore_region(self._frame_background)
        self.ax.draw(canvas.get_renderer())
        self._background = canvas.copy_from_bbox(self.ax.bbox)
        self._draw_transient_artists()
        canvas.blit(self.fig.bbox)

    def _draw_transient_artists(self):
        for artist in (self.brush, self.free_preview):
            if artist is not None and artist.get_visible():
                self.ax.draw_artist(artist)

    def _draw_brush(self):
        """Refresh the eraser or rectangle preview without redrawing the point cloud."""
        if self._background is not None and self.fig.canvas.supports_blit:
            self.fig.canvas.restore_region(self._background)
            self._draw_transient_artists()
            self.fig.canvas.blit(self.ax.bbox)
        else:
            self.fig.canvas.draw_idle()

    def _throttled_redraw(self):
        self._queue_view()

    def _height_home_limits(self):
        selected = self.index.xz[self.index.alive & self.index.display_mask]
        vertices = [v for key in (*self.POLYGON_KEYS, "obstacle_segments")
                    for shape in self.annotations.get(key, []) for v in shape]
        if vertices:
            selected = np.vstack((selected, np.asarray(vertices)))
        if not len(selected):
            selected = self.index.xz
        low, high = selected.min(axis=0), selected.max(axis=0)
        pad = np.maximum((high-low)*0.02, 0.1)
        return ((low[0]-pad[0], high[0]+pad[0]), (low[1]-pad[1], high[1]+pad[1]))

    def _home(self):
        self._home_limits = self._height_home_limits()
        self.ax.set_xlim(self._home_limits[0])
        self.ax.set_ylim(self._home_limits[1])
        self._flush_view()

    def _set_mode(self, m):
        self._finish_erase()
        self._clear_selection()
        self._clear_current()
        self._cancel_wall_start()
        self.mode = m
        if self.brush is not None:
            self.brush.set_visible(False)
        self._set_help_text(m)
        self._update_title()

    def _update_title(self):
        r = self.ERASE_RADIUS_M
        count = np.count_nonzero(self.index.alive & self.index.display_mask)
        layer = (f"height=[{self.display_height_min:g}, {self.display_height_max:g}]m, ground Y={self.display_ground_y:g}m"
                 if self.height_filter_enabled else "all heights")
        selected = f" | selected={self.selected[0]} #{self.selected[1]+1}" if self.selected else ""
        hint = {"select": "left=select / right=clear; Delete=remove",
                "free": "left=two opposite corners / right=cancel; z=undo corner",
                "wallseg": "left=endpoint / right=cancel; z=undo start",
                "erase": "left drag=erase; [/] brush radius"}.get(
                    self.mode, "left=vertex / right=close; z=undo vertex")
        self.ax.set_title(
            f"MODE: {self.mode}   erased={len(self.erased)}   brush={r:.2f}m"
            f"   ({hint})\n"
            f"Display: {count} points | {layer}{selected}",
            fontsize=10)
        self.fig.canvas.draw_idle()

    def _redraw_existing(self):
        from matplotlib.patches import Polygon
        self._clear_selection()
        for artist, _, _ in self.poly_artists:
            artist.remove()
        self.poly_artists.clear()
        for mode in (*self.POLYGON_KEYS, "obstacle_segments"):
            polys = self.annotations.get(mode, [])
            if mode == "obstacle_segments":
                for index, seg in enumerate(polys):
                    (line,) = self.ax.plot([seg[0][0], seg[1][0]],
                                           [seg[0][1], seg[1][1]],
                                           color="#111111", linewidth=5,
                                           solid_capstyle="round", zorder=3)
                    self.poly_artists.append((line, "obstacle_segments", index))
                continue
            for index, v in enumerate(polys):
                arr = np.asarray(v)
                color = self.COLORS.get(mode, "gray")
                patch = self.ax.add_patch(Polygon(
                    arr, closed=True, fill=True, alpha=0.25,
                    facecolor=color, edgecolor=color, zorder=3))
                self.poly_artists.append((patch, mode, index))

    def _clear_selection(self):
        if self.selection_artist is not None:
            self.selection_artist.remove()
            self.selection_artist = None
        self.selected = None
        self._selection_hits = []
        self._selection_click_px = None

    def _select_at(self, event):
        """Pick geometry in screen pixels so line selection works at every zoom."""
        from matplotlib.path import Path as MplPath
        point = self.ax.transData.transform((event.xdata, event.ydata))
        near_edges, interiors = [], []
        for order, (_artist, key, index) in enumerate(self.poly_artists):
            coords = np.asarray(self.annotations[key][index], dtype=float)
            pixels = self.ax.transData.transform(coords)
            is_wall = key == "obstacle_segments"
            start = pixels[:-1] if is_wall else pixels
            end = pixels[1:] if is_wall else np.roll(pixels, -1, axis=0)
            delta = end - start
            length2 = np.sum(delta*delta, axis=1)
            t = np.clip(np.sum((point-start)*delta, axis=1) / np.maximum(length2, 1e-12), 0, 1)
            distance = float(np.min(np.linalg.norm(point-(start+t[:,None]*delta), axis=1)))
            if distance <= 9:
                near_edges.append((distance, -order, key, index))
            elif not is_wall and MplPath(pixels).contains_point(point):
                interiors.append((key, index))
        hits = [(key, index) for _, _, key, index in sorted(near_edges)] + list(reversed(interiors))
        choice = 0
        if (hits and hits == self._selection_hits and self.selected in hits and
                self._selection_click_px is not None and np.linalg.norm(point-self._selection_click_px) <= 5):
            choice = (hits.index(self.selected)+1) % len(hits)
        self._clear_selection()
        if hits:
            self.selected = hits[choice]
            self._selection_hits, self._selection_click_px = hits, point
            key, index = self.selected
            coords = np.asarray(self.annotations[key][index], dtype=float)
            if key != "obstacle_segments":
                coords = np.vstack((coords, coords[0]))
            self.selection_artist, = self.ax.plot(coords[:,0], coords[:,1], color="#00bcd4",
                                                  linewidth=3, marker="o", markersize=5, zorder=6)
            name = self.LABELS_ZH["wallseg" if key == "obstacle_segments" else key] if self._has_cjk else key
            self._set_help_text(message=(f"已选中 {name} 第 {index+1} 项；Delete 删除，Ctrl+Z 恢复；重复点击可轮换重叠项。" if self._has_cjk else
                                         f"Selected {name} #{index+1}. Delete removes; Ctrl+Z restores. Click again to cycle overlaps."))
        else:
            self._set_help_text(message="未选中标注，请点击区域内部或靠近墙线。" if self._has_cjk else
                                "No annotation selected. Click inside an area or near a wall.")
        self._update_title()

    # ------------------------------------------------------------------ 交互
    def _on_click(self, event):
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 2:
            self._view_timer.stop()
            self._view_pending = False
            self._pan = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim(),
                         self.ax.transData.inverted().frozen())
            self._pan_shift = (0., 0.)
            if self.fig.canvas.supports_blit:
                buffer = np.asarray(self.fig.canvas.buffer_rgba())
                x0, y0, x1, y1 = np.round(self.ax.bbox.extents).astype(int)
                top, bottom = buffer.shape[0]-y1, buffer.shape[0]-y0
                self._pan_rect = (top, bottom, x0, x1)
                self._pan_image = buffer[top:bottom, x0:x1].copy()
            return
        if self.mode == "erase":
            self._on_click_erase(event)
            return
        if self.mode == "select":
            if event.button == 1:
                self._select_at(event)
            elif event.button == 3:
                self._clear_selection()
                self._set_help_text("select")
                self._update_title()
            return
        if self.mode == "wallseg":
            self._on_click_wallseg(event)
            return
        if self.mode == "free":
            self._on_click_free(event)
            return
        if event.button == 1:
            self.current.append((float(event.xdata), float(event.ydata)))
            self._draw_current()
        elif event.button == 3:
            self._close_current()

    def _on_release(self, event):
        if event.button == 2:
            if self._pan is not None:
                x0, y0, (xl0, xl1), (zl0, zl1), inverse = self._pan
                sx, sy = self._pan_shift
                dx, dz = inverse.transform((x0+sx, y0+sy)) - inverse.transform((x0, y0))
                self.ax.set_xlim(xl0-dx, xl1-dx)
                self.ax.set_ylim(zl0-dz, zl1-dz)
            self._pan = None
            self._pan_image = None
            self._flush_view()
        elif event.button == 1:
            self._finish_erase()

    def _on_motion(self, event):
        if self._pan is not None:
            self._on_motion_pan(event)
        elif self.mode == "free":
            self._on_motion_free(event)
        else:
            self._on_motion_erase(event)

    def _on_click_free(self, event):
        if event.button == 3:
            self._clear_current()
            return
        if event.button != 1:
            return
        pt = (float(event.xdata), float(event.ydata))
        if not self.current:
            self.current = [pt]
            self._draw_current()
            self._set_help_text(message="已选第一角点；移动鼠标预览，再点对角点完成自由区。右键或 z 取消。" if self._has_cjk else
                                "First corner set. Move to preview, then click the opposite corner. Right click or z cancels.")
            return
        x0, z0 = self.current[0]
        x1, z1 = pt
        if abs(x1-x0) <= 1e-9 or abs(z1-z0) <= 1e-9:
            self._set_help_text(message="自由区需要面积：第二点的 X、Z 都要与第一点不同。请重新点对角点。" if self._has_cjk else
                                "A free area needs nonzero width and height. Choose a different opposite corner.")
            return
        x_min, x_max = sorted((x0, x1))
        z_min, z_max = sorted((z0, z1))
        self.current = [[x_min,z_min], [x_max,z_min], [x_max,z_max], [x_min,z_max]]
        self._close_current()
        self._set_help_text("free")

    def _on_motion_free(self, event):
        if len(self.current) != 1:
            return
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            if self.free_preview is not None and self.free_preview.get_visible():
                self.free_preview.set_visible(False)
                self._draw_brush()
            return
        from matplotlib.patches import Rectangle
        if self.free_preview is None:
            self.free_preview = Rectangle((0,0), 0, 0, facecolor=self.COLORS['free'],
                                          edgecolor=self.COLORS['free'], alpha=.25,
                                          linewidth=1.5, linestyle='--', zorder=4,
                                          animated=self.fig.canvas.supports_blit)
            self.ax.add_patch(self.free_preview)
        x0, z0 = self.current[0]
        self.free_preview.set_bounds(min(x0,event.xdata), min(z0,event.ydata),
                                     abs(event.xdata-x0), abs(event.ydata-z0))
        self.free_preview.set_visible(True)
        self._draw_brush()

    def _clear_free_preview(self):
        if self.free_preview is not None:
            self.free_preview.remove()
            self.free_preview = None

    def _on_motion_pan(self, event):
        if self._pan is None or event.x is None or event.y is None:
            return
        x0, y0, (xl0, xl1), (zl0, zl1), inverse = self._pan
        self._pan_shift = (event.x-x0, event.y-y0)
        if self._pan_image is not None:
            # Translate cached pixels while dragging. Newly exposed areas are
            # filled with fresh source points on release; no geometry is edited.
            buffer = np.asarray(self.fig.canvas.buffer_rgba())
            top, bottom, left, right = self._pan_rect
            target = buffer[top:bottom, left:right]
            source = self._pan_image
            if target.shape != source.shape:
                return
            target[:] = 255
            sx, sy = int(round(event.x-x0)), int(round(y0-event.y))
            h, w = source.shape[:2]
            tx, ty = max(0,sx), max(0,sy)
            ox, oy = max(0,-sx), max(0,-sy)
            width, height = w-abs(sx), h-abs(sy)
            if width > 0 and height > 0:
                target[ty:ty+height, tx:tx+width] = source[oy:oy+height, ox:ox+width]
            self._background = None
            self.fig.canvas.blit(self.ax.bbox)
            return
        dx, dz = inverse.transform((event.x, event.y)) - inverse.transform((x0, y0))
        self.ax.set_xlim(xl0-dx, xl1-dx)
        self.ax.set_ylim(zl0-dz, zl1-dz)
        self._queue_view()

    def _on_motion_erase(self, event):
        if self.mode != "erase":
            return
        from matplotlib.patches import Circle
        if event.inaxes is not self.ax or event.xdata is None:
            if self.brush is not None:
                self.brush.set_visible(False)
                self._draw_brush()
            self._stroke_last = None
            return
        if self.brush is None:
            self.brush = Circle((0, 0), self.ERASE_RADIUS_M, fill=False,
                                edgecolor="#cc0000", linewidth=1.2, alpha=0.8,
                                animated=self.fig.canvas.supports_blit)
            self.ax.add_patch(self.brush)
        self.brush.center = (event.xdata, event.ydata)
        self.brush.set_radius(self.ERASE_RADIUS_M)
        self.brush.set_visible(True)
        if self._erasing:
            self._apply_erase(event.xdata, event.ydata)
        self._draw_brush()

    def _apply_erase(self, x, z):
        end = (float(x), float(z))
        start = self._stroke_last if self._stroke_last is not None else end
        self.erased.update(int(v) for v in self.index.erase_stroke(start, end, self.ERASE_RADIUS_M))
        self._stroke_last = end

    def _finish_erase(self):
        if not self._erasing:
            return
        self._erasing = False
        added = self.erased - self._erase_before
        if added:
            self._record(("erase", added))
        self._erase_before = None
        self._stroke_last = None
        self._rebuild_points()
        self._update_title()

    def _on_click_erase(self, event):
        if event.button == 1:
            self._erasing = True
            self._erase_before = set(self.erased)
            self._stroke_last = None
            self._apply_erase(event.xdata, event.ydata)
            self._on_motion_erase(event)
        elif event.button == 3:
            self._finish_erase()

    def _record(self, action):
        self._history.append(action)
        self._history = self._history[-100:]
        self._redo_history.clear()

    def _apply_action(self, action, undo):
        if action[0] == "erase":
            if undo:
                self.erased.difference_update(action[1])
            else:
                self.erased.update(action[1])
            self._rebuild_points()
        else:
            kind, key, index, value = action
            remove = (kind == "add") == undo
            if remove:
                self.annotations[key].pop(index)
            else:
                self.annotations[key].insert(index, value)
            self._redraw_existing()
        self._update_title()

    def _undo(self):
        self._finish_erase()
        if self._history:
            action = self._history.pop()
            self._apply_action(action, True)
            self._redo_history.append(action)

    def _redo(self):
        if self._redo_history:
            action = self._redo_history.pop()
            self._apply_action(action, False)
            self._history.append(action)

    # ------------------------------------------------------------------ 多边形
    def _draw_current(self):
        self._clear_free_preview()
        if self.current_line is not None:
            self.current_line.remove()
            self.current_line = None
        for lb in self.vertex_labels:
            lb.remove()
        self.vertex_labels = []
        if not self.current:
            self.fig.canvas.draw_idle()
            return
        color = self.COLORS.get(self.mode, "gray")
        xs = [p[0] for p in self.current] + [self.current[0][0]]
        zs = [p[1] for p in self.current] + [self.current[0][1]]
        (self.current_line,) = self.ax.plot(
            xs, zs, color=color, linewidth=1.4, linestyle="--",
            marker="o", markersize=6, markerfacecolor=color, alpha=0.9, zorder=4)
        for i, (x, z) in enumerate(self.current):
            self.vertex_labels.append(self.ax.annotate(
                str(i + 1), (x, z), textcoords="offset points", xytext=(6, 6),
                fontsize=9, color=color, fontweight="bold", zorder=5))
        self.fig.canvas.draw_idle()

    def _clear_current(self):
        self._clear_free_preview()
        self.current = []
        if self.current_line is not None:
            self.current_line.remove()
            self.current_line = None
        for lb in self.vertex_labels:
            lb.remove()
        self.vertex_labels = []
        self._update_title()

    def _close_current(self):
        from matplotlib.patches import Polygon
        if len(self.current) < 3:
            print("polygon needs >=3 vertices, discarded")
            self._clear_current()
            return
        try:
            import map2d_data as m2d
            m2d.validate_polygon(self.current)
        except ValueError as e:
            print(f"invalid polygon ({e}), discarded")
            self._clear_current()
            return
        value = [list(p) for p in self.current]
        index = len(self.annotations[self.mode])
        self.annotations[self.mode].append(value)
        self._record(("add", self.mode, index, value))
        arr = np.asarray(self.current)
        color = self.COLORS.get(self.mode, "gray")
        patch = self.ax.add_patch(Polygon(arr, closed=True, fill=True, alpha=0.25,
                                          facecolor=color, edgecolor=color, zorder=3))
        self.poly_artists.append((patch, self.mode, index))
        print(f"saved {self.mode} polygon ({len(self.current)} vertices)")
        self._clear_current()

    def _undo_point(self):
        if self.current:
            self.current.pop()
            self._draw_current()
        elif self.wall_start is not None:
            self._cancel_wall_start()

    def _on_click_wallseg(self, event):
        if event.button == 3:
            self._cancel_wall_start()
            return
        if event.button != 1:
            return
        pt = (float(event.xdata), float(event.ydata))
        if self.wall_start is None:
            self.wall_start = pt
            (self.wall_start_marker,) = self.ax.plot(
                [pt[0]], [pt[1]], marker="o", markersize=8,
                markerfacecolor="#444444", markeredgecolor="red", zorder=4)
            self.fig.canvas.draw_idle()
        else:
            seg = [list(self.wall_start), list(pt)]
            index = len(self.annotations["obstacle_segments"])
            self.annotations.setdefault("obstacle_segments", []).append(seg)
            self._record(("add", "obstacle_segments", index, seg))
            (line,) = self.ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                                   color="#111111", linewidth=5,
                                   solid_capstyle="round", zorder=3)
            self.poly_artists.append((line, "obstacle_segments", index))
            print(f"saved wall segment ({len(self.annotations['obstacle_segments'])} total)")
            self._cancel_wall_start()

    def _cancel_wall_start(self):
        self.wall_start = None
        if self.wall_start_marker is not None:
            self.wall_start_marker.remove()
            self.wall_start_marker = None
        self._update_title()

    def _del_poly(self):
        key = "obstacle_segments" if self.mode == "wallseg" else self.mode
        if key in (*self.POLYGON_KEYS, "obstacle_segments") and self.annotations[key]:
            self._delete_annotation(key, len(self.annotations[key])-1)
        else:
            self._set_help_text(message="当前类型没有可删除项；也可用 Select 选中标注后点 Delete。" if self._has_cjk else
                                "No item in this drawing type. Use Select, then Delete to remove a specific annotation.")

    def _delete_selected(self):
        if self.selected is None:
            self._set_help_text(message="请先点 Select / 7，再点击要删除的区域或墙线。" if self._has_cjk else
                                "Choose Select / 7, then click the area or wall to delete.")
            return
        self._delete_annotation(*self.selected)

    def _delete_annotation(self, key, index):
        self._finish_erase()
        value = self.annotations[key].pop(index)
        self._record(("delete", key, index, value))
        self._redraw_existing()
        self._update_title()
        self._set_help_text(message="已删除标注；Ctrl+Z / Undo 可恢复，Ctrl+Y / Redo 可重做。" if self._has_cjk else
                            "Annotation deleted. Ctrl+Z / Undo restores it; Ctrl+Y / Redo repeats the deletion.")

    def _save(self):
        self._finish_erase()
        self.annotations["erased_point_ids"] = sorted(int(i) for i in self.erased)
        import os, tempfile
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix=self.out_path.name+".", dir=self.out_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=2)
            os.replace(path, self.out_path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
        print(f"saved -> {self.out_path} (erased {len(self.erased)} points)")

    # ------------------------------------------------------------------ 键盘/滚轮
    def _on_key(self, event):
        # Text input owns its keys: typing '1' must not switch to Free mode.
        if any(getattr(box, "capturekeystrokes", False)
               for box in (self.radius_box, *self.height_boxes.values())):
            return
        k = event.key
        if k is None:
            return
        if k in self.MODE_KEYS:
            self._set_mode(self.MODE_KEYS[k])
        elif k == "ctrl+z":
            self._undo()
        elif k in ("ctrl+y", "ctrl+shift+z"):
            self._redo()
        elif k == "h":
            self._home()
        elif k == "escape":
            self._clear_selection()
            self._clear_current()
            self._cancel_wall_start()
        elif k in ("z", "backspace"):
            self._undo_point()
        elif k == "x":
            self._delete_selected() if self.selected is not None else self._del_poly()
        elif k == "delete":
            self._delete_selected()
        elif k == "w":
            self._save()
        elif k in ("?", "f1"):
            self._show_help()
        elif k == "[":
            self._scale_erase_radius(1/1.5)
        elif k == "]":
            self._scale_erase_radius(1.5)
        elif k == "q":
            self._save()
            import matplotlib.pyplot as plt
            plt.close(self.fig)

    def _on_close(self, event):
        self._view_timer.stop()
        self._save()
        if self._help_fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._help_fig)

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or self._pan is not None:
            return
        factor = 1.0 / 1.25 if event.button == "up" else 1.25
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        self.ax.set_xlim([xd - (xd - xlim[0]) * factor, xd + (xlim[1] - xd) * factor])
        self.ax.set_ylim([yd - (yd - ylim[0]) * factor, yd + (ylim[1] - yd) * factor])
        self._throttled_redraw()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="二维规划地图人工标注（XZ 俯视）")
    p.add_argument("--source-json", required=True,
                   help="导出的 source_map.json（含 landmarks 全局 XYZ）")
    p.add_argument("--out", default="annotations.json", help="标注输出文件")
    p.add_argument("--annotations", default=None, help="载入已有标注继续编辑")
    p.add_argument("--ground-y", type=float, default=None,
                   help="候选障碍的地面 Y（全局地图坐标；不给则不标红）")
    p.add_argument("--height-min", type=float, default=0.5, help="候选障碍高度下限，不控制灰色底图")
    p.add_argument("--height-max", type=float, default=5.0, help="候选障碍高度上限，不控制灰色底图")
    p.add_argument("--display-ground-y", type=float, default=None,
                   help="启用底图高度切片：地面在全局地图中的 Y；不自动估计地面")
    p.add_argument("--display-height-min", type=float, default=None,
                   help="底图相对地面高度下限，默认 -0.15m；需同时给 display-ground-y")
    p.add_argument("--display-height-max", type=float, default=None,
                   help="底图相对地面高度上限，默认 +0.15m；需同时给 display-ground-y")
    p.add_argument("--hide-candidates", action="store_true", help="关闭红点着色；不更改候选障碍生成策略")
    p.add_argument("--erase-radius-m", type=float, default=0.3,
                   help="初始擦除半径，单位米，范围 0.05–3.00，默认 0.30；界面可随时调整")
    p.add_argument("--display-mode", choices=("auto", "points", "density"), default="auto",
                   help="auto：概览像素聚合，放大后原始点；points：强制全点；density：始终聚合")
    args = p.parse_args(argv)
    if not np.isfinite(args.erase_radius_m) or not 0.05 <= args.erase_radius_m <= 3.0:
        p.error("--erase-radius-m 必须在 0.05–3.00 米之间")
    if args.display_ground_y is None and (args.display_height_min is not None or args.display_height_max is not None):
        p.error("显示高度范围需要显式提供 --display-ground-y（地图地面 Y，不一定是 0）")
    display_min = -0.15 if args.display_height_min is None else args.display_height_min
    display_max = 0.15 if args.display_height_max is None else args.display_height_max
    if (not np.all(np.isfinite([display_min, display_max])) or display_min > display_max or
            (args.display_ground_y is not None and not np.isfinite(args.display_ground_y))):
        p.error("显示地面/高度范围非法：须为有限数，且下限不能大于上限")

    data = json.loads(Path(args.source_json).read_text(encoding="utf-8"))
    import map2d_data as m2d
    m2d._validate_source_json(data)
    pts = np.array([[lm["pose"]["x"], lm["pose"]["y"], lm["pose"]["z"]]
                    for lm in data["landmarks"]], dtype=np.float64)
    ids = np.array([lm["id"] for lm in data["landmarks"]], dtype=np.int64)
    candidates = None
    cand_ids = None
    if args.ground_y is not None:
        if not np.all(np.isfinite([args.ground_y, args.height_min, args.height_max])) or args.height_min > args.height_max:
            p.error("地面/高度区间非法")
        h = args.ground_y - pts[:, 1]
        mask = (h >= args.height_min) & (h <= args.height_max)
        candidates = pts[mask]
        cand_ids = ids[mask]

    annotations = {"free": [], "obstacle": [], "unknown": [], "candidate_remove": [],
                   "obstacle_segments": []}
    existing = Path(args.annotations) if args.annotations else Path(args.out)
    if existing.is_file():
        loaded = json.loads(existing.read_text(encoding="utf-8"))
        print(f"恢复标注: {existing}")
    elif args.annotations:
        p.error(f"标注文件不存在: {existing}")
    else:
        loaded = {}
    if loaded:
        for k in annotations:
            annotations[k] = list(loaded.get(k, []))
        if "erased_point_ids" in loaded:
            annotations["erased_point_ids"] = list(loaded["erased_point_ids"])

    editor = Annotator(pts, candidates, annotations, Path(args.out),
                       landmark_ids=ids, cand_ids=cand_ids, display_mode=args.display_mode,
                       erase_radius_m=args.erase_radius_m, display_ground_y=args.display_ground_y,
                       display_height_min=display_min, display_height_max=display_max,
                       show_candidates=not args.hide_candidates)
    import matplotlib.pyplot as plt
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
