"""
output 层：把 decision 的状态和 features 的移动量"呈现"给用户。

两个出口：
    draw(frame, state, movement)  在画面上绘制（英文标签 + 移动量 + 可选的走势图）
    log(state, movement)          控制台英文日志（限频，默认 0.5 秒一条）

为什么全部用英文：
    cv2.putText 自带 Hershey 字体不支持中文，画出来是方块；
    统一英文让画面和控制台一致（需要中文展示时再引 Pillow 渲染）。

debug 模式：
    画面角落画一个 ~320x48 的"移动量走势图"：近几秒的 movement 折线，
    两条虚线标出 still_threshold / moving_threshold，中间是迟滞带。
    用途是让用户自己坐在摄像头前动一动/停一停，观察静止和活动时
    movement 大概落在什么区间，从而定出合适的两个阈值。

坐姿角度 + 坐姿状态 + 提醒横幅 + 骨架（新增）：
    draw(..., posture, reminder, pose, posture_state, posture_reminder) 可选参数。
    posture: PostureFeatures.update() 的返回值，至少含 'head_neck_angle'，
            髋可见时另含 'torso_angle'/'back_curvature'。在状态标签下追加一行
            `Torso 8.2°  Neck 5.1°  Back 6.0°`；**key 缺失（该角度算不出，
            如髋不可见）时显示 `N/A`，保证这一行始终出现**，而不是整行消失。
    reminder: SedentaryAlert 触发的久坐提醒文案（英文）。画成顶部横幅
            （cv2 绘制），停留 reminder_hold_sec 秒。
    posture_state: PostureDecision.posture（PostureState 枚举）。在角度行下方
            固定画一行 `Posture: GOOD / SLUMPED / N/A`，颜色随状态（绿/琥珀/灰）。
    posture_reminder: PostureAlert 触发的不良坐姿提醒文案（英文）。画成
            第二条横幅，位于久坐横幅正下方（琥珀色区分），同样停留
            reminder_hold_sec 秒。
    still_elapsed_sec / slump_elapsed_sec: SedentaryAlert / PostureAlert 的
            elapsed_sec（当前连续静坐/弓背秒数）。非 None 时在坐姿状态行下方
            画一行 `Still 12.3/20s  Slump 7.8/10s` 计时（阈值来自构造参数
            duration_limit_sec / posture_duration_limit_sec），方便测试时看进度。
    pose:    pose_estimation.detect_pose() 的返回值 {'landmarks': [(x,y,z,vis)x17], 'image_size'}。
            画成骨架叠加层（关键点 + 连接线），默认开启，可用 draw_skeleton=False 关闭。
            骨架只画躯干 + 四肢关键点（COCO 的 5-16，绿色），**不画头部/面部点
            （0-4）及其连线**——脸只用于角度计算，画面骨架不含脸。
            例外：耳朵点（3/4）单独用黄色画出，因为它不参与骨架连线、也不属于
            躯干四肢，单独标出来方便确认坐姿角度里"耳-肩-髋"链路的耳端有没有
            被检测到（head_neck_angle / back_curvature 都要用到耳）。
    角度说明: 画面左下角固定绘制"耳-肩-髋三点连线角度定义"英文说明，
            文字来自 features.ANGLE_LEGEND（单一来源，output 只渲染），
            不随帧变化、无需参数。

本模块只消费 decision.State/PostureState + float + 四个 dict（posture/久坐文案/
不良坐姿文案/pose 数据）。
例外：仅 import features.ANGLE_LEGEND 这一个**数据常量**用于渲染角度说明，
不 import features 的逻辑/类。
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

from decision import State, PostureState
from features import ANGLE_LEGEND

# 骨架：17 个 COCO 关键点之间怎么连线（RTMPose 输出的 COCO 17 点）。
# 只保留躯干 + 四肢连线；头部/面部 0-4 不画（脸只用于角度计算）。
POSE_CONNECTIONS = frozenset([
    (5, 6), (5, 7), (7, 9),         # 肩 + 左臂
    (6, 8), (8, 10),                # 右臂
    (5, 11), (6, 12), (11, 12),     # 躯干
    (11, 13), (13, 15),             # 左腿
    (12, 14), (14, 16),             # 右腿
])

# 横幅面板尺寸（顶部居中，半透明底条）
_BANNER_H = 56

# 状态颜色（BGR）
_COLOR_STILL = (60, 200, 60)    # 绿
_COLOR_MOVING = (60, 60, 220)   # 红
_COLOR_UNKNOWN = (180, 180, 180)  # 灰

_STATE_LABEL = {
    State.STILL: "STILL",
    State.MOVING: "MOVING",
    State.UNKNOWN: "NO PERSON",
}

# 坐姿状态标签（英文，Hershey 画不了中文）
_POSTURE_LABEL = {
    PostureState.GOOD: "Posture: GOOD",
    PostureState.SLUMPED: "Posture: SLUMPED",
    PostureState.UNKNOWN: "Posture: N/A",
}

# 坐姿状态颜色（BGR）：绿=坐姿好，琥珀=弓背/不良坐姿，灰=无人/数据不足
_COLOR_GOOD = (60, 200, 60)
_COLOR_SLUMPED = (0, 140, 255)
_COLOR_POSTURE_UNKNOWN = (180, 180, 180)

# 走势图面板尺寸
_CHART_W, _CHART_H = 320, 48


class FrameRenderer:
    """画面/控制台的输出器。"""

    def __init__(self,
                 still_threshold: float = 0.05,
                 moving_threshold: float = 0.10,
                 debug: bool = False,
                 log_interval_sec: float = 0.5,
                 chart_seconds: float = 3.0,
                 reminder_hold_sec: float = 8.0,
                 draw_skeleton: bool = True,
                 duration_limit_sec: float = 1200.0,
                 posture_duration_limit_sec: float = 300.0) -> None:
        """
        参数:
            still_threshold / moving_threshold: 迟滞阈值，走势图上画两条线用。
            debug:         True 时在画面上画走势图。
            log_interval_sec: 控制台日志的最小间隔（秒），避免每帧刷屏。
            chart_seconds: 走势图覆盖的时间跨度（秒）。
            reminder_hold_sec: 提醒横幅在画面上停留的秒数（久坐/不良坐姿共用）。
            draw_skeleton: 是否把姿态识别的骨架（关键点 + 连线）叠到画面上，
                         默认 True。想看纯数字/需要省 CPU 时用 --no-skeleton 关闭。
            duration_limit_sec: 久坐提醒阈值（秒），画面上"Still x/..s"的进度用。
            posture_duration_limit_sec: 不良坐姿提醒阈值（秒），画面上
                         "Slump x/..s"的进度用。
        """
        self.still_threshold = still_threshold
        self.moving_threshold = moving_threshold
        self.debug = debug
        self.log_interval_sec = log_interval_sec
        self.chart_seconds = chart_seconds
        self.reminder_hold_sec = reminder_hold_sec
        self.draw_skeleton = draw_skeleton
        self.duration_limit_sec = duration_limit_sec
        self.posture_duration_limit_sec = posture_duration_limit_sec

        self._chart: Deque[tuple] = deque()  # (ts, movement) 只存 movement 非 None
        self._last_log_ts = 0.0
        self._banner_text: Optional[str] = None   # 当前要显示的久坐提醒横幅
        self._banner_until = 0.0                  # 久坐横幅显示到哪个时刻（monotonic）
        self._posture_banner_text: Optional[str] = None   # 当前要显示的不良坐姿横幅
        self._posture_banner_until = 0.0                  # 不良坐姿横幅显示到哪个时刻

    # ---------- 画面 ----------

    def draw(self, frame: np.ndarray, state: State,
             movement: Optional[float],
             posture: Optional[dict] = None,
             reminder: Optional[str] = None,
             pose: Optional[dict] = None,
             posture_state: Optional[PostureState] = None,
             posture_reminder: Optional[str] = None,
             still_elapsed_sec: Optional[float] = None,
             slump_elapsed_sec: Optional[float] = None) -> np.ndarray:
        """在 frame 上叠加状态信息，返回新帧（也在原帧上就地画）。

        参数:
            state / movement: 同原有签名。
            posture: PostureFeatures.update() 的返回值（至少含 head_neck_angle；
                     torso/back 在髋不可见时缺失）。非 None 时在状态标签下追加
                     一行 `Torso x.x°  Neck x.x°  Back x.x°`，缺失的角度显示 N/A。
            reminder: SedentaryAlert 触发的久坐提醒文案（英文）。非 None 时记录
                     横幅并停留 reminder_hold_sec 秒。
            pose:    pose_estimation.detect_pose() 的返回值（landmarks + image_size）。
                     非 None 且 draw_skeleton=True 时，把骨架叠到画面上。
            posture_state: PostureDecision.posture（PostureState 枚举）。非 None 时
                     在角度行下方画一行 `Posture: GOOD/SLUMPED/N/A`（颜色随状态）。
            posture_reminder: PostureAlert 触发的不良坐姿提醒文案（英文）。
                     非 None 时记录第二条横幅（久坐横幅下方，琥珀色区分）。
            still_elapsed_sec: SedentaryAlert.elapsed_sec（当前连续静坐秒数）。
                     非 None 时画一行 `Still x.x/..s` 计时。
            slump_elapsed_sec: PostureAlert.elapsed_sec（当前连续弓背秒数）。
                     非 None 时同一行画 `Slump x.x/..s` 计时。
        """
        # 骨架层（画在状态/横幅等文字之下，避免遮挡文字）
        if self.draw_skeleton and pose is not None:
            self._draw_skeleton(frame, pose)

        # 记录提醒横幅（触发帧传入一次，之后自己倒计时消失）
        if reminder is not None:
            self._banner_text = reminder
            self._banner_until = time.monotonic() + self.reminder_hold_sec
        if posture_reminder is not None:
            self._posture_banner_text = posture_reminder
            self._posture_banner_until = time.monotonic() + self.reminder_hold_sec

        color = _COLOR_STILL if state is State.STILL else (
            _COLOR_MOVING if state is State.MOVING else _COLOR_UNKNOWN)

        # 状态标签（英文，Hershey 画不了中文）
        label = _STATE_LABEL[state]
        if movement is not None:
            label += f"  mv={movement:.3f}"
        cv2.putText(frame, label, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # 坐姿角度（英文，紧跟状态标签下方）。
        # posture 至少含 head_neck_angle；torso/back 在髋不可见时缺失 → 显示 N/A
        if posture is not None:
            def _fmt(key: str) -> str:
                v = posture.get(key)
                return f"{v:.1f}°" if v is not None else "N/A"
            posture_line = (
                f"Torso {_fmt('torso_angle')}  "
                f"Neck {_fmt('head_neck_angle')}  "
                f"Back {_fmt('back_curvature')}"
            )
            cv2.putText(frame, posture_line, (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                        cv2.LINE_AA)

        # 坐姿状态（英文，紧跟角度行下方，常显；N/A = 无人/数据不足）
        if posture_state is not None:
            pcolor = (_COLOR_GOOD if posture_state is PostureState.GOOD else
                      _COLOR_SLUMPED if posture_state is PostureState.SLUMPED else
                      _COLOR_POSTURE_UNKNOWN)
            cv2.putText(frame, _POSTURE_LABEL[posture_state], (12, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, pcolor, 1, cv2.LINE_AA)

        # 计时行：Still/Slump 已保持秒数 vs 提醒阈值（方便测试看进度）
        if still_elapsed_sec is not None and slump_elapsed_sec is not None:
            timer_line = (
                f"Still {still_elapsed_sec:.1f}/{self.duration_limit_sec:.0f}s"
                f"  Slump {slump_elapsed_sec:.1f}/"
                f"{self.posture_duration_limit_sec:.0f}s")
            cv2.putText(frame, timer_line, (12, 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                        cv2.LINE_AA)

        # 两条提醒横幅（英文，cv2 Hershey）：久坐在上、不良坐姿在下（琥珀色）
        now = time.monotonic()
        if self._banner_text is not None and now < self._banner_until:
            self._draw_banner(frame, self._banner_text, y_top=0)
        elif now >= self._banner_until:
            self._banner_text = None
        if self._posture_banner_text is not None and now < self._posture_banner_until:
            self._draw_banner(frame, self._posture_banner_text,
                              y_top=_BANNER_H, accent=_COLOR_SLUMPED)
        elif now >= self._posture_banner_until:
            self._posture_banner_text = None

        if self.debug:
            self._record_chart(movement)
            self._draw_chart(frame)

        # 角度定义说明（英文，左下角）
        self._draw_angle_legend(frame)

        return frame

    # ---------- 骨架叠加 ----------

    def _draw_skeleton(self, frame: np.ndarray, pose: dict) -> None:
        """把姿态识别的骨架（关键点 + 连接线）叠到画面上。

        pose: {'landmarks': [(x,y,z,vis) x17], 'image_size': (w,h)}
        只画 visibility >= 0.3 的点/线，低可见度的残缺点不画，避免噪点乱连。
        """
        landmarks = pose.get('landmarks')
        image_size = pose.get('image_size')
        if not landmarks or image_size is None:
            return

        w, h = image_size
        visibility_min = 0.3

        def px(pid: int) -> Optional[tuple]:
            """关键点归一化坐标 -> 像素坐标；可见度不足返回 None。"""
            if pid >= len(landmarks):
                return None
            x, y, _z, vis = landmarks[pid]
            if vis < visibility_min:
                return None
            return int(x * w), int(y * h)

        # 关键点：只画躯干 + 四肢（COCO 的 5-16，绿色）；头部/面部 0-4 不画
        for pid in range(5, 17):
            p = px(pid)
            if p is not None:
                cv2.circle(frame, p, 4, (0, 255, 0), -1, cv2.LINE_AA)

        # 耳朵点（3/4）：单独用黄色画出。它们不参与骨架连线，但坐姿角度
        # （head_neck_angle / back_curvature）依赖耳点，标出来方便确认检测。
        for pid in (3, 4):
            p = px(pid)
            if p is not None:
                cv2.circle(frame, p, 4, (0, 215, 255), -1, cv2.LINE_AA)

        # 连接线（两端都可见才画，防止线条插到残缺点）
        for a, b in POSE_CONNECTIONS:
            pa, pb = px(a), px(b)
            if pa is not None and pb is not None:
                cv2.line(frame, pa, pb, (255, 0, 0), 2, cv2.LINE_AA)

    def log(self, state: State, movement: Optional[float],
            posture_state: Optional[PostureState] = None) -> None:
        """控制台英文日志，按 log_interval_sec 限频。"""
        now = time.monotonic()
        if now - self._last_log_ts < self.log_interval_sec:
            return
        self._last_log_ts = now

        posture_str = ""
        if posture_state is not None:
            posture_str = f"  posture={posture_state.value}"
        if movement is None:
            print(f"[output] state={state.value}  mv=N/A "
                  f"(no person / low data){posture_str}")
        else:
            print(f"[output] state={state.value}  mv={movement:.4f}{posture_str}")

    # ---------- 走势图（debug） ----------

    def _record_chart(self, movement: Optional[float]) -> None:
        now = time.monotonic()
        if movement is not None:
            self._chart.append((now, movement))
        cutoff = now - self.chart_seconds
        while self._chart and self._chart[0][0] < cutoff:
            self._chart.popleft()

    def _draw_chart(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        x0 = max(w - _CHART_W - 10, 0)
        y0 = 10
        x1, y1 = x0 + _CHART_W, y0 + _CHART_H

        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)  # 底板
        cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 120, 120), 1)

        # 上限：取 moving_threshold 的 2 倍（至少 0.1），防数值大时曲线顶出面板
        vmax = max(0.1, self.moving_threshold * 2)
        inner_w, inner_h = _CHART_W - 4, _CHART_H - 4

        def sy(v: float) -> int:
            return y1 - 2 - int(round((v / vmax) * inner_h))

        # 迟滞带（still ~ moving 之间）淡色填充
        y_still = sy(self.still_threshold)
        y_moving = sy(self.moving_threshold)
        cv2.rectangle(frame, (x0 + 2, min(y_still, y_moving)),
                      (x1 - 2, max(y_still, y_moving)), (40, 40, 80), -1)

        # 两条阈值虚线
        self._dashed_line(frame, (x0 + 2, y_still), (x1 - 2, y_still),
                          (90, 220, 90))
        self._dashed_line(frame, (x0 + 2, y_moving), (x1 - 2, y_moving),
                          (90, 90, 220))

        # movement 折线
        if len(self._chart) >= 2:
            pts = []
            for ts, mv in self._chart:
                t_frac = (ts - self._chart[0][0]) / self.chart_seconds
                px = x0 + 2 + int(round(t_frac * inner_w))
                pts.append((min(max(px, x0 + 2), x1 - 2), sy(mv)))
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)],
                          False, (255, 255, 255), 1, cv2.LINE_AA)

    # ---------- 角度定义说明 ----------

    def _draw_angle_legend(self, frame: np.ndarray) -> None:
        """画面左下角画"耳-肩-髋三点连线的角度定义"说明（英文，半透明底）。

        文字来自 features.ANGLE_LEGEND（features 层定义语义，本层只渲染）。
        """
        lines = ANGLE_LEGEND
        h, w = frame.shape[:2]

        scale = 0.45
        line_h = 17
        pad_x, pad_y = 8, 5
        text_w = 0
        for line in lines:
            (tw, _th), _base = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            text_w = max(text_w, tw)

        x0 = 10
        y0 = h - (len(lines) * line_h + 2 * pad_y) - 10
        x1 = x0 + text_w + 2 * pad_x
        y1 = y0 + len(lines) * line_h + 2 * pad_y

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            y = y0 + pad_y + line_h * (i + 1) - 6
            color = (60, 220, 60) if i > 0 else (230, 230, 230)
            cv2.putText(frame, line, (x0 + pad_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    # ---------- 提醒横幅（英文） ----------

    def _draw_banner(self, frame: np.ndarray, text: str,
                     y_top: int = 0,
                     accent: tuple = (60, 200, 60)) -> None:
        """在画面顶部画一条提醒横幅（半透明底条 + 文字）。

        y_top:  横幅顶部 y 坐标（0 = 第一条，_BANNER_H = 第二条）。
        accent: 下边线/文字颜色（久坐=绿，不良坐姿=琥珀）。
        """
        h, w = frame.shape[:2]

        # 底条（半透明深色，保证文字可读）
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_top), (w, y_top + _BANNER_H),
                      (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.line(frame, (0, y_top + _BANNER_H), (w, y_top + _BANNER_H),
                 accent, 2)

        # 文字（Hershey 支持英文），居中
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = max((w - tw) // 2, 10)
        y = y_top + (_BANNER_H + th) // 2
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, accent, 2,
                    cv2.LINE_AA)

    @staticmethod
    def _dashed_line(img, pt1, pt2, color, dash=5, gap=3) -> None:
        """在图上画一条虚线（cv2 没有内建虚线）。"""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = int(np.hypot(x2 - x1, y2 - y1))
        if dist == 0:
            return
        dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
        i = 0
        while i < dist:
            s = i
            e = min(i + dash, dist)
            cv2.line(img,
                     (int(x1 + dx * s), int(y1 + dy * s)),
                     (int(x1 + dx * e), int(y1 + dy * e)),
                     color, 1)
            i += dash + gap
