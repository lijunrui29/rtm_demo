"""
output 层：把 decision 的状态和 features 的移动量"呈现"给用户。

两个出口：
    draw(frame, state, movement)  在画面上绘制（英文标签 + 移动量 + 可选的走势图）
    log(state, movement)          控制台英文日志（限频，默认 0.5 秒一条）
    export_pose_json(schema, path) 把规范 schema（pose_schema 输出）导出为 JSON，
                                供对接机器人/肉眼对比两边关节命名

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
            髋可见时另含 'torso_angle'/'back_curvature'，双侧肩在时含
            'neck_compression'（比值，无单位）。在状态标签下追加一行
            `Torso 8.2°  Neck 5.1°  Back 6.0°  Head 0.75`；**key 缺失（该
            角度算不出，如髋不可见/单侧链路无肩宽）时显示 `N/A`，保证这一
            行始终出现**，而不是整行消失。
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
            画成骨架叠加层（关键点 + 连接线 + pid/置信度标签），默认开启，
            可用 draw_skeleton=False 关闭。
            画 3-16 号点：躯干/四肢 5-16 绿、耳 3/4 黄（坐姿角度链路用）；
            面部 0-2（鼻/眼）不画——机器人控制台（figurobot-console）没有
            面部关节，画了没对照。置信度 <0.3 的"没检测到"点画灰色空心圈
            示意位置。每个点旁标 `pid:置信度`，字色按置信度分档（与
            pose_schema status 一致）：>=0.5 绿、0.3~0.5 黄（低置信度）、
            <0.3 灰（没检测到）。连线只有躯干 + 四肢（头/脸不连线），且
            两端都 >=0.3 才画，避免噪点乱连。
    角度说明: 画面左下角固定绘制"耳-肩-髋三点连线角度定义"英文说明，
            文字来自 features.ANGLE_LEGEND（单一来源，output 只渲染），
            不随帧变化、无需参数。

CVA / FSA 姿态风险指标叠加（新增）：
    draw_cva_overlay(frame, cva, cva_level) 可选调用（main 在 draw/draw_schema
    后调用）。cva: ErgonomicRiskFeatures.update() 的返回值；cva_level:
    decision.CvaRisk.level（CvaLevel 枚举）。在计时行下方画 CVA 主指标
    （按分级着色）；FSA 是**辅助指标**，仅当 CVA 判定为"中重度"及以上时以
    更小灰白字附注（标注 aux，含相对趋势），与 CVA 分开呈现、不暗示两者
    权重相同。CVA 无有效值时显示 `CVA --  N/A`。
    免责声明：本系统输出的 CVA/FSA 角度指标反映体表姿态模式，不能替代医学
    影像诊断或临床评估（分级阈值参考 Mostafaee et al. 2022 观察性分组）。

本模块只消费 decision.State/PostureState/CvaLevel + float + 五个 dict
（posture/久坐文案/不良坐姿文案/pose 数据/CVA-FSA 特征）。
例外：仅 import features.ANGLE_LEGEND 这一个**数据常量**用于渲染角度说明，
不 import features 的逻辑/类。
"""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

from decision import State, PostureState, CvaLevel
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

# 规范关节状态颜色（draw_schema 面板用，BGR）
_STATUS_COLOR = {
    "ok": (60, 200, 60),        # 绿
    "low_conf": (0, 210, 255),  # 黄
    "na": (150, 150, 150),      # 灰
    "offline": (0, 80, 220),    # 红
    "error": (0, 60, 220),      # 红
}

# CVA 分级颜色（draw_cva_overlay 用，BGR）：绿=正常、黄=轻度、橙=中重度、
# 红=重度、灰=未知
_CVA_LEVEL_COLOR = {
    CvaLevel.NORMAL: (60, 200, 60),
    CvaLevel.MILD: (0, 210, 255),
    CvaLevel.MODERATE_SEVERE: (0, 165, 255),
    CvaLevel.SEVERE: (0, 60, 220),
    CvaLevel.UNKNOWN: (180, 180, 180),
}


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
                     torso/back 在髋不可见时缺失；neck_compression 在单侧链路
                     无肩宽时缺失）。非 None 时在状态标签下追加一行
                     `Torso x.x°  Neck x.x°  Back x.x°  Head x.xx`，缺失的
                     项显示 N/A（颈压缩为比值，显示两位小数，无 °）。
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

        # 坐姿特征（英文，紧跟状态标签下方）。
        # posture 至少含 head_neck_angle；torso/back 在髋不可见时缺失、
        # neck_compression 在单侧链路（无肩宽）时缺失 → 都显示 N/A。
        # 颈压缩是比值（耳-肩竖直间距/肩宽），两位小数、无 °，见左下角图例。
        if posture is not None:
            def _fmt(key: str) -> str:
                v = posture.get(key)
                return f"{v:.1f}°" if v is not None else "N/A"
            def _fmt_ratio(key: str) -> str:
                v = posture.get(key)
                return f"{v:.2f}" if v is not None else "N/A"
            posture_line = (
                f"Torso {_fmt('torso_angle')}  "
                f"Neck {_fmt('head_neck_angle')}  "
                f"Back {_fmt('back_curvature')}  "
                f"Head {_fmt_ratio('neck_compression')}"
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

    # ---------- CVA/FSA 姿态风险指标叠加（2026-09-01 新增） ----------

    def draw_cva_overlay(self, frame: np.ndarray, cva: Optional[dict],
                         cva_level: Optional[CvaLevel] = None) -> None:
        """在画面左侧叠加 CVA 主指标 +（仅中重度及以上时的）FSA 辅助行。

        cva: ErgonomicRiskFeatures.update() 的返回值，含
             'cva_deg'/'cva_valid'/'fsa_deg'/'fsa_valid'/'fsa_trend_deg'。
        cva_level: decision.CvaRisk.level（CvaLevel 枚举）。
        CVA 主行画在计时行 (12,112) 下方 (12,140)，按分级着色：
            NORMAL 绿 / MILD 黄 / MODERATE_SEVERE 橙 / SEVERE 红 / UNKNOWN 灰。
            cva_deg 为 None（从未可算）→ `CVA --  N/A`（灰）。
        FSA 是辅助指标：仅当 cva_level 为 MODERATE_SEVERE/SEVERE 且 fsa_deg
        可显示时，以更小的灰白字画在 (12,162)，标注 aux —— 与 CVA 分开呈现、
        不暗示两者权重相同（decision 层不读 FSA，这里只做报告附注）。
        """
        if cva is None or cva_level is None:
            return

        color = _CVA_LEVEL_COLOR.get(cva_level, (180, 180, 180))
        cva_deg = cva.get('cva_deg')
        cva_line = f"CVA {cva_deg:.1f}°  {cva_level.value}" \
            if cva_deg is not None else "CVA --  N/A"
        cv2.putText(frame, cva_line, (12, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # FSA 辅助行（仅中重度及以上附注；无 fsa 值不画）
        if cva_level in (CvaLevel.MODERATE_SEVERE, CvaLevel.SEVERE):
            fsa_deg = cva.get('fsa_deg')
            if fsa_deg is not None:
                trend = cva.get('fsa_trend_deg')
                trend_s = "" if trend is None else f" trend {trend:+.1f}°"
                fsa_line = f"FSA {fsa_deg:.1f}°{trend_s}  aux"
                cv2.putText(frame, fsa_line, (12, 162),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1,
                            cv2.LINE_AA)

    # ---------- 骨架叠加 ----------

    def _draw_skeleton(self, frame: np.ndarray, pose: dict) -> None:
        """把姿态识别的骨架（关键点 + 连接线 + pid/置信度标签）叠到画面上。

        pose: {'landmarks': [(x,y,z,vis) x17], 'image_size': (w,h)}
        画 3-16 号点（测试时能看到编号+置信度）；面部点 0-2（鼻/眼）不画——
        机器人控制台（figurobot-console SKELETON）没有面部关节，画了没对照。
            点颜色按部位——躯干/四肢 5-16 绿、耳 3/4 黄（坐姿角度链路用）；
            置信度 <0.3 的"没检测到"点画灰色空心小圈示意位置。
            标签字色按置信度分档（与 pose_schema status 一致）：
                >=0.5 绿、0.3~0.5 黄（低置信度）、<0.3 灰（没检测到）。
        连线仍只连躯干/四肢，且两端都可见(>=0.3)才画，避免噪点乱连。
        """
        landmarks = pose.get('landmarks')
        image_size = pose.get('image_size')
        if not landmarks or image_size is None:
            return

        w, h = image_size
        # 与 features.visibility_min / pose_schema 的置信度分档保持一致
        VIS_LOW = 0.3    # < 此值 = 没检测到（灰）
        VIS_OK = 0.5     # >= 此值 = 置信度足够（绿）

        def px(pid: int) -> Optional[tuple]:
            """关键点归一化坐标 -> 像素坐标（不过滤可见度，分级用颜色表达）。"""
            if pid >= len(landmarks):
                return None
            x, y, _z, _vis = landmarks[pid]
            return int(x * w), int(y * h)

        def vis_of(pid: int) -> float:
            return landmarks[pid][3] if pid < len(landmarks) else 0.0

        # 关键点：画 3-16 号（连线见下）。面部 0-2（鼻/眼）不画——机器人控制台
        # 没有面部关节（figurobot-console SKELETON 头部只有 头Y/颈P），画了没对照。
        # 躯干/四肢 5-16 绿；耳 3/4 黄（head_neck_angle / back_curvature 依赖
        # 耳点，单独标出）。
        # 没检测到(<0.3)的点画灰色空心小圈——示意模型认为的位置，方便定位丢点。
        for pid in range(3, 17):
            p = px(pid)
            if p is None:
                continue
            if vis_of(pid) < VIS_LOW:
                cv2.circle(frame, p, 3, (150, 150, 150), 1, cv2.LINE_AA)
            elif pid <= 4:
                cv2.circle(frame, p, 4, (0, 215, 255), -1, cv2.LINE_AA)  # 黄：耳
            else:
                cv2.circle(frame, p, 4, (0, 255, 0), -1, cv2.LINE_AA)    # 绿：躯干/四肢

        # 标签：每个画出的点（3-16）右下角标 `pid:置信度`，字色按置信度分档（黑描边保证可读）。
        for pid in range(3, 17):
            p = px(pid)
            if p is None:
                continue
            vis = vis_of(pid)
            if vis >= VIS_OK:
                color = (60, 200, 60)      # 绿：置信度够
            elif vis >= VIS_LOW:
                color = (0, 215, 255)      # 黄：低置信度
            else:
                color = (150, 150, 150)    # 灰：没检测到
            label = f"{pid}:{vis:.2f}"
            tx = min(p[0] + 6, w - 42)     # 不超出画面右缘（约 40px 宽的标签）
            ty = p[1] + 12
            cv2.putText(frame, label, (tx, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 连接线（两端都可见才画，防止线条插到残缺点）
        for a, b in POSE_CONNECTIONS:
            if vis_of(a) < VIS_LOW or vis_of(b) < VIS_LOW:
                continue
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

    # ---------- 27 关节读数面板（默认显示，--no-show-schema 关闭） ----------

    def draw_schema(self, frame: np.ndarray, schema: Optional[dict] = None) -> None:
        """画面右下角叠加 27 个规范关节的读数面板（测试用）。

        schema: pose_schema.human_adapter / robot_adapter 的输出 dict。
                右侧竖向列出 27 个 DOF 的 servo_id / 规范名 / 位置 / 角度 / 状态：
                    position 为归一化坐标 (x,y)（关键点数据），na 显示 --；
                    角度 na 显示 --；状态颜色 ok=绿 low_conf=黄 na=灰 offline/error=红。
        Hershey 字体不支持中文，这里用英文规范名；zh_name 在 JSON 导出里带。
        本层只渲染传入的 dict，不 import pose_schema（保持分层）。
        """
        if not schema:
            return
        joints = schema.get("joints")
        if not joints:
            return
        items = list(joints.values())
        h, w = frame.shape[:2]

        header = "Schema (27 DOF)  sv/name/pos/angle/status"
        scale = 0.40
        pad_x, pad_y = 8, 6

        def _row(j: dict) -> str:
            a = j.get("angle_deg")
            ang = "--" if a is None else f"{a:.1f}"
            pos = j.get("position")
            if pos is None or pos.get("x") is None or pos.get("y") is None:
                pos_s = "--"
            else:
                pos_s = f"({pos['x']:.3f},{pos['y']:.3f})"
            return (f"{j['servo_id']:>2} {j['canonical']:<18} "
                    f"{pos_s:>16} {ang:>7} {j.get('status', 'na')}")

        rows = [_row(j) for j in items]

        text_w = 0
        for line in [header] + rows:
            (tw, _th), _base = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            text_w = max(text_w, tw)

        n = len(rows) + 1  # 标题 + 27 行
        line_h = 15
        # 面板装不下（小画面）时压缩行高
        line_h = min(line_h, max(10, (h - 2 * pad_y - 16) // n))

        x0 = max(w - text_w - 2 * pad_x - 8, 0)
        y1 = h - 8
        y0 = y1 - n * line_h - 2 * pad_y

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + text_w + 2 * pad_x, y1),
                      (25, 25, 35), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + text_w + 2 * pad_x, y1),
                      (110, 110, 130), 1)

        cv2.putText(frame, header, (x0 + pad_x, y0 + pad_y + line_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), 1,
                    cv2.LINE_AA)
        for i, j in enumerate(items):
            y = y0 + pad_y + line_h * (i + 2) - 4
            color = _STATUS_COLOR.get(j.get("status"), (150, 150, 150))
            cv2.putText(frame, rows[i], (x0 + pad_x, y),
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


# ---------- 骨架数据导出（规范 schema → JSON） ----------

def export_pose_json(schema: dict, path: Optional[str] = None) -> str:
    """把规范 schema（pose_schema.human_adapter / robot_adapter 的输出）导出为 JSON。

    用对齐后的规范命名（shoulder_left_pitch 等），方便未来直接对接机器人侧 /
    肉眼对比两边数据。path 非 None 时同时写盘（UTF-8）。返回 JSON 字符串。
    """
    text = json.dumps(schema, ensure_ascii=False, indent=2)
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
