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
    fhp_state / cva_proxy_deg: FhpDecision 的头前伸状态（FhpState 枚举）与
            CvaProxyFeatures 的 proxy 平滑角度。fhp_state 非 None 时在**坐姿状态
            行正下方**画一行 `Head Fwd: Normal/Slight/Obvious/N/A (73.2°)`
            （颜色随状态，绿/琥珀/红/灰），好让"坐姿好不好"和"头有没有前伸"
            并排对照；本层只呈现，不判断（头前伸的完整读数仍在 draw_fhp_overlay）。
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

头前伸 CVA-like proxy 叠加 + 逐帧 CSV（新增，2026-09-23）：
    draw_fhp_overlay(frame, cva, fhp_state, decision=None, ...) 可选调用（main
    在 draw_cva_overlay 后调用）。cva: features.CvaProxyFeatures.update() 的
    返回值（含像素几何 cva_proxy_geom）；fhp_state: decision.FhpState 枚举。
    在 CVA 行下方画三行：
        `CVA Proxy: 65.8°  raw 64.9°`（平滑值 + 原始值，按状态着色）
        `FHP State: Normal / Slight / Obvious / N/A`
        `Confidence 0.87`（该帧 proxy 用到的关键点平均置信度）
    另把 proxy 的**几何**画到画面上（耳中点、C7 代理点、两点连线、过 C7 代理
    的水平虚线 + 到耳中点的竖直虚线），几何直接取自 features 给的
    cva_proxy_geom 像素坐标 —— 本层只画不算（分层约束：output 不重算特征）。
    术语：proxy ≠ 临床 CVA（C7 用双肩中点近似、耳点非 tragus），画面上标
    `proxy` 以提示不是标准 CVA。
    FHP 提醒横幅画在第三行横幅位置（久坐、不良坐姿之下），停留
    reminder_hold_sec 秒，语义与另两条一致。
    draw_skeleton_frame(frame, pose) 是骨架叠加的公开入口（骨架本是 draw()
    的一部分）：标定/实验脚本只想看关键点、不想伪造 state 时用。
    FhpCsvLogger(path) 逐帧写 CSV（列名见 FHP_CSV_COLUMNS，含用户要求的
    timestamp / cva_proxy / original_fhp_score / combined_score / fhp_state /
    keypoint_confidence），供离线分析：original_fhp_score = 原有头颈角判据的
    得分（head_neck_angle / 阈值，>=1.0 = 该旧规则已触发），combined_score =
    FhpDecision 的合成得分（两判据取大，配置权重决定用哪几路）。纯记录，
    不做任何判断。

本模块只消费 decision.State/PostureState/CvaLevel/FhpState + float + 五个 dict
（posture/久坐文案/不良坐姿文案/pose 数据/CVA-FSA 特征）。
例外：仅 import features.ANGLE_LEGEND 这一个**数据常量**用于渲染角度说明，
不 import features 的逻辑/类。
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

from decision import State, PostureState, CvaLevel, FhpState, FrontFhpState
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

# 头前伸状态标签/颜色（draw_fhp_overlay 用，BGR）：绿=正常、琥珀=轻微、
# 红=明显、灰=数据不足。用 Normal/Slight/Obvious 三词，便于和 CSV 对齐。
_FHP_STATE_LABEL = {
    FhpState.NORMAL: "Normal",
    FhpState.SLIGHT_FHP: "Slight",
    FhpState.OBVIOUS_FHP: "Obvious",
    FhpState.UNKNOWN: "N/A",
}
_FHP_STATE_COLOR = {
    FhpState.NORMAL: (60, 200, 60),
    FhpState.SLIGHT_FHP: (0, 140, 255),
    FhpState.OBVIOUS_FHP: (0, 60, 220),
    FhpState.UNKNOWN: (180, 180, 180),
}

# 左侧文字块的 y 行位（像素）。自上而下：状态 28 / 坐姿读数 56 / 坐姿状态 84 /
# **头前伸摘要 112** / 计时 140 / CVA 168 / FSA(aux) 190 / 头前伸详情 214 起。
# 摘要行（112）紧贴坐姿状态行，让"坐姿好不好"和"头有没有前伸"能一眼对照
# —— 两者是两条独立判据，同一帧可以一个 GOOD 一个 Slight（见 FhpDecision 说明）。
_FHP_SUMMARY_Y = 112
_TIMER_Y = 140
_CVA_MAIN_Y = 168
_CVA_AUX_Y = 190

# 头前伸详情块起始 y（在 CVA 行 168 / FSA 行 190 之下），行距 22
_FHP_TEXT_Y = 214
_FHP_LINE_H = 22

# 摘要行前缀（状态词复用 _FHP_STATE_LABEL，与画面/CSV 用词一致）
_FHP_SUMMARY_PREFIX = "Head Fwd:"

# 正面实验行的标签/颜色（draw_front_fhp_overlay 用，BGR）：绿=未前伸、
# 红=有前伸、灰=数据不足。**刻意用 Normal/FHP 两个词**（不用 45° 那条线的
# Slight/Obvious）：两条线判的不是一个量，画面用词分开，避免看混。
_FRONT_FHP_STATE_LABEL = {
    FrontFhpState.NORMAL: "Normal",
    FrontFhpState.FHP: "FHP",
    FrontFhpState.UNKNOWN: "N/A",
}
_FRONT_FHP_STATE_COLOR = {
    FrontFhpState.NORMAL: (60, 200, 60),
    FrontFhpState.FHP: (0, 60, 220),
    FrontFhpState.UNKNOWN: (180, 180, 180),
}

# 正面实验行的 y 行位：与 45° proxy 详情块**共用同一个槽**（214/236）。
# 两条线现在分属两个入口（main_demo.py / main_front_demo.py），同屏互斥 ——
# 见 CLAUDE.md「机位路由」，所以这里直接用别名，别再错开一段空白。
_FRONT_FHP_Y = _FHP_TEXT_Y   # 214：左栏 28/56/84/140/168/190/214/236
_FRONT_FHP_PREFIX = "Front FHP (exp):"

# 关节读数面板默认只列**上半身** DOF（坐姿场景下腿/踝基本在画面外，
# 满屏 na 只是噪声）。按规范名前缀过滤，不 import pose_schema（保持分层）。
# 隐藏：hip_* / knee_* / ankle_*（12 个腿 DOF）→ 面板剩 15 个上半身 DOF。
# 注意：只影响**画面面板**，schema 本身与 --dump-schema 导出的 JSON 仍是完整
# 27 DOF（那是给人体→机器人映射用的，不能删）。
_SCHEMA_HIDDEN_PREFIXES = ("hip_", "knee_", "ankle_")


def _visible_joints(joints: dict, upper_only: bool = True) -> list:
    """按规范名前缀过滤面板要显示的 DOF（默认只留上半身）。"""
    items = list(joints.values())
    if not upper_only:
        return items
    return [j for j in items
            if not str(j.get("canonical", "")).startswith(_SCHEMA_HIDDEN_PREFIXES)]

# proxy 几何叠加用色（BGR）：耳中点青、C7 代理点品红、参考虚线灰
_FHP_HEAD_COLOR = (255, 220, 0)
_FHP_C7_COLOR = (230, 0, 230)
_FHP_REF_COLOR = (110, 110, 110)
# 头前伸提醒横幅的强调色（琥珀，与不良坐姿横幅同色系）
_COLOR_FHP_ACCENT = (0, 140, 255)

# 逐帧 CSV 列（前 6 列是需求里点名的，其余是补充的调试/分析列）
FHP_CSV_COLUMNS = (
    "timestamp",             # 该帧时间戳（秒；实时=monotonic，视频=播放进度）
    "cva_proxy",             # CVA-like proxy（平滑后，度）
    "original_fhp_score",    # 原有头颈角判据得分 = head_neck_angle/阈值（>=1.0 即旧规则触发）
    "combined_score",        # FhpDecision 合成得分（两路加权取大；>=1.0 → 至少 Slight）
    "fhp_state",             # NORMAL / SLIGHT_FHP / OBVIOUS_FHP / UNKNOWN
    "keypoint_confidence",   # proxy 用到的关键点平均置信度
    "frame_index",           # 帧序号（从 0 起）
    "time_epoch",            # 墙上时钟 ISO 时间，便于和录像对时
    "cva_proxy_raw",         # 未平滑 proxy（debug 用，看抖动幅度）
    "ratio_proxy",           # proxy 相对阈值的位置 = slight_threshold/proxy（>=1.0 即过阈值）
    "ratio_head",            # 同 original_fhp_score，便于与 ratio_proxy 并列比较
    "valid",                 # proxy 本帧是否有效（0/1）
    "pts",                   # 本帧用到的关键点标签，如 ear_mid+sh_mid
    "posture_label",         # 实验标注（测试脚本用；实时 demo 留空）
)


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
        self._fhp_banner_text: Optional[str] = None       # 当前要显示的头前伸横幅
        self._fhp_banner_until = 0.0                      # 头前伸横幅显示到哪个时刻

    # ---------- 画面 ----------

    def draw(self, frame: np.ndarray, state: State,
             movement: Optional[float],
             posture: Optional[dict] = None,
             reminder: Optional[str] = None,
             pose: Optional[dict] = None,
             posture_state: Optional[PostureState] = None,
             posture_reminder: Optional[str] = None,
             still_elapsed_sec: Optional[float] = None,
             slump_elapsed_sec: Optional[float] = None,
             fhp_state: Optional[FhpState] = None,
             cva_proxy_deg: Optional[float] = None) -> np.ndarray:
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
            fhp_state: FhpDecision.update() 的返回（FhpState 枚举）。非 None 时
                     在**坐姿状态行正下方**画一行摘要 `Head Fwd: Normal/Slight/
                     Obvious/N/A`（颜色随状态：绿/琥珀/红/灰），后面可跟 proxy
                     角度（见 cva_proxy_deg）。只呈现状态，不在本层判头前伸。
                     注意：这与 draw_fhp_overlay 的详情块是**同一个状态**的两处
                     呈现（摘要紧邻坐姿、详情含 raw/conf/计时），不是两条判据。
            cva_proxy_deg: features.CvaProxyFeatures 的 cva_proxy_deg（平滑值，
                     度）。与 fhp_state 同时给时附在摘要行尾 `(64.8°)`，方便
                     边看坐姿边看头前伸的程度；None 时只画状态词。
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

        # 头前伸摘要行（紧贴坐姿状态行）：只呈现 decision 给的状态，不在此判断。
        # 用 _FHP_STATE_LABEL 同一套词（Normal/Slight/Obvious），与画面详情块、
        # CSV 的 fhp_state 对齐；颜色与详情块一致，便于两处互认。
        if fhp_state is not None:
            fhp_color = _FHP_STATE_COLOR.get(fhp_state, (180, 180, 180))
            fhp_line = (f"{_FHP_SUMMARY_PREFIX} "
                        f"{_FHP_STATE_LABEL.get(fhp_state, 'N/A')}")
            if cva_proxy_deg is not None:
                fhp_line += f" ({cva_proxy_deg:.1f}°)"
            cv2.putText(frame, fhp_line, (12, _FHP_SUMMARY_Y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, fhp_color, 1, cv2.LINE_AA)

        # 计时行：Still/Slump 已保持秒数 vs 提醒阈值（方便测试看进度）
        if still_elapsed_sec is not None and slump_elapsed_sec is not None:
            timer_line = (
                f"Still {still_elapsed_sec:.1f}/{self.duration_limit_sec:.0f}s"
                f"  Slump {slump_elapsed_sec:.1f}/"
                f"{self.posture_duration_limit_sec:.0f}s")
            cv2.putText(frame, timer_line, (12, _TIMER_Y),
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
        CVA 主行画在计时行下方 (12, _CVA_MAIN_Y)，按分级着色：
            NORMAL 绿 / MILD 黄 / MODERATE_SEVERE 橙 / SEVERE 红 / UNKNOWN 灰。
            cva_deg 为 None（从未可算）→ `CVA --  N/A`（灰）。
        FSA 是辅助指标：仅当 cva_level 为 MODERATE_SEVERE/SEVERE 且 fsa_deg
        可显示时，以更小的灰白字画在 (12, _CVA_AUX_Y)，标注 aux —— 与 CVA
        分开呈现、不暗示两者权重相同（decision 层不读 FSA，这里只做报告附注）。
        """
        if cva is None or cva_level is None:
            return

        color = _CVA_LEVEL_COLOR.get(cva_level, (180, 180, 180))
        cva_deg = cva.get('cva_deg')
        cva_line = f"CVA {cva_deg:.1f}°  {cva_level.value}" \
            if cva_deg is not None else "CVA --  N/A"
        cv2.putText(frame, cva_line, (12, _CVA_MAIN_Y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # FSA 辅助行（仅中重度及以上附注；无 fsa 值不画）
        if cva_level in (CvaLevel.MODERATE_SEVERE, CvaLevel.SEVERE):
            fsa_deg = cva.get('fsa_deg')
            if fsa_deg is not None:
                trend = cva.get('fsa_trend_deg')
                trend_s = "" if trend is None else f" trend {trend:+.1f}°"
                fsa_line = f"FSA {fsa_deg:.1f}°{trend_s}  aux"
                cv2.putText(frame, fsa_line, (12, _CVA_AUX_Y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1,
                            cv2.LINE_AA)

    # ---------- 头前伸 CVA-like proxy 叠加（2026-09-23 新增） ----------

    def draw_fhp_overlay(self, frame: np.ndarray, cva: Optional[dict],
                         fhp_state: Optional[FhpState] = None,
                         decision=None,
                         alert_elapsed_sec: Optional[float] = None,
                         duration_limit_sec: Optional[float] = None,
                         fhp_reminder: Optional[str] = None,
                         show_geometry: bool = True,
                         posture_label: str = "") -> None:
        """叠加头前伸（CVA-like proxy）三行读数 + proxy 几何 + 提醒横幅。

        参数:
            cva: features.CvaProxyFeatures.update() 的返回值，含
                 'cva_proxy_deg'（平滑值，度）、'cva_proxy_raw_deg'（原始值）、
                 'cva_proxy_valid'、'cva_proxy_conf'、'cva_proxy_pts'、
                 'cva_proxy_geom'（像素坐标 {'head': (x,y), 'c7': (x,y)}）。
            fhp_state: decision.FhpState 枚举（决定三行的着色）。None 时整块不画。
            decision: decision.FhpDecision 实例**或**同名字段的 dict，用来读
                 ratio_proxy / ratio_head / combined_score 附在读数行尾（可选）。
                 只读属性，不做判断（分层：output 不算特征、不做决策）。
            alert_elapsed_sec / duration_limit_sec: FhpAlert 的已持续秒数与该
                 实例的触发阈值（秒），非 None 时在末行附 `FHP 12.3/300s` 进度。
            fhp_reminder: FhpAlert 触发的提醒文案（英文）。非 None 时记录第三
                 条横幅（久坐、不良坐姿横幅下方），停留 reminder_hold_sec 秒。
            show_geometry: 是否把 proxy 的几何（耳中点、C7 代理点、连线、过
                 C7 代理的水平/竖直虚线）画在画面上。几何用 cva_proxy_geom
                 给的像素坐标，本层不重算（--no-geometry 可关）。
            posture_label: 实验标注（如 '1'/'2'/'3'），非空时附在读数行里，
                 方便边测边核对当前在摆哪个姿势；正式 demo 传空串。

        画面文字（英文，Hershey 画不了中文）：
            CVA Proxy: 65.8°  raw 64.9°
            FHP State: Normal / Slight / Obvious / N/A
            Confidence 0.87  ratio 1.03/1.11
        术语提醒：这是 proxy（C7 用双肩中点近似、耳点非 tragus），不是临床 CVA。
        """
        if cva is None or fhp_state is None:
            return

        # 提醒横幅（先记录，和 draw() 的两条同一套停留逻辑）
        if fhp_reminder is not None:
            self._fhp_banner_text = fhp_reminder
            self._fhp_banner_until = time.monotonic() + self.reminder_hold_sec

        # proxy 几何（画在文字之下，避免压住读数）
        if show_geometry:
            geom = cva.get('cva_proxy_geom')
            if geom and geom.get('head') and geom.get('c7'):
                self._draw_fhp_geometry(frame, geom['head'], geom['c7'])

        color = _FHP_STATE_COLOR.get(fhp_state, (180, 180, 180))
        proxy = cva.get('cva_proxy_deg')
        raw = cva.get('cva_proxy_raw_deg')
        conf = cva.get('cva_proxy_conf')

        # 第 1 行：proxy 值（平滑 + 原始）
        if proxy is None:
            line1 = "CVA Proxy: N/A"
        else:
            line1 = f"CVA Proxy: {proxy:.1f}°"
            if raw is not None:
                line1 += f"  raw {raw:.1f}°"

        # 第 2 行：状态（Normal / Slight / Obvious / N/A）
        line2 = f"FHP State: {_FHP_STATE_LABEL.get(fhp_state, 'N/A')}"

        # 第 3 行：置信度 + 比值 + 计时 + 标注
        bits = []
        if conf is not None:
            bits.append(f"Confidence {conf:.2f}")
        if decision is not None:
            rp = _score_of(decision, 'ratio_proxy')
            ob = _score_of(decision, 'obvious_ratio')
            if rp is not None:
                bits.append(f"ratio {rp:.2f}" +
                            (f"/{ob:.2f}" if ob is not None else ""))
        if alert_elapsed_sec is not None and duration_limit_sec is not None:
            bits.append(f"FHP {alert_elapsed_sec:.1f}/{duration_limit_sec:.0f}s")
        if posture_label:
            bits.append(f"label {posture_label}")
        line3 = "  ".join(bits)

        # 关键点标签（哪几个点参与了 proxy），放在状态行之后、同色小字
        pts = cva.get('cva_proxy_pts')

        for i, txt in enumerate((line1, line2, line3)):
            if not txt:
                continue
            _put_text_outlined(frame, txt, (12, _FHP_TEXT_Y + i * _FHP_LINE_H),
                               0.55 if i < 2 else 0.45,
                               color if i < 2 else (200, 200, 200))
        if pts:
            _put_text_outlined(frame, f"proxy pts: {pts}",
                               (12, _FHP_TEXT_Y + 3 * _FHP_LINE_H),
                               0.40, (170, 170, 170))

        # 第三/四条横幅（久坐 0、不良坐姿 _BANNER_H、头前伸 2*_BANNER_H）
        now = time.monotonic()
        if self._fhp_banner_text is not None and now < self._fhp_banner_until:
            self._draw_banner(frame, self._fhp_banner_text,
                              y_top=2 * _BANNER_H, accent=_COLOR_FHP_ACCENT)
        elif now >= self._fhp_banner_until:
            self._fhp_banner_text = None

    def _draw_fhp_geometry(self, frame: np.ndarray, head: tuple,
                           c7: tuple) -> None:
        """画 proxy 的几何：耳中点 → C7 代理点 的连线 + 过 C7 的水平虚线。

        head / c7 是 **像素坐标**（features 层算好的，本层只画）。
        proxy 角 = 连线与水平方向的夹角，所以补一条过 C7 代理点的水平虚线
        和一条从耳中点垂到该水平线的竖直虚线，让画面上的夹角一眼能对上读数。
        """
        hx, hy = int(head[0]), int(head[1])
        cx, cy = int(c7[0]), int(c7[1])
        cv2.line(frame, (cx, cy), (hx, hy), _FHP_HEAD_COLOR, 2, cv2.LINE_AA)
        # 水平参考线：从 C7 代理点伸向耳中点一侧（画面上的"水平"= 像素水平；
        # 角度是在像素空间算的，两者一致），长度取 |Δx| 的 1.25 倍留点余量
        span = max(abs(hx - cx), 12)
        ex = cx + int(span * 1.25) * (1 if hx >= cx else -1)
        self._dashed_line(frame, (cx, cy), (ex, cy), _FHP_REF_COLOR)
        # 竖直虚线：耳中点垂到水平参考线（右角三角，夹角即 proxy）
        self._dashed_line(frame, (hx, hy), (hx, cy), _FHP_REF_COLOR)
        # 端点 + 标签
        cv2.circle(frame, (hx, hy), 5, _FHP_HEAD_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, _FHP_C7_COLOR, -1, cv2.LINE_AA)
        _put_text_outlined(frame, "ear_mid", (hx + 8, hy - 6), 0.40,
                           _FHP_HEAD_COLOR)
        _put_text_outlined(frame, "C7 proxy", (cx + 8, cy + 16), 0.40,
                           _FHP_C7_COLOR)

    # ---------- 正面 Ear–Shoulder proxy 实验行（2026-09-23 新增；只显示不提醒） ----------

    def draw_front_fhp_overlay(self, frame: np.ndarray, esp: Optional[dict],
                               front_state: Optional[FrontFhpState] = None,
                               decision=None,
                               show_geometry: bool = True) -> None:
        """叠加**正面机位** Ear–Shoulder displacement proxy 的两档实验行。

        ⚠ 这是**实验显示行**：不是 45° 那条线的另一处呈现，两者特征/方向/
        阈值都不同（见 decision.FrontFhpState）。**本层只画**传入的状态与读数，
        不判断、不提醒（decision.FrontFhpDecision 也不产出提醒）。

        参数:
            esp: features.EarShoulderProxyFeatures.update() 的返回值，含
                 'ear_shoulder_proxy'（EMA 平滑值）/ 'ear_shoulder_proxy_raw' /
                 'ear_shoulder_proxy_valid' / 'ear_shoulder_proxy_conf' /
                 'ear_shoulder_proxy_geom'（{'ear_mid','sh_mid','sh_width_px'}，
                 像素坐标）。
            front_state: decision.FrontFhpState 枚举（决定着色）。None 时整块不画。
            decision: decision.FrontFhpDecision 实例**或**同名字段的 dict，
                 只用来读 normal_max 附在第二行（只读属性，不做判断）。
            show_geometry: 是否画位移几何（耳中点水平/竖直虚线到肩中点，
                 与 --no-fhp-geometry 共用开关）。

        画面文字（英文，Hershey 画不了中文）：
            Front FHP (exp): Normal  ES 0.041  raw 0.039
            bound 0.065 (in-sample)  conf 0.87  display only, no alert
        """
        if esp is None or front_state is None:
            return

        if show_geometry:
            geom = esp.get('ear_shoulder_proxy_geom')
            if geom and geom.get('ear_mid') and geom.get('sh_mid'):
                self._draw_front_geometry(frame, geom['ear_mid'], geom['sh_mid'])

        color = _FRONT_FHP_STATE_COLOR.get(front_state, (180, 180, 180))
        proxy = esp.get('ear_shoulder_proxy')
        raw = esp.get('ear_shoulder_proxy_raw')
        conf = esp.get('ear_shoulder_proxy_conf')
        degenerate = bool(esp.get('ear_shoulder_proxy_degenerate'))

        # 第 1 行：状态 + 位移指标（平滑值，附原始值）
        line1 = (f"{_FRONT_FHP_PREFIX} "
                 f"{_FRONT_FHP_STATE_LABEL.get(front_state, 'N/A')}")
        if proxy is not None:
            line1 += f"  ES {proxy:.3f}"
        if raw is not None:
            line1 += f"  raw {raw:.3f}"

        # 第 2 行：口径与来源标注（灰白小字）——阈值是 in-sample 种子值，
        # 且本行不参与任何提醒，必须在画面上写明，避免被当判据用。
        bound = _score_of(decision, 'normal_max')
        bits = [f"bound {bound:.3f}" if bound is not None else "bound --",
                "in-sample"]
        if conf is not None:
            bits.append(f"conf {conf:.2f}")
        if degenerate:
            bits.append("view degenerate")
        bits.append("display only, no alert")
        line2 = "  ".join(bits)

        _put_text_outlined(frame, line1, (12, _FRONT_FHP_Y), 0.55, color)
        _put_text_outlined(frame, line2, (12, _FRONT_FHP_Y + _FHP_LINE_H),
                           0.40, (200, 200, 200))

    def _draw_front_geometry(self, frame: np.ndarray, ear_mid: tuple,
                             sh_mid: tuple) -> None:
        """画正面位移几何：耳中点 → 肩中点 的水平/竖直位移（两条虚线）。

        ear_mid / sh_mid 是**像素坐标**（features 层算好的，本层只画）。
        该 proxy = |Δx| / 肩宽，所以画一条过肩中点的水平虚线到耳中点正下方、
        再画一条竖直虚线连到耳中点 —— 夹角无关，看的就是这段水平位移相对
        肩宽有多大（与 45° 那条线的"夹角"几何区分开）。
        """
        ex, ey = int(ear_mid[0]), int(ear_mid[1])
        sx, sy = int(sh_mid[0]), int(sh_mid[1])
        # 水平位移（proxy 的分子）：从肩中点沿 y=sy 拉到耳中点所在的 x
        self._dashed_line(frame, (sx, sy), (ex, sy), _FHP_REF_COLOR)
        # 竖直连接（耳中点 → 水平线），画出来才知道两个中点差多少高度
        self._dashed_line(frame, (ex, sy), (ex, ey), _FHP_REF_COLOR)
        cv2.circle(frame, (ex, ey), 5, _FHP_HEAD_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy), 5, _FHP_C7_COLOR, -1, cv2.LINE_AA)
        _put_text_outlined(frame, "ear_mid", (ex + 8, ey - 6), 0.40,
                           _FHP_HEAD_COLOR)
        _put_text_outlined(frame, "sh_mid", (sx + 8, sy + 16), 0.40,
                           _FHP_C7_COLOR)
        # 双箭头标出 |Δx|（就是 proxy 的分子），一眼对上读数
        cv2.arrowedLine(frame, (sx, sy - 6), (ex, sy - 6), _FHP_HEAD_COLOR, 1,
                        cv2.LINE_AA, tipLength=0.15)
        cv2.arrowedLine(frame, (ex, sy - 6), (sx, sy - 6), _FHP_HEAD_COLOR, 1,
                        cv2.LINE_AA, tipLength=0.15)

    # ---------- 骨架叠加 ----------

    def draw_skeleton_frame(self, frame: np.ndarray, pose: Optional[dict]) -> None:
        """只叠骨架（关键点 + 连线 + pid/置信度标签），不画状态/角度/横幅。

        骨架本来是 draw() 的一部分；标定/实验脚本只想要骨架（核对参与 proxy
        计算的关键点），又不想为了调 draw() 伪造一个 State，用这个入口。
        受 draw_skeleton 开关控制（--no-skeleton 时什么都不画）。
        """
        if self.draw_skeleton and pose is not None:
            self._draw_skeleton(frame, pose)

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

    def draw_schema(self, frame: np.ndarray, schema: Optional[dict] = None,
                    upper_only: bool = True) -> None:
        """画面右下角叠加规范关节的读数面板（测试用）。

        schema: pose_schema.human_adapter / robot_adapter 的输出 dict。
                右侧竖向列出各 DOF 的 servo_id / 规范名 / 位置 / 角度 / 状态：
                    position 为归一化坐标 (x,y)（关键点数据），na 显示 --；
                    角度 na 显示 --；状态颜色 ok=绿 low_conf=黄 na=灰 offline/error=红。
        upper_only: True（默认）只列**上半身** DOF（过滤 hip_/knee_/ankle_ 共 12 个
                腿 DOF）—— 坐姿场景里腿脚基本在画面外，列出来全是 na。
                置 False 显示全部 27 个。**只影响本面板**：传入的 schema 与
                --dump-schema 导出的 JSON 始终是完整 27 DOF（给映射用，不能删）。
        Hershey 字体不支持中文，这里用英文规范名；zh_name 在 JSON 导出里带。
        本层只渲染传入的 dict，不 import pose_schema（保持分层）。
        """
        if not schema:
            return
        joints = schema.get("joints")
        if not joints:
            return
        items = _visible_joints(joints, upper_only)
        if not items:
            return
        h, w = frame.shape[:2]

        total = len(joints)
        header = (f"Schema upper-body ({len(items)}/{total} DOF)"
                  if upper_only else f"Schema ({total} DOF)") + \
            "  sv/name/pos/angle/status"
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

        n = len(rows) + 1  # 标题 + 各 DOF 行（默认 15 行上半身）
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


# ---------- 绘制小工具 ----------

def _put_text_outlined(frame: np.ndarray, text: str, org: tuple,
                       scale: float = 0.5, color: tuple = (230, 230, 230),
                       thickness: int = 1) -> None:
    """画一行带黑色描边的文字（视频画面上比纯色文字好认，深色/浅色背景都清楚）。

    描边用**上下左右各偏移 1px 的 4 次细笔画**，不用「同一位置先粗后细画两遍」：
    后者在 OpenCV 5.0（LINE_AA）上会把最后一个字重复画在字符串末尾（画面右侧
    多出一个淡淡的字影），实测粗笔画（thickness>=2）重叠同一个 baseline 就触发。
    4 次偏移的都是 thickness=1，不触发该问题（2026-09-23 实拍对比确认）。
    """
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(frame, text, (org[0] + dx, org[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness,
                    cv2.LINE_AA)
    cv2.putText(frame, text, org,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


def _score_of(src, name: str) -> Optional[float]:
    """从决策对象或 dict 里取一个只读数值（取不到返回 None）。

    draw_fhp_overlay / FhpCsvLogger 需要读 decision.FhpDecision 的
    ratio_proxy / ratio_head / combined_score / obvious_ratio；写成对象或 dict
    都能取，省得调用方为了打印再包一个 dict（同时保持本层不做判断）。
    """
    if src is None:
        return None
    if isinstance(src, dict):
        return src.get(name)
    return getattr(src, name, None)


# ---------- 头前伸逐帧 CSV ----------

class FhpCsvLogger:
    """逐帧记录头前伸相关读数的 CSV 写入器（只记录，不做任何判断）。

    列见 FHP_CSV_COLUMNS：前 6 列是需求点名的
    `timestamp, cva_proxy, original_fhp_score, combined_score, fhp_state,
    keypoint_confidence`，后面是为离线分析补的（原始未平滑 proxy、比值、
    有效性、用到的关键点、实验标注等）。

    用法（上下文管理器，异常退出也会 flush/close）：
        with FhpCsvLogger("fhp.csv") as lg:
            lg.log(ts, cva, state, decision, frame_index=i)

    说明:
        - `original_fhp_score` 是**原有**头颈角判据的得分（head_neck_angle /
          阈值，>=1.0 表示旧规则会判 FHP），和 `cva_proxy` 并列，方便直接比
          A（旧判据）/ B（proxy）/ C（合成）三条线。
        - 三列里任一缺失写空字符串，不写 0 —— 避免把"没数据"当成"读数为 0"。
        - 默认每 flush_every 行 flush 一次，实时跑时中途断电也留得住数据。
    """

    def __init__(self, path: str, extra_columns=(), flush_every: int = 30) -> None:
        """
        参数:
            path:          CSV 输出路径（UTF-8，覆盖写）。
            extra_columns: 追加到标准列之后的列名（如实验脚本的 state_a/state_b），
                           log(..., extra={...}) 按这些名字取值。
            flush_every:   每写多少行 flush 一次。
        """
        self.path = path
        self.extra_columns = tuple(extra_columns)
        self.flush_every = max(1, int(flush_every))
        self._n = 0
        self._closed = False
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(list(FHP_CSV_COLUMNS) + list(self.extra_columns))

    def __enter__(self) -> "FhpCsvLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def log(self, timestamp: float,
            cva: Optional[dict] = None,
            fhp_state: Optional[FhpState] = None,
            decision=None,
            frame_index: Optional[int] = None,
            posture_label: str = "",
            extra: Optional[dict] = None) -> None:
        """写一行。

        参数:
            timestamp:   该帧时间戳（秒）。实时=time.monotonic()，视频=播放进度秒。
            cva:         features.CvaProxyFeatures.update() 的返回值。
            fhp_state:   decision.FhpState 枚举。
            decision:    decision.FhpDecision 实例或同名字段的 dict（读
                         combined_score / ratio_proxy / ratio_head）。None 时
                         这三列留空（例如只想记 proxy 原始值做标定时）。
            frame_index: 帧序号（可选）。
            posture_label: 实验标注（'1'/'2'/'3' 或 Normal/… 名字），实时 demo 留空。
            extra:       追加列的值（键 = 构造时给的 extra_columns 名）。
        """
        cva = cva or {}
        rp = _score_of(decision, 'ratio_proxy')
        rh = _score_of(decision, 'ratio_head')
        row = [
            _num(timestamp),
            _num(cva.get('cva_proxy_deg')),
            _num(rh),                      # original_fhp_score = 原有头颈角得分
            _num(_score_of(decision, 'combined_score')),
            fhp_state.value if isinstance(fhp_state, FhpState) else (
                '' if fhp_state is None else str(fhp_state)),
            _num(cva.get('cva_proxy_conf')),
            '' if frame_index is None else int(frame_index),
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            _num(cva.get('cva_proxy_raw_deg')),
            _num(rp),
            _num(rh),
            '' if cva.get('cva_proxy_valid') is None else int(bool(
                cva.get('cva_proxy_valid'))),
            cva.get('cva_proxy_pts') or '',
            posture_label or '',
        ]
        for name in self.extra_columns:
            v = (extra or {}).get(name)
            if isinstance(v, FhpState):
                row.append(v.value)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                row.append(_num(v))
            else:
                row.append('' if v is None else str(v))

        self._writer.writerow(row)
        self._n += 1
        if self._n % self.flush_every == 0:
            self._fh.flush()

    def flush(self) -> None:
        """把缓冲区落盘（实时跑着中途要读回 CSV 出报告时先调一次）。"""
        if not self._closed:
            self._fh.flush()

    def close(self) -> None:
        """flush 并关闭文件（重复调用安全）。"""
        if self._closed:
            return
        self._fh.flush()
        self._fh.close()
        self._closed = True


def _num(v) -> str:
    """数值列格式化：None → 空字符串（不写 0，避免"没数据"被当读数），
    其余统一 6 位有效数字（角度够用，也能看出抖动）。"""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)


def selftest_schema_upper_only() -> None:
    """自测关节面板的"只显示上半身"过滤：腿/踝过滤掉、肩/肘/腕/躯干保留，
    upper_only=False 时回到全部，且过滤不改动传入的 schema（这里用造的
    dict，不 import pose_schema —— 本层只渲染传入的数据）。"""
    names = ("waist_pitch", "neck_pitch", "shoulder_left_pitch",
             "elbow_left_pitch", "wrist_right_pitch", "chest_roll",
             "hip_left_pitch", "knee_left_pitch", "ankle_right_roll")
    schema = {"joints": {n: {"canonical": n, "servo_id": i}
                         for i, n in enumerate(names, 1)}}
    kept = [j["canonical"] for j in _visible_joints(schema["joints"], True)]
    dropped = [c for c in schema["joints"] if c not in kept]
    assert dropped == ["hip_left_pitch", "knee_left_pitch", "ankle_right_roll"], \
        dropped
    assert len(kept) == 6 and "shoulder_left_pitch" in kept and "neck_pitch" in kept
    assert len(_visible_joints(schema["joints"], False)) == len(names), \
        "关掉过滤应显示全部"
    assert len(schema["joints"]) == len(names), "过滤不能改动传入的 schema"

    # 面板能画：黑底帧上跑一遍不报错（含 status/position 缺失的降级路径）
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    FrameRenderer(draw_skeleton=False).draw_schema(blank, schema)
    FrameRenderer(draw_skeleton=False).draw_schema(blank.copy(), schema,
                                                   upper_only=False)
    print("  selftest_schema_upper_only: OK")


def _summary_row_pixels(frame: np.ndarray, near: Optional[tuple] = None,
                        y: int = _FHP_SUMMARY_Y) -> int:
    """文字行（y 上下 12px 带）里的像素数：near=None 时数所有非黑像素；
    给了 BGR 颜色则数"接近该色"的像素（抗锯齿会把边缘混色，所以按距离判定，
    不要求精确相等）。y 默认是 45° 线的摘要行；正面实验行传 _FRONT_FHP_Y
    （两个入口互斥，那块槽位在两边的含义不同，但一行都不会同时出现）。"""
    band = frame[y - 12:y + 6]
    if near is None:
        return int(np.count_nonzero(band.any(axis=2)))
    # int32 起步：差值平方最大 255²=65025，int16 会溢出成负数
    diff = band.astype(np.int32) - np.array(near, dtype=np.int32)
    return int(np.count_nonzero(np.sqrt((diff ** 2).sum(axis=2)) < 40))


def selftest_fhp_summary_line() -> None:
    """自测 draw() 里的头前伸摘要行：不传 fhp_state 时该行不画，传了则画在
    坐姿状态行下方、颜色随状态。用黑底帧数像素，不碰摄像头。"""
    frame_probe = lambda **kw: _summary_row_pixels(
        FrameRenderer(draw_skeleton=False).draw(
            np.zeros((400, 640, 3), dtype=np.uint8), State.STILL, 0.0,
            posture={'head_neck_angle': 12.0}, posture_state=PostureState.GOOD,
            **kw))

    blank = frame_probe()
    assert blank == 0, f"不给 fhp_state 时摘要行不该有像素，实际 {blank}"
    for st in (FhpState.NORMAL, FhpState.SLIGHT_FHP, FhpState.OBVIOUS_FHP,
               FhpState.UNKNOWN):
        n = frame_probe(fhp_state=st, cva_proxy_deg=64.8)
        assert n > 0, f"{st} 摘要行没画出来"
        # 颜色必须取该状态的规定色（绿/琥珀/红/灰），不能一律白字
        want = _FHP_STATE_COLOR[st]
        got = _summary_row_pixels(
            FrameRenderer(draw_skeleton=False).draw(
                np.zeros((400, 640, 3), dtype=np.uint8), State.STILL, 0.0,
                posture={'head_neck_angle': 12.0},
                posture_state=PostureState.GOOD, fhp_state=st,
                cva_proxy_deg=64.8), near=want)
        assert got > 0, f"{st} 摘要行颜色不是 {want}"
    print("  selftest_fhp_summary_line: OK")


def selftest_front_fhp_line() -> None:
    """自测 draw_front_fhp_overlay（正面 Ear–Shoulder 两档**实验显示行**）：
    不传状态时整块不画、传了就画、颜色随状态、无读数/退化帧照画（判 UNKNOWN
    是 decision 的事，output 只呈现）、几何开关生效、且几何区域与文字行不重叠。
    用黑底帧数像素，不碰摄像头。"""
    from decision import FrontFhpDecision

    def esp(proxy=0.03, raw=None, valid=True, degenerate=False, geom=None):
        return {'ear_shoulder_proxy': proxy,
                'ear_shoulder_proxy_raw': proxy if raw is None else raw,
                'ear_shoulder_proxy_valid': valid,
                'ear_shoulder_proxy_conf': 0.87,
                'ear_shoulder_proxy_pts': 'ear_mid+sh_mid',
                'ear_shoulder_proxy_geom': geom,
                'ear_shoulder_proxy_parts': {},
                'ear_shoulder_proxy_degenerate': degenerate}

    geom = {'ear_mid': (300.0, 90.0), 'sh_mid': (280.0, 170.0),
            'sh_width_px': 460.0}
    # 几何整段画在 y≈90~170，避开文字行（_FRONT_FHP_Y=214 / +22=236 的取像素带
    # 是 [202,220) 与 [224,242)），这样数几何像素不会被文字串进来
    geom_box = lambda f: int(np.count_nonzero(
        f[60:200, 260:360].any(axis=2)))

    def render(state, payload, show_geometry=False):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        FrameRenderer(draw_skeleton=False).draw_front_fhp_overlay(
            frame, payload, state, FrontFhpDecision(),
            show_geometry=show_geometry)
        return frame

    # 1) 不给状态 / 不给特征 → 什么都不画
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    fr = FrameRenderer(draw_skeleton=False)
    fr.draw_front_fhp_overlay(blank, esp(), None, FrontFhpDecision())
    fr.draw_front_fhp_overlay(blank, None, FrontFhpState.NORMAL,
                              FrontFhpDecision())
    assert int(np.count_nonzero(blank)) == 0, "状态或特征缺失时不该画任何像素"

    # 2) 三种状态都画得出来，且颜色取该状态的规定色（绿/红/灰）
    for st in (FrontFhpState.NORMAL, FrontFhpState.FHP, FrontFhpState.UNKNOWN):
        f = render(st, esp())
        assert _summary_row_pixels(f, y=_FRONT_FHP_Y) > 0, f"{st} 实验行没画出来"
        assert _summary_row_pixels(f, near=_FRONT_FHP_STATE_COLOR[st],
                                   y=_FRONT_FHP_Y) > 0, \
            f"{st} 实验行颜色不是 {_FRONT_FHP_STATE_COLOR[st]}"
        # 第二行（口径标注）也要有：bound/in-sample/display only
        assert _summary_row_pixels(f, y=_FRONT_FHP_Y + _FHP_LINE_H) > 0, \
            f"{st} 缺第二行口径标注"

    # 3) 无读数 / 退化帧照画（读数是 options：本层只呈现，判定在 decision）
    for payload in (esp(proxy=None, valid=False),
                    esp(proxy=0.20, valid=False, degenerate=True),
                    esp(proxy=None)):
        f = render(FrontFhpState.UNKNOWN, payload)
        assert _summary_row_pixels(f, y=_FRONT_FHP_Y) > 0, \
            "无读数/退化帧也应画出 N/A（不能让整行消失）"

    # 4) 几何开关：开→有像素，关→没有（画的是 features 给的像素坐标，不重算）
    assert geom_box(render(FrontFhpState.NORMAL, esp(geom=geom),
                           show_geometry=True)) > 0
    assert geom_box(render(FrontFhpState.NORMAL, esp(geom=geom),
                           show_geometry=False)) == 0
    # 没有 geom（预热/退化时 features 可能给 None）也不能崩
    assert geom_box(render(FrontFhpState.NORMAL, esp(geom=None),
                           show_geometry=True)) == 0

    # 5) 与 45° 详情块**共用槽位**（两个入口互斥：main_demo.py 只跑 45°、
    #    main_front_demo.py 只跑正面，同屏不会同时出现两条线的文字）
    assert _FRONT_FHP_Y == _FHP_TEXT_Y, \
        "正面实验行应与 45° 详情块共用首行槽位（入口互斥，见 CLAUDE.md 机位路由）"
    assert _FRONT_FHP_Y > _CVA_AUX_Y, "正面实验行要在 CVA/FSA 行之下，别叠在一起"
    print("  selftest_front_fhp_line: OK")


def selftest_fhp_output() -> None:
    """自测 FhpCsvLogger（不依赖 cv2 绘制）：列名、空值不写 0、追加列、
    dict/对象两种传参、重复 close 安全。CSV 写在临时目录。"""
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "fhp_selftest.csv")
    with FhpCsvLogger(path, extra_columns=("state_a", "state_b")) as lg:
        # dict 传参 + 缺值：combined_score/ratio 缺失应为空
        lg.log(1.5, {'cva_proxy_deg': 65.0, 'cva_proxy_raw_deg': 64.0,
                     'cva_proxy_conf': 0.9, 'cva_proxy_valid': True,
                     'cva_proxy_pts': 'ear_mid+sh_mid'},
               FhpState.SLIGHT_FHP, {'ratio_proxy': 1.03, 'ratio_head': 2.0,
                                     'combined_score': 2.0},
               frame_index=0, posture_label="2",
               extra={'state_a': FhpState.NORMAL, 'state_b': 0.5})
        # 全空：整行都是空列，不应写 0
        lg.log(2.5, None, None, None, frame_index=1)

    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]
    assert rows[0][:6] == ["timestamp", "cva_proxy", "original_fhp_score",
                           "combined_score", "fhp_state",
                           "keypoint_confidence"], rows[0][:6]
    r1 = dict(zip(rows[0], rows[1]))
    assert r1["timestamp"] == "1.5" and r1["cva_proxy"] == "65"
    assert r1["original_fhp_score"] == "2" and r1["combined_score"] == "2"
    assert r1["fhp_state"] == "SLIGHT_FHP"
    assert r1["state_a"] == "NORMAL" and r1["state_b"] == "0.5"
    assert r1["posture_label"] == "2"
    r2 = dict(zip(rows[0], rows[2]))
    for col in ("cva_proxy", "original_fhp_score", "combined_score",
                "keypoint_confidence", "cva_proxy_raw", "valid", "pts"):
        assert r2[col] == "", f"{col} 空值应写空串，实际 {r2[col]!r}"
    assert r2["fhp_state"] == "" and r2["frame_index"] == "1"

    # 对象传参（FhpDecision）+ 重复 close
    from decision import FhpDecision
    d = FhpDecision()
    d.update({'cva_proxy_deg': 55.0, 'cva_proxy_conf': 0.9})
    lg2 = FhpCsvLogger(path)
    lg2.log(3.0, {'cva_proxy_deg': 55.0}, FhpState.OBVIOUS_FHP, d, frame_index=2)
    lg2.close()
    lg2.close()   # 重复 close 安全
    with open(path, encoding="utf-8") as f:
        last = [r for r in csv.reader(f) if r][-1]
    # combined_score = ratio_proxy = 68.17/55 ≈ 1.23945（对象属性读法）
    assert last[3].startswith("1.239"), last[3]
    assert last[4] == "OBVIOUS_FHP"
    print("  selftest_fhp_output: OK")


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


if __name__ == "__main__":
    # 自测（不需要摄像头；绘制部分用黑底帧数像素，其余靠人工跑 demo 看）
    selftest_fhp_output()
    selftest_fhp_summary_line()
    selftest_front_fhp_line()
    selftest_schema_upper_only()
    print("output selftest: ALL PASSED")
