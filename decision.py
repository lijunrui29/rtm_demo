"""
decision 层：把 features 层给出的特征映射成状态，再决定"要不要提醒"。

两类判断（都带迟滞，防边界抖动）：

1) 静止/在动（消费移动量）：
    movement <= still_threshold   -> STILL（静止）
    movement >= moving_threshold  -> MOVING（在动）
    still_threshold < movement < moving_threshold -> 保持当前状态（迟滞带）

2) 坐姿/弓背（消费 PostureFeatures 的角度 dict）：
    对每个可见角度算 ratio = 角度 / 阈值，取最大者 worst：
    worst >= 1.0            -> SLUMPED（弓背/头前伸/前倾，至少一个超限）
    worst <= hysteresis_ratio -> GOOD（都明显低于阈值）
    中间 -> 迟滞带，保持当前状态。髋不可见时只按 head_neck_angle /
    neck_compression 判。
    特殊项 neck_compression（颈压缩，见 features.py）是"越小越弓背"：
    耸肩+低头会压扁耳-肩竖直间距，正面摄像头也看得到。它和其它角度方向
    相反，ratio 取倒数（阈值/值），其余迟滞语义完全一致。

3) CVA 分级（消费 ErgonomicRiskFeatures 的平滑颅椎角）：
    CvaRisk 按分级阈值把 CVA 映射为 NORMAL / MILD / MODERATE_SEVERE / SEVERE，
    阈值默认 55/50/44（Mostafaee et al. 2022 观察性分组，**非临床诊断标准**；
    本系统用肩点近似 C7，量值需实拍自标定）。CVA 已由 features 层做滑动窗口
    中位数平滑，这里不额外加迟滞。FSA（前伸肩角）是辅助指标，本层**不读取**
    —— 只在 CVA 中重度及以上时由 output 层附注显示。

4) 头前伸分级（消费 CvaProxyFeatures 的 CVA-like proxy，可选消费原有头颈角）：
    FhpDecision 把 proxy 映射为 NORMAL / SLIGHT_FHP / OBVIOUS_FHP 三档，两级
    阈值（slight / obvious）是**项目自标定的实验阈值**，按机位取
    CVA_PROXY_VIEW_PRESETS（**不是临床 CVA 阈值，别套 50° 之类的文献数**）。
    原有头颈角（PostureFeatures.head_neck_angle）作为**弱判据**一起参与：
    它能靠 A/B/C 权重把状态从 NORMAL 抬到 SLIGHT，但**不会单独判出 OBVIOUS**
    （只有 proxy 有两级阈值）。两路合成一个 combined_score，也写进逐帧 CSV。

"持续满多久才算数"这类时间规则，由 _DurationAlert 基类的三个子类实现：
    SedentaryAlert 累计"连续静止"时长（消费 StillnessDecision.State）；
    PostureAlert    累计"连续不良坐姿"时长（消费 PostureDecision.PostureState）；
    FhpAlert        累计"连续头前伸"时长（消费 FhpDecision.FhpState）。
三者不改 StillnessDecision / PostureDecision 的接口。

接口约定（保持稳定，别改签名）：
    State 枚举：STILL / MOVING / UNKNOWN
    StillnessDecision(still_threshold=0.05, moving_threshold=0.10)
    update(movement: Optional[float]) -> State
    state（只读属性）

    PostureState 枚举：GOOD / SLUMPED / UNKNOWN
    PostureDecision(head_neck_threshold=30.0, torso_threshold=30.0,
                    back_threshold=25.0, neck_threshold=0.45,
                    hysteresis_ratio=0.8)
    update(posture: Optional[dict]) -> PostureState
    posture（只读属性）

    SedentaryAlert(duration_limit_sec=1200.0)            久坐提醒
    update(state, timestamp=None) -> Optional[str]       触发时返回提醒文案（英文），否则 None
    elapsed_sec（只读属性）                               当前连续久坐秒数

    PostureAlert(duration_limit_sec=300.0)               不良坐姿提醒
    update(posture_state, timestamp=None) -> Optional[str]
    elapsed_sec（只读属性）                               当前连续不良坐姿秒数

    CvaLevel 枚举：NORMAL / MILD / MODERATE_SEVERE / SEVERE / UNKNOWN
    CvaRisk(normal_threshold=55.0, mild_threshold=50.0, severe_threshold=44.0)
    update(cva_deg: Optional[float]) -> CvaLevel        颅椎角分级（不读 FSA）
    level（只读属性）

    FhpState 枚举：NORMAL / SLIGHT_FHP / OBVIOUS_FHP / UNKNOWN
    FhpDecision(slight_threshold=68.17, obvious_threshold=61.42,
                hysteresis_ratio=0.8, min_conf=0.0, head_neck_threshold=30.0,
                weight_proxy=1.0, weight_head=1.0)
    update(cva: Optional[dict], posture: Optional[dict] = None) -> FhpState
    state / combined_score / ratio_proxy / ratio_head（只读属性）

    FhpAlert(duration_limit_sec=300.0, min_state=FhpState.SLIGHT_FHP)
    update(fhp_state, timestamp=None) -> Optional[str]
    elapsed_sec（只读属性）
"""

from __future__ import annotations

import enum
from typing import Optional


class State(enum.Enum):
    """人员状态。"""
    STILL = "STILL"
    MOVING = "MOVING"
    UNKNOWN = "UNKNOWN"


class StillnessDecision:
    """基于移动量的静止/在动判断（迟滞 + 无人兜底）。"""

    def __init__(self,
                 still_threshold: float = 0.05,
                 moving_threshold: float = 0.10) -> None:
        """
        参数:
            still_threshold:  移动量 <= 此值判为静止。
            moving_threshold: 移动量 >= 此值判为在动。
                              两个阈值之间的区间是"迟滞带"：保持当前状态。
                              默认 0.05 / 0.10 是瞎猜的起始值，
                              用 --debug 标定后调这两个参数。
        """
        if moving_threshold < still_threshold:
            raise ValueError("moving_threshold 不能小于 still_threshold")
        self.still_threshold = still_threshold
        self.moving_threshold = moving_threshold
        self._state = State.UNKNOWN

    @property
    def state(self) -> State:
        return self._state

    def update(self, movement: Optional[float]) -> State:
        """输入一帧的移动量，更新并返回当前状态。

        movement 为 None（没检测到人 / 数据不足）→ UNKNOWN。
        迟滞带内的处理：
            - 当前不是 UNKNOWN → 保持当前状态；
            - 当前是 UNKNOWN（首次/无人刚恢复）→ 默认 MOVING（fail-safe：
              宁可先判定"在动"，也不在没有充分证据时断言"静止"）。
        """
        if movement is None:
            self._state = State.UNKNOWN
            return self._state

        if movement <= self.still_threshold:
            self._state = State.STILL
        elif movement >= self.moving_threshold:
            self._state = State.MOVING
        else:
            # 迟滞带：保持当前状态；UNKNOWN 时兜底为 MOVING
            if self._state is State.UNKNOWN:
                self._state = State.MOVING
        return self._state


class _DurationAlert:
    """连续处于"触发状态"满时长 → 触发一次提醒的通用基类（锁存 + 冻结）。

    子类只需做两件事：
        1) __init__ 里通过 super().__init__(duration_limit_sec, message_template)
           给出"连续满多少秒触发"和触发文案模板（模板里用 {dur} 占位，
           如 "Been sitting for {dur}. ..."）；
        2) 实现 _state_kind(value)，把输入状态归为三类：
               "ACCUM"  累计时长（触发状态，如 STILL / SLUMPED）
               "RESET"  清零并重新上膛（如 MOVING / GOOD —— 恢复后重新计时）
               "FREEZE" 冻结不计、不清零（如 UNKNOWN —— 检测丢失不冤枉用户）
    累计/锁存/文案生成都收敛在这里，避免 SedentaryAlert / PostureAlert 复制逻辑。

    锁存语义：触发后保持触发状态期间不重复触发，只有 RESET 清零之后
    才允许再次触发（即"满时长提醒一次，恢复后再满再提醒"）。
    """

    def __init__(self, duration_limit_sec: float, message_template: str) -> None:
        """
        参数:
            duration_limit_sec: 连续处于触发状态满多少秒触发提醒。
            message_template:   触发文案模板，{dur} 会被替换为时长字符串。
        """
        if duration_limit_sec <= 0:
            raise ValueError("duration_limit_sec 必须 > 0")
        self.duration_limit_sec = duration_limit_sec
        self._message_template = message_template

        self._elapsed_sec = 0.0     # 当前连续触发状态秒数
        self._armed = True          # 是否允许触发（RESET 清零后重新上膛）
        self._last_ts = None        # 上次 update 的时间戳（增量计时的起点）

    @property
    def elapsed_sec(self) -> float:
        """当前连续触发状态秒数（只读）。"""
        return self._elapsed_sec

    def _state_kind(self, value) -> str:
        """把输入状态归为 "ACCUM" / "RESET" / "FREEZE" 三类（子类实现）。"""
        raise NotImplementedError

    def update(self, value, timestamp: Optional[float] = None) -> Optional[str]:
        """喂入一帧的状态，累计时长；触发提醒时返回英文文案，否则 None。

        参数:
            value:     子类约定的状态值（如 State / PostureState）。
            timestamp: 该帧的单调时间戳（秒），默认 time.monotonic()。
                       与 features 层约定一致：视频回放/自测时请传入模拟时间戳。

        返回:
            str 提醒文案（仅触发的那一帧返回），其余 None。
        """
        import time
        now = timestamp if timestamp is not None else time.monotonic()

        kind = self._state_kind(value)

        if kind == "RESET":
            # 状态恢复正常（起身/坐直）：清零计时，重新上膛
            self._elapsed_sec = 0.0
            self._armed = True
            self._last_ts = None
            return None

        if kind == "FREEZE":
            # 检测丢失：冻结不计，但不清零（不冤枉用户）
            self._last_ts = None
            return None

        # "ACCUM"：按时间戳增量累计
        if self._last_ts is not None:
            self._elapsed_sec += max(0.0, now - self._last_ts)
        self._last_ts = now

        if self._armed and self._elapsed_sec >= self.duration_limit_sec:
            # 触发一次，锁存（直到 RESET 清零后才重新允许）
            self._armed = False
            # 时长显示：≥ 1 分钟用分钟，否则用秒（自测用秒级阈值也看得懂）
            sec = self.duration_limit_sec
            dur_str = f"{sec / 60.0:.0f} min" if sec >= 60 else f"{sec:.0f} s"
            return self._message_template.format(dur=dur_str)

        return None


class SedentaryAlert(_DurationAlert):
    """久坐提醒：累计"连续静止"时长，满了就触发一次提醒。

    消费 StillnessDecision.State（本模块的枚举），不在 decision 层之外重复
    判断静止/在动：

        STILL   累计连续静止时长（按传入 timestamp 的增量）。
        MOVING  立即清零（人动了 = 起身，重新开始计时）。
        UNKNOWN 冻结不计，但不清零（检测丢帧/短暂出框不冤枉用户）。

    累计 ≥ duration_limit_sec → 触发一次提醒，返回英文文案；
    锁存语义由 _DurationAlert 基类统一处理（详见该类 docstring）。

    接口约定（保持稳定，别改签名）：
        SedentaryAlert(duration_limit_sec=1200.0)   # 1200 秒 = 20 分钟
        update(state: State, timestamp=None) -> Optional[str]
        elapsed_sec（只读属性）  当前连续久坐秒数
    """

    def __init__(self, duration_limit_sec: float = 1200.0) -> None:
        """
        参数:
            duration_limit_sec: 连续静止满多少秒触发提醒。默认 1200 = 20 分钟。
        """
        super().__init__(duration_limit_sec,
                         message_template="Been sitting for {dur}. "
                                          "Time to stand up and move!")

    def _state_kind(self, state: State) -> str:
        if state is State.MOVING:
            return "RESET"
        if state is State.UNKNOWN:
            return "FREEZE"
        return "ACCUM"  # STILL


class PostureState(enum.Enum):
    """坐姿状态。"""
    GOOD = "GOOD"
    SLUMPED = "SLUMPED"
    UNKNOWN = "UNKNOWN"


class PostureDecision:
    """基于坐姿角度/比值判断坐姿好坏（迟滞，消费 PostureFeatures 的 dict）。

    对当前可见的每个特征算 ratio，取最大者 worst：
        worst >= 1.0              -> SLUMPED（弓背/头前伸/前倾，至少一个超限）
        worst <= hysteresis_ratio -> GOOD（所有特征都明显优于阈值）
        两者之间                    -> 迟滞带，保持当前状态（防边界抖动）

    特殊项 neck_compression（颈压缩，比值，越小越弓背）：方向与其它角度相反，
    ratio 取倒数（阈值/值）—— 值 <= 阈值 时 ratio >= 1 → SLUMPED，其余逻辑
    （<= 0.8 恢复、迟滞带保持）与其它角度完全一致。

    只对当前可见的特征判断：髋不可见（人只露出上半身）时，torso/back 两个
    角度缺失，只按 head_neck_angle / neck_compression 判；posture 为 None
    （无人/数据不足）→ UNKNOWN。

    迟滞带内默认 GOOD（fail-safe：没有充分证据时不打扰用户；与 StillnessDecision
    的"未证实不判静止"方向相反但同理）。

    四个阈值是起始猜测（neck_compression 的 0.45 也是），用实际画面
    （--debug 看角度读数）标定后调整。见 CLAUDE.md「阈值（标定经验）」。

    接口约定（保持稳定，别改签名；新增参数只加带默认值的关键字参数）：
        PostureDecision(head_neck_threshold=30.0, torso_threshold=30.0,
                        back_threshold=25.0, neck_threshold=0.45,
                        hysteresis_ratio=0.8)
        update(posture: Optional[dict]) -> PostureState
        posture（只读属性）
    """

    # 角度 dict 的 key -> (对应阈值属性名, 是否"低于阈值=不良").
    # 前三者"越大越不良"（ratio = 角度/阈值）；neck_compression 是"越小越弓背"
    # （ratio = 阈值/角度），迟滞语义保持一致（ratio <= 0.8 = 明显恢复）。
    _ANGLE_TO_THRESHOLD = {
        'head_neck_angle': ('head_neck_threshold', False),
        'torso_angle': ('torso_threshold', False),
        'back_curvature': ('back_threshold', False),
        'neck_compression': ('neck_threshold', True),
    }

    def __init__(self,
                 head_neck_threshold: float = 30.0,
                 torso_threshold: float = 30.0,
                 back_threshold: float = 25.0,
                 neck_threshold: float = 0.45,
                 hysteresis_ratio: float = 0.8) -> None:
        """
        参数:
            head_neck_threshold: 头前伸角阈值（度）。0 = 头在肩正上方。
            torso_threshold:     躯干前倾角阈值（度）。0 = 身体竖直。
            back_threshold:      背部弯曲角阈值（度）。0 = 背直。
            neck_threshold:      颈压缩阈值（比值，无单位；耳-肩竖直间距/肩宽）。
                                 值 <= 此值判弓背（耸肩+低头压扁间距）。默认 0.45
                                 是起始猜测：正面摄像头下的"坐直/弓背"读数随取景
                                 和身体比例变化，用 --debug 标定（见 CLAUDE.md）。
            hysteresis_ratio:    恢复判 GOOD 的比值（默认 0.8：特征掉到阈值
                                 的 80% 以下才算真正恢复，防止边界抖动）。
        """
        if hysteresis_ratio <= 0 or hysteresis_ratio > 1.0:
            raise ValueError("hysteresis_ratio 必须在 (0, 1] 内")
        if neck_threshold <= 0:
            raise ValueError("neck_threshold 必须 > 0")
        self.head_neck_threshold = head_neck_threshold
        self.torso_threshold = torso_threshold
        self.back_threshold = back_threshold
        self.neck_threshold = neck_threshold
        self.hysteresis_ratio = hysteresis_ratio
        self._posture = PostureState.UNKNOWN

    @property
    def posture(self) -> PostureState:
        return self._posture

    def update(self, posture: Optional[dict]) -> PostureState:
        """喂入一帧的坐姿特征，更新并返回当前坐姿状态。

        参数:
            posture: PostureFeatures.update() 的返回值（特征 dict），至少含
                     'head_neck_angle'；髋不可见时缺 'torso_angle' /
                     'back_curvature'；单侧链路时缺 'neck_compression'；
                     无人/数据不足时为 None。

        返回:
            PostureState：GOOD / SLUMPED / UNKNOWN。
        """
        if posture is None:
            self._posture = PostureState.UNKNOWN
            return self._posture

        # 对当前可见的特征算 ratio，取最坏（最大）的那个。
        # 普通角度：ratio = 值/阈值（越大越不良）；
        # 颈压缩（below_is_bad）：ratio = 阈值/值（越小越弓背），值 <= 0 时按
        # 无限大处理（头完全压到肩上 = 最大不良），避免除零。
        ratios = []
        for key, (attr, below_is_bad) in self._ANGLE_TO_THRESHOLD.items():
            value = posture.get(key)
            if value is not None:
                threshold = getattr(self, attr)
                if not below_is_bad:
                    ratios.append(value / threshold)
                else:
                    ratios.append(threshold / value if value > 1e-9
                                  else float('inf'))
        if not ratios:
            self._posture = PostureState.UNKNOWN
            return self._posture

        worst = max(ratios)
        if worst >= 1.0:
            self._posture = PostureState.SLUMPED
        elif worst <= self.hysteresis_ratio:
            self._posture = PostureState.GOOD
        else:
            # 迟滞带：保持当前状态；UNKNOWN 时兜底为 GOOD（fail-safe）
            if self._posture is PostureState.UNKNOWN:
                self._posture = PostureState.GOOD
        return self._posture


class PostureAlert(_DurationAlert):
    """不良坐姿提醒：累计"连续 SLUMPED"时长，满了就触发一次提醒。

    消费 PostureDecision.PostureState（本模块的枚举），与 SedentaryAlert 对称：

        SLUMPED  累计连续不良坐姿时长（按传入 timestamp 的增量）。
        GOOD     立即清零（坐直/起身 = 姿势恢复，重新开始计时）。
        UNKNOWN  冻结不计，但不清零（检测丢帧不冤枉用户）。

    累计 ≥ duration_limit_sec → 触发一次提醒，返回英文文案；
    锁存语义由 _DurationAlert 基类统一处理。

    假设人坐在座位上：站立/走动时角度自然回到 0（GOOD）会清零计时，不会误
    累计；行走时偶发的角度尖峰持续时间很短，除非阈值调得过低，否则不足以触发。

    接口约定（保持稳定，别改签名）：
        PostureAlert(duration_limit_sec=300.0)   # 300 秒 = 5 分钟
        update(posture_state: PostureState, timestamp=None) -> Optional[str]
        elapsed_sec（只读属性）  当前连续不良坐姿秒数
    """

    def __init__(self, duration_limit_sec: float = 300.0) -> None:
        """
        参数:
            duration_limit_sec: 连续不良坐姿满多少秒触发提醒。默认 300 = 5 分钟。
        """
        super().__init__(duration_limit_sec,
                         message_template="Poor posture for {dur}. "
                                          "Sit up straight!")

    def _state_kind(self, posture_state: PostureState) -> str:
        if posture_state is PostureState.GOOD:
            return "RESET"
        if posture_state is PostureState.UNKNOWN:
            return "FREEZE"
        return "ACCUM"  # SLUMPED


class CvaLevel(enum.Enum):
    """CVA（颅椎角）风险分级。

    分级阈值来自 Mostafaee et al. 2022 观察性分组，**非临床诊断标准**；
    本系统用肩点近似 C7（见 features.ErgonomicRiskFeatures），量值需实拍自标定。
    """
    NORMAL = "NORMAL"                     # CVA >= 55°
    MILD = "MILD"                         # 50° <= CVA < 55°
    MODERATE_SEVERE = "MODERATE_SEVERE"   # 44° <= CVA < 50°
    SEVERE = "SEVERE"                     # CVA < 44°
    UNKNOWN = "UNKNOWN"                   # 无有效 CVA（无人/数据不足）


class CvaRisk:
    """CVA 分级（消费 ErgonomicRiskFeatures 的平滑 cva_deg，**不读 FSA**）。

    分级阈值默认 55/50/44（度），来自 Mostafaee et al. 2022 观察性分组：
        cva >= normal_threshold              -> NORMAL（正常）
        normal_threshold > cva >= mild_threshold -> MILD（轻度头前伸）
        mild_threshold > cva >= severe_threshold -> MODERATE_SEVERE（中重度）
        cva < severe_threshold               -> SEVERE（重度头前伸，建议触发
                                                 高优先级预警；预警机制暂未实现）
    说明：文献阈值针对真实 C7-耳连线，本系统用肩点近似 C7，存在稳定系统偏差，
    **不能直接套用文献阈值** —— 请用 --cva-*-threshold（或构造参数）按实拍标定。
    CVA 已由 features 层做滑动窗口中位数平滑，等级一般不会单帧跳变，这里不额外
    加迟滞。

    FSA（前伸肩角）是辅助指标，本层**不读取**：它只由 output 层在 CVA 判定为
    MODERATE_SEVERE 及以上时附注显示，供报告参考肩部代偿，不参与任何判断。

    接口约定（保持稳定，别改签名）：
        CvaRisk(normal_threshold=55.0, mild_threshold=50.0,
                severe_threshold=44.0)
        update(cva_deg: Optional[float]) -> CvaLevel
        level（只读属性）
    """

    def __init__(self,
                 normal_threshold: float = 55.0,
                 mild_threshold: float = 50.0,
                 severe_threshold: float = 44.0) -> None:
        """
        参数:
            normal_threshold:  CVA >= 此值判正常（默认 55°）。
            mild_threshold:    CVA >= 此值判轻度头前伸（默认 50°）。
            severe_threshold:  CVA >= 此值判中重度头前伸，更低判重度（默认 44°）。
        """
        if not (normal_threshold > mild_threshold > severe_threshold):
            raise ValueError("CVA 阈值需满足 normal_threshold > mild_threshold "
                             "> severe_threshold")
        self.normal_threshold = normal_threshold
        self.mild_threshold = mild_threshold
        self.severe_threshold = severe_threshold
        self._level = CvaLevel.UNKNOWN

    @property
    def level(self) -> CvaLevel:
        return self._level

    def update(self, cva_deg: Optional[float]) -> CvaLevel:
        """喂入一帧的平滑 CVA，更新并返回当前分级。

        cva_deg 为 None（无人 / 从未有有效 CVA）→ UNKNOWN。
        """
        if cva_deg is None:
            self._level = CvaLevel.UNKNOWN
            return self._level
        if cva_deg >= self.normal_threshold:
            self._level = CvaLevel.NORMAL
        elif cva_deg >= self.mild_threshold:
            self._level = CvaLevel.MILD
        elif cva_deg >= self.severe_threshold:
            self._level = CvaLevel.MODERATE_SEVERE
        else:
            self._level = CvaLevel.SEVERE
        return self._level


# ---------------------------------------------------------------------------
# FhpDecision / FhpAlert：头前伸三档（消费 CvaProxyFeatures 的 CVA-like proxy）
# （2026-09-23 新增：用项目自标定的实验阈值，不用文献临床阈值）
# ---------------------------------------------------------------------------

class FhpState(enum.Enum):
    """头前伸状态（三档 + 未知）。

    NORMAL       正常（proxy 在 normal 侧，且原有头颈角没超限）
    SLIGHT_FHP   轻度头前伸
    OBVIOUS_FHP  明显头前伸
    UNKNOWN      无有效读数（无人 / 关键点不足 / 置信度低于门控）
    """
    NORMAL = "NORMAL"
    SLIGHT_FHP = "SLIGHT_FHP"
    OBVIOUS_FHP = "OBVIOUS_FHP"
    UNKNOWN = "UNKNOWN"


# CVA-like proxy 的实验阈值（度，越小越前伸）：机位 -> (slight, obvious)
# 标定来源（2026-09-23）：front.csv / f45.csv，1 人 × 2 机位 × 3 次重复，
#   slight = 该机位上 Normal 组 vs (Slight+Obvious) 组的 Youden 最优阈值；
#   obvious = 该机位 Slight 组 vs Obvious 组的 Youden 最优阈值。
# 复算结论（Normal vs FHP）：45 机位 AUC 0.997 / front 0.935。
# 这些是**项目自标定的实验阈值**，不是临床 CVA 阈值：
#   * cva_proxy 用肩点近似 C7、取的是"耳"不是耳屏，量值与文献 CVA 不可比；
#   * 样本只有 1 人，换人/换椅子高度/换取景都要重标定；
#   * 阈值只对**像素空间**的 cva_proxy 有效（features.CvaProxyFeatures），
#     对归一化空间的 ErgonomicRiskFeatures.cva_deg 无效（两者差 w/h 纵横比）。
CVA_PROXY_VIEW_PRESETS = {
    "45":    (68.17, 61.42),
    "front": (84.82, 79.32),
}

# proxy 的迟滞带默认值（**不要沿用 PostureDecision 的 0.8**，见 FhpDecision 说明）。
# 标定依据（同一批数据）：proxy 的可用区间只有几十度、且"坐直"和"前伸"的差
# 只有几度，迟滞带必须窄，恢复点还得落在真实 Normal 读数区间内：
#   45 机位: Normal 读数 [69.23, 77.37]（中位 73.27），Slight [59.23, 67.11]
#            → 恢复点 = 68.17/0.98 = 69.56°，正好落在 Normal 区间内 ✓
#            → 若沿用 0.8：恢复点 = 85.21°，Normal 一次都到不了（最大 77.37）
#              → 一旦判过 SLIGHT 就永不恢复、FhpAlert 也无法重新上膛
#   正面   : Normal [83.47, 88.03]（中位 86.26），Slight [79.80, 87.38]（有重叠）
#            → 恢复点 = 84.82/0.98 = 86.55°，落在 Normal 的 IQR 内 ✓
# 带内保持 = 防止读数在阈值附近抖动时状态来回跳（宽度 ≈ 1.4~1.7°，约等于
# 同一姿势内的读数离散度）；单帧抖动另有 features 的 EMA 平滑兜底。
CVA_PROXY_HYSTERESIS_RATIO = 0.98


class FhpDecision:
    """CVA-like proxy → 头前伸三档（项目自标定阈值 + 迟滞 + 可选置信度门控）。

    判据主体是 proxy（越小越前伸），两级阈值给出三档：
        proxy >= slight_threshold                 -> NORMAL
        obvious_threshold <= proxy < slight_threshold -> SLIGHT_FHP
        proxy <  obvious_threshold                -> OBVIOUS_FHP
    实现上用 ratio = slight_threshold / proxy 做归一化（proxy 到 0 时 ratio → ∞，
    头完全压到肩水平 = 最严重），状态机与 PostureDecision 同一套语义：
        ratio >= 1.0            -> 至少 SLIGHT（1.0 = slight 阈值本身）
        ratio <= hysteresis_ratio -> NORMAL（真正恢复才降档，防边界抖动）
        中间                    -> 迟滞带，保持当前状态（UNKNOWN 时兜底 NORMAL：
                                   没证据不判前伸、不打扰用户）
    OBVIOUS 只由 proxy 给出（只有 proxy 有两级阈值）。

    原有头颈角（PostureFeatures 的 head_neck_angle，度，越大越前伸）作为**弱
    判据**一起参与，两路合成 combined_score：
        ratio_head     = head_neck_angle / head_neck_threshold（>=1 = 该判据自己超限）
        combined_score = max(weight_proxy * ratio_proxy, weight_head * ratio_head)
    即默认是"两者取更严重"（OR）。权重就是 A/B/C 三种配置的开关：
        weight_proxy=0, weight_head=1  -> A：只用原有头颈角（只能到 SLIGHT）
        weight_proxy=1, weight_head=0  -> B：只用 CVA proxy
        weight_proxy=1, weight_head=1  -> C：两者都用（默认）
    为什么合成用 max 而不是加权平均：两者都只是"体表姿态的 2D 投影代理"，没有
    共同的量纲/标定，加权平均出来的数没有物理含义；max 表达的是"任一判据认为
    前伸就是前伸"，可解释、可复现，也不引入机器学习。

    置信度门控 min_conf：proxy 本帧的平均关键点置信度低于它就不判（UNKNOWN）。
    默认 0.0 = 不额外门控 —— features.CvaProxyFeatures 已经用 visibility_min
    (0.3) 门控过一次，这里只是给"喂进来的特征门控更松"的场景留的兜底。

    降级语义：FhpDecision 不自己滤 None 值 —— proxy 的"维持上一有效值"是
    features 层的职责（cva_proxy_deg 在丢帧时给旧值、valid=False）。所以丢帧时
    状态会**保持**而不是跳 UNKNOWN，这与 features 层"降级维持、不乱报"一致。

    接口约定（保持稳定，别改签名；新增参数只加带默认值的关键字参数）：
        FhpDecision(slight_threshold=68.17, obvious_threshold=61.42,
                    hysteresis_ratio=0.98, min_conf=0.0,
                    head_neck_threshold=30.0,
                    weight_proxy=1.0, weight_head=1.0)
        update(cva: Optional[dict], posture: Optional[dict] = None) -> FhpState
        state / combined_score / ratio_proxy / ratio_head（只读属性）
    """

    def __init__(self,
                 slight_threshold: float = CVA_PROXY_VIEW_PRESETS["45"][0],
                 obvious_threshold: float = CVA_PROXY_VIEW_PRESETS["45"][1],
                 hysteresis_ratio: float = CVA_PROXY_HYSTERESIS_RATIO,
                 min_conf: float = 0.0,
                 head_neck_threshold: float = 30.0,
                 weight_proxy: float = 1.0,
                 weight_head: float = 1.0) -> None:
        """
        参数:
            slight_threshold:  NORMAL/SLIGHT 边界（proxy 度；低于它判至少 SLIGHT）。
                               默认取 45° 机位标定值（见 CVA_PROXY_VIEW_PRESETS）。
            obvious_threshold: SLIGHT/OBVIOUS 边界（proxy 度，必须小于 slight）。
            hysteresis_ratio:  恢复判 NORMAL 的比值（默认 0.98，**别直接套
                               PostureDecision 的 0.8**）。proxy 的可分区间很窄，
                               0.8 会让恢复点跑到 85° 以上（45 机位坐直才 69~77°），
                               一旦判过 SLIGHT 就永不恢复 —— 见
                               CVA_PROXY_HYSTERESIS_RATIO 的标定依据。
            min_conf:          proxy 置信度下限，低于它判 UNKNOWN；0.0 = 不门控。
            head_neck_threshold: 原有头颈角阈值（度），合成 combined_score 用。
            weight_proxy / weight_head: 两路权重（默认各 1.0 = OR）。置 0 即关闭
                               该路（对应 A/B/C 三种配置，见类 docstring）。
        """
        if slight_threshold <= obvious_threshold:
            raise ValueError("slight_threshold 必须大于 obvious_threshold")
        if obvious_threshold <= 0:
            raise ValueError("obvious_threshold 必须 > 0")
        if hysteresis_ratio <= 0 or hysteresis_ratio > 1.0:
            raise ValueError("hysteresis_ratio 必须在 (0, 1] 内")
        if head_neck_threshold <= 0:
            raise ValueError("head_neck_threshold 必须 > 0")
        if weight_proxy < 0 or weight_head < 0 or (weight_proxy == 0 and weight_head == 0):
            raise ValueError("weight_proxy / weight_head 需 >= 0 且不能同时为 0")
        self.slight_threshold = slight_threshold
        self.obvious_threshold = obvious_threshold
        self.hysteresis_ratio = hysteresis_ratio
        self.min_conf = min_conf
        self.head_neck_threshold = head_neck_threshold
        self.weight_proxy = weight_proxy
        self.weight_head = weight_head

        self._state = FhpState.UNKNOWN
        self._ratio_proxy: Optional[float] = None
        self._ratio_head: Optional[float] = None
        self._combined: Optional[float] = None

    # ---------- 只读属性 ----------

    @property
    def state(self) -> FhpState:
        return self._state

    @property
    def ratio_proxy(self) -> Optional[float]:
        """proxy 的归一化严重度：1.0 = 正好在 slight 阈值上，越大越前伸。"""
        return self._ratio_proxy

    @property
    def ratio_head(self) -> Optional[float]:
        """原有头颈角的归一化严重度：1.0 = 正好在 head_neck_threshold 上。"""
        return self._ratio_head

    @property
    def combined_score(self) -> Optional[float]:
        """两路合成严重度 = max(w_proxy·ratio_proxy, w_head·ratio_head)。

        逐帧写进 CSV，便于离线比较 A（只有原有指标）/ B（只有 proxy）/ C（合成）
        三种配置 —— 本类的状态机就是用这枚数在跑，所以 CSV 里的数与画面一致。
        """
        return self._combined

    @property
    def obvious_ratio(self) -> float:
        """OBVIOUS 档的 combined 边界（= slight/obvious，默认约 1.11）。"""
        return self.slight_threshold / self.obvious_threshold

    # ---------- 判断 ----------

    def update(self, cva: Optional[dict],
               posture: Optional[dict] = None) -> FhpState:
        """喂入一帧的 CVA-like proxy 特征（和可选的原有坐姿特征），返回头前伸状态。

        参数:
            cva:     features.CvaProxyFeatures.update() 的返回值（读
                     'cva_proxy_deg' / 'cva_proxy_conf'）；None 表示无人/无数据。
            posture: features.PostureFeatures.update() 的返回值，只读
                     'head_neck_angle'（原有 FHP 判据）；None 或缺 key 表示
                     该帧头颈角不可用（此时只按 proxy 判）。

        返回:
            FhpState：NORMAL / SLIGHT_FHP / OBVIOUS_FHP / UNKNOWN。
        """
        if cva is None:
            return self._reset(FhpState.UNKNOWN)

        proxy = cva.get('cva_proxy_deg')
        if proxy is None:
            return self._reset(FhpState.UNKNOWN)

        conf = cva.get('cva_proxy_conf')
        if self.min_conf > 0 and conf is not None and conf < self.min_conf:
            # 关键点置信度太低：宁可不判（UNKNOWN），也不拿糊掉的读数报警
            return self._reset(FhpState.UNKNOWN)

        # proxy → ratio（越小越前伸；proxy=0 表示耳与肩同高 = 最严重）
        self._ratio_proxy = (self.slight_threshold / proxy if proxy > 1e-9
                             else float('inf'))

        angle = None if posture is None else posture.get('head_neck_angle')
        self._ratio_head = (None if angle is None
                            else angle / self.head_neck_threshold)

        proxy_term = self.weight_proxy * self._ratio_proxy
        head_term = 0.0 if self._ratio_head is None else self.weight_head * self._ratio_head
        self._combined = max(proxy_term, head_term)

        if proxy_term >= self.obvious_ratio:
            # 只有 proxy 有两级阈值：OBVIOUS 必须由它给出
            self._state = FhpState.OBVIOUS_FHP
        elif self._combined >= 1.0:
            self._state = FhpState.SLIGHT_FHP
        elif self._combined <= self.hysteresis_ratio:
            self._state = FhpState.NORMAL
        elif self._state is FhpState.UNKNOWN:
            # 迟滞带 + 还没有历史状态 → 兜底 NORMAL（fail-safe：不打扰用户）
            self._state = FhpState.NORMAL
        # 其余情况：迟滞带内保持当前状态
        return self._state

    def _reset(self, state: FhpState) -> FhpState:
        """清掉本帧的比值（无有效读数时不留下旧比值误导调用方）。"""
        self._ratio_proxy = None
        self._ratio_head = None
        self._combined = None
        self._state = state
        return state


class FhpAlert(_DurationAlert):
    """头前伸提醒：累计"连续判为头前伸"时长，满了触发一次提醒。

    消费 FhpDecision.FhpState（本模块枚举），与 PostureAlert 对称：
        NORMAL                    立即清零（坐直了 = 恢复，重新计时）
        SLIGHT_FHP / OBVIOUS_FHP  累计（是否计入取决于 min_state）
        UNKNOWN                   冻结不计也不清零（丢帧不冤枉用户）

    min_state 决定"多严重才开始计时"：默认 SLIGHT_FHP（轻度就算，因为提醒的
    目的是让人及时坐直）；只想对明显头前伸报警就传 OBVIOUS_FHP。

    接口约定（保持稳定，别改签名）：
        FhpAlert(duration_limit_sec=300.0, min_state=FhpState.SLIGHT_FHP)
        update(fhp_state: FhpState, timestamp=None) -> Optional[str]
        elapsed_sec（只读属性）
    """

    # 严重度排序（min_state 比较用）；UNKNOWN 不参与
    _SEVERITY = {FhpState.NORMAL: 0, FhpState.SLIGHT_FHP: 1,
                 FhpState.OBVIOUS_FHP: 2}

    def __init__(self, duration_limit_sec: float = 300.0,
                 min_state: FhpState = FhpState.SLIGHT_FHP) -> None:
        """
        参数:
            duration_limit_sec: 连续头前伸满多少秒触发提醒。默认 300 = 5 分钟。
            min_state: 达到这个严重度才计入时长（SLIGHT_FHP 或 OBVIOUS_FHP）。
        """
        if min_state not in self._SEVERITY or min_state is FhpState.NORMAL:
            raise ValueError("min_state 只能是 SLIGHT_FHP 或 OBVIOUS_FHP")
        super().__init__(duration_limit_sec,
                         message_template="Forward head posture for {dur}. "
                                          "Sit up straight!")
        self.min_state = min_state

    def _state_kind(self, state: FhpState) -> str:
        if state is FhpState.NORMAL:
            return "RESET"
        if state is FhpState.UNKNOWN:
            return "FREEZE"
        # 没到 min_state 的那一档 = 姿势好转了 → 清零重新计时
        if self._SEVERITY[state] < self._SEVERITY[self.min_state]:
            return "RESET"
        return "ACCUM"


# ---------------------------------------------------------------------------
# FrontFhpDecision：正面机位 Ear–Shoulder displacement proxy 的**两档**分类
# （2026-09-23 新增：**只显示不提醒**的实验分支，与上面 FhpDecision 那条线独立）
# ---------------------------------------------------------------------------

class FrontFhpState(enum.Enum):
    """正面 Ear–Shoulder displacement proxy 的两档状态（+ 未知）。

    NORMAL   该 proxy 在"未前伸"一侧
    FHP      该 proxy 在"有前伸"一侧（**不再分轻重**）
    UNKNOWN  无有效读数（无人 / 关键点不足 / 预热中 / 视角退化）

    **不是** FhpState 的分支、也不要互换：FhpState 是 45° 那条线上
    CvaProxyFeatures 的三档，本枚举是正面机位上 EarShoulderProxyFeatures 的两档，
    两者特征不同、方向相反（FhpState 的 proxy 越小越前伸，本枚举的 proxy 越大
    越前伸），阈值也不可互换（见 CLAUDE.md「机位路由」）。
    """
    NORMAL = "NORMAL"
    FHP = "FHP"
    UNKNOWN = "UNKNOWN"


# 正面 Ear–Shoulder displacement proxy 的**上界**（无单位比值，越大越前伸）：
#     proxy <  FRONT_PROXY_NORMAL_MAX -> NORMAL
#     proxy >= FRONT_PROXY_NORMAL_MAX -> FHP（边界归更前伸的一档）
# 为什么只有一档：2026-09-23 两轮真机实采（front_p1_live1/live2，同人同机位、
# 每档 3 段重复）证明 **Slight 与 Obvious 分不开** —— Slight 段 #5（0.1033）比
# Obvious 段 #1（0.0889）还大、两档区间互相穿插，再分一档就是给噪声贴标签。
# 合并成 Normal / FHP 后同一批数据 SNR = 3.23（Normal 0.0268 vs FHP 0.0970），
# 可分性才站得住。
# 取值来源：front.csv / front_p1_live1.csv / front_p1_live2.csv 的
# 「Normal vs (Slight+Obvious)」两档 in-sample 中点（0.065 落在两次实采中点附近）。
# ⚠ 这是**给显示行用的种子值**，不是判据、不是医学标准：
#   * 同人同机位两轮的"坐直"基线在 0.067 ↔ 0.027 之间漂移 → **不可跨次沿用**，
#     换人 / 换椅子 / 换距离 / 换取景都必须重标（--front-fhp-normal-max 覆盖，
#     标定脚本 test_front_ear_shoulder_proxy.py 就是干这个的）；
#   * 两档区间仍有重叠（Normal 上界与 FHP 下界交叠），只能看趋势、别当判据；
#   * 该量只在**正面**机位有意义：同一批数据的 45° 块反而不单调
#     （Slight 0.339 > Obvious 0.306）→ 斜对机位保持原有
#     CvaProxyFeatures + FhpDecision 逻辑，**别把本阈值套过去**；
#   * 本类不产出任何提醒（没有对应的 *_Alert）。
FRONT_PROXY_NORMAL_MAX = 0.065


class FrontFhpDecision:
    """正面 Ear–Shoulder displacement proxy → 两档状态（**只显示，不提醒**）。

    判据只有一个上界（本指标越大越前伸，与 FhpDecision 的 proxy 方向相反）：
        proxy <  normal_max -> NORMAL
        proxy >= normal_max -> FHP（含等号：边界归更前伸的一档，与
                               PostureDecision(worst >= 1.0) / CvaRisk(>= 阈值)
                               / FhpDecision(ratio >= 1.0) 同一约定）

    刻意做得比 FhpDecision 简单（**不是简化漏了，是设计如此**）：
        * 无迟滞、无跨帧状态：这条线只挂一行读数，没有提醒要"上膛/防抖"，
          抖动由 features 层的 EMA 兜住（EarShoulderProxyFeatures τ=1.0s）；
          加迟滞反而会让画面上的读数与判定不同源、更难核对。
        * 无持续时长、无 _DurationAlert：接提醒是另一个决定，用户明确要求
          "只显示不提醒"，要接的时候再单独提。
        * 不复用 FhpDecision：两者特征/方向/阈值/机位都不同（见 FrontFhpState
          的说明），混用会让 45° 线与正面线互相污染。

    退化语义：**本帧不可用就 UNKNOWN，不沿用旧判定**。features 层在退化帧
    （肩宽塌陷）会保留上一有效平滑值、并把 ear_shoulder_proxy_valid 置 False；
    本类把 valid=False 当作"没有读数"（画面显示 N/A），因为侧身/人走开时继续
    显示上一次的 Normal 会误导观众。预热期（不足 min_frames）同样落在 valid=False。
    注：标定脚本 test_front_ear_shoulder_proxy.py 的 HUD 直接对（可能是维持值的）
    平滑读数分类，那是为了测量时看趋势；生产显示行以 valid 为准，更保守。

    接口约定（保持稳定，别改签名；新增参数只加带默认值的关键字参数）：
        FrontFhpDecision(normal_max=0.065, min_conf=0.0)
        update(esp: Optional[dict]) -> FrontFhpState
        state / proxy / proxy_raw / ratio / normal_max（只读属性）
    """

    def __init__(self, normal_max: float = FRONT_PROXY_NORMAL_MAX,
                 min_conf: float = 0.0) -> None:
        """
        参数:
            normal_max: NORMAL 上界（无单位比值，见 FRONT_PROXY_NORMAL_MAX）。
                        默认 0.065 是**实采 1 人 × 3 次的 in-sample 种子值**，
                        换条件必须用 --front-fhp-normal-max 覆盖后重标。
            min_conf:   proxy 置信度下限，低于它判 UNKNOWN；0.0 = 不门控
                        （features 已按 visibility_min=0.3 门控过一次）。
        """
        if normal_max <= 0:
            raise ValueError("normal_max 必须 > 0")
        if min_conf < 0:
            raise ValueError("min_conf 必须 >= 0")
        self.normal_max = normal_max
        self.min_conf = min_conf

        self._state = FrontFhpState.UNKNOWN
        self._proxy: Optional[float] = None
        self._proxy_raw: Optional[float] = None
        self._ratio: Optional[float] = None

    # ---------- 只读属性 ----------

    @property
    def state(self) -> FrontFhpState:
        return self._state

    @property
    def proxy(self) -> Optional[float]:
        """本帧用于判定的 proxy（EMA 平滑值，无单位）。"""
        return self._proxy

    @property
    def proxy_raw(self) -> Optional[float]:
        """本帧原始 proxy（未平滑，debug 用，看抖动幅度）。"""
        return self._proxy_raw

    @property
    def ratio(self) -> Optional[float]:
        """相对阈值的位置：1.0 = 正好在 normal_max 上，>= 1.0 即 FHP 侧。"""
        return self._ratio

    # ---------- 判断 ----------

    def update(self, esp: Optional[dict]) -> FrontFhpState:
        """喂入一帧的 Ear–Shoulder proxy 特征，返回两档状态。

        参数:
            esp: features.EarShoulderProxyFeatures.update() 的返回值（读
                 'ear_shoulder_proxy' / 'ear_shoulder_proxy_raw' /
                 'ear_shoulder_proxy_valid' / 'ear_shoulder_proxy_conf'）；
                 None 表示无人/无数据。

        返回:
            FrontFhpState：NORMAL / FHP / UNKNOWN。
        """
        if esp is None:
            return self._reset(FrontFhpState.UNKNOWN)

        self._proxy_raw = esp.get('ear_shoulder_proxy_raw')
        proxy = esp.get('ear_shoulder_proxy')
        if proxy is None:
            return self._reset(FrontFhpState.UNKNOWN)
        if not esp.get('ear_shoulder_proxy_valid'):
            # 预热中 / 视角退化（肩宽塌陷）：不拿维持值当读数，直接 N/A
            return self._reset(FrontFhpState.UNKNOWN)

        conf = esp.get('ear_shoulder_proxy_conf')
        if self.min_conf > 0 and conf is not None and conf < self.min_conf:
            return self._reset(FrontFhpState.UNKNOWN)

        self._proxy = proxy
        self._ratio = proxy / self.normal_max
        # 越大越前伸：到/过阈值即 FHP（含等号）
        self._state = (FrontFhpState.FHP if self._ratio >= 1.0
                       else FrontFhpState.NORMAL)
        return self._state

    def _reset(self, state: FrontFhpState) -> FrontFhpState:
        """清掉本帧读数（无有效读数时不留下旧比值/旧 proxy 误导调用方）。"""
        self._proxy = None
        self._proxy_raw = None
        self._ratio = None
        self._state = state
        return state


# ---------------------------------------------------------------------------
# 自测：纯 stdlib 合成数据，验证久坐计时/锁存逻辑（main_demo.py --selftest 汇总调用）
# ---------------------------------------------------------------------------

def selftest_sedentary() -> None:
    """合成数据自测 SedentaryAlert：计时、触发、锁存、MOVING 清零重上膛。"""
    # 1) 累计计时：静止 2+3 秒 = 5 秒 → 触发一次
    a = SedentaryAlert(duration_limit_sec=5.0)
    assert a.update(State.STILL, timestamp=0.0) is None, "0s 不应触发"
    assert a.update(State.STILL, timestamp=2.0) is None, "2s 不应触发"
    msg = a.update(State.STILL, timestamp=5.0)
    assert msg is not None, "满 5s 应触发提醒"
    assert "5 s" in msg, f"秒级阈值应显示 '5 s'，实际 {msg!r}"

    # 2) 锁存：触发后继续保持静止 → 不重复触发
    assert a.update(State.STILL, timestamp=6.0) is None, "锁存期间不应重复触发"

    # 3) UNKNOWN：冻结不计，但不清零
    assert a.update(State.UNKNOWN, timestamp=6.5) is None
    assert a.elapsed_sec >= 5.0, "UNKNOWN 不应清零计时"

    # 4) MOVING：清零并重新上膛，再坐满一次能再触发
    assert a.update(State.MOVING, timestamp=7.0) is None
    assert a.elapsed_sec == 0.0, "MOVING 应清零计时"
    a.update(State.STILL, timestamp=8.0)
    a.update(State.STILL, timestamp=10.0)
    msg2 = a.update(State.STILL, timestamp=13.0)
    assert msg2 is not None, "起身后再坐满应能再次触发"

    # 5) UNKNOWN 冻结：不计 UNKNOWN 期间的时间，也不清零已累计的时长
    a2 = SedentaryAlert(duration_limit_sec=10.0)
    a2.update(State.STILL, timestamp=0.0)
    a2.update(State.STILL, timestamp=2.0)     # 已累计 2s
    a2.update(State.UNKNOWN, timestamp=5.0)   # 冻结：不计，但不清零
    assert a2.elapsed_sec == 2.0, f"UNKNOWN 不应清零已累计时长，应 2s，实际 {a2.elapsed_sec}"
    a2.update(State.STILL, timestamp=6.0)     # 锚点重置，恢复后的第一帧只建锚不计
    assert a2.elapsed_sec == 2.0, f"UNKNOWN 期间不应计时，应 2s，实际 {a2.elapsed_sec}"
    a2.update(State.STILL, timestamp=8.0)     # 恢复后继续累计 2s
    assert a2.elapsed_sec == 4.0, f"恢复后应继续累计，应 4s，实际 {a2.elapsed_sec}"

    print("  selftest_sedentary: OK")


def selftest_posture_decision() -> None:
    """合成数据自测 PostureDecision：超限→SLUMPED、恢复→GOOD、迟滞带保持、
    髋缺失（只有 head_neck）仍能判、无人→UNKNOWN、迟滞带内 UNKNOWN→GOOD、
    颈压缩（越小越弓背）超限/恢复/迟滞/除零保护。"""
    # 默认阈值：head_neck 30 / torso 30 / back 25，迟滞 0.8（恢复阈值 = 0.8 × 阈值）
    pd = PostureDecision()

    # 1) 全角度明显低于阈值 → GOOD
    assert pd.update({'head_neck_angle': 5.0, 'torso_angle': 5.0,
                      'back_curvature': 5.0}) is PostureState.GOOD, \
        "全角度小应为 GOOD"

    # 2) 背部弯曲超限（back=30 > 25）→ SLUMPED
    assert pd.update({'head_neck_angle': 5.0, 'torso_angle': 5.0,
                      'back_curvature': 30.0}) is PostureState.SLUMPED, \
        "任一角度超限应为 SLUMPED"

    # 3) 迟滞带：back=21 在 [0.8×25=20, 25) 之间 → 保持 SLUMPED
    assert pd.update({'head_neck_angle': 5.0, 'torso_angle': 5.0,
                      'back_curvature': 21.0}) is PostureState.SLUMPED, \
        "迟滞带内应保持 SLUMPED"

    # 4) 明显恢复：back=15 < 20 → GOOD
    assert pd.update({'head_neck_angle': 5.0, 'torso_angle': 5.0,
                      'back_curvature': 15.0}) is PostureState.GOOD, \
        "明显恢复应为 GOOD"

    # 5) 髋缺失：只剩 head_neck_angle（torso/back 缺失）→ 仍能判
    pd2 = PostureDecision()
    assert pd2.update({'head_neck_angle': 40.0}) is PostureState.SLUMPED, \
        "髋缺失、头前伸超限应为 SLUMPED"
    assert pd2.update({'head_neck_angle': 20.0}) is PostureState.GOOD, \
        "髋缺失、头恢复应为 GOOD"

    # 6) 无人 → UNKNOWN
    assert pd.update(None) is PostureState.UNKNOWN, "无人应为 UNKNOWN"

    # 7) 迟滞带内且当前 UNKNOWN → 兜底 GOOD（fail-safe：没证据不打扰）
    pd3 = PostureDecision()
    assert pd3.update({'head_neck_angle': 25.0}) is PostureState.GOOD, \
        "迟滞带内 UNKNOWN 应兜底为 GOOD"

    # 8) SLUMPED 后回到迟滞带 → 保持 SLUMPED（不轻易掉回 GOOD）
    pd4 = PostureDecision()
    pd4.update({'head_neck_angle': 40.0})  # SLUMPED
    assert pd4.update({'head_neck_angle': 26.0}) is PostureState.SLUMPED, \
        "SLUMPED 后回到迟滞带应保持 SLUMPED"

    # 9) 颈压缩（越小越弓背）：0.30 < 默认阈值 0.45 → SLUMPED；0.70 → GOOD
    pd5 = PostureDecision()
    assert pd5.update({'neck_compression': 0.30}) is PostureState.SLUMPED, \
        "颈压缩低于阈值应为 SLUMPED"
    assert pd5.update({'neck_compression': 0.70}) is PostureState.GOOD, \
        "颈压缩明显高于阈值应为 GOOD"

    # 10) 颈压缩迟滞带：0.50 → ratio=0.45/0.50=0.9 ∈ [0.8,1) → 保持 SLUMPED
    pd6 = PostureDecision()
    pd6.update({'neck_compression': 0.30})  # SLUMPED
    assert pd6.update({'neck_compression': 0.50}) is PostureState.SLUMPED, \
        "颈压缩迟滞带内应保持 SLUMPED"

    # 11) 颈压缩 = 0（头完全压到肩上）→ 不除零，按最大不良处理 → SLUMPED
    pd7 = PostureDecision()
    assert pd7.update({'neck_compression': 0.0}) is PostureState.SLUMPED, \
        "颈压缩为 0 应判 SLUMPED"

    # 12) 与其它角度联动：颈压缩正常但 head_neck 超限 → SLUMPED
    pd8 = PostureDecision()
    assert pd8.update({'neck_compression': 0.70,
                       'head_neck_angle': 40.0}) is PostureState.SLUMPED, \
        "任一特征超限（含颈压缩）应为 SLUMPED"

    print("  selftest_posture_decision: OK")


def selftest_posture_alert() -> None:
    """合成数据自测 PostureAlert：累计触发、锁存、GOOD 清零重上膛、UNKNOWN 冻结。"""
    # 1) 累计计时：SLUMPED 3+2 秒 = 5 秒 → 触发一次
    a = PostureAlert(duration_limit_sec=5.0)
    assert a.update(PostureState.GOOD, timestamp=0.0) is None, "GOOD 不应计时"
    assert a.update(PostureState.SLUMPED, timestamp=1.0) is None, "1s 不应触发"
    assert a.update(PostureState.SLUMPED, timestamp=4.0) is None, "4s 不应触发"
    msg = a.update(PostureState.SLUMPED, timestamp=6.0)
    assert msg is not None, "满 5s 应触发提醒"
    assert "5 s" in msg, f"秒级阈值应显示 '5 s'，实际 {msg!r}"

    # 2) 锁存：触发后继续保持 SLUMPED → 不重复触发
    assert a.update(PostureState.SLUMPED, timestamp=7.0) is None, \
        "锁存期间不应重复触发"

    # 3) UNKNOWN：冻结不计，但不清零
    assert a.update(PostureState.UNKNOWN, timestamp=7.5) is None
    assert a.elapsed_sec >= 5.0, "UNKNOWN 不应清零计时"

    # 4) GOOD：清零并重新上膛，再弓背满一次能再触发
    assert a.update(PostureState.GOOD, timestamp=8.0) is None
    assert a.elapsed_sec == 0.0, "GOOD 应清零计时"
    a.update(PostureState.SLUMPED, timestamp=9.0)
    a.update(PostureState.SLUMPED, timestamp=11.0)
    msg2 = a.update(PostureState.SLUMPED, timestamp=14.0)
    assert msg2 is not None, "坐直后再弓背满时长应能再次触发"

    # 5) UNKNOWN 冻结：不计 UNKNOWN 期间的时间，也不清零已累计的时长
    a2 = PostureAlert(duration_limit_sec=10.0)
    a2.update(PostureState.SLUMPED, timestamp=0.0)
    a2.update(PostureState.SLUMPED, timestamp=2.0)     # 已累计 2s
    a2.update(PostureState.UNKNOWN, timestamp=5.0)     # 冻结：不计，但不清零
    assert a2.elapsed_sec == 2.0, \
        f"UNKNOWN 不应清零已累计时长，应 2s，实际 {a2.elapsed_sec}"
    a2.update(PostureState.SLUMPED, timestamp=6.0)     # 恢复后第一帧只建锚不计
    assert a2.elapsed_sec == 2.0, \
        f"UNKNOWN 期间不应计时，应 2s，实际 {a2.elapsed_sec}"
    a2.update(PostureState.SLUMPED, timestamp=8.0)     # 恢复后继续累计 2s
    assert a2.elapsed_sec == 4.0, \
        f"恢复后应继续累计，应 4s，实际 {a2.elapsed_sec}"

    print("  selftest_posture_alert: OK")


def selftest_cva_risk() -> None:
    """合成数据自测 CvaRisk：各级边界（55/50/44，含等号）、None→UNKNOWN、阈值可配、
    非法阈值顺序报错。"""
    r = CvaRisk()

    # 1) 分级边界（含等号：>= 阈值归入上级）
    assert r.update(60.0) is CvaLevel.NORMAL, "60° 应为 NORMAL"
    assert r.update(55.0) is CvaLevel.NORMAL, "55° 边界应含等号 → NORMAL"
    assert r.update(54.9) is CvaLevel.MILD, "54.9° 应为 MILD"
    assert r.update(50.0) is CvaLevel.MILD, "50° 边界应含等号 → MILD"
    assert r.update(49.9) is CvaLevel.MODERATE_SEVERE, "49.9° 应为 MODERATE_SEVERE"
    assert r.update(44.0) is CvaLevel.MODERATE_SEVERE, "44° 边界应含等号 → MODERATE_SEVERE"
    assert r.update(43.9) is CvaLevel.SEVERE, "43.9° 应为 SEVERE"
    assert r.update(0.0) is CvaLevel.SEVERE, "0° 应为 SEVERE（不除零、不崩溃）"
    assert r.update(None) is CvaLevel.UNKNOWN, "None 应为 UNKNOWN"

    # 2) 阈值可配（按实拍标定后传入）
    r2 = CvaRisk(normal_threshold=50.0, mild_threshold=45.0, severe_threshold=40.0)
    assert r2.update(48.0) is CvaLevel.MILD, "配置后 48° 应为 MILD"

    # 3) 非法阈值顺序应报错
    try:
        CvaRisk(normal_threshold=40.0, mild_threshold=45.0, severe_threshold=44.0)
        raise AssertionError("阈值乱序应抛 ValueError")
    except ValueError:
        pass

    print("  selftest_cva_risk: OK")


def _cva(deg, conf=0.9):
    """构造一帧 CvaProxyFeatures 的返回（只填 FhpDecision 用到的 key）。"""
    return {'cva_proxy_deg': deg, 'cva_proxy_conf': conf, 'cva_proxy_valid': True}


def selftest_fhp_decision() -> None:
    """合成数据自测 FhpDecision：三档边界（含等号）、迟滞、A/B/C 权重、
    原有头颈角只能抬到 Slight、置信度门控、None→UNKNOWN、参数校验、
    以及 CVA_PROXY_VIEW_PRESETS 本身自洽。"""
    # 0) preset 自洽：每机位 (slight, obvious) 递减、都在 (0, 90]
    for view, (slight, obvious) in CVA_PROXY_VIEW_PRESETS.items():
        assert slight > obvious > 0, f"{view} 预设应满足 slight > obvious > 0"
        assert slight <= 90.0, f"{view} 预设的角应 <= 90°"

    d = FhpDecision()   # 默认 = 45° 机位（68.17 / 61.42）

    # 1) 三档：明显正常 / 明显前伸各来一发（边界等号见下）
    assert d.update(_cva(90.0)) is FhpState.NORMAL, "90° 应 NORMAL"
    assert d.update(_cva(30.0)) is FhpState.OBVIOUS_FHP, "明显前伸 → OBVIOUS"
    # 边界等号：正好落在阈值上算"已超限"，与 PostureDecision(worst >= 1.0) /
    # CvaRisk(>= 阈值) 同一约定
    assert FhpDecision().update(_cva(68.17)) is FhpState.SLIGHT_FHP, \
        "正好落在 slight 阈值上（ratio=1.0）→ 含等号，判 SLIGHT"
    assert FhpDecision().update(_cva(61.42)) is FhpState.OBVIOUS_FHP, \
        "正好落在 obvious 阈值上（ratio=obvious_ratio）→ 含等号，判 OBVIOUS"
    assert FhpDecision().update(_cva(68.0)) is FhpState.SLIGHT_FHP, "略低于 slight → SLIGHT"

    # 2) 无读数 → UNKNOWN（并把比值清空）
    assert d.update(None) is FhpState.UNKNOWN, "cva=None 应 UNKNOWN"
    assert d.ratio_proxy is None and d.combined_score is None, \
        "UNKNOWN 时不应留着上一帧的比值"
    assert d.update({'cva_proxy_deg': None, 'cva_proxy_conf': None}) is FhpState.UNKNOWN

    # 3) 迟滞：SLIGHT 后在迟滞带（ratio ∈ (0.98, 1.0)）内保持 SLIGHT，真恢复才 NORMAL
    #    45 机位默认下迟滞带 ≈ proxy ∈ (68.17, 69.56)，约 1.4° 宽
    d2 = FhpDecision()
    assert d2.update(_cva(65.0)) is FhpState.SLIGHT_FHP
    assert d2.update(_cva(68.2)) is FhpState.SLIGHT_FHP, \
        "ratio≈0.9996 刚过 slight 阈值，在迟滞带内 → 保持 SLIGHT（不来回跳）"
    assert d2.update(_cva(69.0)) is FhpState.SLIGHT_FHP, "ratio≈0.988 仍在带内 → 保持"
    assert d2.update(_cva(75.0)) is FhpState.NORMAL, \
        "ratio≈0.909 掉出迟滞带 → 真正恢复（45 机位坐直读数 69~77° 就是这个档）"
    assert d2.update(_cva(90.0)) is FhpState.NORMAL

    # 4) 迟滞带 + 全新实例（无历史）→ 兜底 NORMAL
    d3 = FhpDecision()
    assert d3.update(_cva(69.0)) is FhpState.NORMAL, "迟滞带内 UNKNOWN 应兜底 NORMAL"

    # 4b) 迟滞带的自洽性（回归保护，0.8 就是踩了这个坑）：
    #     恢复点必须落在该机位**坐直时真实出现过**的读数区间内，否则一旦判过
    #     SLIGHT 就永不恢复；带宽也不该超过坐直读数的离散范围。
    #     {机位: (Normal 读数 min, max)}，来自 f45.csv / front.csv。
    #     注：正面机位 Normal 与 Slight 有重叠（恢复点 86.55° 落在 Normal 的
    #     IQR 内、但低于 Slight 上限 87.38°），所以这里只校验"可恢复"+"带宽合理"。
    for view, (normal_lo, normal_hi) in {"45": (69.23, 77.37),
                                         "front": (83.47, 88.03)}.items():
        slight = CVA_PROXY_VIEW_PRESETS[view][0]
        recover = slight / CVA_PROXY_HYSTERESIS_RATIO
        assert recover >= slight, "恢复点必须比触发点更靠正常侧"
        assert recover <= normal_hi, \
            f"{view} 机位恢复点 {recover:.1f}° 超出坐直读数上限 {normal_hi}° → 永不恢复"
        band = recover - slight
        assert band <= normal_hi - normal_lo, \
            f"{view} 迟滞带宽 {band:.2f}° 不该超过坐直读数区间宽度"

    # 5) OBVIOUS 掉回 SLIGHT：proxy 越过 obvious 边界但 ratio 仍 >= 1
    d4 = FhpDecision()
    assert d4.update(_cva(55.0)) is FhpState.OBVIOUS_FHP
    assert d4.update(_cva(65.0)) is FhpState.SLIGHT_FHP, \
        "proxy 回到 obvious 之上、slight 之下 → 降一档到 SLIGHT"

    # 6) 合成：CVA 正常但原有头颈角超限 → 抬到 SLIGHT（弱判据的作用）
    d5 = FhpDecision()
    st = d5.update(_cva(88.0), {'head_neck_angle': 40.0})
    assert st is FhpState.SLIGHT_FHP, f"头颈角 40°>30° 应把 NORMAL 抬到 SLIGHT，实际 {st}"
    assert abs(d5.combined_score - 40.0 / 30.0) < 1e-9, d5.combined_score
    assert abs(d5.ratio_proxy - 68.17 / 88.0) < 1e-9, d5.ratio_proxy

    # 7) 原有头颈角**不能**单独判出 OBVIOUS（只有 proxy 有两级阈值）
    d6 = FhpDecision()
    st = d6.update(_cva(88.0), {'head_neck_angle': 89.0})   # ratio_head ≈ 2.97
    assert st is FhpState.SLIGHT_FHP, \
        f"头颈角再大也只到 SLIGHT（OBVIOUS 只由 proxy 给出），实际 {st}"

    # 8) A / B / C 三种权重配置
    best = {'cva_proxy_deg': 55.0, 'cva_proxy_conf': 0.9}   # proxy 单独看是 OBVIOUS
    a = FhpDecision(weight_proxy=0.0, weight_head=1.0)      # A：只用原有头颈角
    assert a.update(best, {'head_neck_angle': 10.0}) is FhpState.NORMAL, \
        "A 配置不看 proxy（头颈角正常）→ NORMAL"
    assert a.update(best, {'head_neck_angle': 40.0}) is FhpState.SLIGHT_FHP
    assert a.update(best, None) is FhpState.NORMAL
    b = FhpDecision(weight_proxy=1.0, weight_head=0.0)      # B：只用 proxy
    assert b.update(best, {'head_neck_angle': 89.0}) is FhpState.OBVIOUS_FHP, \
        "B 配置只看 proxy → OBVIOUS（头颈角不影响）"
    c = FhpDecision()                                       # C：两者都用（默认）
    assert c.update(best, {'head_neck_angle': 10.0}) is FhpState.OBVIOUS_FHP

    # 9) 置信度门控：低于 min_conf → UNKNOWN，不拿糊掉的读数报警
    d7 = FhpDecision(min_conf=0.5)
    assert d7.update(_cva(55.0, conf=0.3)) is FhpState.UNKNOWN, "低置信应 UNKNOWN"
    assert d7.update(_cva(55.0, conf=0.8)) is FhpState.OBVIOUS_FHP
    # 默认 min_conf=0（不额外门控，features 层已按 visibility_min 滤过）
    assert FhpDecision().update(_cva(55.0, conf=0.01)) is FhpState.OBVIOUS_FHP

    # 10) 非法参数
    for kwargs in (dict(slight_threshold=50.0, obvious_threshold=60.0),
                   dict(obvious_threshold=0.0),
                   dict(hysteresis_ratio=0.0), dict(hysteresis_ratio=1.5),
                   dict(head_neck_threshold=0.0),
                   dict(weight_proxy=0.0, weight_head=0.0),
                   dict(weight_proxy=-1.0)):
        try:
            FhpDecision(**kwargs)
            raise AssertionError(f"{kwargs} 应抛 ValueError")
        except ValueError:
            pass

    print("  selftest_fhp_decision: OK")


def selftest_fhp_alert() -> None:
    """合成数据自测 FhpAlert：累计触发、锁存、NORMAL 清零、UNKNOWN 冻结、
    min_state=OBVIOUS 时 SLIGHT 不计入（视为已好转）。"""
    # 1) 默认 SLIGHT 起算：累计满 5s → 触发一次
    #    注意首帧只记时间起点、不计时长（与 features 层同一约定），
    #    所以 t=1s 起算、t=6s 才刚好满 5s。
    a = FhpAlert(duration_limit_sec=5.0)
    assert a.update(FhpState.NORMAL, timestamp=0.0) is None
    assert a.update(FhpState.SLIGHT_FHP, timestamp=1.0) is None
    assert a.update(FhpState.OBVIOUS_FHP, timestamp=3.0) is None
    assert a.update(FhpState.SLIGHT_FHP, timestamp=5.0) is None, "累计才 4s，未满"
    msg = a.update(FhpState.OBVIOUS_FHP, timestamp=6.0)
    assert msg is not None and "5 s" in msg, msg
    assert abs(a.elapsed_sec - 5.0) < 1e-9, a.elapsed_sec
    #    SLIGHT 与 OBVIOUS 都算触发状态（=min_state），在两档之间来回不打断计时

    # 2) 锁存：继续保持 → 不重复触发（仍继续计时）
    assert a.update(FhpState.OBVIOUS_FHP, timestamp=7.0) is None, "锁存期间不应重复"

    # 3) UNKNOWN：冻结不计也不清零
    assert a.update(FhpState.UNKNOWN, timestamp=9.0) is None
    assert a.elapsed_sec >= 5.0, "UNKNOWN 不应清零"
    assert not a.update(FhpState.UNKNOWN, timestamp=12.0), "UNKNOWN 期间不计时"
    assert abs(a.elapsed_sec - 6.0) < 1e-9, \
        f"冻结前累计到 6.0s，冻结后不再变，实际 {a.elapsed_sec}"

    # 4) NORMAL：清零重上膛，再满一次能再触发
    assert a.update(FhpState.NORMAL, timestamp=13.0) is None
    assert a.elapsed_sec == 0.0
    a.update(FhpState.SLIGHT_FHP, timestamp=14.0)
    a.update(FhpState.SLIGHT_FHP, timestamp=16.0)
    assert a.update(FhpState.SLIGHT_FHP, timestamp=19.0) is not None

    # 5) min_state=OBVIOUS：SLIGHT 不算（视为好转，清零）
    b = FhpAlert(duration_limit_sec=5.0, min_state=FhpState.OBVIOUS_FHP)
    b.update(FhpState.OBVIOUS_FHP, timestamp=0.0)
    b.update(FhpState.OBVIOUS_FHP, timestamp=2.0)
    assert abs(b.elapsed_sec - 2.0) < 1e-9, b.elapsed_sec
    assert b.update(FhpState.SLIGHT_FHP, timestamp=3.0) is None
    assert b.elapsed_sec == 0.0, "SLIGHT 未达 min_state → 清零"
    b.update(FhpState.OBVIOUS_FHP, timestamp=4.0)
    assert b.update(FhpState.OBVIOUS_FHP, timestamp=9.0) is not None, "应能触发"

    # 6) min_state 非法
    for bad in (FhpState.NORMAL, FhpState.UNKNOWN):
        try:
            FhpAlert(min_state=bad)
            raise AssertionError(f"min_state={bad} 应抛 ValueError")
        except ValueError:
            pass

    print("  selftest_fhp_alert: OK")


def _esp(proxy, raw=None, valid=True, conf=0.9, degenerate=False):
    """合成一帧 EarShoulderProxyFeatures.update() 的返回值（自测用）。"""
    return {'ear_shoulder_proxy': proxy,
            'ear_shoulder_proxy_raw': proxy if raw is None else raw,
            'ear_shoulder_proxy_valid': valid,
            'ear_shoulder_proxy_conf': conf,
            'ear_shoulder_proxy_pts': 'ear_mid+sh_mid',
            'ear_shoulder_proxy_geom': None,
            'ear_shoulder_proxy_parts': {},
            'ear_shoulder_proxy_degenerate': degenerate}


def selftest_front_fhp_decision() -> None:
    """合成数据自测 FrontFhpDecision：方向（越大越前伸）、只有一个上界、
    边界含等号、**只有两档**（不产生第三档）、valid/degenerate → UNKNOWN、
    None → UNKNOWN、无迟滞（显示行不做状态保持）、参数校验，
    以及**不影响 45° 那条线**（FhpDecision 行为不变）。"""
    # 0) 枚举自洽：确实只有两档（+ UNKNOWN），没有 Slight/Obvious 细分
    assert [s.name for s in FrontFhpState] == ["NORMAL", "FHP", "UNKNOWN"], \
        "正面线只有两档：Slight/Obvious 已被实采证明分不开（见 FRONT_PROXY_NORMAL_MAX）"

    d = FrontFhpDecision()   # 默认上界 0.065

    # 1) 方向：**越大越前伸**（与 FhpDecision 的 proxy 方向相反）
    assert d.update(_esp(0.02)) is FrontFhpState.NORMAL, "小位移应 NORMAL"
    assert d.update(_esp(0.20)) is FrontFhpState.FHP, "大位移应 FHP"
    assert abs(d.ratio - 0.20 / 0.065) < 1e-9, d.ratio

    # 2) 边界含等号：正好落在上界 → 归更前伸的一档
    assert FrontFhpDecision().update(_esp(0.065)) is FrontFhpState.FHP, \
        "正好落在 normal_max 上（ratio=1.0）→ 含等号，判 FHP"
    assert FrontFhpDecision().update(_esp(0.0649)) is FrontFhpState.NORMAL, "略低于上界 → NORMAL"

    # 3) 只有两档：中间值（0.08）和很大的值（0.30）都是同一档 FHP
    assert FrontFhpDecision().update(_esp(0.08)) is FrontFhpState.FHP
    assert FrontFhpDecision().update(_esp(0.30)) is FrontFhpState.FHP

    # 4) 无读数 → UNKNOWN，且清掉上一帧读数
    d2 = FrontFhpDecision()
    assert d2.update(_esp(0.20)) is FrontFhpState.FHP
    assert d2.update(None) is FrontFhpState.UNKNOWN, "esp=None 应 UNKNOWN"
    assert d2.proxy is None and d2.ratio is None, "UNKNOWN 不应留着上一帧的读数"
    assert d2.update({'ear_shoulder_proxy': None,
                      'ear_shoulder_proxy_valid': True}) is FrontFhpState.UNKNOWN

    # 5) 退化/预热帧（valid=False）→ UNKNOWN，**即使平滑值里还有维持的旧读数**
    assert d2.update(_esp(0.20, valid=False, degenerate=True)) is FrontFhpState.UNKNOWN, \
        "视角退化帧不能沿用旧判定（侧身时继续显示 Normal 会误导）"
    assert d2.proxy is None, "无效帧不留读数"

    # 6) 无迟滞：跨过阈值立刻回 NORMAL（显示行不做状态保持；抖动由 features EMA 兜）
    d3 = FrontFhpDecision()
    assert d3.update(_esp(0.070)) is FrontFhpState.FHP
    assert d3.update(_esp(0.060)) is FrontFhpState.NORMAL, \
        "显示行无迟滞带：读数回到上界之下就立刻显示 NORMAL"
    assert d3.update(_esp(0.0651)) is FrontFhpState.FHP, "再跨上去立刻 FHP"

    # 7) 置信度门控（默认 0.0 = 不额外门控，features 已按 visibility_min 滤过）
    d4 = FrontFhpDecision(min_conf=0.5)
    assert d4.update(_esp(0.20, conf=0.3)) is FrontFhpState.UNKNOWN, "低置信应 UNKNOWN"
    assert d4.update(_esp(0.20, conf=0.8)) is FrontFhpState.FHP
    assert FrontFhpDecision().update(_esp(0.20, conf=0.01)) is FrontFhpState.FHP

    # 8) 阈值可覆盖 + 非法参数
    assert FrontFhpDecision(normal_max=0.10).update(_esp(0.07)) is FrontFhpState.NORMAL, \
        "覆盖上界后 0.07 在 NORMAL 侧"
    for kwargs in (dict(normal_max=0.0), dict(normal_max=-0.1), dict(min_conf=-0.1)):
        try:
            FrontFhpDecision(**kwargs)
            raise AssertionError(f"{kwargs} 应抛 ValueError")
        except ValueError:
            pass

    # 9) 与 45° 那条线互不影响：FhpDecision 的三档/方向仍照旧（回归保护）
    assert FhpDecision().update(_cva(55.0)) is FhpState.OBVIOUS_FHP, \
        "新增正面线不得改变 45° 线的判定"
    assert FhpDecision().update(_cva(90.0)) is FhpState.NORMAL

    print("  selftest_front_fhp_decision: OK")


if __name__ == "__main__":
    selftest_sedentary()
    selftest_posture_decision()
    selftest_posture_alert()
    selftest_cva_risk()
    selftest_fhp_decision()
    selftest_fhp_alert()
    selftest_front_fhp_decision()
    print("decision selftest: ALL PASSED")
