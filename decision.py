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
    中间 -> 迟滞带，保持当前状态。髋不可见时只按 head_neck_angle 判。

"持续满多久才算数"这类时间规则，由 _DurationAlert 基类的两个子类实现：
    SedentaryAlert 累计"连续静止"时长（消费 StillnessDecision.State）；
    PostureAlert    累计"连续不良坐姿"时长（消费 PostureDecision.PostureState）。
二者不改 StillnessDecision / PostureDecision 的接口。

接口约定（保持稳定，别改签名）：
    State 枚举：STILL / MOVING / UNKNOWN
    StillnessDecision(still_threshold=0.05, moving_threshold=0.10)
    update(movement: Optional[float]) -> State
    state（只读属性）

    PostureState 枚举：GOOD / SLUMPED / UNKNOWN
    PostureDecision(head_neck_threshold=30.0, torso_threshold=30.0,
                    back_threshold=25.0, hysteresis_ratio=0.8)
    update(posture: Optional[dict]) -> PostureState
    posture（只读属性）

    SedentaryAlert(duration_limit_sec=1200.0)            久坐提醒
    update(state, timestamp=None) -> Optional[str]       触发时返回提醒文案（英文），否则 None
    elapsed_sec（只读属性）                               当前连续久坐秒数

    PostureAlert(duration_limit_sec=300.0)               不良坐姿提醒
    update(posture_state, timestamp=None) -> Optional[str]
    elapsed_sec（只读属性）                               当前连续不良坐姿秒数
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
    """基于三个坐姿角度判断坐姿好坏（迟滞，消费 PostureFeatures 的角度 dict）。

    对当前可见的每个角度算 ratio = 角度 / 阈值，取最大者 worst：
        worst >= 1.0              -> SLUMPED（弓背/头前伸/前倾，至少一个超限）
        worst <= hysteresis_ratio -> GOOD（所有角度都明显低于阈值）
        两者之间                    -> 迟滞带，保持当前状态（防边界抖动）

    只对当前可见的角度判断：髋不可见（人只露出上半身）时，torso/back 两个
    角度缺失，只按 head_neck_angle 判；posture 为 None（无人/数据不足）→ UNKNOWN。

    迟滞带内默认 GOOD（fail-safe：没有充分证据时不打扰用户；与 StillnessDecision
    的"未证实不判静止"方向相反但同理）。

    三个阈值是起始猜测，用实际画面（--debug 看角度读数）标定后调整。

    接口约定（保持稳定，别改签名）：
        PostureDecision(head_neck_threshold=30.0, torso_threshold=30.0,
                        back_threshold=25.0, hysteresis_ratio=0.8)
        update(posture: Optional[dict]) -> PostureState
        posture（只读属性）
    """

    # 角度 dict 的 key -> 对应阈值属性名
    _ANGLE_TO_THRESHOLD = {
        'head_neck_angle': 'head_neck_threshold',
        'torso_angle': 'torso_threshold',
        'back_curvature': 'back_threshold',
    }

    def __init__(self,
                 head_neck_threshold: float = 30.0,
                 torso_threshold: float = 30.0,
                 back_threshold: float = 25.0,
                 hysteresis_ratio: float = 0.8) -> None:
        """
        参数:
            head_neck_threshold: 头前伸角阈值（度）。0 = 头在肩正上方。
            torso_threshold:     躯干前倾角阈值（度）。0 = 身体竖直。
            back_threshold:      背部弯曲角阈值（度）。0 = 背直。
            hysteresis_ratio:    恢复判 GOOD 的比值（默认 0.8：角度掉到阈值
                                 的 80% 以下才算真正恢复，防止边界抖动）。
        """
        if hysteresis_ratio <= 0 or hysteresis_ratio > 1.0:
            raise ValueError("hysteresis_ratio 必须在 (0, 1] 内")
        self.head_neck_threshold = head_neck_threshold
        self.torso_threshold = torso_threshold
        self.back_threshold = back_threshold
        self.hysteresis_ratio = hysteresis_ratio
        self._posture = PostureState.UNKNOWN

    @property
    def posture(self) -> PostureState:
        return self._posture

    def update(self, posture: Optional[dict]) -> PostureState:
        """喂入一帧的坐姿角度，更新并返回当前坐姿状态。

        参数:
            posture: PostureFeatures.update() 的返回值（角度 dict，单位：度），
                     至少含 'head_neck_angle'；髋不可见时缺 'torso_angle' /
                     'back_curvature'；无人/数据不足时为 None。

        返回:
            PostureState：GOOD / SLUMPED / UNKNOWN。
        """
        if posture is None:
            self._posture = PostureState.UNKNOWN
            return self._posture

        # 对当前可见的角度算 ratio，取最坏（最大）的那个
        ratios = []
        for key, attr in self._ANGLE_TO_THRESHOLD.items():
            angle = posture.get(key)
            if angle is not None:
                ratios.append(angle / getattr(self, attr))
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
    髋缺失（只有 head_neck）仍能判、无人→UNKNOWN、迟滞带内 UNKNOWN→GOOD。"""
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


if __name__ == "__main__":
    selftest_sedentary()
    selftest_posture_decision()
    selftest_posture_alert()
    print("decision selftest: ALL PASSED")
