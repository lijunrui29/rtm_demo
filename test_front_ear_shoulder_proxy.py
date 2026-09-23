"""
正面机位 FHP/posture 实验分支：Ear–Shoulder displacement proxy（**独立脚本**）。

=== 为什么有这条分支（2026-09-23 排查结论）===
正面机位下现有的角度类指标都测不到头前伸：
    * CVA-like proxy（耳-肩连线 vs 水平线）在正面被 ~280px 的耳-肩**竖直**间距压掉
      —— 实测 Δx 19/29/53px 经 atan 只剩 3.9°/6.5°/13.9°，Normal 与 Slight 只差 2.7°
      （组内 σ ≈ 1.2°）；
    * posture 的 head_neck 被正面投影压缩（16.8/21.3/30.0°，而阈值是 30° → 只够得到
      Obvious 的尾巴），torso/back 因髋不可见全程 N/A，neck_compression 从不到阈值。
**所以正面不靠调阈值硬修**（那是在拟合噪声），改用另一条信号做实验验证：
    Ear–Shoulder displacement proxy = |耳中点x − 肩中点x| ÷ 肩宽（像素空间）
本脚本是这条信号的**测量验证工具**：测量、对比、如实报 overlap —— 不拟合阈值、
不进生产 decision。

=== 它是什么 / 不是什么（命名纪律，别越界）===
    * 只是**图像平面里的相对水平位移**（2D image-plane displacement），用肩宽归一化
      掉远近/体型；
    * **不是** 3D 前伸距离、**不是** Forward Head Distance(FHD)、**不是** clinical FHD。
      正面投影下 Δx 变大既可能来自头前伸，也可能来自低头/头部转动/整个人塌下去
      （本批数据髋全程不可见，proxy 分不清"头相对躯干前伸"和"整个人塌下去"）；
    * 只能说 **Ear–Shoulder displacement proxy**。
**方向与 cva_proxy_deg 相反**：本指标**越大越前伸**（cva_proxy_deg 越小越前伸）——
两列并排看时别搞反。

=== 跑法（在 rtm_demo/ 目录下，用装了 cv2/onnxruntime 的解释器）===
    python test_front_ear_shoulder_proxy.py --camera 0                  # 摄像头，正面机位
    python test_front_ear_shoulder_proxy.py --camera 0 --session 0.6m   # 标记距离/椅子/取景
    python test_front_ear_shoulder_proxy.py --video clip.mp4            # 视频文件
    python test_front_ear_shoulder_proxy.py --analyze-only a.csv b.csv  # 只出报告（不开摄像头）
    python test_front_ear_shoulder_proxy.py --selftest                  # 合成数据自测

按键（窗口里）：
    1 = Normal（坐直）   2 = 轻微前伸   3 = 明显前伸
    0 = 清除当前标签（这一帧不参与统计）  s = 立刻出报告并写盘  q / ESC = 结束

每个姿势保持 15~20 秒、每种至少重复 2~3 次（段首 1 秒是过渡帧，用 --skip-sec 丢掉）。

=== CSV（默认 front_ear_shoulder_proxy.csv）===
前 6 列来自 output.FhpCsvLogger（沿用现有角度线的口径，**只作对照**）：
    timestamp, cva_proxy, original_fhp_score, combined_score, fhp_state,
    keypoint_confidence
本脚本追加的列：
    view, person, session,                     实验条件标签（分块统计用）
    ear_shoulder_proxy,                        本实验指标：EMA 平滑值（越大越前伸）
    ear_shoulder_proxy_raw,                    本实验指标：本帧原始值（测量验证看这列）
    ear_shoulder_proxy_valid,                  本帧能否用于统计（预热/退化/降级时 0）
    ear_shoulder_proxy_degenerate,             视角退化（肩宽塌陷，侧身）→ 1
    shoulder_width, shoulder_width_norm,       肩宽（像素 / 归一化）
    ear_mid_x, shoulder_mid_x,                 耳中点 x、肩中点 x（像素）
    left/right_ear_confidence,                 四个参与点的**原始**置信度
    left/right_shoulder_confidence,            （被门控掉的也照写，便于看为什么不算）
    head_neck_angle,                           现有 posture 的特征（只作对照）
    posture_state,                             现有 PostureDecision 的状态（默认阈值，不改）
    exp_front_state                            **实验**分档（两档 NORMAL/FHP，
                                               见 EXP_THRESHOLDS，非判据）
缺数据写空字符串、不写 0（"没数据"≠"读数 0"）。

=== 口径：判定只做两档（Normal / 有前伸）===
2026-09-23 实采（同一人同机位、每档 3 段重复）证明 **Slight 与 Obvious 分不开**：
Slight 三段 0.0872/0.1033/0.0835 vs Obvious 0.0889/0.1156/0.1033 —— 跨段交错、
信噪比 0.55，head_neck 与 cva_proxy 也分不开同样这两档。所以判定口径定为
**Normal / FHP（有前伸）**两档；键盘 1/2/3 照旧，用来做**诊断细分**
（报告同时给"两档合并口径"和"三档明细"），1=坐直、2=轻度前伸、3=明显前伸。

=== 实验阈值（**不是医学标准，别当判据**）===
见 EXP_THRESHOLDS：只在 front 机位给**一个上界**（本指标越大越前伸）。**in-sample**：
只为让"实验分级"这一列有东西看。实测"坐直"基线本身就会漂移（同人同机位两轮
0.067 ↔ 0.027），**这个数不可跨次沿用**。换人/换距离/换取景都要重标，
用 --front-normal-max 覆盖，别回来改这里的数字。

=== 边界（本轮不许越界）===
    * 不改 decision.py / main_demo.py / output.py / capture.py / web_demo.py；
      45° 机位继续走 CvaProxyFeatures + FhpDecision（那条线不动）；
    * 本脚本不消费生产判据、生产判据也不消费本脚本的指标 —— 只有多人多次验证
      稳定之后，才谈把这条信号接进正式 decision。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from enum import Enum
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from features import (EarShoulderProxyFeatures, PostureFeatures,   # noqa: E402
                      CvaProxyFeatures)
from decision import (PostureDecision, FhpDecision, FhpState,     # noqa: E402
                      CVA_PROXY_VIEW_PRESETS)
from output import FrameRenderer, FhpCsvLogger                    # noqa: E402
from pose_estimation import detect_pose                           # noqa: E402

# 键盘标签 → 姿势名（1/2/3 都保留，判定只做两档，三档用于诊断细分）
LABEL_KEYS = {"1": "Normal", "2": "Slight_FHP", "3": "Obvious_FHP"}
LABEL_ORDER = ("Normal", "Slight_FHP", "Obvious_FHP")

# 兼容老 CSV（posture_label 写的是 0/1/2/3，0=未标注）：1/2/3 与键盘同一套含义
LABEL_FROM_NUMBER = {"1": "Normal", "2": "Slight_FHP", "3": "Obvious_FHP"}

# **判定口径只做两档**（2026-09-23 选定）：Slight 与 Obvious 实测分不开（见模块
# docstring），所以"有前伸"= 标签 2+3 合并成一档 FHP。三档明细仍照常统计。
FHP_LABELS = ("Slight_FHP", "Obvious_FHP")


class FrontState(Enum):
    """正面实验指标的**两档**状态（本脚本内部用）。

    **不是生产状态枚举**，与 decision.FhpState 无关（那边是 45° 角度线的
    三档 + UNKNOWN，还带迟滞；这里只是测量脚本里的一列对照）。
    """

    NORMAL = "NORMAL"
    FHP = "FHP"


# 实验阈值（**非医学标准**）：正面机位、两档口径，**一个上界**
# （本指标**越大越前伸**，别照抄 cva_proxy 那套"越小越前伸"）：
#     proxy <  normal_max → Normal
#     proxy >= normal_max → FHP（边界归更前伸的一档）
# 取值 0.065 = 离线回放（front.csv：Normal 均值 0.0401 与 Slight+Obvious 合并均值
# 0.0893）的中点。**in-sample**，只为给"实验分级"一列对照，**不是判据**。
# ⚠ 两轮真机实采（front_p1_live1/live2，同人同机位）显示"坐直"基线在 0.067↔0.027
# 之间漂移 —— **这个数不可跨次沿用**；换人/换机位/换取景必须重标
# （用 --front-normal-max 覆盖，别改这里的数字）。
# 45° 机位不给阈值：那条线用 CvaProxyFeatures + FhpDecision（本脚本只作对照）。
EXP_THRESHOLDS = {"front": 0.065}

# 追加到 output.FhpCsvLogger 标准列之后的列（见模块 docstring 的口径表）
EXTRA_COLUMNS = (
    "view", "person", "session",
    "ear_shoulder_proxy", "ear_shoulder_proxy_raw",
    "ear_shoulder_proxy_valid", "ear_shoulder_proxy_degenerate",
    "shoulder_width", "shoulder_width_norm", "ear_mid_x", "shoulder_mid_x",
    "left_ear_confidence", "right_ear_confidence",
    "left_shoulder_confidence", "right_shoulder_confidence",
    "head_neck_angle", "posture_state", "exp_front_state",
)

# 机位自检（**只提醒、绝不自动切换**）：块内归一化肩宽中位数低于此值时，报告里提醒
# "这段取景看起来不是明显正面"。**不做自动判别机位**——本项目实测（front.csv /
# f45.csv）肩宽归一化 front 0.365~0.377 vs 45° 0.280~0.350（**重叠**，且随距离/姿势变），
# 耳间距 266~319 vs 253~296px 同样重叠，没有可靠的自动机位判别特征。
# 所以机位必须由人用 --view 声明（明显正面才用正面分支，斜对保持原有逻辑），
# 这里只做一次"你声明的和取景看起来一致吗"的提醒。0.30 是这两次录制的观测之间取的。
FRONT_NORM_WIDTH_WARN = 0.30

# 显示窗口名（imshow 与 getWindowProperty 必须用同一个字符串：用户点右上角关窗后要能退出）
WINDOW_NAME = "Front Ear-Shoulder displacement proxy (experiment)"


# ---------- CSV 读入与统计 ----------

def _f(s):
    """CSV 字符串 → float（空/坏值 → None，不填 0）。"""
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_csv(path: str) -> list:
    """读逐帧 CSV（可以是本脚本写的，也可以是老标定脚本写的）。"""
    with open(path, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)]


def label_of(row: dict) -> str:
    """行的姿势标签名；兼容老 CSV 的数字标签（1/2/3）与空（未标注）。"""
    lab = (row.get("posture_label") or "").strip()
    if lab in LABEL_ORDER:
        return lab
    return LABEL_FROM_NUMBER.get(lab, "")


def meta_of(row: dict, col: str, default: str = "") -> str:
    """取实验条件标签（view/person/session），空 → default。"""
    return (row.get(col) or "").strip() or default


def block_key(row: dict, default_view: str) -> tuple:
    """分块键：(机位, 人, session) —— 换人/换距离各跑一遍就是不同块。"""
    return (meta_of(row, "view", default_view),
            meta_of(row, "person", "?"),
            meta_of(row, "session", "-"))


def raw_ok(row: dict) -> bool:
    """该帧能否进 raw 统计：raw 有值，且不是视角退化帧。"""
    if _f(row.get("ear_shoulder_proxy_raw")) is None:
        return False
    return (row.get("ear_shoulder_proxy_degenerate") or "0").strip() \
        not in ("1", "true", "True")


def smoothed_ok(row: dict) -> bool:
    """该帧能否进平滑值统计：raw 可用 + 本帧真的出了平滑读数（esp_valid=1）。"""
    return raw_ok(row) and (row.get("ear_shoulder_proxy_valid") or "0").strip() \
        in ("1", "true", "True")


def drop_run_heads(rows: list, skip_sec: float) -> tuple:
    """丢掉每个标签段开头 skip_sec 秒（切姿势时的过渡帧）。

    段 = 连续相同标签的行。传入的 rows 应已按 (机位, 人, session) 分好块。
    返回 (保留行, 丢弃帧数)。时间用 CSV 的 timestamp 列。
    """
    if skip_sec <= 0:
        return list(rows), 0
    kept, dropped = [], 0
    prev_label, run_start = None, None
    for r in rows:
        lab = label_of(r)
        t = _f(r.get("timestamp"))
        if lab != prev_label:
            prev_label, run_start = lab, t
        if lab and t is not None and run_start is not None \
                and (t - run_start) < skip_sec:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def stats(vals: list) -> dict:
    """一组数的 n / mean / std(样本, n<2 记 0) / min / max / median。空 → n=0。"""
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None,
                "min": None, "max": None, "median": None}
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    s = sorted(vals)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {"n": n, "mean": mean, "std": math.sqrt(var),
            "min": s[0], "max": s[-1], "median": med}


def all_rows(_row: dict) -> bool:
    """「不挑帧」的 usable 判定（对照口径的列用它：只要该列有值就算）。"""
    return True


def values(rows: list, column: str, usable=raw_ok) -> list:
    """挑出 rows 里 column 的读数（只保留 usable 判定的帧）。"""
    return [v for v in (_f(r.get(column)) for r in rows if usable(r))
            if v is not None]


def by_label(rows: list) -> dict:
    """{标签: [行, ...]}，只保留 LABEL_ORDER 里的标签。"""
    out = {lab: [] for lab in LABEL_ORDER}
    for r in rows:
        lab = label_of(r)
        if lab:
            out[lab].append(r)
    return out


def separation(rows: list, column: str, lo_labels, hi_labels,
               usable=raw_ok) -> dict:
    """两组读数能否被一条阈值切开（本指标**越大越前伸**）。

    lo_labels: 应更小（更接近坐直）的一组；hi_labels: 应更大（更前伸）的一组。
    比 Normal vs FHP 时 lo=Normal、hi=FHP；比 Slight vs Obvious 时
    lo=Slight、hi=Obvious。

    返回 dict(separated, cut, gap, overlap_lo, overlap_hi, lo_max, hi_min,
              gap_std)。separated=True 时可用阈值区间是 (lo_max, hi_min]；
    separated=False 时 overlap_lo/overlap_hi 是重叠区间的两端。
    gap_std = 间距 ÷ 两组平均 σ（信噪比），< 2 基本没法用。
    """
    groups = by_label(rows)
    lo = [v for lab in lo_labels for v in values(groups[lab], column, usable)]
    hi = [v for lab in hi_labels for v in values(groups[lab], column, usable)]
    if not lo or not hi:
        return {"separated": None, "cut": None, "gap": None, "overlap_lo": None,
                "overlap_hi": None, "lo_max": None, "hi_min": None,
                "gap_std": None}
    lo_s, hi_s = stats(lo), stats(hi)
    lo_max, hi_min = lo_s["max"], hi_s["min"]
    sep = lo_max < hi_min
    noise = (lo_s["std"] + hi_s["std"]) / 2.0
    return {"separated": sep,
            "cut": 0.5 * (lo_max + hi_min) if sep else None,
            "gap": hi_min - lo_max,
            "overlap_lo": None if sep else hi_min,
            "overlap_hi": None if sep else lo_max,
            "lo_max": lo_max, "hi_min": hi_min,
            "gap_std": (abs(hi_s["mean"] - lo_s["mean"]) / noise) if noise else None}


def run_means(rows: list, column: str, usable=raw_ok) -> dict:
    """{标签: [(段内帧数, 段内均值), ...]} —— 每个连续标签段 = 一次重复。

    段间一致性是"同一个人重测一次稳不稳"的最小单位（比组内 σ 更接近验证要问的
    问题：换一次摆姿势还一样吗）。
    """
    out = {lab: [] for lab in LABEL_ORDER}
    cur_label, cur_vals = None, []
    for r in rows:
        lab = label_of(r)
        v = _f(r.get(column)) if (lab and usable(r)) else None
        if lab != cur_label:
            if cur_label and cur_vals:
                out[cur_label].append((len(cur_vals), sum(cur_vals) / len(cur_vals)))
            cur_label, cur_vals = lab, []
        if v is not None:
            cur_vals.append(v)
    if cur_label and cur_vals:
        out[cur_label].append((len(cur_vals), sum(cur_vals) / len(cur_vals)))
    return out


def thresholds_for(view: str, args) -> tuple:
    """该机位的实验阈值 → (float|None, 来源说明)。None = 本脚本不给它定阈值。

    两档口径：只有**一个上界**（Normal 的上界，超过即 FHP）。
    只有"明显正面"的机位才给（45°/斜对机位保持原有逻辑，本脚本不给它定正面阈值）；
    命令行显式覆盖时按覆盖值走（用于自标定新条件）。
    """
    normal_max = (args.front_normal_max if args.front_normal_max is not None
                  else EXP_THRESHOLDS.get(view))
    if normal_max is None:
        return None, f"{view} 机位没有实验阈值（斜对机位保持原有逻辑，别硬套）"
    if args.front_normal_max is not None:
        return normal_max, "命令行 --front-normal-max 覆盖"
    return normal_max, f"{view} 机位实验预设（**in-sample**，见脚本 EXP_THRESHOLDS）"


def classify(proxy, normal_max):
    """实验**两档**分档（**in-sample 阈值，非医学标准、非判据**）：**越大越前伸**。

        proxy <  normal_max → NORMAL
        proxy >= normal_max → FHP（边界归更前伸的一档）

    只做两档：实采证明三档里的 Slight/Obvious 分不开（见模块 docstring），
    三档只保留在报告的诊断明细里。
    proxy/threshold 缺失 → None。不做迟滞、不做时长规则 —— 只为和"当前 Posture"并排看。
    """
    if proxy is None or normal_max is None:
        return None
    return FrontState.NORMAL if proxy < normal_max else FrontState.FHP


def pooled(rows: list, column: str, labels, usable=raw_ok) -> list:
    """把几个标签的读数合并成一档（两档口径用：FHP = Slight + Obvious）。"""
    groups = by_label(rows)
    return [v for lab in labels for v in values(groups[lab], column, usable)]


def two_tier_midpoint(rows: list, column: str, usable=raw_ok):
    """本批数据的**两档中点**（in-sample，纯诊断，**不参与分级**）。

    返回 (normal均值, fhp均值, 中点) 或 None。只在两档都有读数时给。
    用途：告诉你"这一批数据自己的分界大概在哪"，而不是拿它去判定 ——
    判定用的阈值只来自 EXP_THRESHOLDS / 命令行（避免用数据拟合阈值来让结果好看）。
    """
    n = values(by_label(rows)["Normal"], column, usable)
    f = pooled(rows, column, FHP_LABELS, usable)
    if not n or not f:
        return None
    m_n, m_f = sum(n) / len(n), sum(f) / len(f)
    return m_n, m_f, 0.5 * (m_n + m_f)


def state_shares(rows: list, column: str) -> dict:
    """{标签: {取值: 帧数}}，看某个状态列在每个姿势段的分布。"""
    out = {lab: {} for lab in LABEL_ORDER}
    for r in rows:
        lab = label_of(r)
        if not lab:
            continue
        v = (r.get(column) or "").strip() or "(空)"
        out[lab][v] = out[lab].get(v, 0) + 1
    return out


def _pct(s: dict) -> str:
    """把 {取值: 帧数} 打成 "取值=n(n%)" 的形式（按帧数降序）。"""
    tot = sum(s.values()) or 1
    return "  ".join(f"{k}={v}({100.0 * v / tot:.0f}%)"
                     for k, v in sorted(s.items(), key=lambda kv: -kv[1]))


def _fn(v, nd=4) -> str:
    return "--" if v is None else f"{v:.{nd}f}"


# ---------- 数据源 / 绘制 ----------

class _Source:
    """统一数据源：摄像头（capture.FrameSource：read() → 帧或 None）和视频文件
    （cv2.VideoCapture：read() → (ok, frame)）都包成 read() → (ok, frame) + with。

    摄像头走 capture 层的 CameraCapture（不在本脚本里重造），视频文件本脚本直接
    用 cv2。与 test_fhp_cva_proxy.py 的 _Source 同构（两个实验脚本各自独立，谁
    都可以单独跑/单独删）。
    """

    def __init__(self, args) -> None:
        self.args = args
        self.fps = 0.0
        if args.video:
            import cv2
            self._cap = cv2.VideoCapture(args.video)
            if not self._cap.isOpened():
                raise SystemExit(f"打不开视频文件: {args.video}")
            self.fps = args.video_fps or self._cap.get(cv2.CAP_PROP_FPS) or 0.0
            if not self.fps or self.fps <= 0:
                self.fps = 30.0
                print("[warn] 读不到视频帧率，按 30fps 折算时间戳"
                      "（--video-fps 可指定）")
            n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.desc = (f"video:{args.video} ({self.fps:.1f}fps"
                         + (f", {n}帧)" if n else ")"))
            self._src = None
        else:
            from capture import CameraCapture
            self.idx = args.camera if args.camera is not None else 0
            self._src = CameraCapture(self.idx, width=1280, height=720)
            self.desc = f"camera:{self.idx}"
            self._cap = None
        self._t0 = time.monotonic()

    def __enter__(self) -> "_Source":
        if self._src is not None:
            self._src.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._src is not None:
            self._src.release()
        if self._cap is not None:
            self._cap.release()

    def read(self):
        """返回 (是否读到, 帧)。读不到（视频结束/摄像头掉线）→ (False, None)。"""
        if self._cap is not None:
            ok, frame = self._cap.read()
            return bool(ok) and frame is not None, frame
        frame = self._src.read()
        return frame is not None, frame

    def stamp(self, idx: int) -> float:
        """该帧时间戳（秒）：视频用播放时钟（可复现），实时用单调时钟（EMA 需要）。"""
        if self._cap is not None:
            return idx / self.fps
        return time.monotonic() - self._t0


def _text(frame, txt, org, scale=0.5, color=(230, 230, 230)):
    """带黑描边的文字（画面实时看，深浅背景都要清楚）。"""
    import cv2
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(frame, txt, (org[0] + dx, org[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                cv2.LINE_AA)


def run_capture(args) -> None:
    """主流程：采集 → 打标签 → 逐帧写 CSV → 结束时出报告。"""
    import cv2

    thr, thr_src = thresholds_for(args.view, args)

    # 复用生产链路做**对照**读数：现有角度线（CvaProxyFeatures+FhpDecision，
    # 按本机位预设）与现有 posture 线（默认阈值）。本实验指标来自
    # features.EarShoulderProxyFeatures —— 三者都只记录，不参与本脚本的判定。
    esp = EarShoulderProxyFeatures(
        smooth_tau_sec=args.smooth_tau,
        min_shoulder_width_norm=args.min_shoulder_width)
    cva = CvaProxyFeatures()
    post = PostureFeatures()
    slight, obvious = CVA_PROXY_VIEW_PRESETS[args.view]
    dec_ref = FhpDecision(slight_threshold=slight, obvious_threshold=obvious)
    dec_post = PostureDecision()          # 生产默认阈值，本脚本不改
    ren = FrameRenderer(draw_skeleton=not args.no_skeleton)

    src = _Source(args)
    print(f"数据源: {src.desc}   机位 view={args.view}")
    print(f"实验阈值（**非医学标准，in-sample**；两档口径）: "
          + (f"< {thr:.4f} = Normal，>= {thr:.4f} = FHP（{thr_src}）"
             if thr is not None else thr_src))
    print("正在初始化 pose 检测器")
    print("按键：1=Normal  2=Slight FHP  3=Obvious FHP  0=清除标签   "
          "s=出报告  q/ESC=结束")
    print(f"逐帧 CSV: {args.out}")

    label = ""
    label_since = 0.0
    idx = 0
    invalid = []
    degenerate = 0
    with src:
        with FhpCsvLogger(args.out, extra_columns=EXTRA_COLUMNS) as lg:
            while True:
                ok, frame = src.read()
                if not ok:
                    break
                ts = src.stamp(idx)

                pose = detect_pose(frame)
                e = esp.update(pose, ts)
                c = cva.update(pose, ts)
                p = post.update(pose) or {}
                st_ref = dec_ref.update(c, p)
                st_post = dec_post.update(p or None)

                parts = e.get('ear_shoulder_proxy_parts') or {}
                if e.get('ear_shoulder_proxy_raw') is None:
                    invalid.append(idx)
                if e.get('ear_shoulder_proxy_degenerate'):
                    degenerate += 1
                exp_state = classify(e.get('ear_shoulder_proxy'), thr)

                lg.log(ts, c, st_ref, dec_ref, frame_index=idx,
                       posture_label=label,
                       extra={
                           'view': args.view, 'person': args.person,
                           'session': args.session,
                           'ear_shoulder_proxy': e.get('ear_shoulder_proxy'),
                           'ear_shoulder_proxy_raw': e.get('ear_shoulder_proxy_raw'),
                           'ear_shoulder_proxy_valid':
                               int(bool(e.get('ear_shoulder_proxy_valid'))),
                           'ear_shoulder_proxy_degenerate':
                               int(bool(e.get('ear_shoulder_proxy_degenerate'))),
                           'shoulder_width': parts.get('shoulder_width'),
                           'shoulder_width_norm': parts.get('shoulder_width_norm'),
                           'ear_mid_x': parts.get('ear_mid_x'),
                           'shoulder_mid_x': parts.get('shoulder_mid_x'),
                           'left_ear_confidence': parts.get('left_ear_confidence'),
                           'right_ear_confidence': parts.get('right_ear_confidence'),
                           'left_shoulder_confidence':
                               parts.get('left_shoulder_confidence'),
                           'right_shoulder_confidence':
                               parts.get('right_shoulder_confidence'),
                           'head_neck_angle': p.get('head_neck_angle'),
                           'posture_state': st_post.value,
                           'exp_front_state':
                               exp_state.value if exp_state else '',
                       })

                if args.hud:
                    ren.draw_skeleton_frame(frame, pose)
                    _text(frame, f"{src.desc}  view={args.view}  t={ts:6.1f}s  "
                                 f"frame={idx}", (12, 26), 0.55)
                    if label:
                        _text(frame, f"[LABEL] {label}  {ts - label_since:5.1f}s",
                              (12, 50), 0.6, (60, 220, 60))
                    else:
                        _text(frame, "[LABEL] -- (press 1/2/3)",
                              (12, 50), 0.6, (0, 215, 255))
                    # 逐行把三条口径并排显示（需求里的对照行）
                    _text(frame, f"Head-Neck: {_fn(p.get('head_neck_angle'), 1)}",
                          (12, 78), 0.55)
                    _text(frame, "Ear-Shoulder Proxy: "
                                 f"{_fn(e.get('ear_shoulder_proxy'), 4)}"
                                 f" (raw {_fn(e.get('ear_shoulder_proxy_raw'), 4)})",
                          (12, 102), 0.55, (60, 220, 60))
                    _text(frame, f"Current Posture: {st_post.value}",
                          (12, 126), 0.55)
                    _text(frame, "Experimental Front Posture (2-tier): "
                                 + (exp_state.value if exp_state else "n/a"),
                          (12, 150), 0.55, (0, 215, 255))
                    if e.get('ear_shoulder_proxy_degenerate'):
                        _text(frame, "view degenerate: shoulder width collapsed "
                                     "-> frame not counted", (12, 174), 0.5,
                              (0, 0, 255))
                    _text(frame, "1=Normal 2=Slight 3=Obvious 0=clear "
                                 "s=report q=quit", (12, 198), 0.45,
                          (200, 200, 200))
                    cv2.imshow(WINDOW_NAME, frame)
                    k = cv2.waitKey(1) & 0xFF
                    ch = chr(k) if 32 <= k < 127 else ""
                    if k == 27 or ch in ("q", "Q"):
                        break
                    if cv2.getWindowProperty(WINDOW_NAME,
                                             cv2.WND_PROP_VISIBLE) < 1:
                        break       # 点窗口右上角关闭：别让进程后台空转
                    if ch in LABEL_KEYS:
                        label = LABEL_KEYS[ch]
                        label_since = ts
                        print(f"[label] {label} 从 t={ts:.1f}s 开始")
                    elif ch == "0":
                        label = ""
                    elif ch in ("s", "S"):
                        lg.flush()      # 先落盘再读回，否则报告里缺最近几十帧
                        _finish(args, [args.out])
                idx += 1

    cv2.destroyAllWindows()

    print(f"\n共 {idx} 帧，CSV: {args.out}")
    if invalid:
        print(f"[warn] {len(invalid)} 帧算不出 proxy（首帧 {invalid[0]}；常见原因："
              f"人不在画面 / 耳或肩置信度过低 / 只剩单肩）")
    if degenerate:
        print(f"[warn] {degenerate} 帧判视角退化（肩宽塌陷，多出现在侧身）→ 不计入统计")
    _finish(args, [args.out])


# ---------- 报告 ----------

def build_report(rows_all: list, meta: dict, default_view: str) -> list:
    """块 = (机位, 人, session)；每块一份逐姿势统计 + 可分性 + 段间重复性 + 对照。

    rows_all: **未做段首丢弃**的原始行（各块自己丢，避免跨块误判成同一段）。
    """
    lines = [
        "=" * 78,
        "正面机位 Ear-Shoulder displacement proxy —— 实验测量验证报告",
        "=" * 78,
        f"数据来源      : {meta['source']}",
        f"总帧数        : {meta['frames']}（其中未标注 {meta['unlabeled']} 帧不计入统计）",
        f"段首丢弃      : 每段前 {meta['skip_sec']:.1f}s（切姿势的过渡帧）→ 共丢 "
        f"{meta['dropped']} 帧",
        f"平滑          : 一阶 EMA，tau={meta['smooth_tau']:.2f}s（raw 列未平滑）",
        f"肩宽退化门控  : 归一化肩宽 < {meta['min_shoulder_width']:.2f} → 判视角退化、"
        f"不算有效读数（raw 仍记录）",
        "",
        "本指标 = |耳中点x − 肩中点x| ÷ 肩宽（**像素空间**，无单位）。",
        "  * **越大越前伸**（与 cva_proxy_deg 方向相反，别搞反）；",
        "  * 只是**图像平面里的相对水平位移**（2D image-plane displacement），"
        "**不是 3D 前伸距离**，",
        "    不能叫 Forward Head Distance / clinical FHD；",
        "  * 本报告只做测量验证（mean/std/min/max + overlap + 段间重复），"
        "不拟合阈值。",
    ]

    # 分块（保持出现顺序）
    order, blocks = [], {}
    for r in rows_all:
        key = block_key(r, default_view)
        if key not in blocks:
            blocks[key] = []
            order.append(key)
        blocks[key].append(r)

    thr, thr_src = meta['thr'], meta['thr_src']
    for key in order:
        view, person, session = key
        raw_rows = blocks[key]
        kept, dropped = drop_run_heads(raw_rows, meta['skip_sec'])
        groups = by_label(kept)
        n_labeled = sum(len(v) for v in groups.values())
        lines += ["", "=" * 78,
                  f"■ 机位 view={view}   人 person={person}   session={session}",
                  f"  帧 {len(raw_rows)}（丢弃段首 {dropped}）→ 计入统计 "
                  f"{n_labeled} 帧："
                  + "  ".join(f"{lab}={len(groups[lab])}" for lab in LABEL_ORDER),
                  "=" * 78]

        if not n_labeled:
            lines.append("  （这一段没有带标签的帧，跳过）")
            continue

        # 机位自检（启发式，只提醒不切换）：声明的机位和取景看起来一致吗？
        med_norm = stats(values(kept, 'shoulder_width_norm', all_rows))['median']
        if med_norm is not None:
            note = (f"    机位自检（启发式，**只提醒、不自动切换机位**）: 归一化肩宽"
                    f"中位数 {_fn(med_norm, 3)}")
            if med_norm < FRONT_NORM_WIDTH_WARN:
                note += ("  ← 取景看起来**不是明显正面**（本项目实测 front≈0.37 vs "
                         "45°≈0.28~0.35）：正面分支的结论不适用，斜对机位请用原有逻辑"
                         "（--view 45）")
            lines.append(note)

        # [1] raw 读数（测量验证的主口径）
        lines += ["", f"[1] ear_shoulder_proxy_raw（未平滑，主口径；已剔除视角退化帧）",
                  "    姿势           n      mean        std        min        max"
                  "      median"]
        for lab in LABEL_ORDER:
            st = stats(values(groups[lab], 'ear_shoulder_proxy_raw'))
            lines.append(f"    {lab:<12} {st['n']:>5}  {_fn(st['mean'])}  "
                         f"{_fn(st['std'])}  {_fn(st['min'])}  {_fn(st['max'])}  "
                         f"{_fn(st['median'])}")
        means = [stats(values(groups[lab], 'ear_shoulder_proxy_raw'))['mean']
                 for lab in LABEL_ORDER]
        if all(m is not None for m in means):
            mono = means[0] < means[1] < means[2]
            lines.append(f"    单调性（Normal < Slight < Obvious）: "
                         f"{'是' if mono else '否'}   "
                         + " < ".join(_fn(m) for m in means))
        for lo, hi, name in ((("Normal",), ("Slight_FHP",), "Normal → Slight"),
                             (("Slight_FHP",), ("Obvious_FHP",), "Slight → Obvious")):
            sep = separation(kept, 'ear_shoulder_proxy_raw', lo, hi)
            if sep['separated'] is None:
                lines.append(f"    {name}: 数据不足")
                continue
            snap = "分离" if sep['separated'] else "重叠"
            extra = (f"阈值区间 ({_fn(sep['lo_max'])}, {_fn(sep['hi_min'])}]"
                     if sep['separated'] else
                     f"重叠区间 [{_fn(sep['overlap_lo'])}, {_fn(sep['overlap_hi'])}]")
            lines.append(f"    {name}: [{snap}] 间距 {_fn(sep['gap'])}  "
                         f"信噪比 {_fn(sep['gap_std'], 2)}  {extra}")

        # [1b] 两档口径（**判定只做这两档**；三档明细留在 [1] 里作诊断）
        lines += ["", "[1b] 两档口径（**判定只用这两档**：Normal vs 有前伸 = 标签 2+3 "
                      "合并）",
                  "    口径             n      mean        std        min        max"
                  "      median"]
        n_vals = values(groups["Normal"], 'ear_shoulder_proxy_raw')
        f_vals = pooled(kept, 'ear_shoulder_proxy_raw', FHP_LABELS)
        for name, vals in (("Normal", n_vals), ("FHP (2+3)", f_vals)):
            st = stats(vals)
            lines.append(f"    {name:<14} {st['n']:>5}  {_fn(st['mean'])}  "
                         f"{_fn(st['std'])}  {_fn(st['min'])}  {_fn(st['max'])}  "
                         f"{_fn(st['median'])}")
        sep2t = separation(kept, 'ear_shoulder_proxy_raw', ("Normal",), FHP_LABELS)
        if sep2t['separated'] is None:
            lines.append("    两档可分性: 数据不足")
        else:
            snap = "分离" if sep2t['separated'] else "**重叠**"
            extra = (f"可用阈值区间 ({_fn(sep2t['lo_max'])}, {_fn(sep2t['hi_min'])}]"
                     if sep2t['separated'] else
                     f"重叠区间 [{_fn(sep2t['overlap_lo'])}, {_fn(sep2t['overlap_hi'])}]"
                     f" ← 两档在**这一批数据里**都分不开")
            lines.append(f"    两档可分性: [{snap}] 间距 {_fn(sep2t['gap'])}  "
                         f"信噪比 {_fn(sep2t['gap_std'], 2)}  {extra}")
        mid = two_tier_midpoint(kept, 'ear_shoulder_proxy_raw')
        if mid:
            lines.append(f"    本批数据的两档中点（in-sample，**只对本批有效、不参与"
                         f"分级**）: {_fn(mid[2])}"
                         f"   [Normal 均值 {_fn(mid[0])} / FHP 均值 {_fn(mid[1])}]")
        # 两档的段间重复性（每个连续"档"段 = 一次重复；三档的见 [4]）
        tier_segs, prev_tier = {"Normal": [], "FHP": []}, None
        for r in kept:
            lab = label_of(r)
            tier = (None if not lab else
                    ("Normal" if lab == "Normal" else "FHP"))
            if tier != prev_tier:
                if tier:
                    tier_segs[tier].append([])
                prev_tier = tier
            v = _f(r.get('ear_shoulder_proxy_raw')) if (tier and raw_ok(r)) \
                else None
            if v is not None:
                tier_segs[tier][-1].append(v)
        for tier in ("Normal", "FHP"):
            means_ = [sum(v) / len(v) for v in tier_segs[tier] if v]
            if len(means_) < 2:
                continue
            lines.append(f"    {tier:<14} 段间（{len(means_)} 段）: "
                         + "  ".join(f"{_fn(m)}" for m in means_)
                         + f"   极差 {_fn(max(means_) - min(means_))}")

        # [2] 平滑值（只作参考：平滑会吃掉组内离散度，别用它替代 [1]）
        lines += ["", "[2] ear_shoulder_proxy（EMA 平滑值，只取本帧真的出了读数的帧）",
                  "    平滑值只作参考 —— 测可重复性看 [1] 的 raw 与 [4] 的段间极差。",
                  "    姿势           n      mean        std        min        max"]
        for lab in LABEL_ORDER:
            st = stats(values(groups[lab], 'ear_shoulder_proxy', smoothed_ok))
            lines.append(f"    {lab:<12} {st['n']:>5}  {_fn(st['mean'])}  "
                         f"{_fn(st['std'])}  {_fn(st['min'])}  {_fn(st['max'])}")

        # [3] 现有角度线（对照，不作为正面判据）
        lines += ["", "[3] 对照：现有角度类指标（**只记录、不作为正面判据**；正面它们本来就"
                      "测不到头前伸）",
                  "    姿势           head_neck(°)              cva_proxy(°)",
                  "                  mean±std   min  max        mean±std   min  max"]
        for lab in LABEL_ORDER:
            hn = stats(values(groups[lab], 'head_neck_angle', all_rows))
            cp = stats(values(groups[lab], 'cva_proxy', all_rows))
            lines.append(
                f"    {lab:<12} {_fn(hn['mean'], 2):>8}±{_fn(hn['std'], 2):<6}"
                f"{_fn(hn['min'], 1):>6}{_fn(hn['max'], 1):>6}   "
                f" {_fn(cp['mean'], 2):>8}±{_fn(cp['std'], 2):<6}"
                f"{_fn(cp['min'], 1):>6}{_fn(cp['max'], 1):>6}")

        # [4] 段间重复性（每个连续标签段 = 一次重复）
        lines += ["", "[4] 段间重复性（每个连续标签段 = 一次重复；段间极差 ≪ 组内 σ ≪ 组间"
                      "间距 才说明稳）"]
        rm = run_means(kept, 'ear_shoulder_proxy_raw')
        for lab in LABEL_ORDER:
            runs = rm[lab]
            if not runs:
                continue
            means_ = [m for _n, m in runs]
            spread = max(means_) - min(means_)
            sigma = stats(values(groups[lab], 'ear_shoulder_proxy_raw'))['std']
            seg = "  ".join(f"段{i + 1})n={n} mean={_fn(m)}"
                            for i, (n, m) in enumerate(runs))
            ratio = (f"（组内 σ {_fn(sigma)} 的 {spread / sigma:.2f} 倍）"
                     if sigma else "")
            lines.append(f"    {lab:<12} {seg}")
            lines.append(f"    {'':<12} 段间极差 {_fn(spread)}{ratio}")

        # [5] 状态对照
        lines += ["", "[5] 状态对照（逐帧占比）",
                  "    当前 Posture（生产 PostureDecision 默认阈值 30/30/25/0.45，本脚本"
                  "不改它）:"]
        for lab, s in state_shares(kept, 'posture_state').items():
            if s:
                lines.append(f"      {lab:<12} {_pct(s)}")
        # 实验分级：这一块**采集时**是否真的写了分级（45° 机位默认不写 → 不适用），
        # 按数据判断，别拿 CLI 的 --view 去套另一块。
        exp_shares = state_shares(kept, 'exp_front_state')
        if any(k != "(空)" for s in exp_shares.values() for k in s):
            desc = (f"两档：< {thr:.4f} = Normal，>= {thr:.4f} = FHP；{thr_src}；"
                    f"**非医学标准、非判据**" if thr is not None
                    else "**非医学标准、非判据**")
            lines.append(f"    实验正面分档（{desc}:")
            for lab, s in exp_shares.items():
                if s:
                    lines.append(f"      {lab:<12} {_pct(s)}")
        else:
            lines.append("    实验正面分档: 不适用（这块采集时没启用实验阈值；"
                         "45° 机位不给这组阈值）")

    lines += [
        "",
        "=" * 78,
        "怎么用 / 局限（**别越界**）",
        "=" * 78,
        "  * 本脚本只做测量验证，**不拟合阈值**。报告里的实验阈值是同一批数据上的"
        "中点（in-sample），",
        "    只为让「实验正面分档」那一列有东西可看 —— **不是判据、不是医学标准**。",
        "  * **判定只做两档（Normal / 有前伸）**：实采证明 Slight 与 Obvious 分不开"
        "（跨段交错、",
        "    信噪比 0.55），所以 [1b] 是决策口径、[1] 的三档只作诊断明细。",
        "  * 要判这条信号靠不靠得住，看三件事：① 两档（[1b]）的 mean 是否分得开；"
        "② 两档区间是否 overlap；",
        "    ③ 段间极差是否远小于组间间距（[1b]/[4] 都会给；再看段间极差 / 组内 σ）。",
        "  * [1b] 里的「本批数据两档中点」只描述这一批数据的分布，**不参与分级** ——"
        "拿它去判定就是拟合。",
        "  * 下一步 validation（每次实验用 --person / --session 标好条件，报告按块分开）：",
        "      1) 同一个人、同一条件，重复 3 次以上 → 看 [4] 段间极差；",
        "      2) 换 camera distance（例如 0.5m / 0.8m / 1.2m）→ 看块与块的 mean 是否一致"
        "（肩宽归一化本该抵消远近）；",
        "      3) 换人（不同体型/发型/有无眼镜）→ 看块间是否一致；",
        "      4) 换椅子高度 / 取景（画面上人变大变小）→ 看是否还单调；",
        "      5) 每换一个条件重跑一次 `--analyze-only 文件...`，把几份报告并排比。",
        "  * Ear–Shoulder displacement proxy 只是图像平面里的相对水平位移（÷ 肩宽）：",
        "    **不是** 3D 前伸距离，**不能**叫 Forward Head Distance / clinical FHD。",
        "  * 本批数据髋全程不可见 → proxy 分不清「头相对躯干前伸」和「整个人塌下去」。",
        "  * 正面机位的角度类指标（head_neck / cva_proxy）只作对照记录，不据此判定；",
        "    45° 机位继续走 CvaProxyFeatures + FhpDecision（那条线本脚本不参与、不改）。",
        "  * 本脚本**不进生产 decision**：两档口径（方案 C）已定，但要不要接进"
        "decision.py、",
        "    什么时候接，由用户定 —— 至少先过「换距离 / 换人 / 换取景」几轮验证。",
    ]
    return lines


def _finish(args, csv_paths: list) -> None:
    """读回 CSV（可多个文件）→ 出报告 → 写 txt（s 键可随时调用，最后再调一次）。"""
    rows_all = []
    for p in csv_paths:
        rows_all += read_csv(p)
    unlabeled = sum(1 for r in rows_all if not label_of(r))
    thr, thr_src = thresholds_for(args.view, args)
    source = (f"csv:{', '.join(csv_paths)}" if args.analyze_only
              else (f"video:{args.video}" if args.video
                    else f"camera:{args.camera if args.camera is not None else 0}"))
    meta = {"source": source, "frames": len(rows_all), "unlabeled": unlabeled,
            "skip_sec": args.skip_sec, "smooth_tau": args.smooth_tau,
            "min_shoulder_width": args.min_shoulder_width,
            "dropped": 0, "thr": thr, "thr_src": thr_src,
            "view": args.view, "person": args.person, "session": args.session}
    # 段首丢弃按块算（build_report 内部），这里只给总数用于表头
    order, seen = [], set()
    for r in rows_all:
        key = block_key(r, args.view)
        if key not in seen:
            seen.add(key)
            order.append(key)
    dropped = 0
    for key in order:
        blk = [r for r in rows_all if block_key(r, args.view) == key]
        dropped += drop_run_heads(blk, args.skip_sec)[1]
    meta["dropped"] = dropped

    lines = build_report(rows_all, meta, args.view)
    for ln in lines:
        print(ln)
    report_path = args.report or (str(Path(csv_paths[0]).with_suffix(""))
                                  + "_front_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n报告已写出：{report_path}")


def run_analyze_only(args) -> None:
    """不开摄像头，直接对已有的逐帧 CSV 出报告（可给多个文件，按机位/人/条件分块）。"""
    paths = list(args.analyze_only)
    args.report = args.report or (str(Path(paths[0]).with_suffix(""))
                                  + "_front_report.txt")
    _finish(args, paths)


# ---------- CLI ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="正面机位 Ear–Shoulder displacement proxy 实验（独立脚本，"
                    "不影响 demo、不改 decision）")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, nargs="?", const=0, default=None,
                     metavar="INDEX", help="摄像头编号（默认 0）")
    src.add_argument("--video", metavar="PATH", help="视频文件（边播边按 1/2/3 打标签）")
    p.add_argument("--analyze-only", nargs="+", metavar="CSV", default=None,
                   help="只对已有的逐帧 CSV 出报告（可给多个，按块分开统计）")
    p.add_argument("--out", default="front_ear_shoulder_proxy.csv", metavar="PATH",
                   help="逐帧 CSV 输出路径，默认 front_ear_shoulder_proxy.csv")
    p.add_argument("--report", default=None, metavar="PATH",
                   help="报告 txt 路径（默认 <CSV名>_front_report.txt）")
    p.add_argument("--view", default="front", choices=("front", "45"),
                   help="机位标签（本脚本默认 front）。front=正面实验阈值；"
                        "45=只作对照、不给实验阈值")
    p.add_argument("--front-normal-max", type=float, default=None,
                   help="覆盖两档的实验分界（**非医学标准**，默认取机位预设；"
                        "本指标越大越前伸 → 这是 Normal 的上界，达到即判 FHP）")
    p.add_argument("--min-shoulder-width", type=float, default=0.25,
                   help="归一化肩宽下限，低于它判视角退化（侧身），默认 0.25")
    p.add_argument("--smooth-tau", type=float, default=1.0,
                   help="EMA 时间常数（秒），默认 1.0")
    p.add_argument("--person", default="", metavar="TAG",
                   help="实验对象标签（如 p1），报告按人分块")
    p.add_argument("--session", default="", metavar="TAG",
                   help="实验条件标签（如 0.6m/椅A/取景近），报告按条件分块")
    p.add_argument("--skip-sec", type=float, default=1.0,
                   help="丢掉每个标签段开头这么多秒（切姿势的过渡帧），默认 1.0")
    p.add_argument("--video-fps", type=float, default=0.0,
                   help="视频帧率（0=读文件元数据），影响时间戳与平滑步长")
    p.add_argument("--no-hud", dest="hud", action="store_false",
                   help="不开窗口（无人值守/只想采数据；此时无法按键打标签）")
    p.set_defaults(hud=True)
    p.add_argument("--no-skeleton", action="store_true",
                   help="画面不叠骨架（默认叠，便于核对参与计算的关键点）")
    p.add_argument("--selftest", action="store_true",
                   help="合成数据自测统计/报告逻辑（不碰摄像头/模型）")
    return p


def selftest() -> None:
    """合成 CSV 自测：标签映射、统计、段首丢弃、可分性、段间均值、分级、机位自检、
    报告文本（含 45° 块不给实验阈值）。"""
    import os
    import tempfile

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "front_esp_selftest.csv")
    path45 = os.path.join(tmp, "f45_esp_selftest.csv")

    def row(lg, i, label, raw, view="front", degenerate=0, valid=1,
            norm_width=0.42):
        """写一帧：分级列按该机位的实验阈值算（45° 没有阈值 → 留空，与实测一致）。"""
        exp = classify(raw, EXP_THRESHOLDS.get(view))
        lg.log(float(i),
               {'cva_proxy_deg': 80.0, 'cva_proxy_raw_deg': 80.0,
                'cva_proxy_conf': 0.9, 'cva_proxy_valid': True,
                'cva_proxy_pts': 'ear_mid+sh_mid'},
               FhpState.NORMAL,
               {'ratio_proxy': 1.0, 'ratio_head': 0.5, 'combined_score': 1.0},
               frame_index=i, posture_label=label,
               extra={'view': view, 'person': 'p1', 'session': 'selftest',
                      'ear_shoulder_proxy': raw,
                      'ear_shoulder_proxy_raw': raw,
                      'ear_shoulder_proxy_valid': valid,
                      'ear_shoulder_proxy_degenerate': degenerate,
                      'shoulder_width': 500.0, 'shoulder_width_norm': norm_width,
                      'ear_mid_x': 620.0, 'shoulder_mid_x': 600.0,
                      'left_ear_confidence': 0.9, 'right_ear_confidence': 0.9,
                      'left_shoulder_confidence': 0.9,
                      'right_shoulder_confidence': 0.9,
                      'head_neck_angle': 20.0, 'posture_state': 'GOOD',
                      'exp_front_state': exp.value if exp else ''})

    # front 块：Normal 0.040 / Slight 0.060 / Obvious 0.100（每段 4 帧，t=0..12）；
    # Obvious 段中间插一帧视角退化（degenerate=1，不该进统计）、
    # 一帧只出了 raw 没出平滑值（valid=0，只该进 raw 统计）
    with FhpCsvLogger(path, extra_columns=EXTRA_COLUMNS) as lg:
        for i in range(4):
            row(lg, i, "Normal", 0.040)
        for i in range(4, 8):
            row(lg, i, "Slight_FHP", 0.060)
        row(lg, 8, "Obvious_FHP", 0.100)
        row(lg, 9, "Obvious_FHP", 0.900, degenerate=1)   # 侧身塌陷：不算数
        row(lg, 10, "Obvious_FHP", 0.100)
        row(lg, 11, "Obvious_FHP", 0.100, valid=0)       # 预热中：raw 有值
        row(lg, 12, "Obvious_FHP", 0.100)

    # 45° 块（对照）：同一个人同一 session，但 view=45 → 不该有实验分级
    with FhpCsvLogger(path45, extra_columns=EXTRA_COLUMNS) as lg:
        for i in range(3):
            row(lg, i, "Normal", 0.040, view="45")
        for i in range(3, 6):
            row(lg, i, "Obvious_FHP", 0.100, view="45")

    rows = read_csv(path)
    rows45 = read_csv(path45)
    assert len(rows) == 13 and len(rows45) == 6
    assert all(r['view'] == 'front' for r in rows)
    assert all(r['exp_front_state'] == '' for r in rows45), "45° 块不该有实验分级"

    # 1) 标签映射：数字标签（老 CSV 口径）也要认
    assert label_of(rows[0]) == "Normal"
    assert label_of({"posture_label": "2"}) == "Slight_FHP"
    assert label_of({"posture_label": "0"}) == ""
    assert label_of({"posture_label": ""}) == ""

    # 2) 段首丢弃：three 段、skip 1.0s（t 步长 1s）→ 每段丢第 1 帧
    kept, dropped = drop_run_heads(rows, 1.0)
    assert dropped == 3, dropped
    groups = by_label(kept)
    assert [len(groups[l]) for l in LABEL_ORDER] == [3, 3, 4]

    # 3) 统计与过滤：退化帧不进 raw 统计，valid=0 的帧进 raw 统计
    st = stats(values(groups["Normal"], 'ear_shoulder_proxy_raw'))
    assert st["n"] == 3 and abs(st["mean"] - 0.040) < 1e-12 and st["std"] == 0.0
    st_ob = stats(values(groups["Obvious_FHP"], 'ear_shoulder_proxy_raw'))
    assert st_ob["n"] == 3 and st_ob["max"] == 0.100, st_ob   # 0.900 被剔除
    assert stats([])["n"] == 0 and stats([])["mean"] is None
    assert abs(stats([1.0, 2.0, 3.0])["std"] - 1.0) < 1e-12   # 样本标准差

    # 4) 平滑值口径：只取 valid=1 的帧（valid=0 那帧不进平滑统计）
    assert stats(values(groups["Obvious_FHP"], 'ear_shoulder_proxy',
                        smoothed_ok))["n"] == 2

    # 5) 可分性：三档恒定值 → 相邻档都能切开，切点落在两档之间
    sep = separation(kept, 'ear_shoulder_proxy_raw', ("Normal",), ("Slight_FHP",))
    assert sep["separated"] is True and 0.040 < sep["cut"] < 0.060, sep
    sep2 = separation(kept, 'ear_shoulder_proxy_raw',
                      ("Slight_FHP",), ("Obvious_FHP",))
    assert sep2["separated"] is True and 0.060 < sep2["cut"] < 0.100, sep2
    # 重叠情形：把 Normal 段读数改大 → 与 Slight 重叠
    blended = [dict(r) for r in kept]
    for r in blended:
        if label_of(r) == "Normal":
            r["ear_shoulder_proxy_raw"] = "0.065"
    sep3 = separation(blended, 'ear_shoulder_proxy_raw', ("Normal",), ("Slight_FHP",))
    assert sep3["separated"] is False and sep3["overlap_lo"] == 0.060, sep3

    # 6) 段间重复性：每个连续段一条记录
    rm = run_means(kept, 'ear_shoulder_proxy_raw')
    assert [n for n, _m in rm["Normal"]] == [3]
    assert [round(m, 4) for _n, m in rm["Obvious_FHP"]] == [0.100], rm

    # 7) 实验分档：**只做两档**（越大越前伸 → 一个上界；边界归更前伸的一档）
    thr = 0.051
    assert classify(0.040, thr) is FrontState.NORMAL
    assert classify(0.060, thr) is FrontState.FHP
    assert classify(0.100, thr) is FrontState.FHP
    assert classify(0.051, thr) is FrontState.FHP   # 边界归更前伸的一档
    assert classify(None, thr) is None and classify(0.04, None) is None
    # 两档合并口径：FHP = 标签 2+3 合并（合成数据里各 3 帧）
    assert len(pooled(kept, 'ear_shoulder_proxy_raw', FHP_LABELS)) == 6
    mid = two_tier_midpoint(kept, 'ear_shoulder_proxy_raw')
    assert mid and abs(mid[0] - 0.040) < 1e-12 and abs(mid[1] - 0.080) < 1e-12 \
        and abs(mid[2] - 0.060) < 1e-12, mid
    sep2t = separation(kept, 'ear_shoulder_proxy_raw', ("Normal",), FHP_LABELS)
    assert sep2t["separated"] is True and 0.040 < sep2t["cut"] < 0.060, sep2t

    # 8) 阈值来源：45° 机位默认不给实验阈值（要显式覆盖才给）—— 斜对机位保持原有逻辑
    args = build_parser().parse_args(["--selftest"])
    assert thresholds_for("front", args)[0] == EXP_THRESHOLDS["front"]
    assert thresholds_for("45", args)[0] is None
    args45 = build_parser().parse_args(["--front-normal-max", "0.05"])
    assert thresholds_for("45", args45)[0] == 0.05
    assert "命令行" in thresholds_for("45", args45)[1]

    # 9) 报告文本：关键小节与"非医学标准/in-sample/不是 3D"的提醒都在
    meta = {"source": "selftest", "frames": 13, "unlabeled": 0, "skip_sec": 1.0,
            "smooth_tau": 1.0, "min_shoulder_width": 0.25, "dropped": dropped,
            "thr": thr, "thr_src": "front 机位实验预设（**in-sample**）",
            "view": "front", "person": "p1", "session": "selftest"}
    txt = "\n".join(build_report(rows, meta, "front"))
    for need in ("[1] ear_shoulder_proxy_raw", "[1b] 两档口径",
                 "[2] ear_shoulder_proxy", "[3] 对照",
                 "[4] 段间重复性", "[5] 状态对照", "单调性", "怎么用 / 局限",
                 "不是 3D 前伸距离", "in-sample", "非医学标准", "view=front",
                 "person=p1", "机位自检", "FHP (2+3)", "两档可分性"):
        assert need in txt, f"报告缺 {need!r}"
    assert "Normal → Slight: [分离]" in txt and "Slight → Obvious: [分离]" in txt
    assert "两档可分性: [分离]" in txt
    assert "本批数据的两档中点（in-sample" in txt
    assert "实验正面分档（两档：< 0.0510 = Normal，>= 0.0510 = FHP" in txt
    # 合成数据里 Normal 0.040 / FHP 0.060+0.100 → 两档开
    assert "本批数据的两档中点（in-sample，**只对本批有效、不参与分级**）: 0.0600" \
        in txt

    # 10) 多块报告：45° 块不给实验阈值 → "不适用"；取景偏斜时给机位自检提醒
    txt2 = "\n".join(build_report(rows + rows45, meta, "front"))
    assert "view=45" in txt2, "45° 块应单独成节"
    assert "实验正面分档: 不适用" in txt2
    narrow = [dict(r, shoulder_width_norm="0.22") for r in rows]
    assert "取景看起来**不是明显正面**" in "\n".join(
        build_report(narrow, meta, "front"))
    print("  selftest_front_ear_shoulder_proxy: OK")


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        return
    if args.analyze_only:
        run_analyze_only(args)
        return
    if args.camera is None and not args.video:
        build_parser().error("要么给 --camera 0，要么给 --video x.mp4"
                             "（或 --analyze-only CSV / --selftest）")
    run_capture(args)


if __name__ == "__main__":
    main()
