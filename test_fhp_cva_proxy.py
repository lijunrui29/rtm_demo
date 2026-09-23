"""
头前伸 CVA-like proxy 标定/实验脚本（**独立脚本，不影响 main_demo/web_demo**）。

用途：坐在摄像头前（或放一段视频），按姿势分段打标签，记录 proxy 读数，
最后给出每种姿势的 mean / std / min / max 和 CSV，用来**自标定**本项目的
头前伸阈值（阈值按机位、按取景变，别套文献数字）。

跑法（在 rtm_demo/ 目录下，用装了 cv2/onnxruntime 的解释器）：
    python test_fhp_cva_proxy.py --camera 0             # 摄像头，默认 45° 机位
    python test_fhp_cva_proxy.py --camera 0 --view front  # 正面机位（另一套阈值）
    python test_fhp_cva_proxy.py --video clip.mp4       # 视频文件（边播边打标签）
    python test_fhp_cva_proxy.py --analyze-only fhp_cva_proxy.csv   # 只做统计，不开摄像头
    python test_fhp_cva_proxy.py --selftest             # 合成数据自测统计/报告逻辑

按键（窗口里）：
    1 = Normal（坐直）        2 = Slight FHP（轻微前伸）   3 = Obvious FHP（明显前伸）
    0 = 清除当前标签（这一帧不参与统计）   s = 立刻出报告并写盘   q / ESC = 结束

姿势怎么摆：每个姿势保持 15~20 秒、每种至少重复 2~3 次（换姿势时段开头
1 秒是过渡帧，报告里用 --skip-sec 自动丢掉）。摆姿势时头部只动、肩膀尽量别动
——proxy 量的是"耳相对肩"的角度。

设计说明（几条必须记住的口径）：
    * 本脚本**复用生产链路**（features.CvaProxyFeatures → decision.FhpDecision →
      output.draw_fhp_overlay / FhpCsvLogger），不另写一套算法，所以标出来的阈值
      就是产品里真正生效的口径。
    * 这是 **CVA-like proxy，不是临床 CVA**：C7 用双肩中点近似、耳点不是 tragus、
      相机是正面~45° 前侧而不是矢状面。量值不可与文献 CVA 比对。
    * 角度在**像素空间**算（不含画面宽高比）；换分辨率/换宽高比必须重新标定。
    * 三种判据对比（需求里的 A/B/C，报告里给逐帧命中率/误报率）：
        A = 原有头颈角判据（head_neck_angle >= 阈值）
        B = 只用 CVA-like proxy
        C = 两者取大合成（生产默认）
    * 单帧判据对比用**比值**（original_fhp_score / ratio_proxy / combined_score），
      不含迟滞和时长规则——三条线口径一致才可比。带迟滞的状态另有一张分布表。
    * 阈值只在同机位、同取景、同一个人身上成立；换摄像头位置一定要重标。

CSV 列见 output.FHP_CSV_COLUMNS（需求点名的 6 列在前，后面是调试/分析列），
外加本脚本追加的 `state_a` / `state_b`（A、B 两种判据各自的状态机输出，供对比）。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from features import CvaProxyFeatures, PostureFeatures          # noqa: E402
from decision import (FhpDecision, FhpState,                    # noqa: E402
                      CVA_PROXY_VIEW_PRESETS, CVA_PROXY_HYSTERESIS_RATIO)
from output import FrameRenderer, FhpCsvLogger                  # noqa: E402
from pose_estimation import detect_pose                         # noqa: E402

# 键盘标签 → 姿势名（与需求里的 Normal / Slight FHP / Obvious FHP 对应）
LABEL_KEYS = {"1": "Normal", "2": "Slight_FHP", "3": "Obvious_FHP"}
LABEL_ORDER = ("Normal", "Slight_FHP", "Obvious_FHP")
FHP_LABELS = ("Slight_FHP", "Obvious_FHP")   # "应该被判为头前伸"的两个标签

# 报告里逐列统计的量：(CSV 列名, 说明)
METRICS = (
    ("cva_proxy", "CVA proxy 平滑值（度，越小越前伸）"),
    ("cva_proxy_raw", "CVA proxy 原始值（未平滑，度）"),
    ("original_fhp_score", "原有头颈角得分（head_neck/阈值，>=1 旧规则触发）"),
    ("combined_score", "合成得分（>=1 至少 Slight）"),
    ("keypoint_confidence", "proxy 用到关键点的平均置信度"),
)

# 单帧判据对比用的三个比值列（口径一致：>= 1.0 即该判据判 FHP）
SCHEMES = (
    ("A 原有头颈角", "original_fhp_score"),
    ("B CVA-like proxy", "ratio_proxy"),
    ("C 合成（取大）", "combined_score"),
)


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
    """读逐帧 CSV（本脚本自己写的，列名见 output.FHP_CSV_COLUMNS）。"""
    with open(path, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)]


def drop_run_heads(rows: list, skip_sec: float) -> tuple:
    """丢掉每个标签段开头 skip_sec 秒（切姿势时的过渡帧）。

    段 = 连续相同 posture_label 的行。返回 (保留行, 丢弃帧数)。
    时间用 CSV 的 timestamp 列（视频=播放秒数，实时=单调时钟秒数）。
    """
    if skip_sec <= 0:
        return list(rows), 0
    kept, dropped = [], 0
    prev_label, run_start = None, None
    for r in rows:
        lab = r.get("posture_label") or ""
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


def group_by_label(rows: list) -> dict:
    """{标签: [行, ...]}，只保留 LABEL_ORDER 里的标签（未标注的进 ''）。"""
    out = {lab: [] for lab in LABEL_ORDER}
    out[""] = []
    for r in rows:
        lab = r.get("posture_label") or ""
        out.setdefault(lab, []).append(r)
    return out


def separation(groups: dict, key: str = "cva_proxy",
               lo_labels=FHP_LABELS, hi_labels=("Normal",)) -> dict:
    """看两组读数能否用一条阈值切开（proxy 越小越前伸）。

    lo_labels: 读数应更小（更前伸）的一组标签；hi_labels: 应更大的一组。
    比 Normal vs FHP 时 hi=Normal；比 Slight vs Obvious 时 hi=Slight（不是 Normal）。

    返回 dict(separated, cut, gap, overlap_lo, overlap_hi, lo_max, hi_min)。
    separated=True 时可用阈值区间是 (lo_max, hi_min]，建议取中点 cut。
    """
    lo = [v for lab in lo_labels for v in
          (_f(r.get(key)) for r in groups.get(lab, [])) if v is not None]
    hi = [v for lab in hi_labels for v in
          (_f(r.get(key)) for r in groups.get(lab, [])) if v is not None]
    if not lo or not hi:
        return {"separated": None, "cut": None, "gap": None,
                "overlap_lo": None, "overlap_hi": None,
                "lo_max": None, "hi_min": None}
    lo_max, hi_min = max(lo), min(hi)
    sep = lo_max < hi_min
    return {"separated": sep,
            "cut": 0.5 * (lo_max + hi_min) if sep else None,
            "gap": hi_min - lo_max,
            "overlap_lo": None if sep else hi_min,
            "overlap_hi": None if sep else lo_max,
            "lo_max": lo_max, "hi_min": hi_min}


def scheme_rates(rows: list) -> dict:
    """A/B/C 三种判据的逐帧命中/误报（比值口径，>= 1.0 即判 FHP）。

    返回 {判据名: {'hit': 命中率, 'false': 误报率, 'acc': 准确率,
                   'n_fhp': n, 'n_normal': m}}；
    未标注的行不参与。命中率 = FHP 段里判为 FHP 的帧占比；
    误报率 = Normal 段里判为 FHP 的帧占比。
    """
    out = {}
    for name, key in SCHEMES:
        hit_n = hit_ok = norm_n = norm_bad = 0
        for r in rows:
            lab = r.get("posture_label") or ""
            v = _f(r.get(key))
            if v is None or lab not in LABEL_ORDER:
                continue
            flagged = v >= 1.0
            if lab in FHP_LABELS:
                hit_n += 1
                hit_ok += int(flagged)
            else:
                norm_n += 1
                norm_bad += int(flagged)
        total = hit_n + norm_n
        acc = (hit_ok + (norm_n - norm_bad)) / total if total else None
        out[name] = {
            "hit": (hit_ok / hit_n) if hit_n else None,
            "false": (norm_bad / norm_n) if norm_n else None,
            "acc": acc, "n_fhp": hit_n, "n_normal": norm_n,
        }
    return out


def state_shares(rows: list, column: str) -> dict:
    """每个姿势段里各状态占比（带迟滞的状态机输出分布）。

    返回 {标签: {状态: 帧数}}，另含 'n'（该段有效帧数）。
    """
    out = {}
    for lab in LABEL_ORDER:
        rws = [r for r in rows if (r.get("posture_label") or "") == lab]
        states = {}
        for r in rws:
            s = r.get(column) or "?"
            states[s] = states.get(s, 0) + 1
        out[lab] = states
        out[lab]["n"] = len(rws)
    return out


# ---------- 报告 ----------

def _pct(v, nd=1) -> str:
    return "--" if v is None else f"{v * 100:.{nd}f}%"


def _deg(v, nd=1) -> str:
    return "--" if v is None else f"{v:.{nd}f}"


def build_report(rows: list, meta: dict, dropped: int,
                 unlabeled: int) -> list:
    """把统计结果拼成报告文本行（纯函数，方便自测与 --analyze-only 复用）。"""
    groups = group_by_label(rows)
    L = []
    L.append("=" * 90)
    L.append("头前伸 CVA-like proxy 标定报告（proxy ≠ 临床 CVA：C7 用肩中点近似、"
             "耳点非 tragus、像素空间）")
    L.append("=" * 90)
    L.append(f"数据源: {meta['source']}    帧数: {meta['frames']}"
             f"    判别帧: {len(rows)}（丢弃段首过渡 {dropped} 帧，未标注 {unlabeled} 帧）")
    hyst = meta.get("hysteresis_ratio")
    hyst_s = ("" if hyst is None else
              f"    迟滞比: {hyst:g}（恢复点 proxy >= "
              f"{meta['slight'] / hyst:.2f}°）")
    L.append(f"机位: {meta['view']}    阈值: slight <= {meta['slight']:.2f}° / "
             f"obvious <= {meta['obvious']:.2f}°    "
             f"（阈值来源: {meta['threshold_src']}）{hyst_s}")
    L.append(f"合成权重: proxy={meta['weight_proxy']:g} head={meta['weight_head']:g}"
             f"    原有头颈角阈值: {meta['head_neck_threshold']:g}°")
    L.append("")
    L.append("各姿势帧数: " + "  ".join(
        f"{lab}={len(groups.get(lab, []))}" for lab in LABEL_ORDER)
        + f"  未标注={len(groups.get('', []))}")
    L.append("")

    # 每种姿势的 mean/std/min/max
    for key, desc in METRICS:
        L.append(f"[{key}] {desc}")
        L.append("  姿势                n      mean       std       min"
                 "       max    median")
        for lab in LABEL_ORDER:
            vals = [_f(r.get(key)) for r in groups.get(lab, [])]
            st = stats(vals)
            L.append(f"  {lab:<16} {st['n']:>4}  "
                     + (f"{st['mean']:>8.2f}  {st['std']:>8.2f}  "
                        f"{st['min']:>8.2f}  {st['max']:>8.2f}  "
                        f"{st['median']:>8.2f}" if st["n"] else
                        f"{'--':>8}  {'--':>8}  {'--':>8}  {'--':>8}  {'--':>8}"))
        L.append("")

    # 可分性：能不能用一条 proxy 阈值切开
    L.append("可分性（proxy 越小越前伸；用你这次的数据自标定，别套文献数字）:")
    for title, lo_labels, hi_labels in (
            ("Normal vs (Slight+Obvious)", FHP_LABELS, ("Normal",)),
            ("Slight vs Obvious", ("Obvious_FHP",), ("Slight_FHP",))):
        sep = separation(groups, "cva_proxy", lo_labels, hi_labels)
        if sep["separated"] is None:
            L.append(f"  {title}: 数据不足（缺 Normal 或 FHP 段），无法判断")
        elif sep["separated"]:
            L.append(f"  {title}: 可分 —— 阈值区间 [{sep['lo_max']:.2f}, "
                     f"{sep['hi_min']:.2f}]，宽 {sep['gap']:.2f}°，"
                     f"建议取中点 {sep['cut']:.2f}°")
        else:
            L.append(f"  {title}: 有重叠 —— 重叠区 [{sep['overlap_lo']:.2f}, "
                     f"{sep['overlap_hi']:.2f}]，宽 "
                     f"{sep['overlap_hi'] - sep['overlap_lo']:.2f}°"
                     f"（重叠越宽越难分；建议加做重复/换机位，或看下面 A/B/C）")
    L.append("")

    # A/B/C 对比
    L.append("A/B/C 三种判据对比（单帧层面，比值口径、不含迟滞与时长规则）:")
    L.append("  判据                    FHP 帧命中率    Normal 帧误报率    "
             "准确率      n(FHP/Normal)")
    for name, _key in SCHEMES:
        s = scheme_rates(rows)[name]
        L.append(f"  {name:<22} {_pct(s['hit']):>12} {_pct(s['false']):>16} "
                 f"{_pct(s['acc']):>11}     {s['n_fhp']}/{s['n_normal']}")
    L.append("")

    # 带迟滞的状态机分布（生产口径）
    L.append("状态机分布（带迟滞，生产口径；C = fhp_state，A/B 为对照配置）:")
    for col, title in (("fhp_state", "C 合成（默认）"), ("state_b", "B 只 proxy"),
                       ("state_a", "A 只头颈角")):
        shares = state_shares(rows, col)
        pieces = []
        for lab in LABEL_ORDER:
            st = shares.get(lab, {})
            n = st.get("n", 0)
            if not n:
                pieces.append(f"{lab}=--")
                continue
            fhp = st.get("SLIGHT_FHP", 0) + st.get("OBVIOUS_FHP", 0)
            pieces.append(f"{lab}={fhp}/{n} 判 FHP")
        L.append(f"  {title:<14} " + "   ".join(pieces))

    L.append("")
    L.append("怎么用:")
    L.append("  * 把上面「可分」给的阈值中点填进 --cva-proxy-slight-threshold /")
    L.append("    --cva-proxy-obvious-threshold（main_demo.py 与 web_demo.py），"
             "或标定后改 decision.CVA_PROXY_VIEW_PRESETS 的预设值。")
    L.append("  * 阈值只在**同机位、同取景、同一个人**上成立；换摄像头位置/换椅子"
             "高度必须重标。")
    L.append("  * 每种姿势只测了 1 次的话，min/max 就是单次极值，不代表分布；"
             "要下结论请重复 3 次以上、换 2~3 个人。")
    return L


# ---------- 采集 ----------

class _Source:
    """统一数据源：摄像头（capture.FrameSource：read() → 帧或 None）和视频文件
    （cv2.VideoCapture：read() → (ok, frame)）都包成 read() → (ok, frame) + with。

    摄像头走 capture 层的 CameraCapture（不在本脚本里重造），视频文件本脚本直接
    用 cv2（放本地 mp4 标定时不需要机器人摄像头那套）。
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
        """该帧的时间戳（秒）。

        视频用播放时钟（帧号/帧率）——同一段视频每次跑读数一致，便于复现；
        实时摄像头用单调时钟（features 的 EMA 需要真实时间间隔）。
        """
        if self._cap is not None:
            return idx / self.fps
        return time.monotonic() - self._t0


def _text(frame, txt, org, scale=0.5, color=(230, 230, 230)):
    """带黑描边的文字（画面实时看，深浅背景都要清楚）。

    描边用 4 次 1px 偏移的细笔画，不用「先粗后细同一位置两遍」——后者在
    OpenCV 5.0 的 LINE_AA 下会把最后一个字重复画在字符串末尾（见
    output._put_text_outlined 的说明）。
    """
    import cv2
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(frame, txt, (org[0] + dx, org[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                cv2.LINE_AA)


def run_capture(args) -> None:
    """主流程：采集 → 打标签 → 逐帧写 CSV → 结束时出报告。"""
    import cv2

    slight, obvious = CVA_PROXY_VIEW_PRESETS[args.view]
    threshold_src = f"{args.view} 机位实验预设"
    if args.cva_proxy_slight_threshold is not None:
        slight = args.cva_proxy_slight_threshold
        threshold_src = "命令行 --cva-proxy-slight-threshold 覆盖"
    if args.cva_proxy_obvious_threshold is not None:
        obvious = args.cva_proxy_obvious_threshold
        threshold_src = "命令行覆盖"

    # 复用生产链路：同一个特征类 + 同一个决策类（C 生产口径，A/B 为对照配置）
    feats = CvaProxyFeatures()
    post = PostureFeatures()
    common = dict(slight_threshold=slight, obvious_threshold=obvious,
                  head_neck_threshold=args.head_neck_threshold,
                  hysteresis_ratio=args.hysteresis_ratio)
    dec_c = FhpDecision(weight_proxy=args.weight_proxy,
                        weight_head=args.weight_head, **common)
    dec_a = FhpDecision(weight_proxy=0.0, weight_head=1.0, **common)
    dec_b = FhpDecision(weight_proxy=1.0, weight_head=0.0, **common)
    ren = FrameRenderer(draw_skeleton=not args.no_skeleton)

    src = _Source(args)
    print(f"数据源: {src.desc}")
    print(f"机位 {args.view}：slight <= {slight:.2f}°  obvious <= {obvious:.2f}°"
          f"（{threshold_src}）")
    print("正在初始化 pose 检测器")
    print("按键：1=Normal  2=Slight FHP  3=Obvious FHP  0=清除标签   "
          "s=出报告  q/ESC=结束")
    print(f"逐帧 CSV: {args.out}")

    label = ""            # 当前标签（'' = 未标注）
    label_since = 0.0     # 当前标签开始的时刻（窗口里显示已标秒数）
    idx = 0
    warnings = []
    with src:
        with FhpCsvLogger(args.out, extra_columns=("state_a", "state_b")) as lg:
            while True:
                ok, frame = src.read()
                if not ok:
                    break
                ts = src.stamp(idx)

                pose = detect_pose(frame)
                cva = feats.update(pose, ts)
                posture = post.update(pose)
                st_c = dec_c.update(cva, posture)
                st_a = dec_a.update(cva, posture)
                st_b = dec_b.update(cva, posture)

                if cva.get('cva_proxy_deg') is None and \
                        cva.get('cva_proxy_raw_deg') is None:
                    warnings.append(idx)

                lg.log(ts, cva, st_c, dec_c, frame_index=idx,
                       posture_label=label,
                       extra={'state_a': st_a.value, 'state_b': st_b.value})

                if not args.no_hud:
                    ren.draw_skeleton_frame(frame, pose)
                    ren.draw_fhp_overlay(
                        frame, cva, st_c, dec_c,
                        show_geometry=True, posture_label=label)
                    _text(frame, f"{src.desc}  t={ts:6.1f}s  frame={idx}",
                          (12, 26), 0.55, (230, 230, 230))
                    if label:
                        held = ts - label_since
                        _text(frame, f"[LABEL] {label}  {held:5.1f}s",
                              (12, 50), 0.6, (60, 220, 60))
                    else:
                        _text(frame, "[LABEL] -- (press 1/2/3)",
                              (12, 50), 0.6, (0, 215, 255))
                    _text(frame, "1=Normal 2=Slight 3=Obvious 0=clear "
                                 "s=report q=quit", (12, 74), 0.45,
                          (200, 200, 200))
                    cv2.imshow("FHP CVA-like proxy (experiment)", frame)
                    k = cv2.waitKey(1) & 0xFF
                    ch = chr(k) if 32 <= k < 127 else ""
                    if k == 27 or ch in ("q", "Q"):
                        break
                    if ch in LABEL_KEYS:
                        label = LABEL_KEYS[ch]
                        label_since = ts
                        print(f"[label] {label} 从 t={ts:.1f}s 开始")
                    elif ch == "0":
                        label = ""
                    elif ch in ("s", "S"):
                        lg.flush()      # 先落盘再读回，否则报告里缺最近几十帧
                        _finish(args, label)
                idx += 1

    cv2.destroyAllWindows()

    print(f"\n共 {idx} 帧，CSV: {args.out}")
    if warnings:
        print(f"[warn] {len(warnings)} 帧 proxy 无效（首帧 {warnings[0]}；"
              f"常见原因：人不在画面/耳或肩点置信度过低）")
    _finish(args, label)


def _finish(args, label: str) -> None:
    """读回 CSV → 出报告 → 写 txt（s 键可随时调用，最后再调一次）。"""
    rows_all = read_csv(args.out)
    kept, dropped = drop_run_heads(rows_all, args.skip_sec)
    unlabeled = sum(1 for r in rows_all if not (r.get("posture_label") or ""))
    slight, obvious = CVA_PROXY_VIEW_PRESETS[args.view]
    threshold_src = f"{args.view} 机位实验预设"
    if args.cva_proxy_slight_threshold is not None:
        slight = args.cva_proxy_slight_threshold
        threshold_src = "命令行覆盖"
    if args.cva_proxy_obvious_threshold is not None:
        obvious = args.cva_proxy_obvious_threshold
    if args.analyze_only:
        source = f"csv:{args.analyze_only}"
    elif args.video:
        source = f"video:{args.video}"
    else:
        source = f"camera:{args.camera if args.camera is not None else 0}"
    meta = {
        "source": source,
        "frames": len(rows_all),
        "view": args.view,
        "slight": slight, "obvious": obvious,
        "threshold_src": threshold_src,
        "weight_proxy": args.weight_proxy, "weight_head": args.weight_head,
        "head_neck_threshold": args.head_neck_threshold,
        "hysteresis_ratio": args.hysteresis_ratio,
    }
    lines = build_report(kept, meta, dropped, unlabeled)
    for ln in lines:
        print(ln)
    report_path = args.report or (str(Path(args.out).with_suffix(""))
                                  + "_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n报告已写出：{report_path}")


def run_analyze_only(args) -> None:
    """不开摄像头，直接对已有 CSV 出报告（改统计口径时反复用，不用重录）。"""
    args.report = args.report or (str(Path(args.analyze_only).with_suffix(""))
                                  + "_report.txt")
    # 复用 _finish 的报告逻辑，但数据来自 --analyze-only 指定的 CSV
    args.out = args.analyze_only
    _finish(args, "")


# ---------- CLI ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="头前伸 CVA-like proxy 标定/实验（独立脚本，不影响 demo）")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, nargs="?", const=0, default=None,
                     metavar="INDEX", help="摄像头编号（默认 0）")
    src.add_argument("--video", metavar="PATH", help="视频文件（边播边按 1/2/3 打标签）")
    p.add_argument("--analyze-only", metavar="CSV", default=None,
                   help="只对已有的逐帧 CSV 出报告，不开摄像头")
    p.add_argument("--out", default="fhp_cva_proxy.csv", metavar="PATH",
                   help="逐帧 CSV 输出路径，默认 fhp_cva_proxy.csv")
    p.add_argument("--report", default=None, metavar="PATH",
                   help="报告 txt 路径（默认 <CSV名>_report.txt）")
    p.add_argument("--view", choices=sorted(CVA_PROXY_VIEW_PRESETS), default="45",
                   help="机位（45=45° 前侧、front=正面），决定用哪套实验阈值，默认 45")
    p.add_argument("--cva-proxy-slight-threshold", type=float, default=None,
                   help="覆盖 proxy 轻微阈值（度，默认取 --view 预设）")
    p.add_argument("--cva-proxy-obvious-threshold", type=float, default=None,
                   help="覆盖 proxy 明显阈值（度，默认取 --view 预设）")
    p.add_argument("--weight-proxy", type=float, default=1.0,
                   help="合成权重：proxy 一路，默认 1.0")
    p.add_argument("--weight-head", type=float, default=1.0,
                   help="合成权重：原有头颈角一路，默认 1.0")
    p.add_argument("--head-neck-threshold", type=float, default=30.0,
                   help="原有头颈角阈值（度），用于 A/C 对比，默认 30")
    p.add_argument("--hysteresis-ratio", type=float,
                   default=CVA_PROXY_HYSTERESIS_RATIO,
                   help="proxy 迟滞比（状态机口径，影响上面「状态机分布」表），"
                        f"默认 {CVA_PROXY_HYSTERESIS_RATIO:g}")
    p.add_argument("--skip-sec", type=float, default=1.0,
                   help="丢掉每个标签段开头这么多秒（切姿势的过渡帧），默认 1.0")
    p.add_argument("--video-fps", type=float, default=0.0,
                   help="视频帧率（0=读文件元数据），影响时间戳与平滑步长")
    p.add_argument("--no-hud", dest="hud", action="store_false",
                   help="不开窗口（无人值守/只想采数据；此时无法按键打标签）")
    p.set_defaults(hud=True)
    p.add_argument("--no-skeleton", action="store_true",
                   help="画面不叠骨架（默认叠，便于核对参与 proxy 的关键点）")
    p.add_argument("--selftest", action="store_true",
                   help="合成数据自测统计/报告逻辑（不碰摄像头/模型）")
    return p


def selftest() -> None:
    """合成 CSV 自测：统计、段首丢弃、A/B/C 对比、报告文本。"""
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "fhp_selftest.csv")

    def row(lg, i, label, cva, ratio_head, state):
        lg.log(float(i),
               {'cva_proxy_deg': cva, 'cva_proxy_raw_deg': cva,
                'cva_proxy_conf': 0.9, 'cva_proxy_valid': True,
                'cva_proxy_pts': 'ear_mid+sh_mid'},
               state,
               {'ratio_proxy': 68.17 / cva, 'ratio_head': ratio_head,
                'combined_score': max(68.17 / cva, ratio_head)},
               frame_index=i, posture_label=label,
               extra={'state_a': FhpState.NORMAL.value,
                      'state_b': state.value})

    # 12 帧：Normal 88° / Slight 65° / Obvious 55°（各 4 帧，t = 0..11）
    # ratio_head 全 < 1 → A 判据一次都不触发，B/C 在 FHP 段全触发
    with FhpCsvLogger(path, extra_columns=("state_a", "state_b")) as lg:
        for i in range(4):
            row(lg, i, "Normal", 88.0, 0.5, FhpState.NORMAL)
        for i in range(4, 8):
            row(lg, i, "Slight_FHP", 65.0, 0.3, FhpState.SLIGHT_FHP)
        for i in range(8, 12):
            row(lg, i, "Obvious_FHP", 55.0, 0.3, FhpState.OBVIOUS_FHP)

    rows = read_csv(path)
    assert len(rows) == 12

    # 1) 段首丢弃：每段 4 帧、skip 1.0s（t 步长 1s）→ 每段丢第 1 帧、留 3 帧
    kept, dropped = drop_run_heads(rows, 1.0)
    assert dropped == 3, dropped
    groups = group_by_label(kept)
    assert [len(groups[l]) for l in LABEL_ORDER] == [3, 3, 3]

    # 2) 每种姿势的 mean/std/min/max（同段读数恒定 → std=0、min=max=mean）
    st = stats([_f(r["cva_proxy"]) for r in groups["Slight_FHP"]])
    assert st["n"] == 3 and st["mean"] == 65.0 and st["std"] == 0.0
    assert st["min"] == st["max"] == 65.0 and st["median"] == 65.0
    assert stats([])["n"] == 0 and stats([])["mean"] is None
    assert abs(stats([1.0, 2.0, 3.0])["std"] - 1.0) < 1e-12   # 样本标准差

    # 3) 可分性：Normal(88) vs FHP(65,55) 应判"可分"，切点落在 (65, 88]
    sep = separation(groups, "cva_proxy", FHP_LABELS)
    assert sep["separated"] is True and 65.0 < sep["cut"] <= 88.0, sep
    sep2 = separation(groups, "cva_proxy", ("Obvious_FHP",), ("Slight_FHP",))
    assert sep2["separated"] is True and 55.0 < sep2["cut"] <= 65.0, sep2

    # 4) A/B/C：A 全不触发（ratio_head<1，只能靠 Normal 段对分得 4/12）、
    #    B/C 在 FHP 段全触发、Normal 段都不触发（acc=1.0）
    r = scheme_rates(rows)
    a = r["A 原有头颈角"]
    assert a["hit"] == 0.0 and a["false"] == 0.0 and a["n_fhp"] == 8, a
    assert abs(a["acc"] - 4 / 12) < 1e-12, a
    for name in ("B CVA-like proxy", "C 合成（取大）"):
        assert r[name]["hit"] == 1.0 and r[name]["false"] == 0.0, r[name]
        assert abs(r[name]["acc"] - 1.0) < 1e-12, r[name]

    # 5) 状态机分布：C 口径下 Normal 3 帧判 Normal、FHP 段 6 帧全判 FHP
    shares = state_shares(rows, "fhp_state")
    assert shares["Normal"]["NORMAL"] == 4 and shares["Normal"]["n"] == 4
    assert shares["Slight_FHP"]["SLIGHT_FHP"] == 4
    assert shares["Obvious_FHP"]["OBVIOUS_FHP"] == 4

    # 6) 报告文本：关键小节都在，A 的命中率是 0.0%
    meta = {"source": "selftest", "frames": 12, "view": "45",
            "slight": 68.17, "obvious": 61.42, "threshold_src": "预设",
            "weight_proxy": 1.0, "weight_head": 1.0, "head_neck_threshold": 30.0}
    lines = build_report(kept, meta, dropped, 0)
    txt = "\n".join(lines)
    for need in ("[cva_proxy]", "可分性", "A/B/C 三种判据对比", "状态机分布",
                 "怎么用", "Normal=3", "A 原有头颈角", "mean"):
        assert need in txt, f"报告缺 {need!r}"
    assert "68.17" in txt and "61.42" in txt
    print("  selftest_fhp_cva_proxy: OK")


def _synth_pose(ear_dx_px: float, size=(1280, 720), shoulder_y=500.0,
                ear_dy_px=150.0, vis=0.9) -> dict:
    """合成一个人（只有耳/肩点有用）：耳中点相对肩中点水平偏 ear_dx_px 像素。

    真值角度 = atan2(ear_dy_px, ear_dx_px)（像素空间），用来验证 features 的
    CvaProxyFeatures 算出来的 cva_proxy 是否就是这个角度。
    """
    w, h = size
    sx, sy = w / 2.0, shoulder_y
    ex, ey = sx + ear_dx_px, sy - ear_dy_px
    lm = [(0.5, 0.4, 0.0, vis)] * 17          # 其余点随便给（本测试用不到）
    lm[3] = (ex / w, ey / h, 0.0, vis)         # 左耳
    lm[4] = (ex / w, ey / h, 0.0, vis)         # 右耳（与左耳同点 → 中点=该点）
    lm[5] = (sx / w, sy / h, 0.0, vis)         # 左肩
    lm[6] = (sx / w, sy / h, 0.0, vis)         # 右肩（同上）
    return {'landmarks': lm, 'image_size': size}


def selftest_pipeline() -> None:
    """端到端（合成姿态）：真特征类 → 真决策类 → 真 CSV → 真报告。

    用已知几何的合成人验证「features 算出的 cva_proxy == 手算角度」，
    再验证这串链路能产出可读的报告（含三档姿势、A/B/C 对比）。
    """
    import os
    import tempfile

    # 45° 机位实测中位数附近的三个点：Normal 73° / Slight 65° / Obvious 58°
    # 每段 12 帧、帧间隔 1s → EMA(τ=1s) 后半段已收敛，便于断言
    cases = [("Normal", 73.0, 12), ("Slight_FHP", 65.0, 12),
             ("Obvious_FHP", 58.0, 12)]
    path = os.path.join(tempfile.mkdtemp(), "fhp_pipeline.csv")
    feats = CvaProxyFeatures(smooth_tau_sec=1.0)
    post = PostureFeatures()
    dec = FhpDecision()

    def pose_for(deg):
        dx = 150.0 / math.tan(math.radians(deg))
        return _synth_pose(dx)

    t = 0.0
    with FhpCsvLogger(path) as lg:
        i = 0
        for label, deg, n in cases:
            prev_smooth = None
            for k in range(n):
                pose = pose_for(deg)
                cva = feats.update(pose, t)
                posture = post.update(pose)
                st = dec.update(cva, posture)
                raw, smooth = (cva['cva_proxy_raw_deg'],
                               cva['cva_proxy_deg'])
                # 核心几何：特征算出的角度 == 手算的像素空间角度（每帧）
                assert raw is not None and abs(raw - deg) < 0.01, (label, raw, deg)
                if smooth is not None and prev_smooth is not None:
                    # EMA 性质：单调逼近目标，不过冲（用上一帧值夹住）
                    lo, hi = min(prev_smooth, deg), max(prev_smooth, deg)
                    assert lo - 1e-9 <= smooth <= hi + 1e-9, \
                        (label, k, prev_smooth, smooth, deg)
                    prev_smooth = smooth
                elif smooth is not None:
                    prev_smooth = smooth
                if k == n - 1:   # 段末：平滑值应已收敛到真值
                    assert abs(smooth - deg) < 0.05, (label, smooth, deg)
                lg.log(t, cva, st, dec, frame_index=i, posture_label=label)
                t += 1.0
                i += 1

    rows = read_csv(path)
    assert len(rows) == 36
    kept, dropped = drop_run_heads(rows, 1.0)      # 每段丢 1 帧过渡
    assert dropped == 3, dropped
    # 只看每段后 6s（收敛段）：mean 应贴住真值，段内不该乱跳
    conv, dropped6 = drop_run_heads(rows, 6.0)
    assert dropped6 == 18, dropped6
    groups = group_by_label(conv)
    for label, deg, _n in cases:
        st = stats([_f(r['cva_proxy']) for r in groups[label]])
        assert st['n'] == 6 and abs(st['mean'] - deg) < 0.05, (label, st)
        assert st['max'] - st['min'] < 0.2, (label, st)
    sep = separation(groups, "cva_proxy", FHP_LABELS, ("Normal",))
    assert sep['separated'] is True and 65.0 < sep['cut'] <= 73.0, sep
    meta = {"source": "synthetic", "frames": len(rows), "view": "45",
            "slight": 68.17, "obvious": 61.42, "threshold_src": "预设",
            "weight_proxy": 1.0, "weight_head": 1.0, "head_neck_threshold": 30.0}
    txt = "\n".join(build_report(kept, meta, dropped, 0))
    assert "Slight vs Obvious" in txt and "可分" in txt
    print("  selftest_pipeline: OK")


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        selftest_pipeline()
        print("test_fhp_cva_proxy selftest: ALL PASSED")
        return
    if args.analyze_only:
        run_analyze_only(args)
        return
    if args.video is None and args.camera is None:
        build_parser().error("请给 --camera 0 / --video PATH / --analyze-only CSV")
    run_capture(args)


if __name__ == "__main__":
    main()
