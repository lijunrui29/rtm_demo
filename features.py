"""
features 层：从一帧帧的 pose 结果里提取"移动量"这个特征。

FeatureExtractor 维护一个滑动时间窗口（默认 3 秒），跟踪"躯干锚点中心"
（肩中点与髋中点的中点；髋不可见时退化为肩中点），计算归一化的移动量：

    movement = 锚点对"窗口内锚点中位位置"的**中位**位移 / 归一化尺度

只跟踪锚点中心：五官、四肢（耳/肘/腕）的抖动基本不带动肩-髋中心，
不会因为说话、挥手、手指动就被判"在动"。位移统计用中位数而不是
RMS：偶发的一帧检测抖动/孤点不会把移动量顶上去，只有持续的整体位移
（起身、身体大幅度移动）才会让移动量变大 —— 这符合"检测是否静坐
过久"的目标：静坐期间的轻微动作不算"在动"。

其中 归一化尺度 优先 = 躯干高度 mean(‖5-11‖, ‖6-12‖)；髋部点（11/12）
全部不可见时（人只露出上半身）退化为 肩宽 ‖5-6‖（单位与 landmarks 一致，
都是归一化坐标）。归一化后 movement 的量纲是"尺度（躯干高度或肩宽）的比例"：
0.05 表示平均移动了 5%，这样阈值对摄像头距离、人体远近不敏感，可以跨场景复用。

为什么不用"相邻帧之间的平均欧氏距离"（最初想法）：
    - 慢速连续位移：每一帧之间的差都很小，人会"悄悄地移动"，相邻帧差却始终很小；
    - 原地抖动：pose 的噪声让点在原地来回抖，累计路径长度却会很大。
    两种情况下相邻帧差都会把"静止"和"在动"搞反。
相对窗口中位位置算中位位移则两种都能正确区分：抖动是对称噪声，中位位移停在噪声底；
连续漂移会让中位位移变大。

本模块是纯标准库（math/time），不 import cv2 / mediapipe / pose_estimation，
可以独立测试。

接口约定（保持稳定，别改签名）：
    FeatureExtractor(window_seconds=3.0, min_frames=5, min_window_seconds=1.0,
                     visibility_min=0.3, min_valid_points=2)
    update(pose_result, timestamp=None) -> Optional[float]  归一化移动量
    reset()

    PostureFeatures(visibility_min=0.3)                     躯干姿态角度特征
    update(pose_result) -> Optional[dict]  {'head_neck_angle', 可选的
                'torso_angle'/'back_curvature'/'neck_compression'}
                （角度单位：度；neck_compression 为无单位比值）
        取点：整侧链路优先 —— 两侧都齐用中点，否则优先完整单侧链路，
              最后退回中点法；head_neck 只要耳+肩可算即返回，髋缺失时
              torso/back 两个 key 不出现（调用方显示 N/A）；neck_compression
              需双侧肩都在（算肩宽）才出现；耳或肩缺失返回 None（详见类内 docstring）
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, Optional, Tuple

# 躯干 4 个关键点（COCO 17 点标准编号，RTMPose 输出）
# 0 鼻, 1/2 眼, 3 左耳, 4 右耳, 5 左肩, 6 右肩, 7 左肘, 8 右肘,
# 9 左腕, 10 右腕, 11 左髋, 12 右髋, 13 左膝, 14 右膝, 15 左踝, 16 右踝
TAP_IDS = (5, 6, 11, 12)

# 髋部点（11/12）—— 髋部不可见时用肩宽兜底
HIP_IDS = (11, 12)

# 耳朵点（3 左耳 / 4 右耳）—— 坐姿角度特征用
EAR_IDS = (3, 4)

# 肩部点（5 左肩 / 6 右肩）—— 坐姿角度特征用
SHOULDER_IDS = (5, 6)

# 躯干姿态计算用到的全部关键点：耳 3/4 + 肩 5/6 + 髋 11/12
POSTURE_IDS = EAR_IDS + SHOULDER_IDS + HIP_IDS

# 腿部关键点（COCO 17 点）：13/14 膝, 15/16 踝。髋 11/12 属于躯干，不进腿特征。
LEG_IDS = (13, 14, 15, 16)
KNEE_IDS = (13, 14)
ANKLE_IDS = (15, 16)

# 一个点的二维坐标
Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class _WindowedMedianExtractor:
    """滑动窗口 + 中位统计的通用基类：只持有窗口状态与可复用原语。

    各子类的 update() 返回类型不同（躯干版 Optional[float]、腿版 dict），
    所以这里**不做模板方法**，只提供状态（window 参数 + _window deque）和
    原语（_trim / _window_ready / _median / _extract_valid / reset），
    子类各自实现 update()，避免复制窗口/中位逻辑。
    """

    def __init__(self,
                 window_seconds: float = 3.0,
                 min_frames: int = 5,
                 min_window_seconds: float = 1.0,
                 visibility_min: float = 0.3,
                 min_valid_points: int = 2) -> None:
        """
        参数:
            window_seconds:      滑动窗口时长（秒），窗口内的帧参与计算。
            min_frames:          窗口内至少要攒够这么多帧，否则数据不足。
            min_window_seconds:  窗口内首尾帧的最小时间跨度（秒），数据太稀疏
                                 时不乱下结论。
            visibility_min:      某关键点的 visibility 低于此值视为无效点。
            min_valid_points:    一帧里至少要有这么多有效点，才把该帧计入窗口。
        """
        self.window_seconds = window_seconds
        self.min_frames = min_frames
        self.min_window_seconds = min_window_seconds
        self.visibility_min = visibility_min
        self.min_valid_points = min_valid_points

        # 窗口里的每一帧：(timestamp, {点id: (x, y)}, 该帧归一化尺度)
        self._window: deque = deque()

    def reset(self) -> None:
        """清空窗口，重新开始。"""
        self._window.clear()

    def _trim(self, now: float) -> None:
        """按 window_seconds 裁剪窗口内超时的旧帧。"""
        cutoff = now - self.window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _window_ready(self) -> bool:
        """窗口是否已攒够帧数 + 时间跨度（首尾帧差 >= min_window_seconds）。"""
        if len(self._window) < self.min_frames:
            return False
        return self._window[-1][0] - self._window[0][0] >= self.min_window_seconds

    def _extract_valid(self, landmarks: list,
                       ids: Tuple[int, ...]) -> Dict[int, Point]:
        """从 landmarks 里提取 ids 中 visibility >= visibility_min 的点。

        返回 {点id: (x, y)}（归一化坐标），供子类做锚点/尺度计算。
        """
        valid: Dict[int, Point] = {}
        for pid in ids:
            if pid < len(landmarks):
                x, y, _z, vis = landmarks[pid]
                if vis >= self.visibility_min:
                    valid[pid] = (x, y)
        return valid

    @staticmethod
    def _median(values: list) -> float:
        """小数组的中位数（不依赖 statistics，也更快）。"""
        s = sorted(values)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2.0


class FeatureExtractor(_WindowedMedianExtractor):
    """从 pose 序列中提取"躯干移动量"特征。"""

    def __init__(self,
                 window_seconds: float = 3.0,
                 min_frames: int = 5,
                 min_window_seconds: float = 1.0,
                 visibility_min: float = 0.3,
                 min_valid_points: int = 2) -> None:
        """
        参数:
            window_seconds:      滑动窗口时长（秒），窗口内的帧参与计算。
            min_frames:          窗口内至少要攒够这么多帧，否则返回 None。
            min_window_seconds:  窗口内首尾帧的最小时间跨度（秒），同样是为了
                                 数据太稀疏时不乱下结论。
            visibility_min:      某关键点的 visibility 低于此值视为无效点。
            min_valid_points:    一帧里至少要有这么多有效点，才把该帧计入窗口。
        """
        super().__init__(window_seconds=window_seconds,
                         min_frames=min_frames,
                         min_window_seconds=min_window_seconds,
                         visibility_min=visibility_min,
                         min_valid_points=min_valid_points)

    def update(self,
               pose_result: Optional[dict],
               timestamp: Optional[float] = None) -> Optional[float]:
        """喂入一帧的 pose 结果，返回当前归一化移动量。

        参数:
            pose_result: pose_estimation.detect_pose() 的返回值，结构为
                         {'landmarks': [(x, y, z, visibility) x33],
                          'image_size': (w, h)}，未检测到人体时为 None。
            timestamp:   该帧的单调时间戳（秒）。默认用 time.monotonic()。
                         —— 视频回放时请传入模拟时间戳（如 frame_idx / fps），
                            保证窗口按"真实时间"裁剪，不受播放 FPS 影响。

        返回:
            float 归一化移动量（躯干高度的比例），
            None 表示"这一帧不足以判定"（无人 / 有效点太少 / 窗口数据不足）。
        """
        now = timestamp if timestamp is not None else time.monotonic()

        if pose_result is None:
            # 没检测到人：不入窗口、返回 None。窗口保留，不重置。
            return None

        landmarks = pose_result.get('landmarks')
        if not landmarks:
            return None

        # 1) 提取本帧有效的躯干点坐标
        valid = self._extract_valid(landmarks, TAP_IDS)

        # 2) 有效点太少 → 本帧作废，不入窗口
        if len(valid) < self.min_valid_points:
            return None

        # 3) 本帧归一化尺度：优先躯干高度，髋部不可见时用肩宽兜底
        scale = self._frame_scale(valid)
        if scale is None:
            return None

        # 4) 入窗口，并裁剪掉超时的旧帧
        self._window.append((now, valid, scale))
        self._trim(now)

        # 5) 数据量校验
        if not self._window_ready():
            return None

        return self._compute_normalized_movement()

    # ---------- 内部实现 ----------

    @staticmethod
    def _frame_scale(points: Dict[int, Point]) -> Optional[float]:
        """一帧的归一化尺度（长度单位）：优先躯干高度，髋部不可见时用肩宽兜底。

        躯干高度 = mean(‖5-11‖, ‖6-12‖)，只统计两点都有效的侧；
        髋部点（11/12）全部不可见时，退回 mean(‖5-6‖, ‖6-5‖) = ‖5-6‖。
        这是为了兼容"只露出上半身"的场景（人坐着/画面只框到胸口），
        否则髋部永远不可见 → 每帧都被丢弃 → 永远显示"无人/数据不足"。
        """
        lengths = []
        if 5 in points and 11 in points:
            lengths.append(_dist(points[5], points[11]))
        if 6 in points and 12 in points:
            lengths.append(_dist(points[6], points[12]))
        if not lengths:
            # 髋部不可见 → 肩宽兜底（需要同侧肩+髋成对时用躯干高，这里退化为单侧肩距）
            if 5 in points and 6 in points:
                return _dist(points[5], points[6])
            return None
        return sum(lengths) / len(lengths)

    def _compute_normalized_movement(self) -> Optional[float]:
        """基于窗口算归一化移动量（供窗口数据已足时调用）。"""
        # 窗口内每一帧的躯干锚点中心（肩中点与髋中点的中点）
        anchors: list = []
        for _ts, points, _scale in self._window:
            a = self._anchor_center(points)
            if a is not None:
                anchors.append(a)
        if not anchors:
            return None

        # 锚点的中位位置，及各帧对中位的中位位移
        median_x = self._median([x for x, _y in anchors])
        median_y = self._median([y for _x, y in anchors])
        displacements = [math.hypot(x - median_x, y - median_y)
                         for x, y in anchors]
        median_disp = self._median(displacements)

        # 中位归一化尺度（躯干高度或肩宽）
        scale = self._median_torso_scale()
        if scale is None or scale < 1e-4:
            return None  # 尺度过小（离摄像头太远/退化），不作判定

        return median_disp / scale

    @staticmethod
    def _anchor_center(points: Dict[int, Point]) -> Optional[Point]:
        """一帧的躯干锚点中心：肩中点与髋中点的中点。

        髋部点（11/12）不可见时（人只露出上半身/髋在画面外）退化为肩中点，
        保证只露出上半身也能算。五官（耳/眼/鼻）和四肢（肘/腕/膝/踝）不参与，
        所以这些部位的轻微动作不会带动锚点中心 —— 移动量只反映躯干整体移动。
        """
        shoulder_mid = None
        s = [points[pid] for pid in SHOULDER_IDS if pid in points]
        if s:
            shoulder_mid = ((s[0][0] + s[-1][0]) / 2.0,
                            (s[0][1] + s[-1][1]) / 2.0)

        hip_mid = None
        h = [points[pid] for pid in HIP_IDS if pid in points]
        if h:
            hip_mid = ((h[0][0] + h[-1][0]) / 2.0,
                       (h[0][1] + h[-1][1]) / 2.0)

        if shoulder_mid is not None and hip_mid is not None:
            return ((shoulder_mid[0] + hip_mid[0]) / 2.0,
                    (shoulder_mid[1] + hip_mid[1]) / 2.0)
        if shoulder_mid is not None:
            return shoulder_mid
        if hip_mid is not None:
            return hip_mid
        return None

    def _median_torso_scale(self) -> Optional[float]:
        """窗口内各帧归一化尺度的中位数（优先躯干高度，髋部不可见时为肩宽）。"""
        scales = [scale for _ts, _points, scale in self._window]
        if not scales:
            return None
        return self._median(scales)


# ---------------------------------------------------------------------------
# PostureFeatures：坐姿角度特征（新增，纯 stdlib，不依赖 cv2/mediapipe）
# ---------------------------------------------------------------------------

# 三个角度的"定义说明"（英文，供 output 层画到画面左下角）。
# 定义跟着特征走，输出层只引用渲染，不在 output/main 里重复写语义。
# 唯一真实来源是 PostureFeatures.update() 返回的三个角度。
ANGLE_LEGEND = [
    "Ear-shoulder-hip 3-point chain:",
    "  torso   = shoulder->hip  vs vertical",
    "  neck    = ear->shoulder  vs vertical",
    "  back    = 180 - fold angle at shoulder",
    "  head    = ear-shoulder vertical gap / shoulder width",
    "           (smaller = head dropped / hunched)",
]

class PostureFeatures:
    """从单帧 pose 里计算三个坐姿角度 + 一个颈压缩比值，供坐姿识别/展示。

    三个角度（都用图片平面里的归一化坐标 (x, y) 计算，量纲无关，跨摄像头可复用）：
        torso_angle      躯干倾角：肩中→髋中向量 与竖直轴 (0,1) 的夹角。
                         0 = 身体竖直，越大越前倾/倾斜。依赖髋点。
        head_neck_angle  头颈角：耳→肩中向量 与竖直轴 (0,1) 的夹角。
                         0 = 头在肩正上方，越大头越前伸。只需耳+肩，不依赖髋。
        back_curvature   背部弯曲度：耳-肩-髋 三点折线在肩处的内角相对 180°
                         的偏折（180 − 内角）。0 = 耳肩髋基本共线（背直），
                         越大弓背越明显（低头/驼背时折线向内折叠）。依赖髋点。
        neck_compression 颈压缩：耳-肩**竖直间距** ÷ 肩宽（比值，无单位）。
                         0 = 耳与肩同高（头完全压到肩上），越大 = 头越在肩
                         正上方。只需耳+肩+双侧肩（算肩宽），不依赖髋。

    为什么有 neck_compression：正面摄像头下"耸肩+低头"式前弓背在图像平面里
    几乎不改变 x/y —— torso/back/head_neck 三个角度在正面投影都贴 0（见本模块
    开头 "局限" 段），这是现有角度抓不到前弓背的原因。弓背时唯一明显的正面
    信号是耳-肩竖直间距被压缩（肩耸起、头下沉），neck_compression 就是量化
    这个压缩的：它不依赖髋，坐着髋被桌子挡掉时依然可算 —— 恰恰是"坐着弓背"
    最需要的场景。默认阈值等起始值在 decision.py 标定（见 CLAUDE.md）。

    依赖髋点的 torso_angle / back_curvature 在髋不可见（人只露出上半身、
    髋在画面外）时算不出：update() 返回的 dict 不含这两个 key；缺失的 key
    表示 N/A（调用方显示 N/A，不要当 0）。head_neck_angle 只需耳+肩，
    neck_compression 只需耳+肩+双侧肩，髋缺失时依然能算并显示。

    取点策略（整侧链路优先）：
        两侧耳/肩/髋都可见 → 用两侧中点（最稳定，正对摄像头默认路径）；
        只有一侧三点齐 → 只用那一侧的单侧链路（人侧对摄像头时后景髋点
            常被遮挡，混用两侧中点会把角度算歪）；
        都没有完整链路但点数够 → 退回中点法（耳取可见度最高的那一只 +
            肩/髋取中点，耳部可见度门控沿用本模块风格）。
        单侧链路时无双侧肩，肩宽算不出 → neck_compression 缺失（N/A）。

    局限：正面摄像头在图像平面里主要反映左右倾斜和前伸，侧向弓背/低头只能
    部分体现；neck_compression 补上了"耸肩+低头"式前弓背，但仍有残留盲区
    （纯前倾、侧身弓背）。这些特征作为特征先落地、显示，供后续换视角更好的
    摄像头或做坐姿分类时使用。

    接口约定（保持稳定，别改签名）：
        PostureFeatures(visibility_min=0.3)
        update(pose_result: Optional[dict]) -> Optional[dict]
            返回 {'head_neck_angle':°, 可选的 'neck_compression'（比值）、
                  髋可见时另含 'torso_angle':° / 'back_curvature':°}，
                  torso/back 缺失=髋不可见、neck_compression 缺失=单侧链路
                  无肩宽（N/A）；耳或肩缺失返回 None。
    """

    def __init__(self, visibility_min: float = 0.3) -> None:
        """
        参数:
            visibility_min: 某关键点的 visibility 低于此值视为无效点。
        """
        self.visibility_min = visibility_min

    def update(self, pose_result: Optional[dict]) -> Optional[dict]:
        """喂入一帧 pose 结果，返回躯干姿态角度 + 颈压缩（度 / 比值）。

        参数:
            pose_result: pose_estimation.detect_pose() 的返回值，结构为
                         {'landmarks': [(x, y, z, visibility) x33],
                          'image_size': (w, h)}，未检测到人体时为 None。

        返回:
            dict 至少含 'head_neck_angle'（耳+肩可算即返回）；
            'neck_compression' 双侧肩都在时附加（单侧链路无肩宽则缺失）；
            髋可见时另附 'torso_angle' / 'back_curvature'（key 缺失表示该
            帧髋部不可见，调用方应显示 N/A 而不是把它当 0）；
            None 表示耳或肩缺失，任何角度都算不出（无人 / 数据不足）。
        """
        if pose_result is None:
            return None

        landmarks = pose_result.get('landmarks')
        if not landmarks:
            return None

        # 1) 按"整侧链路优先"选出耳/肩/髋三点 + 肩宽
        sel = self._select_points(landmarks)
        if sel is None:
            return None
        ear, shoulder, hip = sel['ear'], sel['shoulder'], sel['hip']

        # 2) 特征计算。head_neck 只依赖耳+肩；neck_compression 只依赖
        #    耳+肩+肩宽（不依赖髋）；torso/back 依赖髋。缺失的 key = N/A。
        result = {
            'head_neck_angle': self._angle_from_vertical(ear, shoulder),
        }
        nc = self._neck_compression(ear, shoulder, sel['shoulder_width'])
        if nc is not None:
            result['neck_compression'] = nc
        if hip is not None:
            result['torso_angle'] = self._angle_from_vertical(shoulder, hip)
            result['back_curvature'] = self._fold_angle(ear, shoulder, hip)
        return result

    # ---------- 内部实现 ----------

    def _select_points(self, landmarks: list) -> Optional[dict]:
        """从 landmarks 里提取可见的躯干姿态点，并决定用哪侧/哪个点。

        再按"整侧链路优先"选出参与角度计算的三点，同时给出肩宽
        （neck_compression 的归一化尺度，双侧肩都在才可算）。

        返回:
            {'ear': Point, 'shoulder': Point, 'hip': Point | None,
             'shoulder_width': float | None}
            或 None（数据不足）。
        """
        # 1) 提取本帧所有可见点（按 visibility_min 门控）
        pts = {}
        for pid in POSTURE_IDS:
            if pid < len(landmarks):
                x, y, _z, vis = landmarks[pid]
                if vis >= self.visibility_min:
                    pts[pid] = (x, y)
        if not pts:
            return None

        side = self._choose_chain_side(pts)
        if side is None:
            return None

        if side == "BOTH":
            ear = self._avg(pts, EAR_IDS)
            shoulder = self._avg(pts, SHOULDER_IDS)
            hip = self._avg(pts, HIP_IDS)
        elif side in ("L", "R"):
            if side == "L":
                ear, shoulder, hip = pts[3], pts[5], pts[11]
            else:
                ear, shoulder, hip = pts[4], pts[6], pts[12]
        else:  # "MID"
            ear = self._pick_best_ear(pts)
            shoulder = self._avg(pts, SHOULDER_IDS)
            hip = self._avg(pts, HIP_IDS)

        # ear / shoulder 必须可见；hip 允许缺失（人只露出上半身/髋在画面外时，
        # head_neck_angle / neck_compression 仍可算，只有 torso/back 无法算）。
        if ear is None or shoulder is None:
            return None

        # 肩宽：双侧肩（5/6）都在才可算；单侧链路（L/R）无肩宽 → None（N/A）。
        shoulder_width = None
        if 5 in pts and 6 in pts:
            shoulder_width = math.hypot(pts[6][0] - pts[5][0],
                                        pts[6][1] - pts[5][1])
        return {"ear": ear, "shoulder": shoulder, "hip": hip,
                "shoulder_width": shoulder_width}

    @staticmethod
    def _choose_chain_side(pts: dict) -> Optional[str]:
        """决定用"左右哪一侧"的耳-肩-髋链路算角度。

        返回:
            "BOTH"  两侧耳/肩/髋都齐，用两侧中点（最稳定）；
            "L"/"R" 只用这一侧完整的单侧三点链；
            "MID"   无完整链路但各点仍够算（正对摄像头、单侧略有遮挡），
                    退回"耳取可见度最高的一只 + 肩/髋取中点"；
            None    数据不够，无法计算。

        优先取整侧链路的动机：人侧对摄像头时，后景那一侧的髋点常被
        身体遮挡不可见，此时若混用两侧中点，角度会算歪。优先让
        "耳朵、肩膀、髋部都在同一侧"的链路参与计算。
        """
        def count(side: str) -> int:
            ids = (3, 5, 11) if side == "L" else (4, 6, 12)
            return sum(1 for pid in ids if pid in pts)

        c_left, c_right = count("L"), count("R")
        if c_left == 3 and c_right == 3:
            return "BOTH"
        if c_left == 3:
            return "L"
        if c_right == 3:
            return "R"
        return "MID" if pts else None

    @staticmethod
    def _avg(pts: dict, ids: Tuple[int, int]) -> Optional[Point]:
        """取 pts 中 ids 内所有可见点的中点（至少 1 个可见）。"""
        pts_l = [pts[pid] for pid in ids if pid in pts]
        if not pts_l:
            return None
        if len(pts_l) == 1:
            return pts_l[0]
        return ((pts_l[0][0] + pts_l[1][0]) / 2.0,
                (pts_l[0][1] + pts_l[1][1]) / 2.0)

    @staticmethod
    def _pick_best_ear(pts: dict) -> Optional[Point]:
        """耳只取可见的那一只（"MID" 分支专用）。"""
        for pid in EAR_IDS:
            if pid in pts:
                return pts[pid]
        return None

    @staticmethod
    def _angle_from_vertical(a: Point, b: Point) -> float:
        """向量 a→b 与竖直轴 (0,1) 的夹角（度，0~180）。"""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    @staticmethod
    def _fold_angle(a: Point, b: Point, c: Point) -> float:
        """耳(a)-肩(b)-髋(c) 折线在 b 处的内角相对 180° 的偏折（度）。

        0 = a/b/c 共线笔直；>0 越大弓背越明显。
        """
        ab = (a[0] - b[0], a[1] - b[1])
        cb = (c[0] - b[0], c[1] - b[1])
        # 内角（0~180 度）
        denom = math.hypot(*ab) * math.hypot(*cb)
        if denom < 1e-9:
            return 0.0
        cos = (ab[0] * cb[0] + ab[1] * cb[1]) / denom
        cos = max(-1.0, min(1.0, cos))  # 数值容差
        angle = math.degrees(math.acos(cos))
        return 180.0 - angle

    @staticmethod
    def _neck_compression(ear: Point, shoulder: Point,
                          shoulder_width: Optional[float]) -> Optional[float]:
        """颈压缩：耳-肩**竖直间距** ÷ 肩宽（无单位比值，越小越弓背）。

        竖直间距 = |shoulder.y − ear.y|（归一化坐标 y 向下；耳在肩上方时
        即"头高出肩多少"）。用肩宽归一化 = 以本人身体尺度为参照，正面摄像头
        下对"耸肩+低头"式前弓背敏感，且不依赖髋（坐着髋被桌子挡也能算）。
        肩宽算不出（单侧链路）→ 返回 None（= N/A）。
        间距为 0（耳与肩同高 = 头完全压到肩上）→ 返回 0.0（最大压缩，
        decision 层按超限处理，不会除零）。
        """
        if shoulder_width is None or shoulder_width < 1e-9:
            return None
        gap = abs(shoulder[1] - ear[1])
        return gap / shoulder_width


# ---------------------------------------------------------------------------
# 自测：纯 stdlib 合成数据，验证各特征计算逻辑（main_demo.py --selftest 汇总调用）
# ---------------------------------------------------------------------------

def _make_pose(pts: dict) -> dict:
    """构造一帧合成 pose：pts 里的点可见，其余点 visibility=0。

    pts: {关键点编号: (x, y)}，x/y 为归一化坐标 0~1。
    返回结构与 pose_estimation.detect_pose() 一致（17 点，z/visibility 补齐）。
    """
    landmarks = []
    for pid in range(17):  # COCO 17 点（RTMPose）
        if pid in pts:
            x, y = pts[pid]
            landmarks.append((x, y, 0.0, 1.0))
        else:
            landmarks.append((0.0, 0.0, 0.0, 0.0))
    return {"landmarks": landmarks, "image_size": (640, 480)}


def _assert_close(actual: float, expected: float, eps: float, msg: str) -> None:
    if abs(actual - expected) > eps:
        raise AssertionError(f"{msg}: 期望 {expected:.4f}，实际 {actual:.4f}")


def selftest_movement() -> None:
    """合成数据自测 FeatureExtractor：静止≈0、明显位移>0.10、五官/四肢轻动仍≈0、
    无人→None、髋缺失兜底。"""
    # 躯干 4 点：肩 (0.4,0.3)/(0.6,0.3)，髋 (0.4,0.7)/(0.6,0.7) → 躯干高 0.4
    base = {5: (0.4, 0.3), 6: (0.6, 0.3), 11: (0.4, 0.7), 12: (0.6, 0.7)}

    # 1) 静止：每帧同一位置 → 移动量 ≈ 0
    fe = FeatureExtractor(window_seconds=5.0, min_frames=3, min_window_seconds=0.1)
    for i in range(5):
        fe.update(_make_pose(base), timestamp=float(i) * 0.1)
    m = fe.update(_make_pose(base), timestamp=0.5)
    assert m is not None, "静止场景不应返回 None"
    _assert_close(m, 0.0, 1e-6, "静止移动量应为 0")

    # 2) 五官/四肢轻动：耳/肘/腕小幅动，肩髋不动 → 移动量 ≈ 0（静坐期间轻微动作不算在动）
    fe = FeatureExtractor(window_seconds=5.0, min_frames=3, min_window_seconds=0.1)
    for i in range(5):
        shake = 0.02 * (1 if i % 2 == 0 else -1)  # 耳/肘/腕在肩髋之间小幅度抖
        moved = {**base,
                 3: (0.5 + shake, 0.2), 4: (0.5 + shake, 0.2),
                 7: (0.35 + shake, 0.45), 8: (0.65 + shake, 0.45),
                 9: (0.3 + shake, 0.55), 10: (0.7 + shake, 0.55)}
        fe.update(_make_pose(moved), timestamp=float(i) * 0.1)
    m = fe.update(_make_pose({**base,
                              3: (0.52, 0.2), 4: (0.52, 0.2),
                              7: (0.37, 0.45), 8: (0.67, 0.45),
                              9: (0.32, 0.55), 10: (0.72, 0.55)}),
                  timestamp=0.5)
    assert m is not None, "五官/四肢轻动场景不应返回 None"
    assert m < 0.02, f"五官/四肢轻动时移动量应很小（<0.02），实际 {m:.4f}"

    # 3) 明显位移：5 帧沿 x 匀速挪 0.2 → 移动量显著 > 0.10
    fe = FeatureExtractor(window_seconds=5.0, min_frames=3, min_window_seconds=0.1)
    for i in range(5):
        shift = 0.05 * i
        moved = {pid: (x + shift, y) for pid, (x, y) in base.items()}
        fe.update(_make_pose(moved), timestamp=float(i) * 0.1)
    m = fe.update(_make_pose({pid: (x + 0.2, y) for pid, (x, y) in base.items()}),
                  timestamp=0.5)
    assert m is not None, "移动场景不应返回 None"
    assert m > 0.10, f"明显位移应 > 0.10，实际 {m:.4f}"

    # 4) 无人 → None
    assert fe.update(None, timestamp=0.6) is None

    # 5) 窗口数据不足：首帧直接返回 None
    fe = FeatureExtractor(window_seconds=5.0, min_frames=3, min_window_seconds=0.1)
    assert fe.update(_make_pose(base), timestamp=0.0) is None

    # 6) 髋部不可见（只露出上半身）→ 肩中点兜底仍能算
    torso_upper = {5: (0.4, 0.3), 6: (0.6, 0.3)}
    fe = FeatureExtractor(window_seconds=5.0, min_frames=3, min_window_seconds=0.1)
    for i in range(5):
        fe.update(_make_pose(torso_upper), timestamp=float(i) * 0.1)
    m = fe.update(_make_pose(torso_upper), timestamp=0.5)
    assert m is not None, "髋不可见、肩中点兜底时不应返回 None"
    _assert_close(m, 0.0, 1e-6, "上半身静止移动量应为 0")

    print("  selftest_movement: OK")


def selftest_posture() -> None:
    """合成数据自测 PostureFeatures：竖直≈0、前伸>0、髋缺失→N/A、点缺失→None、
    颈压缩对弓背敏感（竖直 0.75 / 弓背 0.35）、单侧链路无肩宽→N/A。"""
    post = PostureFeatures()

    # 1) 竖直站立：耳/肩/髋同竖线 → 三个角都 ≈ 0；颈压缩 = 0.75（肩宽 0.2）
    upright = {3: (0.45, 0.2), 4: (0.55, 0.2),
               5: (0.4, 0.35), 6: (0.6, 0.35),
               11: (0.45, 0.8), 12: (0.55, 0.8)}
    r = post.update(_make_pose(upright))
    assert r is not None, "竖直场景不应返回 None"
    for key in ("head_neck_angle", "torso_angle", "back_curvature"):
        _assert_close(r[key], 0.0, 1e-6, f"{key} 竖直应为 0")
    _assert_close(r["neck_compression"], 0.75, 1e-6, "竖直颈压缩应为 0.75")

    # 2) 头前伸：耳向右平 0.15（dx=0.15, dy=0.15）→ head_neck ≈ 45°
    lean = {3: (0.65, 0.2), 4: (0.65, 0.2),
            5: (0.5, 0.35), 6: (0.5, 0.35),
            11: (0.5, 0.8), 12: (0.5, 0.8)}
    r = post.update(_make_pose(lean))
    assert r is not None
    _assert_close(r["head_neck_angle"], 45.0, 0.5, "head_neck 应为 45°")

    # 3) 单侧链路：只有左耳+左肩+左髋 → 仍能算，竖直 → 0；
    #    无双侧肩（无肩宽）→ neck_compression 缺失（= N/A）
    left_only = {3: (0.5, 0.2), 5: (0.5, 0.35), 11: (0.5, 0.8)}
    r = post.update(_make_pose(left_only))
    assert r is not None, "单侧链路不应返回 None"
    for key in ("head_neck_angle", "torso_angle", "back_curvature"):
        _assert_close(r[key], 0.0, 1e-6, f"单侧竖直 {key} 应为 0")
    assert 'neck_compression' not in r, \
        "单侧链路无双侧肩（无肩宽），neck_compression 应为 N/A（缺失）"

    # 4) 髋不可见：只剩耳+肩 → head_neck 仍可算、torso/back 缺失（= N/A）；
    #    但双侧肩在 → neck_compression 可算（正面弓背信号，不依赖髋）。
    #    注：髋不可见时走到 "MID" 分支，耳取单只（_pick_best_ear），两耳故意
    #    居中放置避免单耳的横向偏移干扰 head_neck 断言（颈压缩只看竖直间距，
    #    不受影响）。
    upper = {3: (0.5, 0.2), 4: (0.5, 0.2),
             5: (0.4, 0.35), 6: (0.6, 0.35)}
    r = post.update(_make_pose(upper))
    assert r is not None, "髋不可见时 head_neck 仍应可算"
    assert set(r.keys()) == {"head_neck_angle", "neck_compression"}, \
        f"髋不可见时应只有 head_neck+neck_compression，实际 {sorted(r.keys())}"
    _assert_close(r["head_neck_angle"], 0.0, 1e-6, "上半身竖直 head_neck 应为 0")
    _assert_close(r["neck_compression"], 0.75, 1e-6, "上半身竖直颈压缩应为 0.75")

    # 4b) 弓背（耸肩+低头，髋不可见）：头下沉、肩微抬 → 颈压缩变小
    #     （竖直 0.75 → 弓背 0.35），低于 decision 默认阈值 0.45
    hunched = {3: (0.5, 0.30), 4: (0.5, 0.30),
               5: (0.4, 0.37), 6: (0.6, 0.37)}
    r = post.update(_make_pose(hunched))
    assert r is not None
    _assert_close(r["neck_compression"], 0.35, 1e-6, "弓背颈压缩应约 0.35")
    assert r["neck_compression"] < 0.45, \
        f"弓背颈压缩应低于 decision 默认阈值 0.45，实际 {r['neck_compression']:.3f}"

    # 5) 耳或肩缺失（只剩髋）→ None
    assert post.update(_make_pose({11: (0.5, 0.8), 12: (0.5, 0.8)})) is None

    # 6) 无人 → None
    assert post.update(None) is None

    print("  selftest_posture: OK")


if __name__ == "__main__":
    selftest_movement()
    selftest_posture()
    print("features selftest: ALL PASSED")
