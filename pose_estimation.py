"""
Pose Estimation 核心模块：输入一帧图像，输出 17 个 COCO 关键点。

用 RTMPose（OpenMMLab）的 ONNX 模型做推理，替换原来的 MediaPipe。
只做检测，不做任何判断。返回归一化坐标，不包含绘制/规则逻辑。

模型文件（end2end.onnx）与本次件同目录（rtm_demo/models/ 下），用绝对路径，
避免依赖工作目录。运行前请先装 onnxruntime：
    pip install onnxruntime

RTMPose 是 top-down 模型，本来需要先框出人的包围盒再推理。本项目只检测
坐姿（人基本占满画面、只关心上半身），所以直接把整帧当作"一个人的包围盒"
推理，不需要单独的检测器 —— 这和 MediaPipe 版本的全图推理用法一致。

接口约定（保持稳定，别改签名，与 MediaPipe 版一致）：
    detect_pose(frame_bgr) -> dict | None
        返回 {'landmarks': [(x, y, z, visibility) x17], 'image_size': (w, h)}
        关键点编号是 COCO 17 点标准（不是 MediaPipe 33 点）：
            0 鼻, 1 左眼, 2 右眼, 3 左耳, 4 右耳,
            5 左肩, 6 右肩, 7 左肘, 8 右肘,
            9 左腕, 10 右腕, 11 左髋, 12 右髋,
            13 左膝, 14 右膝, 15 左踝, 16 右踝
        坐姿相关躯干点是 5/6/11/12，耳朵是 3/4。
    detect_pose_with_score(frame_bgr) -> (dict | None, float)
        新加的调试/标定接口：结果与 detect_pose 一致，额外返回本帧整体
        置信度分数（未检测到也返回），供 web 端显示来标定 DETECT_SCORE_THRESHOLD。
    landmarks_to_pixels(landmarks, image_size) -> [(x, y) x17]
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / 'models'
# 用 glob 找 onnx，模型目录（含 20230831/rtmpose_onnx/<name>/）一旦换版本也能找到
_ONNX_FILES = sorted(MODEL_DIR.rglob('end2end.onnx'))
if _ONNX_FILES:
    MODEL_PATH = str(_ONNX_FILES[0])
else:
    MODEL_PATH = str(MODEL_DIR / 'end2end.onnx')  # 占位，get_detector() 里报错

# RTMPose 输入尺寸（与 pipeline.json 的 image_size 一致，顺序 [w, h]）
INPUT_SIZE = (192, 256)

# 归一化参数（ImageNet 统计，RTMPose 训练用；顺序 BGR/RGB 见下）
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# "检测到人"的整体置信度阈值：17 个关键点 SIMCC 峰值（softmax 最大值）的平均。
# 2026-08-16 从 0.3 降到 0.15：坐姿场景下肢/踝部常被截断出画或离得远，
# 这些低分点把平均值拖到 0.3 以下 → 频繁判"未检测到人"。0.15 更易检出。
# 调太高会漏检，调太低会对空场景误检出随机关键点；真机标定可在此改。
DETECT_SCORE_THRESHOLD = 0.15

# 整帧失败时的"人像区域重试"（2026-08-18，实测标定，详见 detect_pose_with_score）：
# 坐着侧身/人偏小时，整帧推理 17 点一起掉到阈值下，但其实人还在画面里、
# 模型只是没锁住。此时用 置信度 >= ANCHOR_CONF 的点（或最近一次成功帧的
# 人像框，时间连续）裁剪放大重推，通常能把 mean 拉回阈值上。
ANCHOR_CONF = 0.2          # 视为"锚点"的最低置信度（实测坐着侧身失败帧最高点 ~0.22）
PERSON_BBOX_VALID_SEC = 1.0  # 上次成功帧人像框的有效期（人不动时沿用；走远/换人自动失效）

_last_person_bbox = None      # (x1, y1, x2, y2)，原图像素
_last_person_bbox_t = 0.0     # time.monotonic()

_sess = None


def get_detector():
    """单例：只初始化一次 onnxruntime 会话。"""
    global _sess
    if _sess is None:
        import onnxruntime as ort
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"找不到 RTMPose 模型：{MODEL_PATH}。请先下载 end2end.onnx"
                "（openmmlab 的 rtmpose-s_simcc-body7_pt-body7_420e-256x192 包），"
                "解压放到 models/ 目录下。")
        _sess = ort.InferenceSession(MODEL_PATH,
                                     providers=['CPUExecutionProvider'])
    return _sess


def _letterbox_dims(w: int, h: int) -> tuple:
    """等比缩放 + 灰边到 INPUT_SIZE 的变换参数。

    返回 (scale, pad_x, pad_y)：原图按 scale 等比缩放后放到 INPUT_SIZE 画布上，
    左上角起始于 (pad_x, pad_y)（输入图像素坐标）。整帧拉伸（16:9 → 3:4）会把
    人形压扁变形，letterbox 保留比例，实测对一般场景整体置信度明显更高
    （2026-08-18：正面帧 mean 0.258 → 0.456）。
    """
    scale = min(INPUT_SIZE[0] / w, INPUT_SIZE[1] / h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    pad_x = (INPUT_SIZE[0] - new_w) // 2
    pad_y = (INPUT_SIZE[1] - new_h) // 2
    return scale, pad_x, pad_y


def _normalize_to_tensor(img: np.ndarray) -> np.ndarray:
    """把已 resize 到 INPUT_SIZE 的 RGB 图归一化成 (1, 3, 256, 192) 张量。

    归一化用 ImageNet 统计（RTMPose 训练配置），通道顺序 RGB。
    """
    img = img.astype(np.float32)
    img = (img - _MEAN) / _STD  # RGB 通道顺序
    # HWC -> CHW，加 batch 维
    img = img.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(img)


def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR 帧 → RTMPose 输入张量 (1, 3, 256, 192)：letterbox 等比缩放 + 灰边。

    RTMPose 训练用的是 ImageNet 统计的 RGB 归一化（pipeline.json 里 to_rgb: true）。
    归一化是对每个通道独立做 (x - mean) / std，通道混不混 RGB 只影响通道顺序。
    为保险按训练配置走 RGB：BGR 先翻转成 RGB，再按 RGB 通道的 mean/std 归一化。
    坐标映射回原图时要用 _letterbox_dims 的 (scale, pad)，见 detect_pose_with_score。
    """
    h, w = frame_bgr.shape[:2]
    scale, pad_x, pad_y = _letterbox_dims(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    canvas = np.zeros((INPUT_SIZE[1], INPUT_SIZE[0], 3), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(
        rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return _normalize_to_tensor(canvas)


def _decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray) -> np.ndarray:
    """把 SIMCC 输出解码成关键点坐标（相对输入图 192x256 的像素坐标）。

    RTMPose 的 SIMCC head 把每个关键点的 x/y 各自编码成一条 softmax 分布
    （长度 = 输入边长 * simcc_split_ratio，这里是 384 / 512）。
    argmax 取峰值位置，再除以 simcc_split_ratio 还原成输入图的像素坐标。

    simcc_x/simcc_y: (batch, 17, 384) / (batch, 17, 512)
    返回: (17, 2) 的 float 数组，顺序 [x, y]（相对输入图，未还原到原图）。
    """
    split_ratio = 2.0  # 与 pipeline.json 的 simcc_split_ratio 一致
    x_bins = simcc_x[0].argmax(axis=1).astype(np.float32) / split_ratio
    y_bins = simcc_y[0].argmax(axis=1).astype(np.float32) / split_ratio
    return np.stack([x_bins, y_bins], axis=1)


def detect_pose(frame_bgr):
    """检测一帧图像中的人体姿态。

    参数:
        frame_bgr: OpenCV 读取的 BGR 图像 (H, W, 3)。

    返回:
        None —— 该帧未检测到人体（score 低于阈值 / 无人）。
        否则返回 dict:
            {
                'landmarks': 长度为 17 的列表，每个元素是 (x, y, z, visibility)
                             - x, y: 归一化坐标 0~1（相对图像宽高）
                             - z:     深度，恒为 0（RTMPose 不输出深度）
                             - visibility: 该点置信度 0~1（SIMCC 峰值即置信度）
                'image_size': (width, height)
            }
        关键点编号对应关系（COCO 17 点标准，不是 MediaPipe 33 点）：
            0 鼻, 1 左眼, 2 右眼, 3 左耳, 4 右耳,
            5 左肩, 6 右肩, 7 左肘, 8 右肘,
            9 左腕, 10 右腕, 11 左髋, 12 右髋,
            13 左膝, 14 右膝, 15 左踝, 16 右踝
    """
    result, _score = detect_pose_with_score(frame_bgr)
    return result


def detect_pose_with_score(frame_bgr):
    """同 detect_pose，额外返回本帧"整体置信度分数"（17 点平均，未检测到也返回）。

    这是新加的调试/标定接口（加能力走"加新方法"，detect_pose 签名保持不变）：
    web 端把分数显示到页面，看它贴着 DETECT_SCORE_THRESHOLD 有多近，据此判断
    阈值该往上还是往下调。

    整帧失败时的"人像区域重试"（2026-08-18）：
    坐着侧身/人偏小时，整帧推理会 17 点一起掉到阈值下（实测失败帧无一 >=0.3，
    但 79% 有 >=0.2 的锚点）。此时用 本帧锚点 或 最近成功帧的人像框 裁剪放大
    重推一次，若重推 mean 过阈就用重推结果（坐标已映射回原图）。正常帧零开销
    （只有失败/临界帧多跑一次推理）。人走远/彻底塌陷的重推也救不回来，仍回 None。

    返回:
        (result, overall_score)：result 与 detect_pose 的返回完全一致（dict | None）；
        overall_score 恒为 float（0~1，越小越像空场景）。未检测到时 result 为 None，
        但分数照样返回，方便看"差多少才够到阈值"。
    """
    sess = get_detector()

    input_img = _preprocess(frame_bgr)
    simcc_x, simcc_y = sess.run(None, {'input': input_img})

    # SIMCC 峰值即该点置信度（softmax 最大值，越接近 1 越可信）
    scores = simcc_x[0].max(axis=1).astype(np.float64)
    coords = _decode_simcc(simcc_x, simcc_y)  # (17, 2)，相对 192x256 输入图

    h, w = frame_bgr.shape[:2]
    scale, pad_x, pad_y = _letterbox_dims(w, h)
    whole = _make_result(h, w, scores, coords,
                         lambda x, y: ((x - pad_x) / scale, (y - pad_y) / scale))

    # 整体置信度 = 各点置信度均值。低于阈值视为"未检测到人"。
    overall_score = float(np.mean(scores))
    if overall_score >= DETECT_SCORE_THRESHOLD:
        _remember_person_bbox(whole)
        return whole, overall_score

    # 整帧失败：人像区域重试（锚点优先，其次最近成功帧的人像框）
    retry = _retry_person_crop(frame_bgr, scores, coords)
    if retry is not None:
        crop_result, crop_score = retry
        _remember_person_bbox(crop_result)
        return crop_result, crop_score
    return None, overall_score


def _make_result(h: int, w: int, scores: np.ndarray, coords: np.ndarray,
                 remap) -> dict:
    """把输入图坐标 (N,2) 用 remap(px, py) -> (原图像素x, y) 映射回原图，
    构建与 detect_pose 一致的结果 dict（归一化坐标 + 该点置信度）。
    """
    landmarks = []
    for i in range(17):
        px, py = remap(float(coords[i, 0]), float(coords[i, 1]))
        landmarks.append((px / w, py / h, 0.0, float(scores[i])))
    return {'landmarks': landmarks, 'image_size': (w, h)}


def _bbox_from_points(frame_bgr: np.ndarray, scores: np.ndarray,
                      coords: np.ndarray) -> tuple | None:
    """从 置信度 >= ANCHOR_CONF 的点 算人像包围盒（原图像素）。

    需要 >=2 个锚点、包围盒足够大，否则返回 None（整帧推理全塌，无从锚定）。
    边距：头顶多留 0.3 倍、左右/底部 0.15 倍（给头留空间）。
    """
    h, w = frame_bgr.shape[:2]
    scale, pad_x, pad_y = _letterbox_dims(w, h)
    xs, ys = [], []
    for i in range(17):
        if scores[i] >= ANCHOR_CONF:
            xs.append((coords[i, 0] - pad_x) / scale)
            ys.append((coords[i, 1] - pad_y) / scale)
    if len(xs) < 2:
        return None
    x1, x2 = max(0.0, min(xs)), min(float(w), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(h), max(ys))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0.0, x1 - bw * 0.15)
    x2 = min(float(w), x2 + bw * 0.15)
    y1 = max(0.0, y1 - bh * 0.30)
    y2 = min(float(h), y2 + bh * 0.15)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return (int(x1), int(y1), int(x2), int(y2))


def _retry_person_crop(frame_bgr: np.ndarray, scores: np.ndarray,
                       coords: np.ndarray):
    """整帧失败时的人像区域重试：返回 (crop_result, crop_overall) 或 None。

    1) 优先用本帧 置信度 >= ANCHOR_CONF 的锚点 框人像；锚点不足时
    2) 用最近一次成功帧缓存的人像框（PERSON_BBOX_VALID_SEC 内有效，
       人静止时位置基本不变，裁剪放大正好补上整帧看不出的信号）；
    3) 裁剪 → 拉伸到输入尺寸 → 重推；重推 mean 过阈才接受。
    """
    global _last_person_bbox, _last_person_bbox_t
    h, w = frame_bgr.shape[:2]
    now = time.monotonic()

    bbox = _bbox_from_points(frame_bgr, scores, coords)
    if bbox is None and _last_person_bbox is not None \
            and now - _last_person_bbox_t <= PERSON_BBOX_VALID_SEC:
        bbox = _last_person_bbox
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    crop = frame_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_tensor = _normalize_to_tensor(
        cv2.resize(crop_rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR))
    simcc_x, simcc_y = get_detector().run(None, {'input': crop_tensor})
    cscores = simcc_x[0].max(axis=1).astype(np.float64)
    ccoords = _decode_simcc(simcc_x, simcc_y)
    crop_overall = float(np.mean(cscores))
    if crop_overall < DETECT_SCORE_THRESHOLD:
        return None  # 裁剪也没救活（人走远/彻底塌陷）

    in_w, in_h = INPUT_SIZE
    result = _make_result(h, w, cscores, ccoords,
                          lambda px, py: (x1 + px * (bw / in_w),
                                          y1 + py * (bh / in_h)))
    return result, crop_overall


def _remember_person_bbox(result: dict) -> None:
    """把一次成功的检测结果里 置信度 >= ANCHOR_CONF 的点 的包围盒缓存下来，
    供后续失败帧的人像区域重试使用（时间连续，人没动）。
    """
    global _last_person_bbox, _last_person_bbox_t
    w, h = result['image_size']
    xs, ys = [], []
    for (x, y, _z, vis) in result['landmarks']:
        if vis >= ANCHOR_CONF:
            xs.append(x * w)
            ys.append(y * h)
    if len(xs) >= 2:
        _last_person_bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        _last_person_bbox_t = time.monotonic()


def landmarks_to_pixels(landmarks, image_size):
    """把归一化坐标转成像素坐标（纯几何换算，不做判断）。

    参数:
        landmarks: detect_pose 返回的 'landmarks'（归一化坐标列表）
        image_size: (width, height)

    返回:
        (x_px, y_px) 元组列表，长度与 landmarks 相同。
    """
    w, h = image_size
    return [(int(x * w), int(y * h)) for x, y, _z, _vis in landmarks]
