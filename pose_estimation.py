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


def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR 帧 → RTMPose 输入张量 (1, 3, 256, 192)，归一化到 ImageNet 统计。

    RTMPose 训练用的是 ImageNet 统计的 RGB 归一化（pipeline.json 里 to_rgb: true）。
    归一化是对每个通道独立做 (x - mean) / std，通道混不混 RGB 只影响通道顺序。
    为保险按训练配置走 RGB：BGR 先翻转成 RGB，再按 RGB 通道的 mean/std 归一化。
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - _MEAN) / _STD  # RGB 通道顺序
    # HWC -> CHW，加 batch 维
    img = img.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(img)


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

    # 整体置信度 = 各点置信度均值。低于阈值视为"未检测到人"。
    overall_score = float(np.mean(scores))
    if overall_score < DETECT_SCORE_THRESHOLD:
        return None, overall_score

    # 归一化到原图坐标：先还原到输入图，再按缩放比映射回原图
    in_w, in_h = INPUT_SIZE
    h, w = frame_bgr.shape[:2]
    scale_x, scale_y = w / in_w, h / in_h

    landmarks = []
    for i in range(17):
        x = coords[i, 0] * scale_x / w
        y = coords[i, 1] * scale_y / h
        landmarks.append((float(x), float(y), 0.0, float(scores[i])))

    return {
        'landmarks': landmarks,
        'image_size': (w, h),
    }, overall_score


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
