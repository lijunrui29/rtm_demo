"""
capture 层：只负责拿到一帧一帧的图像，不关心图像来自哪里。

上层（pose estimation 等）只依赖 FrameSource 的三个方法：
    open()    打开图像源，返回是否成功
    read()    读取一帧，返回 BGR 帧；结束/出错返回 None
    release() 释放图像源

当前只有 CameraCapture（本地摄像头）。以后要换成机器人摄像头，
只需新增一个 FrameSource 子类（如 RTCameraCapture），上层代码不用改。

约定：read() 返回的帧是 BGR 顺序的 numpy 数组（OpenCV 默认格式）。
需要 RGB 的转换（如 mediapipe）留给上层自己做。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Optional

import cv2
import numpy as np

# 一帧图像的别名：BGR 的 numpy 数组
Frame = np.ndarray


class FrameSource(ABC):
    """图像源抽象基类：统一管理"打开/读取/释放"的生命周期。

    子类只需实现 _open/_read/_release 三个方法，
    生命周期状态和 with 语句支持由基类统一处理。
    """

    def __init__(self) -> None:
        self._cap = None          # 子类用它保存底层句柄（如 cv2.VideoCapture）
        self._is_opened = False

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    def open(self) -> bool:
        """打开图像源，返回是否成功。"""
        self.release()  # 重复调用 open 时先清理旧资源
        self._cap = self._open()
        self._is_opened = self._cap is not None
        return self._is_opened

    def read(self) -> Optional[Frame]:
        """读取一帧（BGR）。图像源结束或出错时返回 None。"""
        if not self._is_opened:
            return None
        return self._read()

    def release(self) -> None:
        """关闭图像源并释放资源。"""
        if self._cap is not None:
            self._release()
            self._cap = None
        self._is_opened = False

    def __enter__(self) -> "FrameSource":
        if not self.open():
            raise RuntimeError("图像源打开失败")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ---------- 以下由子类实现 ----------

    @abstractmethod
    def _open(self):
        """打开图像源，成功返回句柄对象，失败返回 None。"""

    @abstractmethod
    def _read(self) -> Optional[Frame]:
        """用 self._cap 读取一帧，返回 BGR 帧；结束/失败返回 None。"""

    @abstractmethod
    def _release(self) -> None:
        """释放 self._cap 指向的底层资源。"""


# Windows 摄像头后端依次尝试顺序：DSHOW → MSMF → 平台默认（None）。
# MSMF 常见"能打开但一直抓不到帧"（cap_msmf.cpp 报 -1072875772）的毛病，
# DSHOW 一般更稳；非 Windows 上 DSHOW/MSMF 打开会失败，自然落到平台默认。
_CAMERA_BACKENDS = (cv2.CAP_DSHOW, cv2.CAP_MSMF, None)
_OPEN_PROBE_RETRIES = 3    # 打开后实抓一帧验证，最多试几次（有的后端 opened() 为真但抓不到）
_READ_RETRIES = 5          # 读帧瞬时失败重试次数（避免一帧抽风把整条链路当"视频结束"）
_READ_RETRY_SLEEP = 0.05   # 每次重试前的等待（秒）


class CameraCapture(FrameSource):
    """本地摄像头图像源（目前用 OpenCV 的 cv2.VideoCapture）。"""

    def __init__(self, index: int = 0,
                 width: Optional[int] = None, height: Optional[int] = None):
        super().__init__()
        self.index = index
        self.width = width      # 可选的期望分辨率，摄像头不一定支持
        self.height = height

    def _open(self):
        for backend in _CAMERA_BACKENDS:
            try:
                cap = (cv2.VideoCapture(self.index, backend)
                       if backend is not None else cv2.VideoCapture(self.index))
            except Exception:
                continue
            if not cap.isOpened():
                cap.release()
                continue
            if self.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # 实抓一帧验证：有的后端能打开但一直读不到帧，那种直接弃用换下一个
            for _ in range(_OPEN_PROBE_RETRIES):
                ok, _ = cap.read()
                if ok:
                    print(f"[capture] 已打开本地摄像头 #{self.index} "
                          f"(backend={backend if backend is not None else 'default'})")
                    return cap
            cap.release()
        print(f"[capture] 打开摄像头失败：编号 {self.index} 不存在或被占用")
        return None

    def _read(self) -> Optional[Frame]:
        for _ in range(_READ_RETRIES):
            ret, frame = self._cap.read()
            if ret:
                return frame
            time.sleep(_READ_RETRY_SLEEP)
        return None

    def _release(self) -> None:
        self._cap.release()
        print("[capture] 摄像头已释放")



