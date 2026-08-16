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


class CameraCapture(FrameSource):
    """本地摄像头图像源（目前用 OpenCV 的 cv2.VideoCapture）。"""

    def __init__(self, index: int = 0,
                 width: Optional[int] = None, height: Optional[int] = None):
        super().__init__()
        self.index = index
        self.width = width      # 可选的期望分辨率，摄像头不一定支持
        self.height = height

    def _open(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            print(f"[capture] 打开摄像头失败：编号 {self.index} 不存在或被占用")
            return None
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print(f"[capture] 已打开本地摄像头 #{self.index}")
        return cap

    def _read(self) -> Optional[Frame]:
        ret, frame = self._cap.read()
        return frame if ret else None

    def _release(self) -> None:
        self._cap.release()
        print("[capture] 摄像头已释放")



