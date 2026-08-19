"""
性能采集（perf_logger 层，桌面版 --perf-log 用）：轻量外设，不影响核心链路。

只在 main_demo.py 装配层挂钩子：每 interval_frames 帧（默认 30 ≈ 1s@30fps）采一次
当前进程的 CPU 占用 / RSS 内存 / 相对时间戳，并记录该帧的 pose 推理耗时
（pose_ms 由 main_demo.py 用 time.monotonic 包 detect_pose 测得，本模块不碰推理）。
写入 CSV（performance_log.csv），独立于 output.py 的渲染/日志逻辑。

为什么独立成模块而不是塞进 output.py / main_demo.py：和现有"一文件一职责"的分层
一致；output.py 是桌面渲染+控制台日志，性能采集是另一个外围职责，混进去会污染；
main_demo.py 只做装配（构造 + 钩子 + finally 收尾），采集/落盘逻辑收敛在这里。

崩溃安全：每行写后 flush，中途异常时已采样的行不丢；close() 幂等；模块内注册
atexit 兜底未捕获的退出（含 Ctrl+C）。

psutil 未安装时构造不抛错，打印警告并置 disabled（--perf-log 静默失效），
避免因为少一个外设依赖把 demo 本身搞崩。

接口约定（保持稳定）：
    PerfLogger(path="performance_log.csv", interval_frames=30)
    should_sample(frame_idx) -> bool   本帧是否该采样（frame_idx 从 0 起）
    sample(frame_idx, pose_ms=None)    写一行；未启用/已关闭时静默 no-op
    close()                            幂等，flush + 关闭
"""

from __future__ import annotations

import atexit
import csv
import time

_PSUTIL_MISSING_HINT = (
    "perf_logger: 未安装 psutil，--perf-log 已忽略。"
    "安装：pip install psutil（已加入 requirements.txt）"
)


class PerfLogger:
    """轻量性能采样器：定时写一行 进程CPU/RSS/时间戳/推理耗时 到 CSV。

    参数:
        path:            CSV 输出路径（默认 performance_log.csv，每次运行覆盖）。
        interval_frames: 每隔多少帧采一次样（默认 30，约 1 秒 @30fps）。
                         cpu_percent() 无参调用返回"自上次调用以来的平均 CPU%"，
                         每 interval_frames 采一次正好得到约一个采样周期的均值。
    """

    def __init__(self, path: str = "performance_log.csv",
                 interval_frames: int = 30) -> None:
        self.path = path
        self.interval_frames = max(1, int(interval_frames))
        self._t0 = time.monotonic()
        self._closed = False

        try:
            import psutil
            self._proc = psutil.Process()
        except ImportError:
            print(_PSUTIL_MISSING_HINT)
            self._proc = None

        if self._proc is None:
            self._file = None
            return

        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["t_mono", "frame", "cpu_percent", "rss_mb", "pose_ms"])
        # 首次调用 cpu_percent 固定返回 0.0，仅用来"起算"，不写行
        self._proc.cpu_percent()
        atexit.register(self.close)

    def should_sample(self, frame_idx: int) -> bool:
        """本帧是否该采样：避开冷启动（frame 0），按 interval 取模。"""
        return self._file is not None and frame_idx >= self.interval_frames \
            and frame_idx % self.interval_frames == 0

    def sample(self, frame_idx: int, pose_ms: float | None = None) -> None:
        """采一行并落盘（每行 flush）。未启用/已关闭时静默 no-op。

        参数:
            frame_idx: 当前帧序号（与 CSV 的 frame 列一致，便于对齐）。
            pose_ms:   该帧 pose 推理耗时（毫秒），由装配层测得；None 则留空。
        """
        if self._file is None or self._closed:
            return
        cpu = self._proc.cpu_percent()                             # 自上次采样以来的平均 CPU%
        rss = self._proc.memory_info().rss / (1024 * 1024)         # MB
        t = time.monotonic() - self._t0
        self._writer.writerow([
            f"{t:.3f}", frame_idx,
            f"{cpu:.1f}", f"{rss:.1f}",
            "" if pose_ms is None else f"{pose_ms:.1f}",
        ])
        self._file.flush()

    def close(self) -> None:
        """幂等收尾：flush + 关闭文件（正常退出 / ESC / Ctrl+C 都调用）。"""
        if self._file is not None and not self._closed:
            try:
                self._file.flush()
                self._file.close()
            finally:
                self._closed = True
