"""
主入口：把 capture → pose → features → decision → output 串成一条链路跑起来。

用法（在 rtm_demo/ 目录下运行）：
    python main_demo.py --camera 0                         # 默认 20s 静坐 / 10s 弓背提醒（测试用短阈值）
    python main_demo.py --camera 0 --debug                 # 摄像头，debug 画走势图
    python main_demo.py --camera 0 --duration-limit 1200 --posture-duration-limit 300  # 生产阈值
    python main_demo.py --no-skeleton                      # 不画骨架（只想看数字/省 CPU）
    python main_demo.py --perf-log                         # 性能采集：每 30 帧写一行 CPU/RSS/推理耗时到 CSV
    python main_demo.py --selftest                          # 合成数据自测（不碰摄像头/mediapipe）

ESC 退出。首帧较慢（惰性初始化 pose 检测器），正常。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 本目录（rtm_demo）内的模块直接 import。
# 注意：RTMPose 模型文件在 models/ 下，路径用绝对路径，不依赖工作目录。
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from capture import CameraCapture                 # noqa: E402
from pose_estimation import detect_pose           # noqa: E402
from features import (FeatureExtractor,           # noqa: E402
                      PostureFeatures,
                      selftest_movement, selftest_posture)
from decision import (StillnessDecision,          # noqa: E402
                      SedentaryAlert, selftest_sedentary,
                      PostureDecision, PostureAlert,
                      selftest_posture_decision, selftest_posture_alert)
from output import FrameRenderer                  # noqa: E402
from perf_logger import PerfLogger                # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="人体静止检测演示（RTMPose）")
    p.add_argument("--camera", type=int, metavar="INDEX",
                   help="本地摄像头编号（默认 0）")
    p.add_argument("--debug", action="store_true",
                   help="debug：画面上画移动量走势图，控制台详细打印")
    p.add_argument("--still-threshold", type=float, default=0.05,
                   help="静止阈值（移动量 <= 此值判静止），默认 0.05")
    p.add_argument("--moving-threshold", type=float, default=0.10,
                   help="在动阈值（移动量 >= 此值判在动），默认 0.10")
    p.add_argument("--window-seconds", type=float, default=3.0,
                   help="滑动窗口时长（秒），默认 3.0")
    p.add_argument("--duration-limit", type=float, default=20.0,
                   help="连续久坐多少秒触发提醒，默认 20（测试用短阈值；生产建议 1200）")
    p.add_argument("--head-neck-threshold", type=float, default=30.0,
                   help="头前伸角阈值（度），超过判不良坐姿，默认 30")
    p.add_argument("--torso-threshold", type=float, default=30.0,
                   help="躯干前倾角阈值（度），超过判不良坐姿，默认 30")
    p.add_argument("--back-threshold", type=float, default=25.0,
                   help="背部弯曲角阈值（度），超过判弓背，默认 25")
    p.add_argument("--neck-threshold", type=float, default=0.45,
                   help="颈压缩阈值（耳-肩竖直间距/肩宽，无单位比值；"
                        "小于此值判弓背，正面摄像头对前弓背敏感），"
                        "默认 0.45（起始猜测，--debug 看 Head 读数标定）")
    p.add_argument("--posture-duration-limit", type=float, default=10.0,
                   help="连续不良坐姿多少秒触发提醒，默认 10（测试用短阈值；生产建议 300）")
    p.add_argument("--reminder-hold", type=float, default=8.0,
                   help="提醒横幅在画面上停留秒数，默认 8")
    p.add_argument("--no-skeleton", dest="draw_skeleton", action="store_false",
                   help="不把姿态骨架（关键点+连线）叠到画面上")
    p.set_defaults(draw_skeleton=True)
    p.add_argument("--perf-log", nargs="?", const="performance_log.csv",
                   default=None, metavar="PATH",
                   help="性能采集：每 30 帧写一行 进程CPU/RSS/相对时间戳/该帧推理耗时"
                        " 到 CSV（默认 performance_log.csv，可指定路径；需已装 psutil）")
    p.add_argument("--selftest", action="store_true",
                   help="用合成数据自测各层，不打开摄像头")
    return p


def run_pipeline(args: argparse.Namespace) -> None:
    """真实链路主循环（单进程，不拆线程）。"""
    import cv2

    # 数据源：本地摄像头
    src = CameraCapture(args.camera if args.camera is not None else 0,
                        width=1280, height=720)

    feats = FeatureExtractor(window_seconds=args.window_seconds)
    post = PostureFeatures()
    dec = StillnessDecision(still_threshold=args.still_threshold,
                            moving_threshold=args.moving_threshold)
    alert = SedentaryAlert(duration_limit_sec=args.duration_limit)
    pdec = PostureDecision(head_neck_threshold=args.head_neck_threshold,
                           torso_threshold=args.torso_threshold,
                           back_threshold=args.back_threshold,
                           neck_threshold=args.neck_threshold)
    palert = PostureAlert(duration_limit_sec=args.posture_duration_limit)
    ren = FrameRenderer(still_threshold=args.still_threshold,
                        moving_threshold=args.moving_threshold,
                        debug=args.debug,
                        reminder_hold_sec=args.reminder_hold,
                        draw_skeleton=args.draw_skeleton,
                        duration_limit_sec=args.duration_limit,
                        posture_duration_limit_sec=args.posture_duration_limit)

    print("正在初始化 pose 检测器")
    dbg_cnt = 0  # 临时标定用计数器（标定完删除）
    frame_idx = 0  # 性能采集用的帧序号
    perf = PerfLogger(args.perf_log) if args.perf_log else None
    try:
        with src:
            while True:
                frame = src.read()
                if frame is None:
                    break

                ts = time.monotonic()

                t_pose = time.monotonic()
                pose = detect_pose(frame)
                pose_ms = (time.monotonic() - t_pose) * 1000.0
                if perf is not None and perf.should_sample(frame_idx):
                    perf.sample(frame_idx, pose_ms=pose_ms)
                frame_idx += 1

                movement = feats.update(pose, ts)
                state = dec.update(movement)
                posture = post.update(pose)
                posture_state = pdec.update(posture)
                # 临时标定：--debug 下每 30 帧（约 1s）打印一次坐姿特征（标定完删除）
                dbg_cnt += 1
                if args.debug and dbg_cnt % 30 == 0:
                    hn = posture.get('head_neck_angle') if posture else None
                    to = posture.get('torso_angle') if posture else None
                    bc = posture.get('back_curvature') if posture else None
                    nc = posture.get('neck_compression') if posture else None
                    a = lambda v: '--' if v is None else f"{v:.0f}"
                    b = lambda v: '--' if v is None else f"{v:.2f}"
                    print(f"[dbg] ang=(hn:{a(hn)} to:{a(to)} bc:{a(bc)} nc:{b(nc)}) "
                          f"post={posture_state.value}", flush=True)
                reminder = alert.update(state, ts)
                posture_reminder = palert.update(posture_state, ts)
                if reminder is not None:
                    print(f"[reminder] {reminder}")
                if posture_reminder is not None:
                    print(f"[posture] {posture_reminder}")
                ren.draw(frame, state, movement, posture, reminder, pose,
                         posture_state, posture_reminder,
                         still_elapsed_sec=alert.elapsed_sec,
                         slump_elapsed_sec=palert.elapsed_sec)
                ren.log(state, movement, posture_state)

                cv2.imshow("Stillness Demo (RTMPose)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        if perf is not None:
            perf.close()  # 正常退出 / ESC / Ctrl+C 都确保 CSV 落盘完整

    cv2.destroyAllWindows()


def selftest() -> None:
    """合成数据自测各层（逻辑在各层自己的 selftest_* 里，此处只汇总）。"""
    print("===== 1. features: movement/stillness =====")
    selftest_movement()
    print("===== 2. features: posture angles =====")
    selftest_posture()
    print("===== 3. decision: sedentary alert =====")
    selftest_sedentary()
    print("===== 4. decision: posture state =====")
    selftest_posture_decision()
    print("===== 5. decision: posture alert =====")
    selftest_posture_alert()
    print("\nALL SELFTESTS PASSED.")


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()
