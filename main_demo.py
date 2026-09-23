"""
侧面（45° 前侧）入口：把 capture → pose → features → decision → output 串成一条链路跑起来。
明显正对摄像头时改用另一个入口 main_front_demo.py（见文件末尾）。

用法（在 rtm_demo/ 目录下运行）：
    python main_demo.py --camera 0                         # 默认 20s 静坐 / 10s 弓背提醒（测试用短阈值）
    python main_demo.py --camera 0 --debug                 # 摄像头，debug 画走势图
    python main_demo.py --camera 0 --duration-limit 1200 --posture-duration-limit 300  # 生产阈值
    python main_demo.py --no-skeleton                      # 不画骨架（只想看数字/省 CPU）
    python main_demo.py --perf-log                         # 性能采集：每 30 帧写一行 CPU/RSS/推理耗时到 CSV
    python main_demo.py --dump-schema                      # 每帧导出人体侧规范 schema JSON（对接机器人用）
    python main_demo.py --no-show-schema                   # 不叠加 27 关节读数面板（默认叠加，测试用）
    python main_demo.py --cva-severe-threshold 40          # 标定 CVA 分级阈值（默认 55/50/44）
    python main_demo.py --view 45 --fhp-csv fhp.csv        # 头前伸 CVA-like proxy：逐帧写 CSV
    python main_demo.py --view 45 --fhp-duration-limit 10  # proxy 判 FHP 连续 10s 提醒（测试用）
    python main_demo.py --selftest                          # 合成数据自测（不碰摄像头）

正面机位（只挂一行 Ear–Shoulder 实验读数，**只显示不提醒**；45° 那条线不装配）：
    python main_front_demo.py --camera 0
    python main_front_demo.py --front-fhp-normal-max 0.05   # 覆盖实验上界（重标后用）

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
                      ErgonomicRiskFeatures,
                      CvaProxyFeatures,
                      EarShoulderProxyFeatures,
                      selftest_movement, selftest_posture, selftest_ergonomic,
                      selftest_cva_proxy, selftest_ear_shoulder_proxy)
from decision import (StillnessDecision,          # noqa: E402
                      SedentaryAlert, selftest_sedentary,
                      PostureDecision, PostureAlert,
                      selftest_posture_decision, selftest_posture_alert,
                      CvaRisk, selftest_cva_risk,
                      FhpDecision, FhpAlert,
                      selftest_fhp_decision, selftest_fhp_alert,
                      CVA_PROXY_VIEW_PRESETS,
                      FrontFhpDecision, FRONT_PROXY_NORMAL_MAX,
                      selftest_front_fhp_decision)
from output import (FrameRenderer, export_pose_json,  # noqa: E402
                    FhpCsvLogger, selftest_fhp_output,
                    selftest_front_fhp_line)
from perf_logger import PerfLogger                # noqa: E402
from pose_schema import human_adapter, selftest_schema  # noqa: E402


def _side_view(value: str) -> str:
    """侧面入口的 --view：只接受 "45"。

    正面机位是**另一个入口**（main_front_demo.py），不是本入口的一个参数值 ——
    机位由入口文件声明（见 CLAUDE.md「机位路由」）。这里给 front 一句指路，
    比 argparse 默认的 "invalid choice" 好认。
    """
    if value == "45":
        return "45"
    if value == "front":
        raise argparse.ArgumentTypeError(
            "正面机位请改用 python main_front_demo.py（本入口只跑 45° 前侧）")
    raise argparse.ArgumentTypeError(f"未知机位 {value!r}（本入口只接受 45）")


def build_parser(front: bool = False) -> argparse.ArgumentParser:
    """构造参数表。

    front=False（默认）：侧面（45° 前侧）入口，注册 45° 那条线的一整组参数。
    front=True        ：正面入口，只注册 Ear–Shoulder 实验显示行的参数；
                        45° 专属参数一个不注册（不装那条线，注册了就是死参数），
                        连 --view 都没有 —— 正面没有机位可选。
    """
    p = argparse.ArgumentParser(
        description=("人体姿态演示：正面入口（Ear–Shoulder 实验显示行，只显示不提醒）"
                     if front else
                     "人体静止检测演示（RTMPose，45° 前侧入口）"))

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
                   help="不把姿态骨架（关键点+连线+pid/置信度标签）叠到画面上")
    p.set_defaults(draw_skeleton=True)
    p.add_argument("--perf-log", nargs="?", const="performance_log.csv",
                   default=None, metavar="PATH",
                   help="性能采集：每 30 帧写一行 进程CPU/RSS/相对时间戳/该帧推理耗时"
                        " 到 CSV（默认 performance_log.csv，可指定路径；需已装 psutil）")
    p.add_argument("--dump-schema", nargs="?", const="pose_schema_dump.json",
                   default=None, metavar="PATH",
                   help="每帧把人体侧规范 schema（pose_schema.human_adapter 输出）"
                        " 导出为 JSON（默认 pose_schema_dump.json，可指定路径），"
                        " 供对接机器人/肉眼对比两边关节命名")
    p.add_argument("--no-show-schema", dest="show_schema", action="store_false",
                   help="不在画面上叠加规范关节读数面板（servo/名称/位置/角度/状态）")
    p.add_argument("--show-all-joints", dest="upper_only", action="store_false",
                   help="读数面板显示全部 27 个 DOF（默认只显示上半身 15 个；"
                        "腿/踝在坐姿画面里基本看不到，默认不列）")
    p.set_defaults(show_schema=True, upper_only=True)
    p.add_argument("--cva-normal-threshold", type=float, default=55.0,
                   help="CVA 正常阈值（度，>= 此值判正常）。默认 55，参考 Mostafaee "
                        "2022 观察性分组，非临床诊断标准；本系统用肩点近似 C7，"
                        "按实拍标定")
    p.add_argument("--cva-mild-threshold", type=float, default=50.0,
                   help="CVA 轻度阈值（度，>= 此值判轻度头前伸），默认 50")
    p.add_argument("--cva-severe-threshold", type=float, default=44.0,
                   help="CVA 中重度阈值（度，>= 此值判中重度、更低判重度），默认 44")
    # ---- 头前伸：两条线互斥，由**入口**决定装哪条（见 CLAUDE.md「机位路由」） ----
    if front:
        # 正面入口：只注册正面 Ear–Shoulder 实验显示行的参数。
        # （原先的 --no-front-es-proxy 已删：45° 线一走，"关掉这行"得到的画面
        #   就等于直接跑侧面入口，这个开关没有独立价值。）
        p.add_argument("--front-fhp-normal-max", type=float, default=None,
                       help="正面实验指标（Ear–Shoulder 水平位移/肩宽）的 Normal 上界，"
                            "超过判 FHP。默认 0.065，是 1 人 × 3 次的 **in-sample 种子值**，"
                            "不可跨次沿用（同人同机位两轮坐直基线 0.067↔0.027）；"
                            "换人/换椅子/换距离/换取景都要用 test_front_ear_shoulder_proxy.py "
                            "重标后用本参数覆盖")
    else:
        # 侧面入口：45° 前侧那条线（CVA-like proxy；proxy ≠ 临床 CVA）
        p.add_argument("--view", type=_side_view, default="45",
                       help="摄像头机位。本入口只跑 45° 前侧 —— 正面机位是另一个"
                            "入口（main_front_demo.py），不是这里的一个参数值")
        p.add_argument("--cva-proxy-slight-threshold", type=float, default=None,
                       help="proxy 轻微头前伸阈值（度，<= 此值判 Slight）。默认取 "
                            "45° 前侧的实验预设 68.17，标定后可用本参数覆盖")
        p.add_argument("--cva-proxy-obvious-threshold", type=float, default=None,
                       help="proxy 明显头前伸阈值（度，<= 此值判 Obvious）。默认取 "
                            "45° 前侧的实验预设 61.42")
        p.add_argument("--fhp-weight-proxy", type=float, default=1.0,
                       help="头前伸合成时 proxy 一路的权重，默认 1.0。"
                            "A/B/C 对比：A=旧判据(--fhp-weight-proxy 0)、"
                            "B=只 proxy(--fhp-weight-head 0)、C=合成(默认 1/1)")
        p.add_argument("--fhp-weight-head", type=float, default=1.0,
                       help="头前伸合成时原有头颈角一路的权重，默认 1.0（见上）")
        p.add_argument("--fhp-hysteresis-ratio", type=float, default=None,
                       help="proxy 恢复判 Normal 的迟滞比，默认 0.98（≈1.4~1.7° 带宽）。"
                            "别用坐姿规则的 0.8：proxy 恢复点会跑到 85° 以上、坐直也"
                            "回不到 Normal（详见 decision.CVA_PROXY_HYSTERESIS_RATIO）")
        p.add_argument("--fhp-duration-limit", type=float, default=10.0,
                       help="proxy 判头前伸连续多少秒触发提醒，默认 10（测试用短阈值；"
                            "生产建议 300，与坐姿提醒口径一致）")
        p.add_argument("--fhp-csv", nargs="?", const="fhp_frames.csv", default=None,
                       metavar="PATH",
                       help="逐帧记录头前伸读数到 CSV（默认 fhp_frames.csv，可指定路径）："
                            "timestamp/cva_proxy/original_fhp_score/combined_score/"
                            "fhp_state/keypoint_confidence 等，供离线分析")
    p.add_argument("--no-fhp-geometry", dest="fhp_geometry", action="store_false",
                   help=("不画 Ear–Shoulder 位移的几何辅助线"
                         "（耳中点/肩中点的水平位移 + 竖直连接 + 位移标注）"
                         if front else
                         "不画 proxy 的几何辅助线（耳中点→C7 代理点连线 + 水平参考虚线）"))
    p.set_defaults(fhp_geometry=True)
    p.add_argument("--selftest", action="store_true",
                   help="用合成数据自测各层，不打开摄像头")
    return p


def run_pipeline(args: argparse.Namespace, front: bool = False) -> None:
    """真实链路主循环（单进程，不拆线程）。

    front=False：45° 前侧入口 —— 装头前伸 CVA-like proxy 那条线（阈值预设 +
                  三档判定 + 提醒 + 逐帧 CSV），也就是今天原有的全部行为。
    front=True ：正面入口 —— 只装 Ear–Shoulder 实验显示行；45° 那条线**一个对象
                  都不建**（不更新、不画、不写 CSV、不提醒），因为它的正面预设
                  实测不可用（坐直段仍误报 ~12%）。

    分支用**显式参数**而不是 args.view：argparse 不校验 default 是否在 choices 里，
    默认值漏改一处就会静默走错分支；而 --selftest 完全不碰 parser 与 run_pipeline，
    没有任何自动安全网会拦住它。
    """
    import cv2

    # parser 与入口必须一致：侧面入口注册 --view，正面入口不注册
    assert hasattr(args, "view") == (not front), \
        "build_parser(front=…) 与 run_pipeline(front=…) 不一致"

    # 数据源：本地摄像头
    src = CameraCapture(args.camera if args.camera is not None else 0,
                        width=1280, height=720)

    feats = FeatureExtractor(window_seconds=args.window_seconds)
    post = PostureFeatures()
    ergo = ErgonomicRiskFeatures()   # CVA/FSA 姿态风险指标
    dec = StillnessDecision(still_threshold=args.still_threshold,
                            moving_threshold=args.moving_threshold)
    alert = SedentaryAlert(duration_limit_sec=args.duration_limit)
    pdec = PostureDecision(head_neck_threshold=args.head_neck_threshold,
                           torso_threshold=args.torso_threshold,
                           back_threshold=args.back_threshold,
                           neck_threshold=args.neck_threshold)
    palert = PostureAlert(duration_limit_sec=args.posture_duration_limit)
    cva_risk = CvaRisk(normal_threshold=args.cva_normal_threshold,
                       mild_threshold=args.cva_mild_threshold,
                       severe_threshold=args.cva_severe_threshold)
    # ---- 头前伸：两条线互斥，按**入口**装配（见 CLAUDE.md「机位路由」） ----
    # 用不到的那条线保持 None；主循环里靠 if front/else 守卫，不建任何多余对象。
    fhp_feats = fdec = falert = fhp_csv = None
    front_feats = front_dec = None
    if front:
        # 正面 Ear–Shoulder displacement proxy：**只显示不提醒**的实验行。
        # 它有自己的 features + decision 类，与 45° 那条线不共用阈值/状态，
        # 也不会产生任何提醒（FrontFhpDecision 没有配套 Alert）。
        front_normal_max = (FRONT_PROXY_NORMAL_MAX
                            if args.front_fhp_normal_max is None
                            else args.front_fhp_normal_max)
        front_feats = EarShoulderProxyFeatures()
        front_dec = FrontFhpDecision(normal_max=front_normal_max)
    else:
        # 45° 前侧那条线：features 出平滑角度，decision 出三档状态 + 提醒，
        # output 只画/只记。阈值取 45° 的实验预设，显式参数可覆盖。
        slight_t, obvious_t = CVA_PROXY_VIEW_PRESETS[args.view]
        if args.cva_proxy_slight_threshold is not None:
            slight_t = args.cva_proxy_slight_threshold
        if args.cva_proxy_obvious_threshold is not None:
            obvious_t = args.cva_proxy_obvious_threshold
        fhp_feats = CvaProxyFeatures()
        fdec = FhpDecision(slight_threshold=slight_t,
                           obvious_threshold=obvious_t,
                           head_neck_threshold=args.head_neck_threshold,
                           weight_proxy=args.fhp_weight_proxy,
                           weight_head=args.fhp_weight_head,
                           **({} if args.fhp_hysteresis_ratio is None else
                              {"hysteresis_ratio": args.fhp_hysteresis_ratio}))
        falert = FhpAlert(duration_limit_sec=args.fhp_duration_limit)
        fhp_csv = FhpCsvLogger(args.fhp_csv) if args.fhp_csv else None
    ren = FrameRenderer(still_threshold=args.still_threshold,
                        moving_threshold=args.moving_threshold,
                        debug=args.debug,
                        reminder_hold_sec=args.reminder_hold,
                        draw_skeleton=args.draw_skeleton,
                        duration_limit_sec=args.duration_limit,
                        posture_duration_limit_sec=args.posture_duration_limit)

    print("正在初始化 pose 检测器")
    if front:
        print(f"[front] 正面 Ear–Shoulder proxy 实验行开启：Normal 上界 "
              f"{front_dec.normal_max:.4f}（< 上界 = Normal，>= = FHP），"
              f"**只显示不提醒**、不参与任何判定")
        print("[front] 该上界是 1 人 × 3 次的 in-sample 种子值（坐直基线跨轮 "
              "0.067↔0.027），换条件必须重标："
              "test_front_ear_shoulder_proxy.py --view front --person x "
              "--session y --analyze-only a.csv b.csv")
        print("[front] 45° 那条线（CVA-like proxy / FHP 提醒 / FHP CSV）在本入口"
              "不装配；要跑 45° 前侧请用 python main_demo.py")
    else:
        print(f"[fhp] view={args.view}  阈值 slight<={slight_t:.2f}° "
              f"obvious<={obvious_t:.2f}°  权重 proxy={args.fhp_weight_proxy:g} "
              f"head={args.fhp_weight_head:g}  提醒 {args.fhp_duration_limit:.0f}s")
        print(f"[fhp] 迟滞比 {fdec.hysteresis_ratio:g}"
              f"（恢复点 proxy >= {slight_t / fdec.hysteresis_ratio:.2f}°）")
        print("[fhp] 阈值是本项目实拍自标的实验值（1 人×2 机位×3 重复），"
              "proxy 用肩点近似 C7、非临床 CVA；换机位/换人必须重标定")
        if (args.fhp_weight_proxy, args.fhp_weight_head) == (0.0, 1.0):
            print("[fhp] A 组：只用原有头颈角判据（proxy 只记录不参与判断）")
        elif (args.fhp_weight_proxy, args.fhp_weight_head) == (1.0, 0.0):
            print("[fhp] B 组：只用 CVA-like proxy 判据")
        else:
            print("[fhp] C 组：proxy + 原有头颈角 取大合成")
        if fhp_csv is not None:
            print(f"[fhp] 逐帧 CSV: {args.fhp_csv}")
    dbg_cnt = 0  # 临时标定用计数器（标定完删除）
    frame_idx = 0  # 性能采集用的帧序号
    perf = PerfLogger(args.perf_log) if args.perf_log else None
    # 每帧的头前伸读数：两条线各写各的，用不到的那条恒为 None
    fhp = fhp_state = fhp_reminder = None
    esp = front_state = None
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

                # 对接机器人测试：--dump-schema 导出 JSON / 画面默认叠加 27 关节读数面板
                # （--no-show-schema 关闭）。
                # human_adapter 是纯数学转换（27 个 DOF 循环），每帧一次足够。
                schema = None
                if args.dump_schema or args.show_schema:
                    schema = human_adapter(pose)
                    if args.dump_schema:
                        export_pose_json(schema, args.dump_schema)

                movement = feats.update(pose, ts)
                state = dec.update(movement)
                posture = post.update(pose)
                posture_state = pdec.update(posture)
                # CVA/FSA 姿态风险指标：features 只算平滑值，decision 只读 CVA 分级
                # （FSA 是辅助指标，不参与任何判断，仅 output 附注显示）
                erg = ergo.update(pose, ts)
                cva_level = cva_risk.update(erg['cva_deg'])
                # 头前伸：两条线各写自己那两个变量，另一条恒为 None（见装配处）
                if front:
                    # 正面 Ear–Shoulder proxy：features 出平滑位移比值，
                    # decision 出两档状态；**只挂一行读数，不产生提醒**
                    esp = front_feats.update(pose, ts)
                    front_state = front_dec.update(esp)
                else:
                    # 45° proxy：features 出平滑角度（+原始值），decision 出三档
                    # 状态；proxy 与原有头颈角按权重取大合成（默认 C 组）
                    fhp = fhp_feats.update(pose, ts)
                    fhp_state = fdec.update(fhp, posture)
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
                if not front:
                    # 45° 的 FHP 提醒：正面入口那条线不装配，这里连对象都没有
                    fhp_reminder = falert.update(fhp_state, ts)
                    if fhp_reminder is not None:
                        print(f"[fhp] {fhp_reminder}")
                ren.draw(frame, state, movement, posture, reminder, pose,
                         posture_state, posture_reminder,
                         still_elapsed_sec=alert.elapsed_sec,
                         slump_elapsed_sec=palert.elapsed_sec,
                         # 坐姿状态行下方再挂一行头前伸摘要（同帧只呈现，不参与
                         # 坐姿判定 —— posture 与 FHP 是两条独立判据）。
                         # 正面入口里 fhp_state 恒为 None，这一行自然不画
                         # （output 层只在 fhp_state 非 None 时画），不必再分支
                         fhp_state=fhp_state,
                         cva_proxy_deg=(fhp or {}).get('cva_proxy_deg'))
                if args.show_schema and schema is not None:
                    ren.draw_schema(frame, schema, upper_only=args.upper_only)
                ren.draw_cva_overlay(frame, erg, cva_level)
                if front:
                    # 正面实验行（只呈现、不提醒、不参与判定）
                    ren.draw_front_fhp_overlay(frame, esp, front_state, front_dec,
                                               show_geometry=args.fhp_geometry)
                else:
                    ren.draw_fhp_overlay(frame, fhp, fhp_state, fdec,
                                         alert_elapsed_sec=falert.elapsed_sec,
                                         duration_limit_sec=args.fhp_duration_limit,
                                         fhp_reminder=fhp_reminder,
                                         show_geometry=args.fhp_geometry)
                if fhp_csv is not None:
                    fhp_csv.log(ts, fhp, fhp_state, fdec, frame_index=frame_idx)
                ren.log(state, movement, posture_state)

                cv2.imshow("Stillness Demo (RTMPose) - front" if front
                           else "Stillness Demo (RTMPose)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        if perf is not None:
            perf.close()  # 正常退出 / ESC / Ctrl+C 都确保 CSV 落盘完整
        if fhp_csv is not None:
            fhp_csv.close()

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
    print("===== 5b. features: CVA/FSA ergonomic risk =====")
    selftest_ergonomic()
    print("===== 5c. decision: CVA grading =====")
    selftest_cva_risk()
    print("===== 5d. features: CVA-like proxy (head forward) =====")
    selftest_cva_proxy()
    print("===== 5e. decision: head-forward grading (proxy) =====")
    selftest_fhp_decision()
    print("===== 5f. decision: head-forward alert =====")
    selftest_fhp_alert()
    print("===== 5g. output: per-frame FHP CSV =====")
    selftest_fhp_output()
    print("===== 5h. features: Ear-Shoulder displacement proxy (front view, experimental) =====")
    selftest_ear_shoulder_proxy()
    print("===== 5i. decision: front-view two-tier FHP (display only) =====")
    selftest_front_fhp_decision()
    print("===== 5j. output: front-view experiment line =====")
    selftest_front_fhp_line()
    print("===== 6. pose_schema: human/robot adapter + JSON export =====")
    selftest_schema()
    print("\nALL SELFTESTS PASSED.")


def main() -> None:
    """侧面（45° 前侧）入口。正面入口见 main_front_demo.py。"""
    args = build_parser(front=False).parse_args()
    if args.selftest:
        selftest()   # 各层自测与机位无关，两个入口共用
        return
    run_pipeline(args, front=False)


if __name__ == "__main__":
    main()
