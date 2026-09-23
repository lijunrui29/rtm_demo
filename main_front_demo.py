"""
正面机位入口：只跑 Ear–Shoulder displacement proxy 的**实验显示行**。

和侧面入口 main_demo.py 的区别（机位由**入口文件**声明，不是命令行参数）：

| | main_demo.py（侧面 / 45° 前侧） | 本入口（明显正面） |
|---|---|---|
| 头前伸线 | CVA-like proxy 三档 + 提醒 + CSV | Ear–Shoulder 两档**只显示** |
| 画的东西 | 45° 读数块 + 几何 + Head Fwd 摘要行 | 正面实验行（两行） |
| 提醒 | 有（FhpAlert） | **无**（没有配套 Alert，不参与任何判定） |

**45° 那条线在本入口一个对象都不装** —— 它的正面预设实测不可用（坐直段仍误报
~12%，恢复点 86.55° 高于坐直中位数 86.28°，见 CLAUDE.md 开放问题②），
装上就是今天实机那次误报的来源。

用法（在 rtm_demo/ 目录下运行）：
    python main_front_demo.py --camera 0
    python main_front_demo.py --camera 0 --front-fhp-normal-max 0.05  # 重标后覆盖上界
    python main_front_demo.py --no-fhp-geometry                       # 不画位移几何
    python main_front_demo.py --selftest                              # 合成数据自测各层

⚠ 上界 0.065 是 1 人 × 3 次的 **in-sample 种子值**，不可跨次沿用（同人同机位两轮
"坐直"基线从 0.067 漂到 0.027）。换人/换椅子/换距离/换取景都要先用
test_front_ear_shoulder_proxy.py 重标。本指标是工程代理，不是临床 FHD。

ESC 退出。主循环与侧面入口共用 main_demo.run_pipeline，没有第二份。
"""

from main_demo import build_parser, run_pipeline, selftest


def main() -> None:
    """正面机位入口：装正面实验显示行，45° 那条线不装配。"""
    args = build_parser(front=True).parse_args()
    if args.selftest:
        selftest()   # 各层自测与机位无关，两个入口共用
        return
    run_pipeline(args, front=True)


if __name__ == "__main__":
    main()
