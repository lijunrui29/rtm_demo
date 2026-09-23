# RTMPose 人体姿态 / 坐姿检测 Demo

基于 RTMPose（OpenMMLab）的人体姿态识别 Demo：检测人体关键点，实时判断「人是否静止」和「坐姿是否端正」，并按时长触发提醒。

> 本项目是**「后续用机器人检测人体姿势、判断后发出提醒」这一目标的简易 Demo 前置版**：先把「检测关键点 → 判姿态 → 按规则提醒」的整条链路在本机 / 网页上跑通，为机器人集成打基础。

- 桌面版：OpenCV 实时窗口，本机摄像头
- Web 版：浏览器打开页面，**用访问者自己的摄像头**（可用于公网演示）

## 功能

| 判定 | 说明 | 提醒 |
|---|---|---|
| 静止 / 在动 / 无人 | 跟踪躯干锚点中心的归一化移动量，迟滞双阈值 | 连续静止满时长触发一次久坐提醒 |
| 坐姿 / 弓背（GOOD / SLUMPED） | 耳-肩-髋三点角度（头颈角 / 躯干角 / 背曲角）超阈值判定 | 连续弓背满时长触发一次不良坐姿提醒 |
| CVA / FSA 姿态风险 | 颅椎角（耳-肩连线 vs 水平线）分级 + 前伸肩角辅助指标 | 无独立提醒（分级仅供显示；SEVERE 预警机制待做） |
| 头前伸（CVA-like proxy） | 耳中点→双肩中点连线与像素水平线的夹角，按机位分阈值的三档（Normal / Slight / Obvious） | 连续头前伸满时长触发一次提醒 |
| 正面 Ear–Shoulder 位移 proxy（实验） | **只在正面入口 `main_front_demo.py` 里**多挂两行：耳中点/肩中点水平位移 ÷ 肩宽，两档 Normal / FHP | **无提醒**（只显示、不参与任何判定；默认上界不可跨次沿用） |

画面上常显：三个角度的实时读数（`Torso / Neck / Back`）、CVA 分级（`CVA 56.3° NORMAL`，中重度及以上时附 `FSA ... aux`）、静止/弓背计时（`Still x.x/20s  Slump x.x/10s`）、状态标签、骨架（COCO 躯干+四肢绿色、耳点黄色圆点）。

## 环境要求

- Windows，Python 3.x
- 依赖：`opencv-python`、`numpy`、`onnxruntime`（模型推理用）
- 模型文件 `end2end.onnx`（rtmpose-s）：见下方「模型文件」

安装依赖（用你自己环境里的 Python）：

```bash
python -m pip install -r requirements.txt
```

下面命令里的 `python` 都指这个装好依赖的解释器。想用虚拟环境隔离，可先 `python -m venv .venv` 激活后再装。

## 模型文件

`models/`（含 `end2end.onnx`，约 21MB）**未提交到仓库**，clone 后需自行准备。

模型是 **RTMPose-s**（body7, 256×192）的 ONNX 导出，原始 PyTorch 权重（官方 mmpose 下载站）：

```
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth
```

`end2end.onnx` 由该权重经 **mmdeploy** 导出（导出得到 `deploy.json / pipeline.json / end2end.onnx` 三个文件，即 mmdeploy SDK 包格式），对应 mmpose 配置为 `projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py`。

准备好后把整个包解压到 `models/` 下任意位置即可——代码用 `rglob('end2end.onnx')` 自动查找，不依赖固定路径。

## 快速开始

在 `rtm_demo/` 目录下运行。命令里的 `python` 指装了依赖的解释器（见「环境要求」）。

### 桌面版

**两个入口，按机位选一个跑**（机位由入口文件声明，不是命令行开关）：

- `main_demo.py` —— **侧面 / 45° 前侧**（原有逻辑，一字未动）
- `main_front_demo.py` —— **明显正面**（只挂正面 Ear–Shoulder 实验读数，只显示、不提醒）

```bash
python main_demo.py --camera 0              # 侧面：摄像头默认 0；ESC 退出
python main_demo.py --camera 0 --debug      # 画面画移动量走势图（用于标定阈值）
python main_demo.py --camera 0 --duration-limit 1200 --posture-duration-limit 300  # 生产阈值（20分钟/5分钟）
python main_demo.py --no-skeleton           # 不画骨架
python main_demo.py --selftest              # 合成数据自测各层（不碰摄像头/模型）

python main_front_demo.py --camera 0        # 正面：只挂 Ear–Shoulder 实验读数（只显示、不提醒）
python main_front_demo.py --selftest        # 同一个自测（各层自测与机位无关）
```

在侧面入口敲 `--view front` 会提示你改用 `main_front_demo.py`；正面入口下 45° 那条线（proxy 阈值、FHP 提醒、FHP CSV）**一个对象都不建**。

### 头前伸标定（独立脚本，不影响 demo）

`test_fhp_cva_proxy.py` 只做「CVA-like proxy 头前伸」的采集与标定：按键标注姿势，实时显示 proxy 读数，退出时给出每种姿势的 mean/std/min/max 和 CSV。

```bash
python test_fhp_cva_proxy.py --camera 0                      # 1/2/3 标注 正常/轻度/明显，0 清除，s 出报告，q 退出
python test_fhp_cva_proxy.py --video sit.mp4 --analyze-only  # 回放视频/已有 CSV，不弹摄像头
python test_fhp_cva_proxy.py --view front                    # 正面机位（用正面那套阈值）
python test_fhp_cva_proxy.py --selftest                      # 合成数据自测（不碰摄像头/模型）
```

输出：CSV 默认写到当前目录 `fhp_cva_proxy.csv`（`--out` 改路径），字段为
`timestamp, cva_proxy, original_fhp_score, combined_score, fhp_state, keypoint_confidence`；
报告默认写到同名的 `fhp_cva_proxy_report.txt`（`--report` 改路径），给出分姿势统计、A/B/C 三种判据对比和建议阈值中点。

### 正面机位的另一条实验信号（独立脚本，同样不影响 demo）

**什么时候用它**：**只有摄像头明显正对着人时**，才用这条新的正面信号；摄像头是斜的（45° 前侧等）时，继续走上面那套 proxy + `FhpDecision`，原逻辑完全不变。机位由**人选用哪个入口文件**显式声明（`main_demo.py` = 45° 前侧 / `main_front_demo.py` = 正面）——本项目**没有**可靠的自动判机位（见「已知范围与限制」），别指望程序自己认。

正面机位下角度类指标会失效：正面把「头往前伸」投影成**深度方向**，耳-肩连线的水平投影只剩十几像素（实测 Δx 19/29/53 px，被 Δy≈280 px 压成 3.9°/6.5°/13.9°），落在阈值区间内分不开；而且髋被桌子挡住时躯干角/背曲角全是 N/A。所以改成量**图像平面里的相对水平位移**：

```
ear_mid_x     = (左耳x + 右耳x) / 2          shoulder_mid_x = (左肩x + 右肩x) / 2
delta_x       = |ear_mid_x − shoulder_mid_x|
shoulder_width = 左肩到右肩的距离
Ear–Shoulder displacement proxy = delta_x / shoulder_width     （像素空间，无单位，越大越前伸）
```

用到的关键点：COCO **3/4（耳）+ 5/6（肩）**（见 `features.py` 的 `EAR_IDS` / `SHOULDER_IDS`）。除以肩宽是为了抵消远近和体型（像素距离本身不能当指标）；分子分母都在**像素空间**量，避免归一化坐标把 x/y 按不同比例缩放。

> ⚠️ 它只是**图像平面里的相对水平位移**，**不是真实的 3D 前伸距离**，**不能叫 Forward Head Distance / clinical FHD**，也不是临床 CVA。

`test_front_ear_shoulder_proxy.py` 是这条信号的采集 / 测量验证脚本（`EarShoulderProxyFeatures` 在 `features.py` 里，**只算不判**）：

```bash
python test_front_ear_shoulder_proxy.py --camera 0                 # 1/2/3 标注 正常/轻度/明显，s 出报告，q 退出
python test_front_ear_shoulder_proxy.py --view front --camera 0    # 声明机位（默认 front；45 机位只作对照，不给实验阈值）
python test_front_ear_shoulder_proxy.py --video clip.mp4           # 回放视频
python test_front_ear_shoulder_proxy.py --analyze-only a.csv b.csv # 对已有 CSV 复算报告（不开摄像头）
python test_front_ear_shoulder_proxy.py --selftest                 # 合成数据自测（不碰摄像头/模型）
```

画面显示四行对照：`Head-Neck`（头颈角）/ `Ear-Shoulder Proxy (raw …)`（本指标 raw 与平滑值）/ `Current Posture`（现有坐姿判定）/ **`Experimental Front Posture (2-tier)`**（本实验指标的两档，标注为实验）。

CSV 默认 `front_ear_shoulder_proxy.csv`，逐帧记 `timestamp, ear_shoulder_proxy(raw), ear_shoulder_proxy(平滑 EMA), shoulder_width, shoulder_width_norm, ear_mid_x, shoulder_mid_x, 四个关键点置信度`（另有有效位/退化位/机位/人/次）。报告默认 `front_ear_shoulder_proxy_report.txt`，按「机位 × 人 × 次」分块给：**[1b] 两档口径（Normal vs 有前伸，判定就看这块）**、[1] 三档明细（诊断用）、两档/相邻档**是否重叠**、**段间重复性**（每次连续标注段算一次重复）、以及现有 `Posture` / `CVA proxy` 的对照读数。

**判定口径只做两档：Normal / 有前伸（FHP）**。实采（同一人同机位、每档 3 段重复）证明轻度与明显**分不开**：Slight 三段 0.0872 / 0.1033 / 0.0835 vs Obvious 0.0889 / 0.1156 / 0.1033 —— 跨段交错、信噪比 0.55，而且 head_neck 与 cva_proxy 也分不开同样这两档。所以细分档位放弃，键盘 `1/2/3` 仍保留作**诊断标签**（报告里三档明细照给），判定只认一条线：

```
proxy <  normal_max → Normal
proxy >= normal_max → FHP（边界归更前伸的一档）
```

阈值（`--front-normal-max`，默认 **0.065**）是**实验阈值**，来自离线回放的 in-sample 中点（Normal 均值 0.0401 与 Slight+Obvious 合并均值 0.0893 的中点），只为让「实验分档」那一行有东西可看：**不是判据、不是医学标准，正式判定不要用它**。脚本自身**只做测量验证、不拟合阈值**——数据重叠就如实记录，别为了好看改数字。

实测（走本脚本的分析路径；前三行是离线回放 `front.csv`/`f45.csv`，后两行是两轮真机实采 `front_p1_live1/2.csv`，都是 1 人。这些录像是**本地实采、未随仓库分发**，只作为下面数字的来源留档）：

| 数据 | Normal | Slight_FHP | Obvious_FHP | 两档（Normal vs 2+3） |
|---|---|---|---|---|
| front 离线回放（1 次） | 0.040 ± 0.010 | 0.061 ± 0.013 | 0.118 ± 0.009 | 单调 ✅ |
| 45° 离线回放（同一个量，作对照） | 0.268 ± 0.028 | 0.339 ± 0.025 | 0.306 ± 0.019 | **不单调 ❌** |
| front 实采第 1 轮 | 0.067 ± 0.006 | 0.072 ± 0.008 | 0.125（段被污染） | 重叠（信噪比 1.03） |
| front 实采第 2 轮 | 0.027 ± 0.024 | 0.092 ± 0.012 | 0.101 ± 0.022 | 重叠（信噪比 3.23） |

三条要点，都是如实记录、没有为了好看调过任何阈值：① 同一个量在 45° 机位**不单调**——两条线各用各的特征，不能互换；② **Slight 与 Obvious 分不开**（就是上面决定只做两档的理由）；③ **「坐直」的基线本身会漂移**：同一个人、同一摄像头、间隔十几分钟，两轮 Normal 从 0.067 掉到 0.027，而取景没变（归一化肩宽中位数 0.401 vs 0.405）——所以**绝对阈值不能跨次沿用**，样本也仍然只有 1 人，换人 / 换椅子 / 换距离都要重采。

#### 桌面版的正面入口也挂这一行（**只显示、不提醒**）

```bash
python main_front_demo.py --camera 0                                   # 一跑就挂上正面实验行
python main_front_demo.py --camera 0 --front-fhp-normal-max 0.05       # 覆盖上界（重标之后用）
python main_front_demo.py --camera 0 --no-fhp-geometry                 # 不画位移几何
```

正面入口的画面上多出两行（`Front FHP (exp): Normal/FHP  ES 0.041  raw 0.039` + `bound 0.065  in-sample  conf 0.87  display only, no alert`），位置与 45° 那条线的详情块**共用同一个槽**（y=214/236，两个入口互斥，同屏不会同时出现），用的是 `EarShoulderProxyFeatures` + `decision.FrontFhpDecision`（两档、无迟滞）——**它只被画出来，不触发提醒、不参与任何判定**；45° 机位那条线（阈值、迟滞、提醒）一行没动，而且**在正面入口里根本不装配**（正面预设实测不可用，见「已知范围与限制」）。上界默认 `decision.FRONT_PROXY_NORMAL_MAX = 0.065`，与脚本那个 `0.065` 同源（in-sample、不可跨次沿用）。两档用词刻意与 45° 线分开（正面 Normal/FHP，45° Normal/Slight/Obvious），免得被当成同一条判据。**正面入口没有 CSV 出口**——要留档只能用 `test_front_ear_shoulder_proxy.py`。

### Web 版（局域网：同一 WiFi 下的其他设备）

服务默认绑定 `0.0.0.0`，局域网内其他设备（手机 / 电脑）打开：

```
http://<你这台电脑的局域网IP>:8080
```

找局域网 IP：命令行运行 `ipconfig`，看「无线局域网适配器 WLAN → IPv4 地址」（形如 `192.168.x.x`）。

> ⚠️ **纯 HTTP 下非本机地址不是安全上下文，浏览器会拒绝摄像头**：局域网设备打开页面只能看到界面，用不了自己的摄像头。要让别人用自己的摄像头，必须走 HTTPS，见下一节「公网演示」。

### Web 版（公网演示：让别人用自己的摄像头）

浏览器 `getUserMedia` 要求 **HTTPS 或 localhost**，纯 HTTP 局域网地址会被浏览器拒绝摄像头授权。用 Cloudflare 快速隧道自动带 HTTPS：

1. **确保 Web 服务在跑**（`rtm_demo/` 目录下）：
   ```bash
   python web_demo.py --port 8080
   ```
2. **另开一个终端**，运行：
   ```bash
   cloudflared tunnel --url http://localhost:8080
   ```
3. 等几秒，输出里的 `https://xxxx.trycloudflare.com` 那一行就是给别人的 URL（每次重新启动都不同）。发出去前**先自己浏览器打开验证**，确认能出页面再发给对方；对方浏览器打开并授权摄像头即可（手机也一样）。

> ⚠️ 隧道 URL 没有鉴权，任何人拿到链接都能访问，只适合临时演示。公网延迟高时，为保持流畅，后端只返回「关键点 + 状态」的小 JSON，骨架由浏览器本地绘制——视频保持本地流畅，骨架跟随会有轻微延迟，属正常。

## 命令行参数（没标「专属」的两个入口都通用；各自的参数表由 `--help` 说了算）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--still-threshold` | 0.05 | 静止阈值：移动量 ≤ 此值判静止 |
| `--moving-threshold` | 0.10 | 在动阈值：移动量 ≥ 此值判在动 |
| `--window-seconds` | 3.0 | 移动量滑动窗口（秒） |
| `--duration-limit` | 20 | 连续静坐多少秒触发久坐提醒（测试短阈值；生产用 1200） |
| `--head-neck-threshold` | 30 | 头颈角阈值（度） |
| `--torso-threshold` | 30 | 躯干角阈值（度） |
| `--back-threshold` | 25 | 背曲角阈值（度） |
| `--neck-threshold` | 0.45 | 颈压缩阈值：耳-肩竖直间距/肩宽（无单位），**小于**此值判弓背（正面摄像头对前弓背敏感） |
| `--posture-duration-limit` | 10 | 连续弓背多少秒触发提醒（测试短阈值；生产用 300） |
| `--reminder-hold` | 8.0 | 提醒横幅停留秒数 |
| `--view` | 45 | **侧面入口专属**，只接受 `45`（给 `front` 会指路去 `main_front_demo.py`）。机位本身由**入口文件**声明：两套读数区间几乎不重叠，跨机位套阈值必错；且目前没有可靠的自动判机位（见「已知范围与限制」） |
| `--cva-proxy-slight-threshold` / `--cva-proxy-obvious-threshold` | 45° 的两档预设 | **侧面入口**：覆盖头前伸 proxy 的两档阈值（默认 45°:68.17/61.42），标定后用这两个参数覆盖。正面那两个预设（84.82/79.32）现在只被标定脚本读 |
| `--fhp-csv` | 关 | **侧面入口**：逐帧写头前伸读数 CSV（默认 `fhp_frames.csv`，列名见 `output.FHP_CSV_COLUMNS`），供离线分析 |
| `--fhp-duration-limit` | 10 | **侧面入口**：proxy 连续判头前伸多少秒触发提醒（测试短阈值；生产用 300） |
| `--fhp-hysteresis-ratio` | 0.98 | **侧面入口**：proxy 恢复 Normal 的迟滞比。**别用坐姿规则的 0.8**（恢复点会跑到 85° 以上，坐直也回不来），原因见 `decision.CVA_PROXY_HYSTERESIS_RATIO` |
| `--fhp-weight-proxy` / `--fhp-weight-head` | 1 / 1 | **侧面入口**：头前伸合成得分的两路权重，A/B/C 对比用（`0`/`1` = 只旧判据，`1`/`0` = 只 proxy，`1`/`1` = 合成） |
| `--front-fhp-normal-max` | 0.065 | **正面入口 `main_front_demo.py` 专属**：正面 Ear–Shoulder 实验行的 Normal 上界（超过判 FHP）。默认值是 1 人 × 3 次的 in-sample 种子，**不可跨次沿用**，换条件用测量脚本重标后覆盖 |
| `--no-fhp-geometry` | 开 | 不画几何辅助线（侧面入口 = proxy 的 C7 连线与水平参考线；正面入口 = Ear–Shoulder 位移几何） |
| `--debug` | 关 | 画移动量走势图 |
| `--no-skeleton` | 开 | 不画骨架 |
| `--show-all-joints` / `--no-show-schema` | 面板开、只上半身 15 个 | 规范关节读数面板（`pose_schema`）：默认只列上半身 15 个 DOF（坐姿画面里腿/踝基本看不到），`--show-all-joints` 列全部 27 个，`--no-show-schema` 关掉整块 |
| `--dump-schema` | 关 | 每帧把人体侧规范 schema 导出成 JSON（默认 `pose_schema_dump.json`），供对接机器人 / 核对关节命名 |
| `--perf-log` | 关 | 每 30 帧写一行进程 CPU / RSS / 推理耗时到 CSV（默认 `performance_log.csv`，需 psutil） |

## 阈值标定

坐姿阈值（角度默认 30 / 30 / 25 度 + 颈压缩默认 0.45）是**起始猜测**。画面左上角直接显示实时读数：Torso / Neck / Back 是角度（越大越弓背），Head 是颈压缩（耳-肩竖直间距/肩宽，**越小越弓背**，正面摄像头下前弓背主要靠它抓）：

1. 坐直保持几秒，记下四个读数；
2. 弓背保持几秒，记下读数；
3. 取「明显弓背」时的边界作为对应阈值：角度取下沿、Head 取上沿（颈压缩从 ~0.6 掉到 ~0.3 左右时，阈值可定在中间偏上，如 0.45）。

`--still-threshold`（默认 0.05）一般够用；若「明显没动却偶发跳 MOVING」，用 `--debug` 看移动量曲线，把它提到 0.06~0.07。模型「检测到人」的阈值是 `pose_estimation.py` 里的 `DETECT_SCORE_THRESHOLD`（默认 0.15）：人常检不出（离得远/光线差）可降到 ~0.12；太低会对空场景误检。

CVA 分级阈值（默认 `--cva-normal-threshold 55 / --cva-mild-threshold 50 / --cva-severe-threshold 44`）来自文献观察性分组，**本系统用肩点近似 C7、量值不可直接套用**，请按实拍标定：画面左侧 `CVA xx.x°` 行是平滑读数，坐直记下高值、头前伸记下低值，把边界定在两档之间。FSA 是辅助指标，仅在 CVA 中重度及以上时随附显示，无需单独标定。

## 项目结构（分层架构）

```
rtm_demo/
├── main_demo.py          # 桌面版【侧面 / 45° 前侧】入口：装配各层、跑主循环
├── main_front_demo.py    # 桌面版【正面】入口（薄壳）：只挂 Ear–Shoulder 实验行，45° 线不装配
├── web_demo.py           # Web 版入口：浏览器摄像头 → /analyze → 回关键点+状态 JSON
├── requirements.txt      # Python 依赖（pip install -r requirements.txt）
├── capture.py            # 图像输入：FrameSource / CameraCapture
├── pose_estimation.py    # RTMPose 推理：17 个 COCO 关键点
├── features.py           # 特征：移动量、坐姿角度、CVA proxy、Ear–Shoulder 位移 proxy（实验，只算不判）
├── decision.py           # 判断：静止/在动、坐姿/弓背、头前伸、计时提醒；正面两档 FrontFhpDecision（只供显示）
├── output.py             # 输出：画面绘制 + 控制台日志 + FHP CSV + 正面实验行（两个桌面入口各用一条）
├── pose_schema.py        # 姿态对齐：人体 COCO17 / 机器人 27 舵机 → 统一 27 DOF 规范 schema（--dump-schema 用，对接机器人）
├── perf_logger.py        # 性能采集（--perf-log）：每 30 帧写 CPU/RSS/推理耗时
├── test_fhp_cva_proxy.py # 头前伸 proxy 标定脚本（独立，不影响 demo）
├── test_front_ear_shoulder_proxy.py  # 正面机位 Ear–Shoulder 位移 proxy 的测量/验证脚本（独立；只测量验证，不拟合阈值）
└── models/               # RTMPose ONNX 模型
```

数据流：`capture → pose → features → decision → output`，各层职责单一、接口稳定。

关键点编号为 **COCO 17 点标准**：0 鼻、3/4 耳、5/6 肩、11/12 髋、9/10 腕等。

## 后续计划

- **提醒方案会扩展**：目前的提醒只是「连续静止 / 弓背**达到时长阈值就提示一次**」这种简单的时间判断；后续会在 `decision.py` 增加更多提示方案（例如结合坐姿程度、动作类型、持续时长等组合出不同的提醒内容与时机）。
- **机器人摄像头**：当前用的是本机 / 浏览器摄像头；后续接入机器人摄像头时，只需在 `capture.py` 新增一个图像源子类，上层判定逻辑不用改。
- **正面 Ear–Shoulder 实验信号：下一步仍只做验证**（显示行已接，接线到此为止）：同一条件重复 ≥3 次看段间极差（已做两轮）、**换摄像头距离**（肩宽归一化本该抵消远近，还没验）、换人 / 换椅子 / 换取景；每次用 `--person` / `--session` 标好条件，`--analyze-only` 多份并排比。要不要再加提醒或当判据，等这些都过了再说。

## 已知范围与限制

- **单 / 多人**：目前按「整帧单一人」处理（top-down 无检测器），适合人占满画面的坐姿场景
- **RULA / REBA 人体工学评估**：未接入，当前判定只基于现有规则
- **CVA / FSA 姿态风险指标**：颅椎角用肩点近似 C7、分级阈值参考 Mostafaee et al. 2022 观察性分组（**非临床诊断标准**），需按本机取景实拍标定。**免责声明：本系统输出的 CVA/FSA 角度指标反映体表姿态模式，不能替代医学影像诊断或临床评估。**
- **头前伸（CVA-like proxy）**：不是标准 CVA——标准 CVA 是矢状面侧拍的 C7–耳屏连线与水平线夹角，本项目的 proxy 是单目 RGB 下用 **肩中点近似 C7、耳中点近似耳屏**、在像素空间量角度，属工程近似，**不代表临床测量**。阈值按机位分开保存（45° / 正面），**跨机位套阈值必错**；且只在同一个人、同取景下成立，换位置要重标。目前样本是 1 人 × 2 机位 × 3 次，正面机位正常/轻度区间有重叠，只能当趋势不能当结论。**免责声明：proxy 输出反映的是表观姿态，不能替代医学影像诊断或临床评估。**
- **机位必须人工声明**：几何上最干净的自动判机位线索是「肩宽被投影压扁多少」（正对 ~0.56、侧身 0.01~0.21），但实测本机取景下**正面 0.365~0.377 与 45° 0.280~0.350 区间重叠**、耳距 266~319 px 与 253~296 px 也重叠，而「肩宽/耳-肩竖直间距」比值又随姿势变（正面 1.66→2.19）——**目前没有可靠的自动机位分类**。所以机位由人**选入口文件**声明（`main_demo.py` = 45° 前侧 / `main_front_demo.py` = 正面），程序只在画面窄肩时提醒（不做自动切换）。摄像头斜对 / 正对本来就是人自己知道的。
- **正面 Ear–Shoulder 位移 proxy（实验，两档口径）**：只做「图像平面里的相对水平位移 ÷ 肩宽」，**不是 3D 前伸距离、不是 clinical FHD**；髋不可见时分不清「头相对躯干前伸」和「整个人塌下去」；**判定只保留 Normal / 有前伸两档**（轻度与明显实测分不开），样本目前 1 人 × 3 次 × 2 机位，且「坐直」基线在两次采集之间就从 0.067 漂到 0.027 —— 换人 / 换椅子 / 换距离都要重采，绝对阈值不可跨次沿用。**正面入口 `main_front_demo.py` 会显示那两行读数（`FrontFhpDecision`，两档、无迟滞、`valid=False` 时显示 N/A），但它只显示不提醒、不参与任何判定；未接入 Web 版；45° 机位那条线未改动（在正面入口里干脆不装配）。**
- **模型**：使用 rtmpose-s（`models/` 内）
- **躯干角度**：髋不可见时躯干角 / 背曲角显示 N/A（头颈角始终可算）
