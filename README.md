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

画面上常显：三个角度的实时读数（`Torso / Neck / Back`）、静止/弓背计时（`Still x.x/20s  Slump x.x/10s`）、状态标签、骨架（COCO 躯干+四肢绿色、耳点黄色圆点）。

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

```bash
python main_demo.py --camera 0              # 摄像头默认 0；ESC 退出
python main_demo.py --camera 0 --debug      # 画面画移动量走势图（用于标定阈值）
python main_demo.py --camera 0 --duration-limit 1200 --posture-duration-limit 300  # 生产阈值（20分钟/5分钟）
python main_demo.py --no-skeleton           # 不画骨架
python main_demo.py --selftest              # 合成数据自测各层（不碰摄像头/模型）
```

### Web 版（本机体验）

```bash
python web_demo.py --port 8080
```

本机浏览器打开 `http://127.0.0.1:8080`，授权摄像头即可。`127.0.0.1` 是本机回环地址，**只有你这台电脑能访问**。

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

## 命令行参数（两个入口通用）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--still-threshold` | 0.05 | 静止阈值：移动量 ≤ 此值判静止 |
| `--moving-threshold` | 0.10 | 在动阈值：移动量 ≥ 此值判在动 |
| `--window-seconds` | 3.0 | 移动量滑动窗口（秒） |
| `--duration-limit` | 20 | 连续静坐多少秒触发久坐提醒（测试短阈值；生产用 1200） |
| `--head-neck-threshold` | 30 | 头颈角阈值（度） |
| `--torso-threshold` | 30 | 躯干角阈值（度） |
| `--back-threshold` | 25 | 背曲角阈值（度） |
| `--posture-duration-limit` | 10 | 连续弓背多少秒触发提醒（测试短阈值；生产用 300） |
| `--reminder-hold` | 8.0 | 提醒横幅停留秒数 |
| `--debug` | 关 | 画移动量走势图 |
| `--no-skeleton` | 开 | 不画骨架 |

## 阈值标定

坐姿三个角度阈值（默认 30 / 30 / 25 度）是**起始猜测**。画面左上角直接显示实时读数：

1. 坐直保持几秒，记下三个角度的读数；
2. 弓背保持几秒，记下读数；
3. 取「明显弓背」时的下沿作为对应阈值。

`--still-threshold`（默认 0.05）一般够用；若「明显没动却偶发跳 MOVING」，用 `--debug` 看移动量曲线，把它提到 0.06~0.07。模型「检测到人」的阈值是 `pose_estimation.py` 里的 `DETECT_SCORE_THRESHOLD`（默认 0.15）：人常检不出（离得远/光线差）可降到 ~0.12；太低会对空场景误检。

## 项目结构（分层架构）

```
rtm_demo/
├── main_demo.py          # 桌面版入口：装配各层、跑主循环
├── web_demo.py           # Web 版入口：浏览器摄像头 → /analyze → 回关键点+状态 JSON
├── requirements.txt      # Python 依赖（pip install -r requirements.txt）
├── capture.py            # 图像输入：FrameSource / CameraCapture
├── pose_estimation.py    # RTMPose 推理：17 个 COCO 关键点
├── features.py           # 特征：移动量、坐姿角度
├── decision.py           # 判断：静止/在动、坐姿/弓背、计时提醒
├── output.py             # 输出：画面绘制 + 控制台日志（桌面版用）
└── models/               # RTMPose ONNX 模型
```

数据流：`capture → pose → features → decision → output`，各层职责单一、接口稳定。

关键点编号为 **COCO 17 点标准**：0 鼻、3/4 耳、5/6 肩、11/12 髋、9/10 腕等。

## 后续计划

- **提醒方案会扩展**：目前的提醒只是「连续静止 / 弓背**达到时长阈值就提示一次**」这种简单的时间判断；后续会在 `decision.py` 增加更多提示方案（例如结合坐姿程度、动作类型、持续时长等组合出不同的提醒内容与时机）。
- **机器人摄像头**：当前用的是本机 / 浏览器摄像头；后续接入机器人摄像头时，只需在 `capture.py` 新增一个图像源子类，上层判定逻辑不用改。

## 已知范围与限制

- **单 / 多人**：目前按「整帧单一人」处理（top-down 无检测器），适合人占满画面的坐姿场景
- **RULA / REBA 人体工学评估**：未接入，当前判定只基于现有规则
- **模型**：使用 rtmpose-s（`models/` 内）
- **躯干角度**：髋不可见时躯干角 / 背曲角显示 N/A（头颈角始终可算）
