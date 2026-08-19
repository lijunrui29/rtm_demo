# CLAUDE.md

项目上下文说明。开发前先读，**任何改动都要保持分层架构**。

## 项目

RTMPose 人体姿态 demo：检测关键点 → 判断「静止 / 在动 / 无人」+ 久坐提醒；耳-肩-髋角度判坐姿 / 弓背（GOOD/SLUMPED）+ 不良坐姿提醒。远期把输入源换成机器人摄像头，决策会变复杂。
旧 MediaPipe 版在 `2026/pose estimation/`，仅历史参考（33 点编号，别混用）。

## 代码（最新在 `rtm_demo/`）

数据流 `capture → pose → features → decision → output`：

| 文件 | 职责 |
|---|---|
| `capture.py` | 图像输入：`FrameSource`(open/read/release，支持 with)，唯一实现 `CameraCapture`（本地摄像头）。换机器人摄像头 = 加一个子类，上层不动 |
| `pose_estimation.py` | RTMPose 推理：`detect_pose(frame)` → 17 个 COCO 关键点或 `None`。整帧 letterbox 推理；分数低于阈值时用「锚点/上次成功帧人像框」裁剪重推。只检测不判断 |
| `features.py` | 特征：`FeatureExtractor`(→移动量 float)、`PostureFeatures`(→角度 dict)。只算不判，纯 stdlib |
| `decision.py` | 判断：`StillnessDecision`(→STILL/MOVING/UNKNOWN)、`SedentaryAlert`(→久坐提醒)、`PostureDecision`(→GOOD/SLUMPED/UNKNOWN)、`PostureAlert`(→弓背提醒)，共享 `_DurationAlert` 基类 |
| `output.py` | 输出：`FrameRenderer.draw/log`，只被桌面版用 |
| `main_demo.py` | 桌面入口：装配各层 + cv2.imshow 主循环 |
| `web_demo.py` | 网页入口：浏览器摄像头 POST `/analyze`，回关键点+状态 JSON，骨架浏览器本地画 |

两条平行线：移动量 `FeatureExtractor → StillnessDecision → SedentaryAlert`；角度 `PostureFeatures → PostureDecision → PostureAlert`。

关键点编号一律 **COCO 17**（0 鼻、3/4 耳、5/6 肩、11/12 髋、9/10 腕）。模型在 `models/`，用 `rglob('end2end.onnx')` 找（绝对路径，不依赖工作目录）。

## 运行

命令里的 `python` 指装了 cv2/numpy/onnxruntime 的解释器（安装方法见 README「环境要求」）。

```bash
python main_demo.py --camera 0                        # 桌面版
python main_demo.py --selftest                        # 合成数据自测各层（不碰摄像头）
python web_demo.py --port 8080                        # Web 版（浏览器摄像头，用法见 README）
```

（在 `rtm_demo/` 目录下运行。）

- CLI 默认是短阈值（静坐 20s / 弓背 10s 提醒，方便看效果）；生产用 `--duration-limit 1200 --posture-duration-limit 300`。
- **Web 摄像头 = 打开页面的访问者自己的浏览器**，无服务器摄像头模式。`getUserMedia` 要 HTTPS/localhost；公网用 `cloudflared tunnel --url http://localhost:8080` 得到 https URL（无鉴权，只适合临时演示）。**后端只回小 JSON、浏览器本地画骨架，别改回回传大图**（公网延迟高会卡成 2~3fps）。

## 分层约束（硬性）

- 职责单一：pose 不做判断、features 不触发提醒/绘制、decision 不画图、output 不重算特征、main 只装配。
- **接口稳定**：各层 docstring 的接口约定别改签名；要加能力就加新类/新方法。新规则一律进 `decision.py`。
- 注释中文，常量/输出文案可英文。

## 阈值（标定经验）

- 检测人：`pose_estimation.py` 的 `DETECT_SCORE_THRESHOLD=0.15`（17 点平均置信度，实测真人 ~0.88 / 空场景 ~0.10）。整帧推理分数低于阈值时**不直接判无人**，先人像区域重试：用本帧置信度 >= `ANCHOR_CONF`(0.2) 的锚点框人再裁重推；锚点不足时用最近成功帧缓存的人像框（1s 有效）。重推仍不过阈才判无人。坐着侧身/人偏小这类「整帧锁不住但人还在」的帧基本能救回；人走远/彻底塌陷救不回。别急着降阈值（太低仍会对空场景误检），先看是不是重试也救不回的目标。
- 静止：`--still-threshold` 默认 0.05，静坐时移动量 ≈ 0；偶发跳 MOVING 提到 0.06~0.07（`--debug` 看移动量曲线再定）。`--moving-threshold` 默认 0.10 一般不动。
- 坐姿角度阈值（`--head-neck-threshold` 30 / `--torso-threshold` 30 / `--back-threshold` 25）是起始猜测：看画面实时 `Torso/Neck/Back` 读数标定，迟滞固定阈值的 80%（`hysteresis_ratio=0.8`）。
- 颈压缩 `--neck-threshold`（默认 0.45，无单位比值 = 耳-肩竖直间距/肩宽）：**越小越弓背**（耸肩+低头会压扁耳肩间距）。正面摄像头下前弓背主要靠它抓——torso/back/neck 三个角在正面投影里几乎不变，且坐着髋常被桌子挡掉只剩 head_neck/颈压缩可判。坐直 ~0.5~0.8、弓背 ~0.3 上下，但随取景和身体比例变化很大，用 `--debug` 看 `Head` 读数标定。
- 侧身退化门控（`features.py` 常量，2026-08-19 实拍标定）：侧身时肩宽被投影压扁（正对 ~0.56 → 侧身 0.01~0.21）、髋又常不可见，产出两个退化值。① `SHOULDER_SCALE_MIN_RATIO=0.5`：髋不可见走肩宽兜底时，肩宽 < 0.5×窗口近期中位尺度 → 该帧作废（移动量返回 None → UNKNOWN），避免静止被放大成 MOVING（站立侧身稳态移动量实测 0.045，贴阈值边；坐着+髋被挡会因尺度塌缩更高）。② `NECK_COMPRESSION_MAX=1.5`：颈压缩比值超此物理上限（耳-肩竖直间距不可能超过肩宽 1.5 倍）→ 视为视角退化 → N/A，避免比值爆炸（实拍 0.5~13.5）把坐姿锁死 GOOD。**局限**：门控只挡「尺度突变/比值爆炸」，不挡「长期稳态侧身坐姿」（髋长期不可见时窗口内尺度一致，移动量仍偏高）——稳态侧身根治要靠侧身专用信号（见后续路线），标定别误以为门控已覆盖。
- 提醒时长：decision.py 类默认 1200s / 300s（生产值），main/web 的 CLI 默认 20 / 10（测试短阈值）。

## 开放问题（不要替用户决定）

RULA / REBA 未接入；模型沿用现有 rtmpose-s；单/多人 = 整帧单一人；keypoint schema = COCO 17。「人像区域重试」（整帧失败时按锚点/上次成功帧人像框裁剪重推，见 `pose_estimation.py`）是无检测器下的临时替代（临时沿用），将来接 RTMDet 后可去掉。demo 里必须临时选的标「临时沿用」，不许默默定死。`web_demo.py` 页面底部「当前规则与范围说明」框按此标注，改这几项时同步更新那里。

## 后续路线

- 坐姿提醒已实现，三个角度阈值待真机标定。
- 机器人摄像头：`capture.py` 加 `FrameSource` 子类，`main_demo.py` 按参数选数据源。
- 多目标/人不在画面中心：`pose_estimation.py` 接 RTMDet 检测器，给 `detect_pose` 加可选 bbox，features/decision/output 不用改。接上后可去掉「人像区域重试」（临时替代，救不回走远/塌陷的帧）。
- 复杂决策全进 `decision.py`，保持 `update()` 类接口稳定，跨帧状态用类内部属性。
