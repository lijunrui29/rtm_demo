"""
web_demo.py：本地 / 公网 Web Demo（RTMPose 姿态/坐姿检测）。

谁打开这个网页，就用谁自己的摄像头做实时姿态识别：
    网页请求摄像头授权 → 视频帧 POST 到后端 → 现有 RTMPose 识别 +
    features/decision 判定 → 后端只返回"关键点 + 状态"的 JSON，
    网页用 canvas 在实时视频上画骨架和状态（不来回传整张图，公网也流畅）。
    后端没有任何摄像头，只做推理和判定。

路由：
    /          页面 HTML
    /analyze   POST：接收浏览器摄像头 JPEG 帧 → 跑现有 pipeline →
               返回 {状态/角度/计时/提醒 + 关键点 landmarks}（纯 JSON，无图片）

复用原则：摄像头源在浏览器侧（capture 层自然不再用），
pose_estimation / features / decision 整条判定链完全复用，核心算法一行没改。
（output.py 的 FrameRenderer 仍由桌面版 main_demo.py 使用；网页端骨架由
浏览器 canvas 画，COCO 连线与 output.py 保持一致。）

公网访问 / HTTPS：
    浏览器 getUserMedia 要求"安全上下文"（HTTPS 或 localhost）。本机直接
    打开对应端口即可；让局域网外的人用自己的摄像头，推荐
    Cloudflare 快速隧道（自动带 HTTPS）：
        cloudflared tunnel --url http://localhost:8080
    会得到一个 https://xxxx.trycloudflare.com 地址，发给他打开，授权摄像头即可。
    服务默认监听 0.0.0.0:8080，隧道把公网 HTTPS 流量转发进来。
    注意：快速隧道 URL 没有鉴权，任何人拿到链接都能访问，只适合临时演示。
    公网延迟高时（尤其大陆到 Cloudflare 海外节点），画面会因帧回传而卡顿——
    已改为后端只回关键点 JSON、浏览器本地画骨架，尽量缓解。

范围说明（沿用现有实现，不做新决定）：
    - 风险判定 = 当前现有规则（静止/在动、坐姿/弓背[头颈角/躯干角/背曲角/颈压缩任一超限]、计时提醒）
    - RULA / REBA：未接入（开放问题，未定）
    - 单 / 多人：沿用"整帧单一人"实现（开放问题，未定，临时沿用）
    - keypoint schema：COCO 17（代码已固定）
    - 模型：models/ 内现有 rtmpose-s（沿用，未换）
    页面底部有同样的说明框。

用法（在 rtm_demo/ 目录下运行）：
    python web_demo.py --port 8080
    浏览器打开页面并授权摄像头即可；Ctrl+C 退出。
    首个访问者的第一帧会初始化 pose 检测器（约 1~2s），之后正常。
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 本目录（rtm_demo）内的模块直接 import。
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import cv2                                      # noqa: E402
import numpy as np                              # noqa: E402
from pose_estimation import detect_pose_with_score   # noqa: E402
from features import (FeatureExtractor,         # noqa: E402
                      PostureFeatures)
from decision import (StillnessDecision,        # noqa: E402
                      SedentaryAlert,
                      PostureDecision, PostureAlert)

# /analyze 单帧最大体积（约几 MB 足够），防止异常大请求拖垮服务
MAX_ANALYZE_BYTES = 4_000_000

# 临时标定：/analyze 帧计数器 + 上次打印时间（每约 8 帧打一行日志，标定完删除）
_dbg_frames = [0]
_dbg_last_ts = [time.monotonic()]


# ---------- 页面 HTML（模块级常量；浏览器端中文 OK，与 cv2 的 Hershey 字体无关） ----------

PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>RTMPose 姿态 / 坐姿检测 Demo</title>
<style>
  body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
         margin: 20px; background: #111; color: #eee; }
  h1 { font-size: 20px; margin: 0 0 12px 0; }
  .container { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }
  .video { flex: 1 1 620px; min-width: 320px; }
  #stage { position: relative; width: 100%; }
  #stage video { width: 100%; display: block; border-radius: 8px; background: #000; }
  #stage canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                  pointer-events: none; }
  .msg { color: #ffb84d; font-size: 13px; margin-top: 6px; min-height: 1.2em; }
  .ctrl { margin-top: 8px; }
  .ctrl button { background: #2a2a2a; color: #eee; border: 1px solid #444;
                 border-radius: 6px; padding: 4px 12px; cursor: pointer; }
  .panel { flex: 0 0 330px; background: #1d1d1d; border-radius: 8px; padding: 14px 16px; }
  .panel h3 { margin: 0 0 8px 0; color: #6ab0ff; }
  .row { display: flex; justify-content: space-between; gap: 12px;
         padding: 6px 0; border-bottom: 1px solid #2a2a2a; font-size: 15px; }
  .row .k { color: #999; }
  .row .v { font-weight: bold; text-align: right; }
  .still  { color: #3ddc84; } .moving { color: #ff5c5c; } .unknown { color: #888; }
  .good   { color: #3ddc84; } .slumped { color: #ff8c00; }
  .reminder { color: #ffb84d; min-height: 1.2em; font-size: 13px; }
  .note { margin-top: 22px; max-width: 960px; background: #1d1d1d;
          border-radius: 8px; padding: 12px 18px; font-size: 13px; color: #bbb; }
  .note h3 { color: #eee; margin: 0 0 8px 0; }
  .note li { margin: 3px 0; }
</style>
</head>
<body>
<h1>RTMPose 姿态 / 坐姿检测 Demo</h1>

<div class="container">
  <div class="video">
    <div id="stage">
      <video id="cam" autoplay playsinline muted></video>
      <canvas id="overlay"></canvas>
      <canvas id="cap" style="display:none"></canvas>
    </div>
    <div id="camMsg" class="msg"></div>
    <div class="ctrl">
      <button id="retryCam" onclick="startBrowser()" style="display:none">重新授权摄像头</button>
    </div>
  </div>

  <div class="panel">
    <h3>实时判定</h3>
    <div class="row"><span class="k">人员状态</span><span class="v" id="state">--</span></div>
    <div class="row"><span class="k">移动量</span><span class="v" id="movement">--</span></div>
    <div class="row"><span class="k">检测分</span><span class="v" id="score">--</span></div>
    <div class="row"><span class="k">躯干点</span><span class="v" id="tvalid">--</span></div>
    <div class="row"><span class="k">坐姿状态</span><span class="v" id="posture">--</span></div>
    <div class="row"><span class="k">头颈角</span><span class="v" id="neck">--</span></div>
    <div class="row"><span class="k">躯干角</span><span class="v" id="torso">--</span></div>
    <div class="row"><span class="k">背曲角</span><span class="v" id="back">--</span></div>
    <div class="row"><span class="k">颈压缩</span><span class="v" id="neckgap">--</span></div>
    <div class="row"><span class="k">连续静坐</span><span class="v" id="still">--</span></div>
    <div class="row"><span class="k">连续弓背</span><span class="v" id="slump">--</span></div>
    <div class="row"><span class="k">久坐提醒</span><span class="v reminder" id="reminder"></span></div>
    <div class="row"><span class="k">坐姿提醒</span><span class="v reminder" id="preminder"></span></div>
  </div>
</div>

<div class="note">
  <h3>当前规则与范围说明</h3>
  <ul>
    <li><b>风险判定 = 当前现有规则</b>：静止 / 在动（移动量双阈值）、坐姿 / 弓背（GOOD / SLUMPED：头颈角 / 躯干角 / 背曲角 / 颈压缩任一超限即弓背）、久坐 / 弓背计时提醒。</li>
    <li><b>摄像头来源</b>：始终取"打开这个网页的人"自己的摄像头（浏览器采集，帧发后端识别，骨架在浏览器本地绘制）。</li>
    <li><b>RULA / REBA</b>：未接入（开放问题，未定）。</li>
    <li><b>单 / 多人</b>：沿用现有"整帧单一人"实现（开放问题，未定，临时沿用）。</li>
    <li><b>Keypoint schema</b>：COCO 17（代码已固定）。</li>
    <li><b>模型</b>：models/ 内现有 rtmpose-s（沿用，未换）。</li>
  </ul>
</div>

<script>
  var video = document.getElementById('cam');
  var cap = document.getElementById('cap');
  var overlay = document.getElementById('overlay');
  var camMsg = document.getElementById('camMsg');
  var retryBtn = document.getElementById('retryCam');

  // COCO 17 骨架连线（只画躯干+四肢 5-16，绿色；耳点 3/4 单独黄色圆点），
  // 与 output.py 的桌面版绘制保持一致。visibility > 0.3 才画。
  var BONES = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
  var VIS_MIN = 0.3;

  function fmt(v, suffix) {
    return (v === null || v === undefined) ? '--' : v + suffix;
  }

  function setPanel(s) {
    var st = document.getElementById('state');
    st.textContent = s.state || '--';
    st.className = 'v ' + String(s.state || '').toLowerCase();
    document.getElementById('movement').textContent = fmt(s.movement, '');
    document.getElementById('score').textContent = fmt(s.score, '');
    var tv = document.getElementById('tvalid');
    tv.textContent = (s.torso_avg === null || s.torso_avg === undefined)
      ? '--' : (s.torso_valid + '/4 均' + s.torso_avg);
    var pst = document.getElementById('posture');
    pst.textContent = s.posture_state || '--';
    pst.className = 'v ' + String(s.posture_state || '').toLowerCase();
    document.getElementById('neck').textContent = fmt(s.head_neck_angle, '°');
    document.getElementById('torso').textContent = fmt(s.torso_angle, '°');
    document.getElementById('back').textContent = fmt(s.back_curvature, '°');
    document.getElementById('neckgap').textContent = fmt(s.neck_compression, '');
    document.getElementById('still').textContent = fmt(s.still_elapsed_sec, ' s');
    document.getElementById('slump').textContent = fmt(s.slump_elapsed_sec, ' s');
    document.getElementById('reminder').textContent = s.reminder || '';
    document.getElementById('preminder').textContent = s.posture_reminder || '';
  }

  // 骨架 + 状态文字画在覆盖层 canvas 上（本地画，不来回传图）
  function drawOverlay(s) {
    var ctx = overlay.getContext('2d');
    var w = overlay.width, h = overlay.height;
    ctx.clearRect(0, 0, w, h);

    var pts = s.landmarks;
    if (pts && s.detected) {
      function P(i) { return { x: pts[i][0] * w, y: pts[i][1] * h, v: pts[i][3] }; }
      // 连线（躯干+四肢）
      ctx.strokeStyle = '#3ddc84'; ctx.fillStyle = '#3ddc84';
      ctx.lineWidth = 3; ctx.lineCap = 'round';
      for (var b = 0; b < BONES.length; b++) {
        var a = P(BONES[b][0]), c = P(BONES[b][1]);
        if (a.v > VIS_MIN && c.v > VIS_MIN) {
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(c.x, c.y); ctx.stroke();
        }
      }
      // 关节点（5-16）
      for (var i = 5; i <= 16; i++) {
        var p = P(i);
        if (p.v > VIS_MIN) { ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2); ctx.fill(); }
      }
      // 耳点 3/4（黄色，坐姿角度依赖它们）
      ctx.fillStyle = '#ffd000';
      for (var j = 3; j <= 4; j++) {
        var e = P(j);
        if (e.v > VIS_MIN) { ctx.beginPath(); ctx.arc(e.x, e.y, 4, 0, Math.PI * 2); ctx.fill(); }
      }
    }

    // 顶部状态文字（带黑描边，视频上可读）
    ctx.font = '16px "Segoe UI", sans-serif';
    ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(0,0,0,0.8)'; ctx.fillStyle = '#fff';
    var line = 'State: ' + (s.state || '--') + '   Posture: ' + (s.posture_state || '--');
    ctx.strokeText(line, 10, 26); ctx.fillText(line, 10, 26);

    // 提醒横幅（有才画）
    var r = s.reminder || s.posture_reminder;
    if (r) {
      ctx.font = '18px "Segoe UI", sans-serif';
      ctx.strokeStyle = 'rgba(0,0,0,0.9)'; ctx.fillStyle = '#ffb84d';
      ctx.strokeText(r, 10, 60); ctx.fillText(r, 10, 60);
    }
  }

  // 页面打开即请求摄像头授权（浏览器会弹授权框）
  startBrowser();

  function startBrowser() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      camMsg.textContent = '当前浏览器不支持摄像头，或页面不是安全上下文（需 HTTPS 或 localhost）。' +
        '局域网纯 HTTP 地址会被浏览器拦截摄像头授权；请用 Cloudflare 隧道 https://xxxx.trycloudflare.com 打开。';
      retryBtn.style.display = 'none';
      return;
    }
    navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 480 } }, audio: false })
      .then(function (stream) {
        video.srcObject = stream;
        retryBtn.style.display = 'none';
        return video.play();
      })
      .then(function () {
        // 覆盖层画布与视频等尺寸（像素），关键点是归一化坐标，乘画布宽高即对齐
        overlay.width = video.videoWidth || 640;
        overlay.height = video.videoHeight || 480;
        requestAnimationFrame(loop);
      })
      .catch(function (err) {
        camMsg.textContent = '摄像头授权失败：' + (err && err.name) +
          '。可点"重新授权"再试。';
        retryBtn.style.display = '';
      });
  }

  // 采集帧 → POST /analyze → 收到关键点 JSON → 本地画骨架
  var FPS = 8, INTERVAL = 1000 / FPS;
  var lastSent = 0, busy = false;

  function loop(now) {
    requestAnimationFrame(loop);
    if (busy || video.readyState < 2) return;
    if (now - lastSent < INTERVAL) return;
    lastSent = now;
    // 采集到固定宽度 480 的小帧再发，控制网络/识别开销（模型本身才 192x256）
    var W = 480;
    var scale = W / video.videoWidth;
    cap.width = W;
    cap.height = Math.max(1, Math.round(video.videoHeight * scale));
    cap.getContext('2d').drawImage(video, 0, 0, cap.width, cap.height);
    busy = true;
    cap.toBlob(function (blob) {
      if (!blob) { busy = false; return; }
      fetch('/analyze', { method: 'POST', body: blob })
        .then(function (r) { return r.json(); })
        .then(function (s) {
          if (s.error) { camMsg.textContent = s.error; }
          else { camMsg.textContent = ''; drawOverlay(s); setPanel(s); }
        })
        .catch(function () { /* 网络抖动：下一帧再试 */ })
        .then(function () { busy = false; });
    }, 'image/jpeg', 0.55);
  }
</script>
</body>
</html>
"""


# ---------- 浏览器摄像头模式：POST /analyze 单帧处理 ----------

_browser_lock = threading.Lock()          # 串行化 /analyze，避免多请求搅乱跨帧状态
_browser_chain = None                     # 惰性建链（首个 /analyze 请求时才初始化）


def build_chain(args: argparse.Namespace):
    """按现有 main_demo 的装配方式建一条 features→decision 判定链。

    返回 (feats, post, dec, alert, pdec, palert)。跨帧状态判定
    （移动量窗口、静坐/弓背计时）依赖链实例常驻，因此只建一条全局共用。
    """
    # 低帧率适配：手机经隧道画面实际约 0.5~1fps，桌面版的 3s/5 帧窗口永远
    # 攒不满 → 移动量恒空。web 入口放宽窗口参数 + 降低可见性门限（0.15，
    # 与 DETECT_SCORE_THRESHOLD 一致；手机画面分数低于桌面 0.3）。
    feats = FeatureExtractor(window_seconds=args.window_seconds,
                             min_frames=args.min_frames,
                             min_window_seconds=args.min_window_seconds,
                             visibility_min=args.visibility_min)
    post = PostureFeatures(visibility_min=args.visibility_min)
    dec = StillnessDecision(still_threshold=args.still_threshold,
                            moving_threshold=args.moving_threshold)
    alert = SedentaryAlert(duration_limit_sec=args.duration_limit)
    pdec = PostureDecision(head_neck_threshold=args.head_neck_threshold,
                           torso_threshold=args.torso_threshold,
                           back_threshold=args.back_threshold,
                           neck_threshold=args.neck_threshold)
    palert = PostureAlert(duration_limit_sec=args.posture_duration_limit)
    return feats, post, dec, alert, pdec, palert


def _torso_stats(pose):
    """躯干 4 点（肩 5/6、髋 11/12）的可见性统计，用于标定。

    移动量（FeatureExtractor）和坐姿角度（PostureFeatures）都要求关键点
    visibility >= 0.3 才计入；电话端整体分数偏低时，躯干点容易被这条门限
    整帧丢掉 → 移动量恒空。返回 (过 0.3 门限的点数 0~4, 4 点平均置信度)，
    pose 为 None 时返回 (0, None)。
    """
    if pose is None:
        return 0, None
    lm = pose.get('landmarks') or []
    vis = [lm[i][3] for i in (5, 6, 11, 12) if i < len(lm)]
    if not vis:
        return 0, None
    return sum(1 for v in vis if v >= 0.3), round(sum(vis) / len(vis), 3)


def analyze_frame(jpeg_bytes: bytes, args: argparse.Namespace) -> dict:
    """浏览器摄像头模式：一帧 JPEG → 现有 pipeline 识别+判定 → 返回 JSON。

    只回状态/角度/计时/提醒 + 归一化关键点 landmarks（小 JSON，不带图），
    由浏览器本地画骨架 —— 公网延迟高时不会因为来回传整张图而卡顿。
    帧数一帧一 POST，链条实例常驻，跨帧累计逻辑照常工作。
    """
    global _browser_chain
    with _browser_lock:
        if _browser_chain is None:
            _browser_chain = build_chain(args)
            print("[web] 浏览器摄像头模式已就绪（首个帧时初始化，模型加载约 1~2s）")
        feats, post, dec, alert, pdec, palert = _browser_chain

        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "无法解码图片"}

        ts = time.monotonic()
        try:
            pose, score = detect_pose_with_score(frame)
            movement = feats.update(pose, ts)
            state_ = dec.update(movement)
            posture = post.update(pose)
            posture_state = pdec.update(posture)
            reminder = alert.update(state_, ts)
            posture_reminder = palert.update(posture_state, ts)
        except Exception as e:
            return {"error": f"pipeline 异常: {type(e).__name__}: {e}"}

        torso_valid, torso_avg = _torso_stats(pose)

        # 临时标定日志：每 8 帧打一行，含实际 fps + 坐姿特征（标定完删除）
        _dbg_frames[0] += 1
        if _dbg_frames[0] % 8 == 0:
            _now = time.monotonic()
            span = _now - _dbg_last_ts[0]
            _dbg_last_ts[0] = _now
            fps = 8 / span if span > 0 else 0.0
            hn = posture.get('head_neck_angle') if posture else None
            to = posture.get('torso_angle') if posture else None
            bc = posture.get('back_curvature') if posture else None
            nc = posture.get('neck_compression') if posture else None
            a = lambda v: '--' if v is None else f"{v:.0f}"
            b = lambda v: '--' if v is None else f"{v:.2f}"
            print(f"[dbg] #{_dbg_frames[0]} fps={fps:.1f} "
                  f"score={score:.3f} detected={pose is not None} "
                  f"torso={torso_valid}/4 avg={torso_avg} "
                  f"mov={movement if movement is None else round(movement, 4)} "
                  f"state={state_.value} post={posture_state.value} "
                  f"ang=(hn:{a(hn)} to:{a(to)} bc:{a(bc)} nc:{b(nc)})",
                  flush=True)

        return {
            "detected": pose is not None,
            "score": round(score, 3),  # 本帧整体置信度（标定 DETECT_SCORE_THRESHOLD 用）
            "torso_valid": torso_valid,   # 躯干点过 0.3 门限的个数（0~4）
            "torso_avg": torso_avg,       # 躯干 4 点平均置信度（判断门限卡没卡）
            "state": state_.value,
            "movement": round(movement, 4) if movement is not None else None,
            "posture_state": posture_state.value,
            "head_neck_angle": _round_angle(posture, "head_neck_angle"),
            "torso_angle": _round_angle(posture, "torso_angle"),
            "back_curvature": _round_angle(posture, "back_curvature"),
            "neck_compression": _round_value(posture, "neck_compression", 2),
            "still_elapsed_sec": round(alert.elapsed_sec, 1),
            "slump_elapsed_sec": round(palert.elapsed_sec, 1),
            "reminder": reminder,
            "posture_reminder": posture_reminder,
            # 归一化关键点 0~1（相对原始帧），浏览器乘画布宽高即对齐
            "landmarks": pose["landmarks"] if pose else None,
            "image_size": list(pose["image_size"]) if pose else None,
        }


def _round_angle(posture, key):
    """posture dict 里取角度，缺 key / 值为 None 时给 None（= 页面上 N/A）。"""
    if posture is None:
        return None
    val = posture.get(key)
    return round(val, 1) if val is not None else None


def _round_value(posture, key, digits):
    """posture dict 里取值并 round 到 digits 位小数；缺 key / 值为 None → None。

    颈压缩是比值（0~1 量级），标定需要两位小数，单独一个 helper（角度仍用
    _round_angle 一位小数 + °）。
    """
    if posture is None:
        return None
    val = posture.get(key)
    return round(val, digits) if val is not None else None


# ---------- HTTP 处理 ----------

class Handler(BaseHTTPRequestHandler):
    """路由：GET / 页面、POST /analyze 识别。"""

    protocol_version = "HTTP/1.1"
    _analyze_args: argparse.Namespace  # 由 main() 在 serve 前注入

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_bytes(PAGE_HTML.encode("utf-8"),
                             "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        """POST /analyze：接收浏览器摄像头的一帧 JPEG，返回识别结果 JSON。"""
        path = self.path.split("?", 1)[0]
        if path != "/analyze":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_ANALYZE_BYTES:
            self._send_json({"error": "请求为空或图片过大"})
            return
        body = self.rfile.read(length)
        self._send_json(analyze_frame(body, self._analyze_args))

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: dict) -> None:
        self._send_bytes(json.dumps(obj).encode("utf-8"),
                         "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        # 只打印"非逐帧"请求，避免 /analyze 每帧刷屏。
        # 注意：send_error 内部会走 log_error，args[0] 可能是 int（状态码），
        # 不能直接拿来 in 判断；先渲染成整行字符串再过滤。
        try:
            line = fmt % args
        except TypeError:
            line = f"{fmt} {args}"
        if "/analyze" in line:
            return
        print(f"[web] {self.address_string()} - {line}")


# ---------- 入口 ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="本地/公网 Web Demo（RTMPose 姿态/坐姿检测，用访问者的摄像头）")
    p.add_argument("--host", type=str, default="0.0.0.0",
                   help="监听地址，默认 0.0.0.0（本机+局域网可访问；公网用 Cloudflare 隧道转发进来）")
    p.add_argument("--port", type=int, default=8080,
                   help="监听端口，默认 8080")
    p.add_argument("--debug", action="store_true",
                   help="debug：画面上画移动量走势图")
    p.add_argument("--still-threshold", type=float, default=0.05,
                   help="静止阈值（移动量 <= 此值判静止），默认 0.05")
    p.add_argument("--moving-threshold", type=float, default=0.07,
                   help="在动阈值（移动量 >= 此值判在动），web 默认 0.07（低帧率下让中等移动也能触发；桌面 main_demo 仍 0.10）")
    p.add_argument("--window-seconds", type=float, default=5.0,
                   help="移动量滑动窗口时长（秒），web 默认 5（手机经隧道帧率 ~0.6fps，窗口需 ≥3 帧间隔≈4.8s 才能算；5 是响应与抗噪的折中；桌面版 main_demo 仍是 3）")
    p.add_argument("--min-frames", type=int, default=2,
                   help="移动量窗口最少帧数，默认 2（低帧率 ~0.6fps 下最快响应；静止实测 mov≈0.02，2 帧中位仍抗噪）")
    p.add_argument("--min-window-seconds", type=float, default=1.0,
                   help="移动量窗口首尾帧最小跨度（秒），默认 1.0（2 帧跨度 ≈1.6s 已满足）")
    p.add_argument("--visibility-min", type=float, default=0.15,
                   help="关键点可见性门限（移动量/坐姿角度要求点分数>=此值），web 默认 0.15（手机画面分数低于桌面 0.3）")
    p.add_argument("--duration-limit", type=float, default=20.0,
                   help="连续久坐多少秒触发提醒，默认 20（测试用短阈值；生产建议 1200）")
    p.add_argument("--head-neck-threshold", type=float, default=30.0,
                   help="头前伸角阈值（度），默认 30")
    p.add_argument("--torso-threshold", type=float, default=30.0,
                   help="躯干前倾角阈值（度），默认 30")
    p.add_argument("--back-threshold", type=float, default=25.0,
                   help="背部弯曲角阈值（度），默认 25")
    p.add_argument("--neck-threshold", type=float, default=0.45,
                   help="颈压缩阈值（耳-肩竖直间距/肩宽，无单位比值；"
                        "小于此值判弓背，正面摄像头对前弓背敏感），"
                        "默认 0.45（起始猜测，看页面 Head/颈压缩读数标定）")
    p.add_argument("--posture-duration-limit", type=float, default=10.0,
                   help="连续不良坐姿多少秒触发提醒，默认 10（测试用短阈值；生产建议 300）")
    p.add_argument("--reminder-hold", type=float, default=8.0,
                   help="提醒横幅在画面上停留秒数，默认 8")
    p.add_argument("--no-skeleton", dest="draw_skeleton", action="store_false",
                   help="不把姿态骨架（关键点+连线）叠到画面上")
    p.set_defaults(draw_skeleton=True)
    return p


def _lan_ip() -> str:
    """尽力取本机局域网 IP；取不到返回空串。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


def main() -> None:
    args = build_parser().parse_args()
    Handler._analyze_args = args
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    print("[web] 谁打开页面，就用谁的摄像头（浏览器采集帧 POST /analyze 识别，骨架本地画）")
    print(f"[web] 监听 {args.host}:{args.port}")
    ip = _lan_ip()
    if args.host in ("0.0.0.0", "::") and ip:
        print(f"[web]   局域网:  http://{ip}:{args.port}/   （注意：纯 HTTP 下浏览器会拒绝摄像头，只能看页面）")
    print(f"[web] 公网演示（别人用自己的摄像头，需要 HTTPS）：")
    print(f"[web]   cloudflared tunnel --url http://localhost:{args.port}")
    print(f"[web]   打开它给出的 https://xxxx.trycloudflare.com 发给别人即可（Ctrl+C 退出）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 正在退出...")
    finally:
        server.server_close()
        print("[web] 已退出")


if __name__ == "__main__":
    main()
