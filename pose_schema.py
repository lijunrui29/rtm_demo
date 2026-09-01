"""
pose_schema 层：把人体 / 机器人两端的姿态数据，统一到同一个"规范关节 schema"。

背景（2026-09-01，为"人体动作映射到机器人关节"做的第一步）：
    人体侧（rtm_demo/pose_estimation.py 的 detect_pose）输出 COCO 17 个关键点的
    **位置**（归一化 2D 坐标 + 置信度，数字编号）。
    机器人侧（figurobot-console/robot_bridge.py 的 /api/pose）输出 27 个舵机的
    **角度**（度，相对零点，servo ID 编号）。
    两边命名、数量、坐标系、语义都不同，直接没法一一对应。
    本模块把两边各自"翻译"进同一个 27 DOF 规范关节空间，字段统一，未来的
    "人体→机器人"映射（mapper）只在这个空间里做，不碰两边原始格式。

规范名 = <部位>_<侧>_<轴>，轴语义沿用 figurobot-console.html 的 SKELETON：
    P=Pitch(x)  R=Roll(z)  Y=Yaw(y)。
关节集合 / 左右 / 轴 / servo_id 均以 figurobot-console.html 的 SKELETON 为唯一依据
（27 DOF；32 个 ID 里缺装 5/10/18/31/32）。每条带 zh_name（该文件的中文关节名，
如 腰Y / 颈P / 左肩P），导出时一并带上，两边命名 1:1 对得上。
servo_id ↔ 部位的映射来自该文件，其注释标注"基于左右对称**推测**"，
真机装配时需按实际动轴确认（见 CLAUDE.md 开放问题）。

本模块是纯标准库（math/json/urllib/time），不 import cv2 / pose_estimation /
features，可独立测试（python pose_schema.py）。

接口约定（保持稳定，别改签名）：
    empty_schema(kind) -> dict                            空的规范 schema
    human_adapter(pose_result) -> dict                    COCO 17 关键点 → 规范 schema
    fetch_robot_pose(base_url=..., timeout=...) -> dict   POST /api/pose 拿机器人原始 JSON
    robot_adapter(pose_json, zero_values=None) -> dict    /api/pose JSON → 规范 schema
    robot_schema_from_bridge(base_url=...) -> dict        fetch + robot_adapter 组合
"""

from __future__ import annotations

import json
import math
import time
import urllib.request

# 规范关节表（27 DOF，与机器人 27 个装配舵机一一对应）。
# 顺序沿用 figurobot-console.html SKELETON：腰→胸→颈/头→左臂→右臂→左腿→右腿。
#   canonical : 规范名（human/robot 两侧共用的命名）
#   servo_id  : 机器人侧舵机 ID（figurobot SKELETON，左右对称推测）
#   coco_ids  : 人体侧用到的 COCO 17 点编号（该 DOF 的信号来源）
#   axis      : 旋转轴（P=x / R=z / Y=y，机器人局部坐标）
#   parent    : 规范关节链上的父节点（"pelvis"=骨盆派生点，非舵机）
CANONICAL_JOINTS = [
    # 腰 / 胸（5 DOF，人体侧只能弱推导 waist_pitch / waist_yaw，其余 na）
    {"canonical": "waist_yaw",   "zh_name": "腰Y", "servo_id": 12, "coco_ids": (5, 6, 11, 12), "axis": "y", "parent": "pelvis"},
    {"canonical": "waist_pitch", "zh_name": "腰P", "servo_id": 15, "coco_ids": (5, 6, 11, 12), "axis": "x", "parent": "waist_yaw"},
    {"canonical": "waist_roll",  "zh_name": "腰R", "servo_id": 14, "coco_ids": (5, 6, 11, 12), "axis": "z", "parent": "waist_pitch"},
    {"canonical": "chest_pitch", "zh_name": "胸P", "servo_id": 16, "coco_ids": (5, 6, 11, 12), "axis": "x", "parent": "waist_roll"},
    {"canonical": "chest_roll",  "zh_name": "胸R", "servo_id": 17, "coco_ids": (5, 6, 11, 12), "axis": "z", "parent": "chest_pitch"},
    # 颈 / 头（2 DOF；颈P 可弱推导，头Y 用鼻-耳水平偏移弱推导）
    {"canonical": "neck_pitch",  "zh_name": "颈P", "servo_id": 2,  "coco_ids": (0, 5, 6), "axis": "x", "parent": "chest_roll"},
    {"canonical": "head_yaw",    "zh_name": "头Y", "servo_id": 1,  "coco_ids": (0, 3, 4), "axis": "y", "parent": "neck_pitch"},
    # 左臂（4 DOF；pitch 类可弱推导，roll 需深度 → na）
    {"canonical": "shoulder_left_pitch", "zh_name": "左肩P", "servo_id": 3,  "coco_ids": (5, 7),    "axis": "x", "parent": "chest_roll"},
    {"canonical": "shoulder_left_roll",  "zh_name": "左肩R", "servo_id": 4,  "coco_ids": (5, 7),    "axis": "z", "parent": "shoulder_left_pitch"},
    {"canonical": "elbow_left_pitch",    "zh_name": "左肘P", "servo_id": 6,  "coco_ids": (5, 7, 9), "axis": "x", "parent": "shoulder_left_roll"},
    {"canonical": "wrist_left_pitch",    "zh_name": "左腕P", "servo_id": 8,  "coco_ids": (9,),      "axis": "x", "parent": "elbow_left_pitch"},
    # 右臂（4 DOF）
    {"canonical": "shoulder_right_pitch", "zh_name": "右肩P", "servo_id": 7,  "coco_ids": (6, 8),    "axis": "x", "parent": "chest_roll"},
    {"canonical": "shoulder_right_roll",  "zh_name": "右肩R", "servo_id": 9,  "coco_ids": (6, 8),    "axis": "z", "parent": "shoulder_right_pitch"},
    {"canonical": "elbow_right_pitch",    "zh_name": "右肘P", "servo_id": 11, "coco_ids": (6, 8, 10), "axis": "x", "parent": "shoulder_right_roll"},
    {"canonical": "wrist_right_pitch",    "zh_name": "右腕P", "servo_id": 13, "coco_ids": (10,),     "axis": "x", "parent": "elbow_right_pitch"},
    # 左腿（6 DOF）
    {"canonical": "hip_left_pitch",  "zh_name": "左髋P", "servo_id": 19, "coco_ids": (11, 13),     "axis": "x", "parent": "pelvis"},
    {"canonical": "hip_left_roll",   "zh_name": "左髋R", "servo_id": 21, "coco_ids": (11, 13),     "axis": "z", "parent": "hip_left_pitch"},
    {"canonical": "hip_left_yaw",    "zh_name": "左髋Y", "servo_id": 29, "coco_ids": (11, 13),     "axis": "y", "parent": "hip_left_roll"},
    {"canonical": "knee_left_pitch", "zh_name": "左膝P", "servo_id": 23, "coco_ids": (11, 13, 15), "axis": "x", "parent": "hip_left_yaw"},
    {"canonical": "ankle_left_pitch", "zh_name": "左踝P", "servo_id": 25, "coco_ids": (13, 15),    "axis": "x", "parent": "knee_left_pitch"},
    {"canonical": "ankle_left_roll",  "zh_name": "左踝R", "servo_id": 27, "coco_ids": (15,),       "axis": "z", "parent": "ankle_left_pitch"},
    # 右腿（6 DOF）
    {"canonical": "hip_right_pitch",  "zh_name": "右髋P", "servo_id": 20, "coco_ids": (12, 14),     "axis": "x", "parent": "pelvis"},
    {"canonical": "hip_right_roll",   "zh_name": "右髋R", "servo_id": 22, "coco_ids": (12, 14),     "axis": "z", "parent": "hip_right_pitch"},
    {"canonical": "hip_right_yaw",    "zh_name": "右髋Y", "servo_id": 30, "coco_ids": (12, 14),     "axis": "y", "parent": "hip_right_roll"},
    {"canonical": "knee_right_pitch", "zh_name": "右膝P", "servo_id": 24, "coco_ids": (12, 14, 16), "axis": "x", "parent": "hip_right_yaw"},
    {"canonical": "ankle_right_pitch", "zh_name": "右踝P", "servo_id": 26, "coco_ids": (14, 16),    "axis": "x", "parent": "knee_right_pitch"},
    {"canonical": "ankle_right_roll",  "zh_name": "右踝R", "servo_id": 28, "coco_ids": (16,),       "axis": "z", "parent": "ankle_right_pitch"},
]

# 舵机 ID ↔ 规范名 的双向查找表（将来 mapper 用）
SERVO_TO_CANONICAL = {j["servo_id"]: j["canonical"] for j in CANONICAL_JOINTS}
CANONICAL_TO_SERVO = {j["canonical"]: j["servo_id"] for j in CANONICAL_JOINTS}

# 置信度 → status 的阈值（对齐 features 的 visibility_min=0.3 门控）
_CONF_OK = 0.5
_CONF_LOW = 0.3

# 机器人角度换算（与 figurobot robot_bridge.py / Three.js 骨架一致）
_POS_MAX = 4095.0
_DEG_MAX = 360.0
_DEFAULT_ZERO = 2048          # zero.ini 缺该舵机时的默认零点（与 HTML updatePose3D 一致）
_ZERO_SCALE = _DEG_MAX / _POS_MAX  # 每 tick 对应多少度


def empty_schema(kind: str) -> dict:
    """构造空的规范 schema：27 个 DOF 全 na，等 adapter 填充。

    kind: "human" 或 "robot"，决定 source.kind。
    """
    assert kind in ("human", "robot"), f"kind 只能是 human/robot，实际 {kind!r}"
    joints = {}
    for j in CANONICAL_JOINTS:
        joints[j["canonical"]] = {
            "canonical": j["canonical"],
            "zh_name": j["zh_name"],   # figurobot SKELETON 的中文关节名（腰Y / 颈P / 左肩P…）
            "servo_id": j["servo_id"],
            "coco_ids": list(j["coco_ids"]),
            "axis": j["axis"],
            "parent": j["parent"],
            "position": None,     # 归一化 (x, y, z=0)，人体侧填；机器人侧无
            "angle_deg": None,    # 度；机器人=相对零点角，人体=2D 弱推导角
            "angle_source": None,  # "robot_read" / "heuristic_2d" / None
            "confidence": 0.0,    # 0~1（人体=源点可见度，机器人=online/error 折算）
            "status": "na",       # ok / low_conf / na / offline / error
            "present_in": [],     # ["human"] / ["robot"]
            "notes": "",
        }
    return {
        "schema_version": "figurobot-pose-0.1",
        "timestamp": 0.0,
        "source": {"kind": kind, "model": "", "online": False},
        "units": {
            "position": "normalized (fraction of frame, x right / y down)",
            "angle": "degrees (robot: relative to zero.ini; human heuristic: "
                     "unsigned magnitude from image vertical; yaw DOFs carry "
                     "sign = image-plane direction, mapper 处理镜像)",
            "servo": "tick (0-4095, robot raw position)",
        },
        "joints": joints,
    }


# ---------------------------------------------------------------------------
# 人体侧 adapter：COCO 17 关键点 → 规范 schema
# ---------------------------------------------------------------------------

def human_adapter(pose_result) -> dict:
    """把 pose_estimation.detect_pose() 的结果翻译成规范 schema。

    参数:
        pose_result: detect_pose() 的返回值 {'landmarks': [(x,y,z,vis) x17],
                     'image_size': (w,h)}，未检测到人体时为 None。
        只读 landmarks，不改动原始输出。

    返回:
        规范 schema。27 个 DOF 里，能填的填 position / 弱推导 angle，
        填不了的保持 na —— 不假装有数据。另附 derived_points（neck 肩中点 /
        pelvis 髋中点 / torso_vector 肩中点→髋中点），供腰部弱推导和将来 mapper 用。
    """
    if pose_result is None:
        return empty_schema("human")
    landmarks = pose_result.get("landmarks")
    if not landmarks:
        return empty_schema("human")

    schema = empty_schema("human")
    schema["timestamp"] = round(time.time(), 3)
    schema["source"]["model"] = "rtmpose-s (COCO 17)"
    schema["source"]["online"] = True

    # 可见点：xy{pid: (x,y)} 与 vis{pid: visibility} 分开存（复用 features 的门控风格）
    xy: dict = {}
    vis: dict = {}
    for pid, lm in enumerate(landmarks[:17]):
        x, y, _, v = lm
        xy[pid] = (x, y)
        vis[pid] = v

    # 推导点：neck=肩中点，pelvis=髋中点，torso_vector=肩中点→髋中点
    neck = _midpoint_if(xy, 5, 6)
    pelvis = _midpoint_if(xy, 11, 12)
    schema["derived_points"] = {}
    if neck is not None:
        schema["derived_points"]["neck"] = {"x": round(neck[0], 4),
                                            "y": round(neck[1], 4), "z": 0.0}
    if pelvis is not None:
        schema["derived_points"]["pelvis"] = {"x": round(pelvis[0], 4),
                                              "y": round(pelvis[1], 4), "z": 0.0}
    if neck is not None and pelvis is not None:
        schema["derived_points"]["torso_vector"] = {
            "dx": round(pelvis[0] - neck[0], 4),
            "dy": round(pelvis[1] - neck[1], 4),
        }

    for joint in CANONICAL_JOINTS:
        entry = schema["joints"][joint["canonical"]]
        entry["present_in"].append("human")

        ang = _heuristic_angle(joint, xy, neck, pelvis)
        anchor = _anchor_point(joint, xy, pelvis)
        computed = False

        if ang is not None:
            entry["angle_deg"] = round(ang, 1)
            entry["angle_source"] = "heuristic_2d"
            entry["notes"] = ("2D 投影推导的无符号幅度角；轴方向映射交给 mapper，"
                              "不假装是机器人真实角度")
            computed = True
        if anchor is not None:
            entry["position"] = {"x": round(anchor[0], 4),
                                 "y": round(anchor[1], 4), "z": 0.0}
            computed = True

        if computed:
            conf = _joint_confidence(joint, vis)
            entry["confidence"] = round(conf, 3)
            entry["status"] = _status_from_conf(conf)
        # else：没信号 → 保持 na / confidence 0
    return schema


def _midpoint_if(xy: dict, a: int, b: int):
    """可见点 a/b 的中点；只有一侧可见时用那一侧的点（与 features._avg 同策略）。"""
    got = [xy[pid] for pid in (a, b) if pid in xy]
    if not got:
        return None
    xs = [p[0] for p in got]
    ys = [p[1] for p in got]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _joint_confidence(joint: dict, vis: dict) -> float:
    """该 DOF 的信号强度 = 源点可见度的最小值（任一源点缺失则不参与取 min）。"""
    confs = [vis[pid] for pid in joint["coco_ids"] if pid in vis]
    return min(confs) if confs else 0.0


def _status_from_conf(conf: float) -> str:
    if conf >= _CONF_OK:
        return "ok"
    if conf >= _CONF_LOW:
        return "low_conf"
    return "na"


def _anchor_point(joint: dict, xy: dict, pelvis):
    """该 DOF 的 position 锚点（归一化坐标）。返回 None 表示无信号。"""
    name = joint["canonical"]
    if name.startswith(("waist_", "chest_")):
        return pelvis                      # 腰/胸的位置锚点 ≈ 髋中点
    if name == "neck_pitch":
        return _midpoint_if(xy, 5, 6)      # 颈 ≈ 肩中点
    if name == "head_yaw":
        return xy.get(0)                   # 鼻
    if "shoulder_left" in name:
        return xy.get(5)
    if "shoulder_right" in name:
        return xy.get(6)
    if "elbow_left" in name:
        return xy.get(7)
    if "elbow_right" in name:
        return xy.get(8)
    if "wrist_left" in name:
        return xy.get(9)
    if "wrist_right" in name:
        return xy.get(10)
    if "hip_left" in name:
        return xy.get(11)
    if "hip_right" in name:
        return xy.get(12)
    if "knee_left" in name:
        return xy.get(13)
    if "knee_right" in name:
        return xy.get(14)
    if "ankle_left" in name:
        return xy.get(15)
    if "ankle_right" in name:
        return xy.get(16)
    return None


def _heuristic_angle(joint: dict, xy: dict, neck, pelvis):
    """人体侧能可靠弱推导的 DOF 角度（度）。

    原则：只推导 2D 投影里几何定义清晰、跟 features 已有角度同源的项
    （躯干/上臂/前臂/大腿/小腿相对竖直的倾角 + 肘/膝折角 + 鼻/肩中点
    水平偏移类 yaw 弱推导），且全部标注 heuristic_2d 由 mapper 处理方向。
    pitch 类返回无符号幅度；yaw 类带符号（图像平面方向，右/下为正）。
    推导不了的（roll / 腕 pitch / 踝 roll 等需深度或手/脚点）一律返回
    None（= na），不瞎编。
    """
    name = joint["canonical"]

    # 颈低头：鼻→颈 向量相对竖直（无符号幅度）
    if name == "neck_pitch":
        if xy.get(0) is not None and neck is not None:
            return _angle_from_vertical(xy[0], neck)

    # 头左右偏航：鼻相对耳连线的水平偏移 / 耳距（带符号，图像平面右偏为正）。
    # 单目 2D 弱推导：方向可靠，幅度受耳点噪声影响大，mapper 需做幅值标定。
    if name == "head_yaw":
        nose = xy.get(0)
        ears = [xy[pid] for pid in (3, 4) if pid in xy]
        if nose is not None and len(ears) == 2:
            width = abs(ears[1][0] - ears[0][0])
            if width > 1e-4:
                mid_x = (ears[0][0] + ears[1][0]) / 2.0
                return math.degrees(math.atan2(nose[0] - mid_x, width))
        return None

    # 躯干前倾：肩中点→髋中点 向量相对竖直（与 features torso_angle 同源）
    if name == "waist_pitch":
        if neck is not None and pelvis is not None:
            return _angle_from_vertical(neck, pelvis)

    # 腰扭转：肩中点相对髋中点的水平偏移 / 肩宽（带符号）。单目弱推导同上。
    if name == "waist_yaw":
        if neck is not None and pelvis is not None \
                and xy.get(5) is not None and xy.get(6) is not None:
            width = abs(xy[6][0] - xy[5][0])
            if width > 1e-4:
                return math.degrees(math.atan2(neck[0] - pelvis[0], width))
        return None

    left = "left" in name
    if name in ("shoulder_left_pitch", "shoulder_right_pitch"):
        sh, el = (5, 7) if left else (6, 8)
        if xy.get(sh) is not None and xy.get(el) is not None:
            return _angle_from_vertical(xy[sh], xy[el])   # 上臂相对竖直的倾角

    if name in ("elbow_left_pitch", "elbow_right_pitch"):
        sh, el, wr = (5, 7, 9) if left else (6, 8, 10)
        if all(xy.get(p) is not None for p in (sh, el, wr)):
            return _fold_deg(xy[sh], xy[el], xy[wr])      # 肘折角（伸直=0）

    if name in ("hip_left_pitch", "hip_right_pitch"):
        hip, knee = (11, 13) if left else (12, 14)
        if xy.get(hip) is not None and xy.get(knee) is not None:
            return _angle_from_vertical(xy[hip], xy[knee])  # 大腿相对竖直

    if name in ("knee_left_pitch", "knee_right_pitch"):
        hip, knee, ank = (11, 13, 15) if left else (12, 14, 16)
        if all(xy.get(p) is not None for p in (hip, knee, ank)):
            return _fold_deg(xy[hip], xy[knee], xy[ank])  # 膝折角（伸直=0）

    if name in ("ankle_left_pitch", "ankle_right_pitch"):
        knee, ank = (13, 15) if left else (14, 16)
        if xy.get(knee) is not None and xy.get(ank) is not None:
            return _angle_from_vertical(xy[knee], xy[ank])  # 小腿相对竖直

    return None


def _angle_from_vertical(p_from, p_to) -> float:
    """向量 p_from→p_to 与竖直轴 (0,1) 的夹角（度，0~90，无符号）。"""
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _fold_deg(a, b, c):
    """折线 a-b-c 在 b 处的内角相对 180° 的偏折（度；伸直=0，越弯越大）。"""
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    denom = math.hypot(*ab) * math.hypot(*cb)
    if denom < 1e-9:
        return None
    cos = (ab[0] * cb[0] + ab[1] * cb[1]) / denom
    cos = max(-1.0, min(1.0, cos))            # 数值容差
    return 180.0 - math.degrees(math.acos(cos))


# ---------------------------------------------------------------------------
# 机器人侧 adapter：/api/pose JSON → 规范 schema（只读，不写回）
# ---------------------------------------------------------------------------

def fetch_robot_pose(base_url: str = "http://127.0.0.1:8888",
                     timeout: float = 5.0) -> dict:
    """POST 机器人桥接的 /api/pose，返回原始 JSON dict。

    base_url: 桥接服务地址（robot_bridge.py 默认 8888 端口）。
    桥接未启动时抛 urllib.error.URLError，由调用方处理。
    """
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/pose",
        data=json.dumps({"fast": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def robot_adapter(pose_json, zero_values=None) -> dict:
    """把 /api/pose 返回的 JSON 翻译成规范 schema。

    参数:
        pose_json: robot_bridge POST /api/pose 的返回（含 joints / ok），
                   或 None（机器人读不到）。
        zero_values: zero.ini 的 {servoId: zeroValue}，缺省时该舵机零点用 2048
                     （与 HTML updatePose3D 一致）。angle_deg 一律为相对零点角。

    返回:
        规范 schema。每个 DOF 填 angle_deg（相对零点）、status、confidence；
        position 机器人侧没有（None）。
    """
    schema = empty_schema("robot")
    schema["timestamp"] = round(time.time(), 3)
    schema["source"]["model"] = "figurobot-bridge"
    schema["source"]["online"] = bool(pose_json and pose_json.get("ok"))

    joints = (pose_json or {}).get("joints") or {}
    zeros = zero_values or {}

    for joint in CANONICAL_JOINTS:
        name = joint["canonical"]
        entry = schema["joints"][name]
        entry["present_in"].append("robot")

        j = joints.get(str(joint["servo_id"])) or {}
        if not j.get("online"):
            # 该舵机在返回里缺了/离线：能区分"装配了但离线"和"压根没装配"
            entry["status"] = "offline" if str(joint["servo_id"]) in joints else "na"
            entry["confidence"] = 0.0
            continue
        err = j.get("error_code")
        if err not in (None, 0):
            entry["status"] = "error"
            entry["confidence"] = 0.6
            continue
        raw = j.get("raw")
        if raw is None:
            entry["status"] = "na"
            entry["confidence"] = 0.0
            continue

        zero = zeros.get(str(joint["servo_id"]), _DEFAULT_ZERO)
        entry["angle_deg"] = round((raw - zero) * _ZERO_SCALE, 1)
        entry["angle_source"] = "robot_read"
        entry["confidence"] = 1.0
        entry["status"] = "ok"
    return schema


def robot_schema_from_bridge(base_url: str = "http://127.0.0.1:8888",
                             timeout: float = 5.0,
                             zero_values=None) -> dict:
    """fetch_robot_pose + robot_adapter 的组合：直接读桥接拿规范 schema。"""
    return robot_adapter(fetch_robot_pose(base_url, timeout), zero_values)


# ---------------------------------------------------------------------------
# 自测：合成数据验证 human/robot adapter + 命名表（main_demo.py --selftest 汇总调用）
# ---------------------------------------------------------------------------

def _make_pose(pts: dict) -> dict:
    """构造一帧合成 pose：pts 里的点可见，其余点 visibility=0。"""
    landmarks = []
    for pid in range(17):
        if pid in pts:
            x, y = pts[pid]
            landmarks.append((x, y, 0.0, 1.0))
        else:
            landmarks.append((0.0, 0.0, 0.0, 0.0))
    return {"landmarks": landmarks, "image_size": (640, 480)}


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def selftest_schema() -> None:
    """合成数据自测：27 DOF 命名表 / human_adapter 推导 / robot_adapter 零点换算 /
    JSON 导出往返。"""
    # ---------- 1) 命名表 ----------
    _assert(len(CANONICAL_JOINTS) == 27, f"规范表应为 27 DOF，实际 {len(CANONICAL_JOINTS)}")
    _assert(len(SERVO_TO_CANONICAL) == 27 and len(CANONICAL_TO_SERVO) == 27,
            "舵机↔规范名双向映射应各 27 条")
    for j in CANONICAL_JOINTS:
        _assert(SERVO_TO_CANONICAL[j["servo_id"]] == j["canonical"], "servo 映射应自洽")
    # zh_name：27 个唯一，且 1:1 对应 figurobot SKELETON 的中文关节名
    zh = {j["canonical"]: j["zh_name"] for j in CANONICAL_JOINTS}
    _assert(len(set(zh.values())) == 27, "zh_name 应 27 个且唯一")
    _assert(zh["waist_yaw"] == "腰Y" and zh["waist_pitch"] == "腰P"
            and zh["neck_pitch"] == "颈P" and zh["head_yaw"] == "头Y"
            and zh["shoulder_left_pitch"] == "左肩P"
            and zh["hip_left_yaw"] == "左髋Y"
            and zh["ankle_right_roll"] == "右踝R",
            "zh_name 应 1:1 对应 figurobot SKELETON 的中文关节名")

    # ---------- 2) human_adapter：竖直站姿 ----------
    # 站直：肩(0.42/0.58, 0.30)、肘微张(0.38/0.62, 0.52)、腕(0.42/0.58, 0.72)、
    # 髋(0.45/0.55, 0.75)、膝(0.45/0.55, 0.88)、踝(0.45/0.55, 0.99)、鼻耳眼在上
    upright = {
        0: (0.50, 0.10), 1: (0.47, 0.10), 2: (0.53, 0.10),
        3: (0.44, 0.12), 4: (0.56, 0.12),
        5: (0.42, 0.30), 6: (0.58, 0.30),
        7: (0.38, 0.52), 8: (0.62, 0.52),
        9: (0.42, 0.72), 10: (0.58, 0.72),
        11: (0.45, 0.75), 12: (0.55, 0.75),
        13: (0.45, 0.88), 14: (0.55, 0.88),
        15: (0.45, 0.99), 16: (0.55, 0.99),
    }
    sch = human_adapter(_make_pose(upright))
    _assert(len(sch["joints"]) == 27, "human schema 应含 27 个 DOF")
    _assert(sch["source"]["kind"] == "human", "human adapter 的 source.kind 应为 human")

    dp = sch["derived_points"]
    _assert("neck" in dp and "pelvis" in dp and "torso_vector" in dp,
            "应产出 neck/pelvis/torso_vector 推导点")
    _assert(abs(dp["neck"]["x"] - 0.50) < 1e-3 and abs(dp["neck"]["y"] - 0.30) < 1e-3,
            f"neck 应为肩中点 (0.50, 0.30)，实际 {dp['neck']}")
    _assert(abs(dp["pelvis"]["x"] - 0.50) < 1e-3, "pelvis 应为髋中点 x=0.50")

    wp = sch["joints"]["waist_pitch"]
    _assert(wp["status"] == "ok" and wp["confidence"] >= 0.99, "竖直站姿 waist_pitch 应 ok")
    _assert(abs(wp["angle_deg"] - 0.0) <= 1.0, f"竖直站姿 waist_pitch 应≈0，实际 {wp['angle_deg']}°")
    _assert(wp["angle_source"] == "heuristic_2d", "人体侧角度应标 heuristic_2d")

    sh_l = sch["joints"]["shoulder_left_pitch"]
    _assert(5.0 < sh_l["angle_deg"] < 20.0, f"上臂微张 shoulder_left_pitch 应 ~10°，实际 {sh_l['angle_deg']}°")
    el_l = sch["joints"]["elbow_left_pitch"]
    _assert(10.0 < el_l["angle_deg"] < 35.0, f"肘微弯 elbow_left_pitch 应 ~21°，实际 {el_l['angle_deg']}°")

    # 正面站直：头 / 腰 yaw 弱推导应为 0（无水平偏移）
    _assert(abs(sch["joints"]["head_yaw"]["angle_deg"]) <= 1.0,
            f"正面 head_yaw 应≈0，实际 {sch['joints']['head_yaw']['angle_deg']}°")
    _assert(abs(sch["joints"]["waist_yaw"]["angle_deg"]) <= 1.0,
            f"正面 waist_yaw 应≈0，实际 {sch['joints']['waist_yaw']['angle_deg']}°")

    # 需要深度/手/脚点的 DOF 角度一律不推导（angle 为 na），但位置锚点仍可有：
    # 语义 = "知道这个关节在哪（position ok），但不知道它的旋转角（angle na）"。
    for na_name in ("waist_roll", "chest_roll", "chest_pitch",
                    "shoulder_left_roll", "wrist_left_pitch", "ankle_left_roll"):
        j = sch["joints"][na_name]
        _assert(j["angle_deg"] is None and j["angle_source"] is None,
                f"{na_name} 的角度应为 na（无可靠 2D 信号），实际 angle={j['angle_deg']}")

    # 转头 + 扭腰：鼻右移 → head_yaw 正；髋右移（肩带相对髋左偏）→ waist_yaw 负
    turned = dict(upright)
    turned[0] = (0.58, 0.10)               # 鼻右偏 0.08，耳距 0.12 → head_yaw≈34°
    turned[11] = (0.48, 0.75)              # 左髋右移 → 髋中点 0.53 > 肩中点 0.50
    turned[12] = (0.58, 0.75)
    tsch = human_adapter(_make_pose(turned))
    _assert(tsch["joints"]["head_yaw"]["angle_deg"] > 3.0,
            f"鼻右偏时 head_yaw 应为正，实际 {tsch['joints']['head_yaw']['angle_deg']}°")
    _assert(tsch["joints"]["waist_yaw"]["angle_deg"] < -3.0,
            f"肩带相对髋左偏时 waist_yaw 应为负，实际 {tsch['joints']['waist_yaw']['angle_deg']}°")
    _assert(sch["joints"]["waist_roll"]["status"] == "ok",
            "waist_roll 位置锚点=髋中点可见，status 应为 ok")
    _assert(sch["joints"]["wrist_left_pitch"]["position"] is not None,
            "wrist_left_pitch 角度 na 但位置锚点应存在（腕点可见）")

    # ---------- 3) human_adapter：无人 ----------
    empty = human_adapter(None)
    _assert(not empty["source"]["online"], "无人时应 source.online=False")
    _assert(all(j["status"] == "na" for j in empty["joints"].values()),
            "无人时所有 DOF 应 na")

    # ---------- 4) robot_adapter：/api/pose 假数据 + 零点换算 ----------
    fake = {"ok": True, "joints": {
        "1": {"raw": 2048, "angle": 180.0, "online": True,  "error_code": 0, "error_text": "正常"},
        "3": {"raw": 3000, "angle": 263.4, "online": True,  "error_code": 0, "error_text": "正常"},
        "6": {"raw": 2048, "angle": 180.0, "online": True,  "error_code": 0x10, "error_text": "未定义(0x10)"},
        "7": {"raw": 2048, "angle": 180.0, "online": False, "error_code": None, "error_text": "无响应"},
        "8": {"raw": None, "angle": None,   "online": True,  "error_code": 0, "error_text": "正常"},
        "10": {"raw": 2048, "angle": 180.0, "online": True,  "error_code": 0, "error_text": "正常"},  # 非装配 ID，应被忽略
    }}
    rsch = robot_adapter(fake, zero_values={"1": 1016, "3": 2048})
    _assert(rsch["source"]["kind"] == "robot", "robot adapter 的 source.kind 应为 robot")

    hy = rsch["joints"]["head_yaw"]
    _assert(hy["status"] == "ok" and hy["angle_source"] == "robot_read",
            "在线无错舵机应 ok + robot_read")
    _assert(abs(hy["angle_deg"] - (2048 - 1016) * _ZERO_SCALE) < 0.1,
            f"head_yaw 应做零点补偿 (2048-1016)*360/4095，实际 {hy['angle_deg']}°")

    sh = rsch["joints"]["shoulder_left_pitch"]
    _assert(abs(sh["angle_deg"] - (3000 - 2048) * _ZERO_SCALE) < 0.1,
            f"shoulder_left_pitch 零点补偿应 ~83.7°，实际 {sh['angle_deg']}°")

    _assert(rsch["joints"]["elbow_left_pitch"]["status"] == "error",
            "报错误码舵机应 status=error")
    _assert(rsch["joints"]["wrist_left_pitch"]["status"] == "na",
            "在线但读不到 raw 应 na")
    _assert(rsch["joints"]["shoulder_right_pitch"]["status"] == "offline",
            "装配位但离线应 offline（servo 7 online:false）")
    _assert(rsch["joints"]["hip_left_pitch"]["status"] == "na",
            "返回里根本没出现的舵机（servo 19）应 na，区别于 offline")

    # ---------- 5) 空 schema ----------
    _assert(empty_schema("robot")["source"]["kind"] == "robot", "empty_schema kind 应为 robot")

    # ---------- 6) JSON 导出往返（output.export_pose_json 已在此路径覆盖） ----------
    from output import export_pose_json
    text = export_pose_json(sch)
    back = json.loads(text)
    _assert(back["schema_version"] == sch["schema_version"], "JSON 导出应可往返")
    _assert(len(back["joints"]) == 27, "JSON 导出应含 27 个 DOF")

    print("  selftest_schema: OK")


if __name__ == "__main__":
    selftest_schema()
    print("pose_schema selftest: ALL PASSED")
