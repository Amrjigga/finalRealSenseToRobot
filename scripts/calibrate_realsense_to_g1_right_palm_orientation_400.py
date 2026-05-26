# ============================================================
# RUN COMMAND - RealSense IMU-stabilized G1 right palm calibration
#
# cd ~/IsaacLab_5
# ./isaaclab.sh -p scripts/calibrate_realsense_to_g1_right_palm_orientation_400.py \
#   --device cuda:0 \
#   --num_samples 400 \
#   --rgb_width 640 \
#   --rgb_height 480 \
#   --fps 30 \
#   --save_images \
#   --use_imu_stabilization
#
# Notes:
# - Use a mounted/rigid camera, not handheld.
# - Save samples only when the hand viz looks stable.
# - Current IMU correction stabilizes camera rotation; remaining noise is mostly depth/landmark palm flips.
# ============================================================

import argparse
import json
import math
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=400)
parser.add_argument("--out_dir", type=str, default=str(Path.home() / "isaac-video-traj-replay" / "data" / "calib_realsense_g1_right_palm_400"))
parser.add_argument("--seed", type=int, default=4507)
parser.add_argument("--rgb_width", type=int, default=640)
parser.add_argument("--rgb_height", type=int, default=480)
parser.add_argument("--fps", type=int, default=30)

parser.add_argument(
    "--use_imu_stabilization",
    action="store_true",
    help="Use D435i accel/gravity to stabilize hand landmarks against camera tilt/roll/pitch.",
)

parser.add_argument("--show_cv", action="store_true", help="Show OpenCV RealSense/MediaPipe window.")
parser.add_argument("--no_realsense", action="store_true", help="Debug robot pose collection without RealSense.")
parser.add_argument("--save_images", action="store_true", help="Save rgb.png and depth.npy per sample.")
parser.add_argument("--hand", type=str, default="right", choices=["right"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG


# ============================================================
# Optional RealSense / MediaPipe imports
# ============================================================

RS_OK = False
MP_OK = False
CV_OK = False

if not args_cli.no_realsense:
    try:
        import pyrealsense2 as rs
        RS_OK = True
    except Exception as e:
        print("[WARN] pyrealsense2 import failed:", e)

    try:
        import cv2
        CV_OK = True
    except Exception as e:
        print("[WARN] cv2 import failed:", e)

    try:
        import mediapipe as mp
        MP_OK = True
    except Exception as e:
        print("[WARN] mediapipe import failed:", e)


# ============================================================
# Input thread
# ============================================================

MP_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

imu_compare_debug_ref = {
    "raw": None,
    "current": None,
    "alt": None,
    "last_print_t": 0.0,
}

robust_landmark_state = {
    "last_good": [None] * 21,
    "missing_count": [999] * 21,
    "last_wrist_offset_from_palm_center": None,
    "frame_idx": 0,
    "last_quality": {},
}

palm_continuity_state = {
    "last_good_frame": None,
    "rejected_count": 0,
}

orientation_debug_ref = {
    "quat": None,
    "last_print_t": 0.0,
}

latest_imu = {
    "enabled": False,
    "initial_gravity": None,
    "gravity": None,
    "frame_count": 0,
    "gyro_enabled": False,
    "orientation_R": None,
    "last_gyro_ts": None,
    "gyro_frame_count": 0,
    "R_depth_from_imu": None,
    "R_imu_from_depth": None,
}

latest_rs = {
    "landmarks_2d": None,
    "landmarks_3d_camera": None,
    "landmarks_3d_isaac": None,
    "palm_frame": None,
    "valid": False,
    "time": 0.0,
    "frame_count": 0,
    "gyro_enabled": False,
    "orientation_R": None,
    "last_gyro_ts": None,
    "gyro_frame_count": 0,
    "R_depth_from_imu": None,
    "R_imu_from_depth": None,
}

input_events = []


def input_thread():
    print("")
    print("INPUT CONTROLS:")
    print("  ENTER     = save current matched right-hand pose and advance")
    print("  s + ENTER = skip current robot pose")
    print("  r + ENTER = regenerate current pose")
    print("  w + ENTER = write/save progress now")
    print("  q + ENTER = write/save progress and quit")
    print("")
    while True:
        try:
            line = input().strip().lower()
            input_events.append(line)
        except EOFError:
            break


threading.Thread(target=input_thread, daemon=True).start()


# ============================================================
# Math helpers
# ============================================================

def quat_normalize(q):
    q = torch.as_tensor(q, dtype=torch.float32)
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)


def quat_wxyz_to_matrix(q):
    q = quat_normalize(q)
    w, x, y, z = q.tolist()

    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


def matrix_to_quat_wxyz(m):
    m = torch.as_tensor(m, dtype=torch.float32)
    m00 = float(m[0, 0])
    m01 = float(m[0, 1])
    m02 = float(m[0, 2])
    m10 = float(m[1, 0])
    m11 = float(m[1, 1])
    m12 = float(m[1, 2])
    m20 = float(m[2, 0])
    m21 = float(m[2, 1])
    m22 = float(m[2, 2])

    tr = m00 + m11 + m22

    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return quat_normalize(torch.tensor([w, x, y, z], dtype=torch.float32))


def rotz(deg):
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def realsense_camera_xyz_to_isaac(p_xyz, align_yaw_deg=-90.0):
    """
    RealSense camera/depth coords are roughly:
      x = image right
      y = image down
      z = forward

    Existing working replay mapping:
      --axis_map forward_z_flip_x
      --align_yaw_deg -90

    This makes:
      Isaac x forward-ish from camera z
      Isaac y flipped camera x
      Isaac z up from -camera y
    then applies yaw alignment.
    """
    x, y, z = p_xyz
    p = torch.tensor([z, -x, -y], dtype=torch.float32)
    return rotz(align_yaw_deg) @ p


def estimate_palm_frame_from_21_landmarks(points_isaac):
    """
    Estimate palm frame from MediaPipe 21 landmarks in Isaac coordinates.
    Returns None if required palm landmarks have invalid/missing depth.
    Required:
      0  wrist
      5  index MCP
      9  middle MCP
      17 pinky MCP
    """
    if points_isaac is None or len(points_isaac) < 21:
        return None

    required = [0, 5, 9, 17]
    for idx in required:
        p = points_isaac[idx]
        if p is None:
            return None
        if not isinstance(p, (list, tuple)):
            return None
        if len(p) != 3:
            return None
        if any(v is None for v in p):
            return None

    clean = []
    for p in points_isaac:
        if p is None or not isinstance(p, (list, tuple)) or len(p) != 3 or any(v is None for v in p):
            clean.append([0.0, 0.0, 0.0])
        else:
            clean.append([float(p[0]), float(p[1]), float(p[2])])

    pts = torch.as_tensor(clean, dtype=torch.float32)

    wrist = pts[0]
    index_mcp = pts[5]
    middle_mcp = pts[9]
    pinky_mcp = pts[17]

    across = index_mcp - pinky_mcp
    across = across / torch.clamp(torch.linalg.norm(across), min=1e-6)

    forward = middle_mcp - wrist
    forward = forward / torch.clamp(torch.linalg.norm(forward), min=1e-6)

    normal = torch.cross(across, forward, dim=0)
    normal = normal / torch.clamp(torch.linalg.norm(normal), min=1e-6)

    forward = torch.cross(normal, across, dim=0)
    forward = forward / torch.clamp(torch.linalg.norm(forward), min=1e-6)

    rot = torch.stack([across, forward, normal], dim=1)
    quat = matrix_to_quat_wxyz(rot)

    return {
        "quat_wxyz": quat.detach().cpu().tolist(),
        "palm_quat_isaac_wxyz": quat.detach().cpu().tolist(),

        "across_xyz": across.detach().cpu().tolist(),
        "forward_xyz": forward.detach().cpu().tolist(),
        "normal_xyz": normal.detach().cpu().tolist(),

        "palm_axes_isaac": {
            "across_xyz": across.detach().cpu().tolist(),
            "forward_xyz": forward.detach().cpu().tolist(),
            "normal_xyz": normal.detach().cpu().tolist(),
        },

        "rotmat": rot.detach().cpu().tolist(),
    }


def deproject_landmarks(depth_frame, intrinsics, landmarks, w, h):
    points_2d = []
    points_3d_cam = []
    points_3d_isaac = []

    for lm in landmarks:
        u = int(round(lm.x * w))
        v = int(round(lm.y * h))
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))

        depth_m = float(depth_frame.get_distance(u, v))

        # Small fallback search if exact pixel has zero depth.
        if depth_m <= 0.0:
            found = False
            for rad in [2, 4, 6, 8]:
                vals = []
                for yy in range(max(0, v - rad), min(h, v + rad + 1)):
                    for xx in range(max(0, u - rad), min(w, u + rad + 1)):
                        d = float(depth_frame.get_distance(xx, yy))
                        if 0.05 < d < 3.0:
                            vals.append(d)
                if vals:
                    depth_m = float(np.median(vals))
                    found = True
                    break
            if not found:
                depth_m = 0.0

        if depth_m > 0.0:
            p_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], depth_m)
            p_isaac = realsense_camera_xyz_to_isaac(p_cam)
            p_cam_list = [float(p_cam[0]), float(p_cam[1]), float(p_cam[2])]
            p_isaac_list = p_isaac.detach().cpu().tolist()
            valid = True
        else:
            p_cam_list = [None, None, None]
            p_isaac_list = [None, None, None]
            valid = False

        points_2d.append({
            "u": u,
            "v": v,
            "x_norm": float(lm.x),
            "y_norm": float(lm.y),
            "z_norm": float(lm.z),
            "depth_m": depth_m,
            "valid_depth": valid,
        })
        points_3d_cam.append(p_cam_list)
        points_3d_isaac.append(p_isaac_list)

    return points_2d, points_3d_cam, points_3d_isaac



def _np_norm(v, eps=1e-8):
    import numpy as _np
    v = _np.asarray(v, dtype=_np.float32)
    n = float(_np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def rotation_matrix_from_vectors_np(a, b):
    """
    Return R such that R @ a ~= b.
    Used for V1 IMU stabilization:
      current gravity direction -> initial gravity direction
    """
    import numpy as _np

    a = _np_norm(a)
    b = _np_norm(b)
    if a is None or b is None:
        return _np.eye(3, dtype=_np.float32)

    v = _np.cross(a, b)
    c = float(_np.dot(a, b))
    s = float(_np.linalg.norm(v))

    if s < 1e-6:
        if c > 0:
            return _np.eye(3, dtype=_np.float32)

        # 180-degree flip fallback.
        axis = _np.array([1.0, 0.0, 0.0], dtype=_np.float32)
        if abs(float(a[0])) > 0.9:
            axis = _np.array([0.0, 1.0, 0.0], dtype=_np.float32)
        v = _np.cross(a, axis)
        v = v / (_np.linalg.norm(v) + 1e-8)
        K = _np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=_np.float32,
        )
        return (_np.eye(3, dtype=_np.float32) + 2.0 * (K @ K)).astype(_np.float32)

    K = _np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=_np.float32,
    )

    R = _np.eye(3, dtype=_np.float32) + K + K @ K * ((1.0 - c) / (s * s))
    return R.astype(_np.float32)


def stabilize_landmarks_with_imu_camera_frame(landmarks_3d_camera):
    """
    V1 IMU stabilization.

    This removes camera tilt/roll/pitch by rotating current camera-frame
    landmarks back toward the initial gravity direction.

    It does NOT solve yaw/head-turn fully. That needs gyro integration V2.
    """
    import numpy as _np

    if not latest_imu.get("enabled"):
        return landmarks_3d_camera

    g0 = latest_imu.get("initial_gravity")
    g = latest_imu.get("gravity")

    if g0 is None or g is None:
        return landmarks_3d_camera

    R_corr = rotation_matrix_from_vectors_np(g, g0)

    out = []
    for p in landmarks_3d_camera:
        if p is None or len(p) != 3 or any(v is None for v in p):
            out.append(None)
            continue

        q = R_corr @ _np.asarray(p, dtype=_np.float32)
        out.append([float(q[0]), float(q[1]), float(q[2])])

    return out



def so3_exp_np(w):
    """
    Rodrigues exponential map.
    w = angular delta vector in radians.
    """
    import numpy as _np

    w = _np.asarray(w, dtype=_np.float32)
    theta = float(_np.linalg.norm(w))

    if theta < 1e-8:
        return _np.eye(3, dtype=_np.float32)

    k = w / theta
    K = _np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=_np.float32,
    )

    R = (
        _np.eye(3, dtype=_np.float32)
        + _np.sin(theta) * K
        + (1.0 - _np.cos(theta)) * (K @ K)
    )
    return R.astype(_np.float32)


def imu_v2_update_from_frames(frames):
    """
    V2 IMU update:
    - accel locks initial gravity and keeps latest gravity
    - gyro integrates full camera rotation, including yaw
    """
    import numpy as _np
    import pyrealsense2 as rs

    # Accel/gravity update
    accel_frame = frames.first_or_default(rs.stream.accel)
    if accel_frame:
        mf = accel_frame.as_motion_frame()
        if mf:
            md = mf.get_motion_data()
            g = [float(md.x), float(md.y), float(md.z)]
            latest_imu["gravity"] = g
            latest_imu["frame_count"] = int(latest_imu.get("frame_count", 0)) + 1

            if latest_imu.get("initial_gravity") is None:
                latest_imu["initial_gravity"] = g
                print("[IMU INIT] initial gravity locked:", g)

    # Gyro integration update
    gyro_frame = frames.first_or_default(rs.stream.gyro)
    if gyro_frame:
        mf = gyro_frame.as_motion_frame()
        if mf:
            ts = float(gyro_frame.get_timestamp()) * 0.001  # ms -> seconds
            md = mf.get_motion_data()

            # RealSense gyro is rad/s.
            w = -_np.array([float(md.x), float(md.y), float(md.z)], dtype=_np.float32)

            if latest_imu.get("orientation_R") is None:
                latest_imu["orientation_R"] = _np.eye(3, dtype=_np.float32)
                latest_imu["last_gyro_ts"] = ts
                latest_imu["gyro_frame_count"] = 0
                print("[IMU V2 INIT] gyro orientation locked")
                return

            last_ts = latest_imu.get("last_gyro_ts")
            if last_ts is None:
                latest_imu["last_gyro_ts"] = ts
                return

            dt = ts - float(last_ts)
            latest_imu["last_gyro_ts"] = ts

            # Reject weird timestamps.
            if dt <= 0.0 or dt > 0.10:
                return

            R = _np.asarray(latest_imu["orientation_R"], dtype=_np.float32)

            # Body-frame gyro update: R initial->current.
            dR = so3_exp_np(w * dt)
            R = dR @ R

            # Keep matrix numerically orthonormal.
            if int(latest_imu.get("gyro_frame_count", 0)) % 30 == 0:
                u, _, vh = _np.linalg.svd(R)
                R = (u @ vh).astype(_np.float32)

            latest_imu["orientation_R"] = R
            latest_imu["gyro_frame_count"] = int(latest_imu.get("gyro_frame_count", 0)) + 1


def stabilize_landmarks_with_imu_v2_camera_frame(landmarks_3d_camera):
    """
    V2 stabilization:
    Uses integrated gyro orientation to rotate current camera-frame landmarks
    back into the initial camera frame.

    This handles yaw better than V1.
    """
    import numpy as _np

    if not latest_imu.get("enabled"):
        return landmarks_3d_camera

    R = latest_imu.get("orientation_R")
    if R is None:
        # fallback to V1 gravity-only correction while gyro initializes
        return stabilize_landmarks_with_imu_camera_frame(landmarks_3d_camera)

    R = _np.asarray(R, dtype=_np.float32)

    # R maps initial IMU frame -> current IMU frame.
    # Landmarks are in depth/camera frame, not IMU frame.
    #
    # Correction:
    #   p_imu_current = R_imu_from_depth @ p_depth_current
    #   p_imu_initial = R.T @ p_imu_current
    #   p_depth_initial = R_depth_from_imu @ p_imu_initial
    R_depth_from_imu = latest_imu.get("R_depth_from_imu")
    R_imu_from_depth = latest_imu.get("R_imu_from_depth")

    if R_depth_from_imu is not None and R_imu_from_depth is not None:
        R_depth_from_imu = _np.asarray(R_depth_from_imu, dtype=_np.float32)
        R_imu_from_depth = _np.asarray(R_imu_from_depth, dtype=_np.float32)
        R_corr = R_depth_from_imu @ R.T @ R_imu_from_depth
    else:
        # fallback: old wrong-ish behavior if extrinsics are unavailable
        R_corr = R.T

    out = []
    for p in landmarks_3d_camera:
        if p is None or len(p) != 3 or any(v is None for v in p):
            out.append(None)
            continue

        q = R_corr @ _np.asarray(p, dtype=_np.float32)
        out.append([float(q[0]), float(q[1]), float(q[2])])

    return out



def imu_v2_update_from_motion_frame(frame):
    """
    Update IMU state from a single RealSense motion frame.

    This is the important V2 fix. RealSense motion frames often arrive as
    individual callback frames, not inside the normal video wait_for_frames()
    frameset.
    """
    import numpy as _np
    import pyrealsense2 as rs

    motion = frame.as_motion_frame()
    if not motion:
        return

    st = motion.get_profile().stream_type()
    md = motion.get_motion_data()

    if st == rs.stream.accel:
        g = [float(md.x), float(md.y), float(md.z)]
        latest_imu["gravity"] = g
        latest_imu["frame_count"] = int(latest_imu.get("frame_count", 0)) + 1

        if latest_imu.get("initial_gravity") is None:
            latest_imu["initial_gravity"] = g
            print("[IMU INIT] initial gravity locked:", g)

    elif st == rs.stream.gyro:
        ts = float(frame.get_timestamp()) * 0.001  # ms -> seconds

        # RealSense gyro is rad/s.
        # If correction is backwards, flip this sign later.
        w = -_np.array([float(md.x), float(md.y), float(md.z)], dtype=_np.float32)

        if latest_imu.get("orientation_R") is None:
            latest_imu["orientation_R"] = _np.eye(3, dtype=_np.float32)
            latest_imu["last_gyro_ts"] = ts
            latest_imu["gyro_frame_count"] = 0
            print("[IMU V2 INIT] gyro orientation locked")
            return

        last_ts = latest_imu.get("last_gyro_ts")
        if last_ts is None:
            latest_imu["last_gyro_ts"] = ts
            return

        dt = ts - float(last_ts)
        latest_imu["last_gyro_ts"] = ts

        if dt <= 0.0 or dt > 0.10:
            return

        R = _np.asarray(latest_imu["orientation_R"], dtype=_np.float32)
        dR = so3_exp_np(w * dt)
        R = dR @ R

        if int(latest_imu.get("gyro_frame_count", 0)) % 30 == 0:
            u, _, vh = _np.linalg.svd(R)
            R = (u @ vh).astype(_np.float32)

        latest_imu["orientation_R"] = R
        latest_imu["gyro_frame_count"] = int(latest_imu.get("gyro_frame_count", 0)) + 1


def start_realsense_motion_module_v2(accel_fps=100, gyro_fps=200, depth_profile=None):
    """
    Start D435i Motion Module directly with a callback.
    Keeps video/depth pipeline separate.
    """
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[IMU WARN] no RealSense device found for motion module")
        return None

    dev = devices[0]

    motion_sensor = None
    for sensor in dev.query_sensors():
        name = sensor.get_info(rs.camera_info.name)
        if "Motion" in name:
            motion_sensor = sensor
            break

    if motion_sensor is None:
        print("[IMU WARN] Motion Module sensor not found")
        return None

    profiles = list(motion_sensor.get_stream_profiles())

    accel_profile = None
    gyro_profile = None

    for prof in profiles:
        if prof.stream_type() == rs.stream.accel and prof.format() == rs.format.motion_xyz32f and prof.fps() == accel_fps:
            accel_profile = prof
        if prof.stream_type() == rs.stream.gyro and prof.format() == rs.format.motion_xyz32f and prof.fps() == gyro_fps:
            gyro_profile = prof

    if accel_profile is None:
        print("[IMU WARN] accel profile", accel_fps, "not found")
    if gyro_profile is None:
        print("[IMU WARN] gyro profile", gyro_fps, "not found")

    open_profiles = [p for p in [accel_profile, gyro_profile] if p is not None]
    if not open_profiles:
        print("[IMU WARN] no usable IMU profiles")
        return None

    latest_imu["enabled"] = True
    latest_imu["gyro_enabled"] = gyro_profile is not None

    # Get official RealSense extrinsics between Motion Module / gyro frame
    # and the depth camera frame. Landmarks are depth/camera-frame points,
    # while gyro integration is IMU-frame rotation.
    if depth_profile is not None and gyro_profile is not None:
        try:
            import numpy as _np
            ex = gyro_profile.get_extrinsics_to(depth_profile)
            R_depth_from_imu = _np.asarray(ex.rotation, dtype=_np.float32).reshape(3, 3)
            latest_imu["R_depth_from_imu"] = R_depth_from_imu
            latest_imu["R_imu_from_depth"] = R_depth_from_imu.T
            print("[IMU V2] depth<-imu extrinsics R:")
            print(R_depth_from_imu)
        except Exception as e:
            print("[IMU WARN] failed to read gyro->depth extrinsics:", e)

    motion_sensor.open(open_profiles)
    motion_sensor.start(lambda frame: imu_v2_update_from_motion_frame(frame))

    print("[IMU V2] Motion Module callback started:",
          "accel", accel_fps if accel_profile else None,
          "gyro", gyro_fps if gyro_profile else None)

    return motion_sensor



def quat_angle_deg_debug(q1, q2):
    import math
    import numpy as _np

    if q1 is None or q2 is None:
        return None

    q1 = _np.asarray(q1, dtype=_np.float32)
    q2 = _np.asarray(q2, dtype=_np.float32)

    n1 = float(_np.linalg.norm(q1))
    n2 = float(_np.linalg.norm(q2))
    if n1 < 1e-8 or n2 < 1e-8:
        return None

    q1 = q1 / n1
    q2 = q2 / n2

    # q and -q are the same rotation, so use abs(dot)
    dot = abs(float(_np.dot(q1, q2)))
    dot = max(-1.0, min(1.0, dot))

    return math.degrees(2.0 * math.acos(dot))


def debug_print_palm_orientation_stability(palm_frame):
    """
    Hold your hand still. Rotate the camera.
    If this angle stays small, saved orientation is stable and the problem is visual.
    If this angle changes a lot, IMU stabilization is not correcting the palm orientation yet.
    """
    import time

    if palm_frame is None:
        return

    q = palm_frame.get("palm_quat_isaac_wxyz", palm_frame.get("quat_wxyz"))
    if q is None:
        return

    if orientation_debug_ref["quat"] is None:
        orientation_debug_ref["quat"] = list(q)
        print("[ORI DEBUG] locked reference palm quat:", orientation_debug_ref["quat"])
        return

    now = time.time()
    if now - float(orientation_debug_ref.get("last_print_t", 0.0)) < 0.75:
        return

    orientation_debug_ref["last_print_t"] = now

    deg = quat_angle_deg_debug(orientation_debug_ref["quat"], q)

    print(
        "[ORI DEBUG]",
        "palm_delta_deg_from_ref=", round(float(deg), 2) if deg is not None else None,
        "imu_gyro_frames=", latest_imu.get("gyro_frame_count"),
        "imu_accel_frames=", latest_imu.get("frame_count"),
    )



def _lm_to_float3(p):
    import numpy as _np
    import torch

    if p is None:
        return None

    try:
        # torch tensor
        if hasattr(p, "detach"):
            arr = p.detach().cpu().numpy().astype("float32").reshape(-1)
        else:
            arr = _np.asarray(p, dtype=_np.float32).reshape(-1)

        if arr.shape[0] < 3:
            return None

        x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
        if not _np.isfinite([x, y, z]).all():
            return None

        return [x, y, z]
    except Exception:
        return None


def _dist3(a, b):
    import math
    if a is None or b is None:
        return None
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def _avg_points(points):
    vals = [p for p in points if p is not None]
    if not vals:
        return None
    n = float(len(vals))
    return [
        sum(float(p[0]) for p in vals) / n,
        sum(float(p[1]) for p in vals) / n,
        sum(float(p[2]) for p in vals) / n,
    ]


def robustify_landmarks_for_viz_and_palm(landmarks_3d_isaac, max_hold_frames=1, max_jump_m=1.00):
    """
    Robust layer after RealSense + IMU correction.

    Goal:
      - show hand viz even if non-critical points have bad depth
      - allow palm orientation when MCPs 5,9,17 are visible
      - do not hard-require wrist 0; estimate it from last wrist offset
      - gate sudden bad-depth jumps
    """
    state = robust_landmark_state
    state["frame_idx"] = int(state.get("frame_idx", 0)) + 1

    raw = []
    for i in range(21):
        p = None
        try:
            p = landmarks_3d_isaac[i]
        except Exception:
            p = None
        raw.append(_lm_to_float3(p))

    out = [None] * 21
    used_last_good = []
    rejected_jumps = []
    raw_valid = []

    # First pass: normal valid points + jump gate + last-good fallback.
    for i in range(21):
        p = raw[i]
        last = state["last_good"][i]

        if p is not None:
            d = _dist3(p, last)
            # If one point teleports too far in one frame, depth is probably bad.
            if d is not None and d > max_jump_m:
                p = None
                rejected_jumps.append(i)

        if p is not None:
            out[i] = p
            state["last_good"][i] = p
            state["missing_count"][i] = 0
            raw_valid.append(i)
        else:
            state["missing_count"][i] = int(state["missing_count"][i]) + 1
            if last is not None and state["missing_count"][i] <= max_hold_frames:
                out[i] = last
                used_last_good.append(i)

    # MCP palm center from index/middle/pinky MCPs.
    palm_center = _avg_points([out[5], out[9], out[17]])

    used_wrist_fallback = False

    # Update wrist offset when wrist + MCPs are available.
    if out[0] is not None and palm_center is not None:
        state["last_wrist_offset_from_palm_center"] = [
            float(out[0][0]) - float(palm_center[0]),
            float(out[0][1]) - float(palm_center[1]),
            float(out[0][2]) - float(palm_center[2]),
        ]

    # If wrist 0 is missing/out of FOV, estimate it from last hand shape.
    if out[0] is None and palm_center is not None and state.get("last_wrist_offset_from_palm_center") is not None:
        off = state["last_wrist_offset_from_palm_center"]
        out[0] = [
            float(palm_center[0]) + float(off[0]),
            float(palm_center[1]) + float(off[1]),
            float(palm_center[2]) + float(off[2]),
        ]
        used_wrist_fallback = True
        used_last_good.append(0)

    # Palm orientation only needs wrist/estimated wrist + MCPs.
    palm_critical = [0, 5, 9, 17]
    has_palm_orientation = all(out[i] is not None for i in palm_critical)

    # Viz can still be drawn with partial hand.
    valid_for_viz = sum(1 for x in raw if x is not None) >= 6

    quality = {
        "frame_idx": state["frame_idx"],
        "raw_valid_count": len(raw_valid),
        "filled_valid_count": sum(1 for x in out if x is not None),
        "raw_valid": raw_valid,
        "used_last_good_points": sorted(set(used_last_good)),
        "rejected_jump_points": sorted(set(rejected_jumps)),
        "used_wrist_fallback": bool(used_wrist_fallback),
        "has_palm_orientation": bool(has_palm_orientation),
        "valid_for_viz": bool(valid_for_viz),
        "palm_critical_status": {
            str(i): bool(out[i] is not None) for i in palm_critical
        },
    }

    state["last_quality"] = quality
    return out, quality


def apply_palm_orientation_continuity_gate(palm_frame, max_jump_deg=75.0):
    """
    Reject one-frame palm orientation flips.

    This handles cases where depth/landmark noise makes the palm axes suddenly
    flip by ~150-180 degrees even though the real hand did not move that much.

    For calibration:
      - display can keep last good palm frame
      - saving should avoid these rejected frames
    """
    if palm_frame is None:
        return None

    q = palm_frame.get("palm_quat_isaac_wxyz", palm_frame.get("quat_wxyz"))
    if q is None:
        return palm_frame

    last = palm_continuity_state.get("last_good_frame")
    if last is None:
        palm_frame["continuity_rejected"] = False
        palm_continuity_state["last_good_frame"] = palm_frame
        return palm_frame

    last_q = last.get("palm_quat_isaac_wxyz", last.get("quat_wxyz"))
    deg = quat_angle_deg_debug(last_q, q)

    if deg is not None and deg > max_jump_deg:
        palm_continuity_state["rejected_count"] = int(palm_continuity_state.get("rejected_count", 0)) + 1

        # Return last good palm orientation instead of the bad flipped one.
        stable = dict(last)
        stable["continuity_rejected"] = True
        stable["rejected_raw_delta_deg"] = float(deg)

        if palm_continuity_state["rejected_count"] % 10 == 1:
            print(
                "[PALM GATE] rejected palm flip:",
                round(float(deg), 2),
                "deg, total=",
                palm_continuity_state["rejected_count"],
            )

        return stable

    palm_frame["continuity_rejected"] = False
    palm_frame["raw_delta_from_last_good_deg"] = float(deg) if deg is not None else None
    palm_continuity_state["last_good_frame"] = palm_frame
    return palm_frame



def stabilize_landmarks_with_imu_v2_alt_camera_frame(landmarks_3d_camera):
    """
    Alternate IMU correction formula for diagnosis.

    Current formula:
      R_corr = R_depth_from_imu @ R.T @ R_imu_from_depth

    Alt formula:
      R_corr = R_depth_from_imu @ R @ R_imu_from_depth

    We compare live and keep whichever gives lower orientation drift.
    """
    import numpy as _np

    if not latest_imu.get("enabled"):
        return landmarks_3d_camera

    R = latest_imu.get("orientation_R")
    if R is None:
        return landmarks_3d_camera

    R = _np.asarray(R, dtype=_np.float32)

    R_depth_from_imu = latest_imu.get("R_depth_from_imu")
    R_imu_from_depth = latest_imu.get("R_imu_from_depth")

    if R_depth_from_imu is not None and R_imu_from_depth is not None:
        R_depth_from_imu = _np.asarray(R_depth_from_imu, dtype=_np.float32)
        R_imu_from_depth = _np.asarray(R_imu_from_depth, dtype=_np.float32)
        R_corr = R_depth_from_imu @ R @ R_imu_from_depth
    else:
        R_corr = R

    out = []
    for p in landmarks_3d_camera:
        if p is None or len(p) != 3 or any(v is None for v in p):
            out.append(None)
            continue

        q = R_corr @ _np.asarray(p, dtype=_np.float32)
        out.append([float(q[0]), float(q[1]), float(q[2])])

    return out



def safe_rs_point_to_isaac_for_debug(p):
    if p is None:
        return None
    try:
        if len(p) != 3:
            return None
        if p[0] is None or p[1] is None or p[2] is None:
            return None
        return realsense_camera_xyz_to_isaac(p)
    except Exception:
        return None



def _np_point3_for_palm_debug(p):
    import numpy as _np
    if p is None:
        return None
    try:
        if len(p) != 3:
            return None
        arr = _np.asarray([float(p[0]), float(p[1]), float(p[2])], dtype=_np.float32)
        if not _np.all(_np.isfinite(arr)):
            return None
        return arr
    except Exception:
        return None


def _quat_wxyz_from_rotmat_np(R):
    import numpy as _np
    R = _np.asarray(R, dtype=_np.float32)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])

    if tr > 0.0:
        S = (tr + 1.0) ** 0.5 * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = _np.asarray([qw, qx, qy, qz], dtype=_np.float32)
    n = float(_np.linalg.norm(q))
    if n < 1e-8 or not _np.all(_np.isfinite(q)):
        return None
    q = q / n
    return [float(v) for v in q]


def loose_estimate_palm_frame_from_21_landmarks(points):
    """
    Loose palm estimator for IMU debug/fallback.

    Uses only:
      0 wrist
      5 index MCP
      9 middle MCP
      17 pinky MCP
    """
    import numpy as _np

    try:
        wrist = _np_point3_for_palm_debug(points[0])
        index = _np_point3_for_palm_debug(points[5])
        middle = _np_point3_for_palm_debug(points[9])
        pinky = _np_point3_for_palm_debug(points[17])
    except Exception:
        return None

    if wrist is None or index is None or middle is None or pinky is None:
        return None

    across = pinky - index
    forward = middle - wrist

    an = float(_np.linalg.norm(across))
    fn = float(_np.linalg.norm(forward))
    if an < 1e-5 or fn < 1e-5:
        return None

    across = across / an
    forward = forward / fn

    normal = _np.cross(across, forward)
    nn = float(_np.linalg.norm(normal))
    if nn < 1e-5:
        return None
    normal = normal / nn

    # Re-orthogonalize forward
    forward = _np.cross(normal, across)
    forward = forward / max(float(_np.linalg.norm(forward)), 1e-8)

    R = _np.stack([across, forward, normal], axis=1)
    q = _quat_wxyz_from_rotmat_np(R)
    if q is None:
        return None

    return {
        "quat_wxyz": q,
        "palm_quat_isaac_wxyz": q,
        "across_xyz": [float(v) for v in across],
        "forward_xyz": [float(v) for v in forward],
        "normal_xyz": [float(v) for v in normal],
        "loose_debug_estimator": True,
    }


def debug_compare_imu_corrections(raw_landmarks_3d_camera, current_landmarks_3d_camera, alt_landmarks_3d_camera):
    """
    Live comparison:
      RAW     = no IMU correction
      CURRENT = current V2 correction
      ALT     = alternate R vs R.T correction

    Hold hand still and rotate camera.
    The best formula is the one with the lowest delta.
    """
    import time

    raw_isaac = [safe_rs_point_to_isaac_for_debug(p) for p in raw_landmarks_3d_camera]
    cur_isaac = [safe_rs_point_to_isaac_for_debug(p) for p in current_landmarks_3d_camera]
    alt_isaac = [safe_rs_point_to_isaac_for_debug(p) for p in alt_landmarks_3d_camera]

    raw_pf = loose_estimate_palm_frame_from_21_landmarks(raw_isaac)
    cur_pf = loose_estimate_palm_frame_from_21_landmarks(cur_isaac)
    alt_pf = loose_estimate_palm_frame_from_21_landmarks(alt_isaac)

    def crit_status(points):
        out = []
        for i in [0, 5, 9, 17]:
            ok = False
            try:
                pp = points[i]
                ok = pp is not None and len(pp) == 3 and pp[0] is not None and pp[1] is not None and pp[2] is not None
            except Exception:
                ok = False
            out.append(f"{i}:{'ok' if ok else 'bad'}")
        return ",".join(out)

    def get_q(pf):
        if pf is None:
            return None
        return pf.get("palm_quat_isaac_wxyz", pf.get("quat_wxyz"))

    raw_q = get_q(raw_pf)
    cur_q = get_q(cur_pf)
    alt_q = get_q(alt_pf)

    if imu_compare_debug_ref["raw"] is None and raw_q is not None:
        imu_compare_debug_ref["raw"] = list(raw_q)
        imu_compare_debug_ref["current"] = list(cur_q) if cur_q is not None else None
        imu_compare_debug_ref["alt"] = list(alt_q) if alt_q is not None else None
        print("[IMU CMP] locked refs")
        return

    now = time.time()
    if now - float(imu_compare_debug_ref.get("last_print_t", 0.0)) < 1.0:
        return
    imu_compare_debug_ref["last_print_t"] = now

    raw_deg = quat_angle_deg_debug(imu_compare_debug_ref["raw"], raw_q)
    cur_deg = quat_angle_deg_debug(imu_compare_debug_ref["current"], cur_q)
    alt_deg = quat_angle_deg_debug(imu_compare_debug_ref["alt"], alt_q)

    print(
        "[IMU CMP]",
        "raw=", round(float(raw_deg), 1) if raw_deg is not None else None,
        "current=", round(float(cur_deg), 1) if cur_deg is not None else None,
        "alt=", round(float(alt_deg), 1) if alt_deg is not None else None,
        "| crit raw", crit_status(raw_isaac),
        "| cur", crit_status(cur_isaac),
        "| alt", crit_status(alt_isaac),
    )


def realsense_thread():
    if args_cli.no_realsense:
        print("[RS] --no_realsense enabled, skipping RealSense thread")
        return

    if not (RS_OK and CV_OK and MP_OK):
        print("[RS] Missing pyrealsense2/cv2/mediapipe. Run with --no_realsense for robot-only debug.")
        return

    print("[RS] starting RealSense pipeline")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args_cli.rgb_width, args_cli.rgb_height, rs.format.bgr8, args_cli.fps)
    config.enable_stream(rs.stream.depth, args_cli.rgb_width, args_cli.rgb_height, rs.format.z16, args_cli.fps)
    if args_cli.use_imu_stabilization:
        latest_imu["enabled"] = True
        print("[IMU V2] using separate Motion Module callback")


    profile = pipeline.start(config)

    depth_profile_for_imu = None
    try:
        depth_profile_for_imu = profile.get_stream(rs.stream.depth)
    except Exception as e:
        print("[IMU WARN] could not get depth stream profile for extrinsics:", e)

    imu_motion_sensor = None
    if args_cli.use_imu_stabilization:
        imu_motion_sensor = start_realsense_motion_module_v2(
            accel_fps=100,
            gyro_fps=200,
            depth_profile=depth_profile_for_imu,
        )
    align = rs.align(rs.stream.color)

    # Official librealsense post-processing chain for D4XX:
    # depth -> disparity -> spatial -> temporal -> depth -> hole filling.
    # Mild settings to reduce holes without too much hand-motion smearing.
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.55)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 2)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.45)
    temporal.set_option(rs.option.filter_smooth_delta, 30)
    if hasattr(rs.option, "persistency_index"):
        temporal.set_option(rs.option.persistency_index, 2)
    else:
        print("[RS WARN] rs.option.persistency_index not available; skipping temporal persistency setting")

    hole_filling = rs.hole_filling_filter(1)


    depth_sensor = profile.get_device().first_depth_sensor()
    latest_rs["depth_scale"] = float(depth_sensor.get_depth_scale())

    try:
        mp_hands = mp.solutions.hands
    except AttributeError:
        from mediapipe.python.solutions import hands as mp_hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("[RS] RealSense + MediaPipe ready")

    try:
        while simulation_app.is_running():
            try:
                frames = pipeline.wait_for_frames(10000)

                if args_cli.use_imu_stabilization:

                    accel_frame = frames.first_or_default(rs.stream.accel)

                    if accel_frame:

                        mf = accel_frame.as_motion_frame()

                        if mf:

                            md = mf.get_motion_data()

                            g = [float(md.x), float(md.y), float(md.z)]

                            latest_imu["gravity"] = g

                            latest_imu["frame_count"] = int(latest_imu.get("frame_count", 0)) + 1

                            if latest_imu.get("initial_gravity") is None:

                                latest_imu["initial_gravity"] = g

                                print("[IMU INIT] initial gravity locked:", g)
            except RuntimeError as e:
                print("[RS WARN] wait_for_frames timeout:", e)
                time.sleep(0.1)
                continue
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            rgb_bgr = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data()).copy()
            h, w = rgb_bgr.shape[:2]
            intr = depth_frame.profile.as_video_stream_profile().intrinsics

            rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_rgb)

            landmarks_2d = None
            landmarks_3d_cam = None
            landmarks_3d_isaac = None
            palm_frame = None
            handedness_label = None
            valid = False

            vis = rgb_bgr.copy()

            if result.multi_hand_landmarks:
                chosen_idx = 0

                if result.multi_handedness:
                    for i, hd in enumerate(result.multi_handedness):
                        label = hd.classification[0].label.lower()
                        if label == "right":
                            chosen_idx = i
                            break

                    handedness_label = result.multi_handedness[chosen_idx].classification[0].label

                hand_lms = result.multi_hand_landmarks[chosen_idx]
                landmarks = hand_lms.landmark

                landmarks_2d, landmarks_3d_cam, landmarks_3d_isaac = deproject_landmarks(
                    depth_frame, intr, landmarks, w, h
                )

                # IMU V2 correction path.
                # deproject_landmarks gives raw camera-frame landmarks in landmarks_3d_cam.
                if args_cli.use_imu_stabilization and landmarks_3d_cam is not None:
                    raw_landmarks_3d_camera_for_imu_cmp = list(landmarks_3d_cam)

                    alt_landmarks_3d_camera_for_imu_cmp = stabilize_landmarks_with_imu_v2_alt_camera_frame(
                        raw_landmarks_3d_camera_for_imu_cmp
                    )

                    current_landmarks_3d_camera_for_imu_cmp = stabilize_landmarks_with_imu_v2_camera_frame(
                        raw_landmarks_3d_camera_for_imu_cmp
                    )

                    # Keep this debug for now. It confirms current IMU correction vs raw/alt.
                    debug_compare_imu_corrections(
                        raw_landmarks_3d_camera_for_imu_cmp,
                        current_landmarks_3d_camera_for_imu_cmp,
                        alt_landmarks_3d_camera_for_imu_cmp,
                    )

                    landmarks_3d_isaac = [
                        safe_rs_point_to_isaac_for_debug(pp)
                        for pp in current_landmarks_3d_camera_for_imu_cmp
                    ]

                # LIVE VIZ LANDMARKS:
                # Keep a copy of live RealSense/IMU-corrected points for drawing.
                # The visual hand shape should use these directly, not robust/fallback points.
                latest_rs["landmarks_3d_isaac_viz_live"] = [
                    _lm_to_float3(_p) for _p in landmarks_3d_isaac
                ]

                # NO-ESTIMATION MODE:
                # Use live RealSense/IMU-corrected landmarks directly.
                # No last-good fill, no wrist estimation, no jump correction.
                live_valid = []
                for _i, _p in enumerate(landmarks_3d_isaac):
                    _pp = _lm_to_float3(_p)
                    landmarks_3d_isaac[_i] = _pp
                    if _pp is not None:
                        live_valid.append(_i)

                palm_critical = [0, 5, 9, 17]
                palm_quality = {
                    "mode": "live_realsense_only_no_estimation",
                    "raw_valid_count": len(live_valid),
                    "filled_valid_count": len(live_valid),
                    "raw_valid": live_valid,
                    "used_last_good_points": [],
                    "rejected_jump_points": [],
                    "used_wrist_fallback": False,
                    "has_palm_orientation": all(landmarks_3d_isaac[i] is not None for i in palm_critical),
                    "valid_for_viz": len(live_valid) >= 6,
                    "palm_critical_status": {
                        str(i): bool(landmarks_3d_isaac[i] is not None) for i in palm_critical
                    },
                }

                valid = bool(palm_quality.get("valid_for_viz", False))

                if valid:
                    palm_frame = None

                    if palm_quality.get("has_palm_orientation", False):
                        palm_frame = estimate_palm_frame_from_21_landmarks(landmarks_3d_isaac)
                        if palm_frame is None:
                            palm_frame = loose_estimate_palm_frame_from_21_landmarks(landmarks_3d_isaac)

                    palm_frame = apply_palm_orientation_continuity_gate(palm_frame, max_jump_deg=75.0)

                    if palm_frame is not None:
                        palm_frame["quality"] = palm_quality

                    latest_rs["palm_frame"] = palm_frame
                    latest_rs["palm_quality"] = palm_quality
                    debug_print_palm_orientation_stability(palm_frame)

                for a, b in MP_CONNECTIONS:
                    pa = landmarks_2d[a]
                    pb = landmarks_2d[b]
                    cv2.line(vis, (pa["u"], pa["v"]), (pb["u"], pb["v"]), (0, 180, 255), 2)

                for p in landmarks_2d:
                    cv2.circle(vis, (p["u"], p["v"]), 3, (0, 255, 0), -1)

            latest_rs.update({
                "rgb": rgb_bgr.copy(),
                "depth": depth_np,
                "depth_scale": latest_rs["depth_scale"],
                "intrinsics": {
                    "width": int(intr.width),
                    "height": int(intr.height),
                    "ppx": float(intr.ppx),
                    "ppy": float(intr.ppy),
                    "fx": float(intr.fx),
                    "fy": float(intr.fy),
                    "model": str(intr.model),
                    "coeffs": [float(x) for x in intr.coeffs],
                },
                "landmarks_2d": landmarks_2d,
                "landmarks_3d_camera": landmarks_3d_cam,
                "landmarks_3d_isaac": landmarks_3d_isaac,
                "palm_frame": palm_frame,
                "handedness": handedness_label,
                "valid": bool(valid),
                "frame_count": latest_rs["frame_count"] + 1,
                "time": time.time(),
            })

            if args_cli.show_cv:
                status = "RIGHT HAND OK" if valid else "NO VALID RIGHT HAND"
                cv2.putText(vis, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if valid else (0, 0, 255), 2)
                cv2.putText(vis, "ENTER in terminal = save | q+ENTER = quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow("RealSense Right Hand Calibration", vis)
                cv2.waitKey(1)

    finally:
        pipeline.stop()
        if 'imu_motion_sensor' in locals() and imu_motion_sensor is not None:
            try:
                imu_motion_sensor.stop()
                imu_motion_sensor.close()
                print("[IMU V2] Motion Module stopped")
            except Exception as e:
                print("[IMU WARN] failed to stop Motion Module:", e)
        if CV_OK:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


threading.Thread(target=realsense_thread, daemon=True).start()


# ============================================================
# Debug draw
# ============================================================

def setup_debug_draw():
    try:
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        print("[VIZ] debug draw enabled")
        return draw
    except Exception as e:
        print("[VIZ] debug draw unavailable:", e)
        return None


def safe_clear_draw(draw):
    if draw is None:
        return
    try:
        draw.clear_points()
    except Exception:
        pass
    try:
        draw.clear_lines()
    except Exception:
        pass


def draw_axis_frame(draw, origin, quat, scale=0.12, z_offset=0.0):
    if draw is None:
        return

    # Keep origin on its current device, usually cuda:0.
    origin = torch.as_tensor(origin, dtype=torch.float32).clone()
    origin[2] += z_offset

    # Build rotation matrix on CPU, then move each axis to origin.device.
    # This matches the working old VR calibration script style.
    quat_cpu = torch.as_tensor(quat, dtype=torch.float32).detach().cpu()
    rot = quat_wxyz_to_matrix(quat_cpu)

    o = tuple(float(v) for v in origin.detach().cpu().tolist())

    x_end = origin + rot[:, 0].to(origin.device) * scale
    y_end = origin + rot[:, 1].to(origin.device) * scale
    z_end = origin + rot[:, 2].to(origin.device) * scale

    draw.draw_lines(
        [o, o, o],
        [
            tuple(float(v) for v in x_end.detach().cpu().tolist()),
            tuple(float(v) for v in y_end.detach().cpu().tolist()),
            tuple(float(v) for v in z_end.detach().cpu().tolist()),
        ],
        [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 0.2, 1.0, 1.0),
        ],
        [4.0, 4.0, 4.0],
    )


def draw_realsense_hand(draw, robot_wrist_pos, scale=0.75):
    """
    Live RealSense-first wrist-relative hand viz.

    This copies the good idea from the older recorded-RealSense viewer:
      world_wrist = robot wrist anchor + visual offset
      rel = landmark - live_wrist
      world_point = world_wrist + rel * scale

    Important:
      - Uses LIVE IMU-corrected RealSense landmarks for shape.
      - Does NOT use robust/fallback landmarks for visual finger shape.
      - Does NOT affect saved palm orientation/calibration data.
      - Keeps the locked 45-degree viz angle.
    """
    if draw is None:
        return

    landmarks = latest_rs.get("landmarks_3d_isaac_viz_live")
    if landmarks is None:
        landmarks = latest_rs.get("landmarks_3d_isaac")

    if landmarks is None or len(landmarks) < 21:
        return

    live = []
    for _p in landmarks:
        try:
            live.append(_lm_to_float3(_p))
        except Exception:
            live.append(None)

    raw_wrist = live[0]
    if raw_wrist is None:
        # Viz should not invent the whole hand. If wrist is not visible,
        # skip this frame instead of distorting the hand shape.
        return

    import torch

    wrist_t = torch.tensor(raw_wrist, dtype=torch.float32, device=args_cli.device)

    # Anchor the drawn hand near the robot wrist.
    base = robot_wrist_pos
    try:
        if len(base.shape) == 2:
            base = base[0]
    except Exception:
        pass

    visual_offset = torch.tensor([0.28, 0.0, 0.08], dtype=torch.float32, device=args_cli.device)
    world_wrist = base + visual_offset

    # VIZ ROTATION: rotate inside the palm plane, around the live palm normal.
    # This turns the hand left/right without flipping palm-up/palm-down.
    palm_plane_rot_deg = 90.0

    def _rotate_vec_around_axis(v, axis, deg):
        import math
        theta = math.radians(float(deg))
        c = math.cos(theta)
        ss = math.sin(theta)

        axis = axis / torch.clamp(torch.linalg.norm(axis), min=1e-6)

        # Rodrigues rotation formula.
        return (
            v * c
            + torch.cross(axis, v, dim=0) * ss
            + axis * torch.dot(axis, v) * (1.0 - c)
        )

    palm_normal_for_viz = None
    try:
        # Use the already-computed/gated palm frame, not raw live MCPs.
        # Raw live MCPs can be missing, which made the previous palm-normal rotation silently skip.
        palm_frame_for_viz = latest_rs.get("palm_frame")
        if palm_frame_for_viz is not None:
            q_raw = palm_frame_for_viz.get("palm_quat_isaac_wxyz", palm_frame_for_viz.get("quat_wxyz"))
            if q_raw is not None:
                q = torch.tensor(q_raw, dtype=torch.float32, device=args_cli.device)
                q = q / torch.clamp(torch.linalg.norm(q), min=1e-8)

                # quat_wxyz_to_matrix already exists in this script.
                r = quat_wxyz_to_matrix(q)

                # Palm frame columns are [across, forward, normal].
                # Normal is the axis that keeps palm-up/down unchanged.
                palm_normal_for_viz = r[:, 2].to(args_cli.device)
                palm_normal_for_viz = palm_normal_for_viz / torch.clamp(torch.linalg.norm(palm_normal_for_viz), min=1e-6)
    except Exception as _e:
        palm_normal_for_viz = None

    # IMPORTANT:
    # Do not let viz rotation jump back to OG when palm_frame disappears.
    # Cache the last good palm-normal axis and keep using it.
    if not hasattr(draw_realsense_hand, "_locked_viz_palm_axis"):
        draw_realsense_hand._locked_viz_palm_axis = None

    if palm_normal_for_viz is not None:
        if draw_realsense_hand._locked_viz_palm_axis is not None:
            prev = draw_realsense_hand._locked_viz_palm_axis.to(args_cli.device)

            # Prevent axis sign flips. +axis with +90 and -axis with +90 look opposite.
            if torch.dot(prev, palm_normal_for_viz) < 0:
                palm_normal_for_viz = -palm_normal_for_viz

            # Light smoothing so the visual rotation axis does not jitter.
            palm_normal_for_viz = 0.90 * prev + 0.10 * palm_normal_for_viz
            palm_normal_for_viz = palm_normal_for_viz / torch.clamp(torch.linalg.norm(palm_normal_for_viz), min=1e-6)

        draw_realsense_hand._locked_viz_palm_axis = palm_normal_for_viz.detach().clone()
    else:
        palm_normal_for_viz = draw_realsense_hand._locked_viz_palm_axis
        if palm_normal_for_viz is not None:
            palm_normal_for_viz = palm_normal_for_viz.to(args_cli.device)

    if not hasattr(draw_realsense_hand, "_dbg_count"):
        draw_realsense_hand._dbg_count = 0
    draw_realsense_hand._dbg_count += 1
    if draw_realsense_hand._dbg_count % 60 == 0:
        print("[VIZ ROT]", "axis_available_or_cached=", palm_normal_for_viz is not None, "deg=", palm_plane_rot_deg)

    hand_shape_scale = 0.65

    point_map = {}
    points = []

    for i, pp in enumerate(live):
        if pp is None:
            continue

        pp_t = torch.tensor(pp, dtype=torch.float32, device=args_cli.device)
        rel = pp_t - wrist_t

        if palm_normal_for_viz is not None:
            rel = _rotate_vec_around_axis(rel, palm_normal_for_viz, palm_plane_rot_deg)

        world = world_wrist + rel * hand_shape_scale
        tup = tuple(float(v) for v in world.detach().cpu().tolist())

        point_map[i] = tup
        points.append(tup)

    if points:
        draw.draw_points(points, [(1.0, 0.45, 0.0, 1.0)] * len(points), [7.0] * len(points))

    starts, ends = [], []
    for a, b in MP_CONNECTIONS:
        if a in point_map and b in point_map:
            starts.append(point_map[a])
            ends.append(point_map[b])

    if starts:
        draw.draw_lines(starts, ends, [(1.0, 0.45, 0.0, 1.0)] * len(starts), [2.5] * len(starts))

class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = G1_INSPIRE_FTP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device, use_fabric=False)
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view([2.5, -2.5, 1.6], [0.0, 0.0, 1.0])

scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.5))
sim.reset()

robot = scene["robot"]
draw = setup_debug_draw()

right_arm_joint_names = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

right_joint_ids, right_joint_names_found = robot.find_joints(right_arm_joint_names)
right_body_ids, right_body_names_found = robot.find_bodies(["right_wrist_yaw_link"])
right_body_id = right_body_ids[0]

print("right_joint_ids:", right_joint_ids)
print("right_joint_names:", right_joint_names_found)
print("right_body_id:", right_body_id, right_body_names_found)

default_q = robot.data.default_joint_pos.clone()
zero_v = torch.zeros_like(default_q)

base_q = default_q.clone()

name_to_local = {name: i for i, name in enumerate(right_joint_names_found)}


def set_local(q, name, value):
    jid = right_joint_ids[name_to_local[name]]
    q[:, jid] = value


# Comfortable fixed arm pose. Only right wrist changes.
set_local(base_q, "right_shoulder_pitch_joint", -0.20)
set_local(base_q, "right_shoulder_roll_joint", -0.35)
set_local(base_q, "right_shoulder_yaw_joint", 0.00)
set_local(base_q, "right_elbow_joint", 0.65)


# ============================================================
# Pose generation + resume state
# ============================================================

out_dir = Path(args_cli.out_dir)
samples_dir = out_dir / "samples"
poses_path = out_dir / "right_robot_poses_400.json"
dataset_path = out_dir / "calibration_dataset.json"
jsonl_path = out_dir / "calibration_dataset.jsonl"
state_path = out_dir / "state.json"

out_dir.mkdir(parents=True, exist_ok=True)
samples_dir.mkdir(parents=True, exist_ok=True)

ROLL_RANGE = (-1.40, 1.40)
PITCH_RANGE = (-1.20, 1.20)
YAW_RANGE = (-1.20, 1.20)


def generate_pose_list():
    rng = random.Random(args_cli.seed)
    poses = []

    coverage_bank = [
        {"mode": "palm_up_candidate", "roll": -1.15, "pitch": -0.75, "yaw": 0.00},
        {"mode": "palm_down_candidate", "roll": 1.15, "pitch": 0.75, "yaw": 0.00},
        {"mode": "palm_forward_candidate", "roll": -0.65, "pitch": 0.75, "yaw": -0.65},
        {"mode": "palm_forward_candidate_2", "roll": -0.65, "pitch": -0.75, "yaw": -0.65},
        {"mode": "side_left_candidate", "roll": -1.20, "pitch": 0.00, "yaw": 0.85},
        {"mode": "side_right_candidate", "roll": 1.20, "pitch": 0.00, "yaw": -0.85},
        {"mode": "diag_1", "roll": -0.90, "pitch": -0.80, "yaw": -0.80},
        {"mode": "diag_2", "roll": -0.90, "pitch": 0.80, "yaw": -0.80},
        {"mode": "diag_3", "roll": 0.90, "pitch": -0.80, "yaw": 0.80},
        {"mode": "diag_4", "roll": 0.90, "pitch": 0.80, "yaw": 0.80},
    ]

    for i in range(args_cli.num_samples):
        if i < len(coverage_bank):
            p = dict(coverage_bank[i])
            p["pose_index"] = i
            poses.append(p)
            continue

        # Mixture distribution: many normal randoms + some extreme wrist poses.
        r = rng.random()

        if r < 0.70:
            mode = "uniform_full_range"
            roll = rng.uniform(*ROLL_RANGE)
            pitch = rng.uniform(*PITCH_RANGE)
            yaw = rng.uniform(*YAW_RANGE)
        elif r < 0.90:
            mode = "extreme_roll"
            roll = rng.choice([-1, 1]) * rng.uniform(0.85, 1.40)
            pitch = rng.uniform(-1.10, 1.10)
            yaw = rng.uniform(-1.10, 1.10)
        else:
            mode = "diagonal_extreme"
            roll = rng.choice([-1, 1]) * rng.uniform(0.70, 1.35)
            pitch = rng.choice([-1, 1]) * rng.uniform(0.55, 1.15)
            yaw = rng.choice([-1, 1]) * rng.uniform(0.45, 1.15)

        poses.append({
            "pose_index": i,
            "mode": mode,
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        })

    return poses


if poses_path.exists():
    poses = json.loads(poses_path.read_text())
    print(f"[RESUME] loaded existing pose list: {poses_path}")
else:
    poses = generate_pose_list()
    poses_path.write_text(json.dumps({
        "description": "Deterministic random right-hand G1 wrist poses for RealSense matching calibration.",
        "hand": "right",
        "seed": args_cli.seed,
        "num_samples": args_cli.num_samples,
        "roll_range": ROLL_RANGE,
        "pitch_range": PITCH_RANGE,
        "yaw_range": YAW_RANGE,
        "poses": poses,
    }, indent=2))
    print(f"[INIT] wrote new pose list: {poses_path}")

samples = []
skipped = []

if dataset_path.exists():
    prev = json.loads(dataset_path.read_text())
    samples = prev.get("samples", [])
    skipped = prev.get("skipped", [])
    print(f"[RESUME] loaded dataset: saved={len(samples)} skipped={len(skipped)}")

done_pose_indices = set()
for s in samples:
    done_pose_indices.add(int(s["pose_index"]))
for s in skipped:
    done_pose_indices.add(int(s["pose_index"]))




# Normalize loaded pose list in case JSON was saved as a dict wrapper.
def _normalize_pose_list_for_resume(x):
    if isinstance(x, dict):
        for key in ("poses", "robot_poses", "right_robot_poses", "pose_list"):
            if key in x and isinstance(x[key], list):
                x = x[key]
                break
        else:
            # If it is a dict of pose_index -> pose dict, use values.
            vals = list(x.values())
            if vals and all(isinstance(v, dict) for v in vals):
                x = vals
            else:
                raise TypeError(f"Could not normalize pose list dict. Keys={list(x.keys())[:10]}")
    if not isinstance(x, list):
        raise TypeError(f"Pose list must be list after normalization, got {type(x)}")
    out = []
    for i, item in enumerate(x):
        if not isinstance(item, dict):
            raise TypeError(f"Pose item {i} is not dict: {type(item)} value={item}")
        if "pose_index" not in item:
            item["pose_index"] = i
        out.append(item)
    return out

poses = _normalize_pose_list_for_resume(poses)
print(f"[RESUME] normalized pose list: {len(poses)} poses")

def next_pose_index():
    for p in poses:
        if int(p["pose_index"]) not in done_pose_indices:
            return int(p["pose_index"])
    return None


pose_idx = next_pose_index()
if pose_idx is None:
    print("[DONE] all poses already completed")
    simulation_app.close()
    raise SystemExit(0)

pose = poses[pose_idx]


def apply_pose(pose_dict):
    q = base_q.clone()
    set_local(q, "right_wrist_roll_joint", pose_dict["roll"])
    set_local(q, "right_wrist_pitch_joint", pose_dict["pitch"])
    set_local(q, "right_wrist_yaw_joint", pose_dict["yaw"])
    robot.write_joint_state_to_sim(q, zero_v)
    robot.set_joint_position_target(q)
    return q


current_q = apply_pose(pose)


# ============================================================
# Save helpers
# ============================================================

def write_dataset(final=False):
    payload = {
        "description": "RIGHT HAND ONLY RealSense-to-G1 palm/wrist orientation calibration. User matches real right palm orientation to random G1 right palm orientation.",
        "hand": "right",
        "created_or_updated_time": time.time(),
        "final": bool(final),
        "target_num_samples": args_cli.num_samples,
        "num_saved_samples": len(samples),
        "num_skipped": len(skipped),
        "seed": args_cli.seed,
        "poses_path": str(poses_path),
        "samples_dir": str(samples_dir),
        "roll_range": ROLL_RANGE,
        "pitch_range": PITCH_RANGE,
        "yaw_range": YAW_RANGE,
        "axis_map": "forward_z_flip_x",
        "align_yaw_deg": -90,
        "samples": samples,
        "skipped": skipped,
    }

    dataset_path.write_text(json.dumps(payload, indent=2))

    with jsonl_path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    state_path.write_text(json.dumps({
        "updated_time": time.time(),
        "saved": len(samples),
        "skipped": len(skipped),
        "next_pose_index": next_pose_index(),
        "final": bool(final),
    }, indent=2))

    print(f"[SAVE] {dataset_path} saved={len(samples)} skipped={len(skipped)} final={final}")


def save_current_capture():
    if not args_cli.no_realsense and not latest_rs["valid"]:
        print("[WARN] no valid right-hand RealSense landmarks. Not saving. Try again.")
        return None

    robot_pos = robot.data.body_link_pos_w[0, right_body_id]
    robot_quat = robot.data.body_link_quat_w[0, right_body_id]
    robot_rot = quat_wxyz_to_matrix(robot_quat.detach().cpu())

    sample_id = f"sample_{len(samples):06d}_pose_{pose_idx:04d}"
    sample_dir = samples_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    rgb_rel = None
    depth_rel = None

    if args_cli.save_images and latest_rs["rgb"] is not None:
        try:
            import cv2
            rgb_path = sample_dir / "rgb.png"
            cv2.imwrite(str(rgb_path), latest_rs["rgb"])
            rgb_rel = str(rgb_path.relative_to(out_dir))
        except Exception as e:
            print("[WARN] failed saving rgb:", e)

    if args_cli.save_images and latest_rs["depth"] is not None:
        try:
            depth_path = sample_dir / "depth.npy"
            np.save(str(depth_path), latest_rs["depth"])
            depth_rel = str(depth_path.relative_to(out_dir))
        except Exception as e:
            print("[WARN] failed saving depth:", e)

    landmarks_2d_path = sample_dir / "landmarks_2d.json"
    landmarks_3d_path = sample_dir / "landmarks_3d.json"
    meta_path = sample_dir / "meta.json"

    landmarks_2d = latest_rs.get("landmarks_2d")
    landmarks_3d_camera = latest_rs.get("landmarks_3d_camera")
    landmarks_3d_isaac = latest_rs.get("landmarks_3d_isaac")

    landmarks_2d_path.write_text(json.dumps(landmarks_2d, indent=2))
    landmarks_3d_path.write_text(json.dumps({
        "camera_xyz_m": landmarks_3d_camera,
        "isaac_xyz_m": landmarks_3d_isaac,
    }, indent=2))

    sample = {
        "sample_id": sample_id,
        "sample_index": len(samples),
        "pose_index": int(pose_idx),
        "hand": "right",

        "robot_pose_mode": pose["mode"],
        "robot_right_wrist_joint_values_roll_pitch_yaw": [
            float(pose["roll"]),
            float(pose["pitch"]),
            float(pose["yaw"]),
        ],
        "robot_right_wrist_yaw_link_pos_w": robot_pos.detach().cpu().tolist(),
        "robot_right_wrist_yaw_link_quat_wxyz": robot_quat.detach().cpu().tolist(),
        "robot_right_wrist_yaw_link_rotmat": robot_rot.detach().cpu().tolist(),

        "realsense_valid": bool(latest_rs.get("valid")),
        "realsense_frame_count": int(latest_rs.get("frame_count", 0)),
        "realsense_frame_age_sec": None if latest_rs.get("time", 0.0) <= 0 else float(time.time() - latest_rs["time"]),
        "realsense_handedness": latest_rs.get("handedness"),
        "realsense_depth_scale": latest_rs.get("depth_scale"),
        "camera_intrinsics": latest_rs.get("intrinsics"),

        "realsense_right_palm_frame": latest_rs.get("palm_frame"),

        "rgb_path": rgb_rel,
        "depth_path": depth_rel,
        "landmarks_2d_path": str(landmarks_2d_path.relative_to(out_dir)),
        "landmarks_3d_path": str(landmarks_3d_path.relative_to(out_dir)),
        "meta_path": str(meta_path.relative_to(out_dir)),

        "axis_map": "forward_z_flip_x",
        "align_yaw_deg": -90,
        "saved_time": time.time(),
    }

    meta_path.write_text(json.dumps(sample, indent=2))
    return sample


print("")
print("================================================")
print("RIGHT HAND REALSENSE -> G1 PALM ORIENTATION CALIBRATION")
print("================================================")
print(f"Target samples: {args_cli.num_samples}")
print(f"Output folder: {out_dir}")
print(f"Saved samples already: {len(samples)}")
print(f"Skipped poses already: {len(skipped)}")
print(f"Next pose index: {pose_idx}")
print("")
print("Match your REAL RIGHT HAND palm orientation to the G1 RIGHT PALM.")
print("Do NOT worry about matching position. Orientation only.")
print("")
print("ENTER      save current matched sample and advance")
print("s + ENTER  skip current pose")
print("r + ENTER  regenerate current pose")
print("w + ENTER  write/save progress")
print("q + ENTER  save progress and quit")
print("================================================")
print("")
print(f"[POSE {pose_idx+1}/{args_cli.num_samples}] saved={len(samples)} skipped={len(skipped)} {pose}")


# ============================================================
# Main loop
# ============================================================

step = 0

while simulation_app.is_running():
    step += 1

    scene.update(sim.get_physics_dt())

    robot.set_joint_position_target(current_q)
    scene.write_data_to_sim()
    sim.step()

    if step % 2 == 0:
        safe_clear_draw(draw)

        robot_pos = robot.data.body_link_pos_w[0, right_body_id]
        robot_quat = robot.data.body_link_quat_w[0, right_body_id]

        draw_axis_frame(draw, robot_pos, robot_quat, scale=0.14, z_offset=0.0)
        draw_realsense_hand(draw, robot_pos, scale=0.75)

        palm_frame = latest_rs.get("palm_frame")
        if palm_frame is not None:
            palm_q_raw = palm_frame.get("palm_quat_isaac_wxyz", palm_frame.get("quat_wxyz"))
            if palm_q_raw is not None:
                palm_q = torch.tensor(palm_q_raw, dtype=torch.float32)
                draw_axis_frame(draw, robot_pos, palm_q, scale=0.14, z_offset=0.22)

    if input_events:
        cmd = input_events.pop(0)

        if cmd == "w":
            write_dataset(final=False)
            continue

        if cmd == "q":
            write_dataset(final=True)
            print("[QUIT] progress saved. You can resume later with the same command.")
            break

        if cmd == "r":
            # Replace this pose with a new random one, but keep same pose_index.
            rng = random.Random(int(time.time() * 1000) % 99999999)
            pose["mode"] = "manual_regenerated"
            pose["roll"] = float(rng.uniform(*ROLL_RANGE))
            pose["pitch"] = float(rng.uniform(*PITCH_RANGE))
            pose["yaw"] = float(rng.uniform(*YAW_RANGE))
            poses[pose_idx] = pose
            poses_path.write_text(json.dumps({
                "description": "Deterministic/random right-hand G1 wrist poses for RealSense matching calibration.",
                "hand": "right",
                "seed": args_cli.seed,
                "num_samples": args_cli.num_samples,
                "roll_range": ROLL_RANGE,
                "pitch_range": PITCH_RANGE,
                "yaw_range": YAW_RANGE,
                "poses": poses,
            }, indent=2))
            current_q = apply_pose(pose)
            print(f"[REGEN] pose {pose_idx+1}/{args_cli.num_samples}: {pose}")
            continue

        if cmd == "s":
            skipped.append({
                "pose_index": int(pose_idx),
                "pose": pose,
                "time": time.time(),
                "reason": "manual_skip",
            })
            done_pose_indices.add(int(pose_idx))
            write_dataset(final=False)
            print(f"[SKIP] pose {pose_idx+1}: {pose}")

            pose_idx = next_pose_index()
            if pose_idx is None:
                write_dataset(final=True)
                print("[DONE] all poses completed")
                break

            pose = poses[pose_idx]
            current_q = apply_pose(pose)
            print(f"[POSE {pose_idx+1}/{args_cli.num_samples}] saved={len(samples)} skipped={len(skipped)} {pose}")
            continue

        # ENTER = save
        sample = save_current_capture()

        if sample is not None:
            samples.append(sample)
            done_pose_indices.add(int(pose_idx))
            print(f"[SAVED] {len(samples)}/{args_cli.num_samples} pose_index={pose_idx} pose={pose}")

            # Save after every sample.
            write_dataset(final=False)

        if len(samples) >= args_cli.num_samples:
            write_dataset(final=True)
            print("[DONE] reached target sample count")
            break

        pose_idx = next_pose_index()
        if pose_idx is None:
            write_dataset(final=True)
            print("[DONE] all poses completed")
            break

        pose = poses[pose_idx]
        current_q = apply_pose(pose)
        print(f"[POSE {pose_idx+1}/{args_cli.num_samples}] saved={len(samples)} skipped={len(skipped)} {pose}")

simulation_app.close()
