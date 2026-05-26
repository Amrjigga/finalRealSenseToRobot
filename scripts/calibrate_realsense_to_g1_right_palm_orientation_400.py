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

latest_imu = {
    "enabled": False,
    "initial_gravity": None,
    "gravity": None,
    "frame_count": 0,
}

latest_rs = {
    "landmarks_2d": None,
    "landmarks_3d_camera": None,
    "landmarks_3d_isaac": None,
    "palm_frame": None,
    "valid": False,
    "time": 0.0,
    "frame_count": 0,
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
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
        latest_imu["enabled"] = True
        print("[IMU] accel stream enabled for V1 gravity stabilization")


    profile = pipeline.start(config)
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

                valid_pts = [p for p in landmarks_3d_isaac if p[0] is not None]
                valid = len(valid_pts) >= 15

                if valid:
                    palm_frame = estimate_palm_frame_from_21_landmarks(landmarks_3d_isaac)

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
    Draw RealSense hand using the same alignment idea as the old VR script:

      world = robot_wrist_pos + (p_isaac - rs_origin) * scale + visual_offset

    This keeps the hand near the robot while preserving the hand's relative 3D shape
    and orientation. Position is only for visualization; calibration is orientation-only.
    """
    if draw is None:
        return

    pts = latest_rs.get("landmarks_3d_isaac")
    if not pts:
        return

    wrist = pts[0]
    if wrist is None or len(wrist) != 3 or any(v is None for v in wrist):
        return

    robot_wrist_pos = torch.as_tensor(robot_wrist_pos, dtype=torch.float32)
    device = robot_wrist_pos.device

    wrist_t = torch.tensor(wrist, dtype=torch.float32, device=device)

    # Same concept as quest_origin in old VR script:
    # lock the first valid wrist as the local origin.
    if latest_rs.get("rs_origin_isaac") is None:
        latest_rs["rs_origin_isaac"] = wrist_t.detach().cpu().tolist()
        print("[RS INIT] RealSense origin locked:", latest_rs["rs_origin_isaac"])

    rs_origin = torch.tensor(latest_rs["rs_origin_isaac"], dtype=torch.float32, device=device)

    # Visual offset above/near robot wrist. Tune only for display comfort.
    # Visual offset from robot wrist.
    # +X = forward, +Y/-Y = side, +Z = up.
    # Move hand forward and slightly down so it sits near the robot wrist,
    # not back by the shoulder.
    visual_offset = torch.tensor([0.28, 0.0, 0.08], dtype=torch.float32, device=device)
    base = robot_wrist_pos + visual_offset

    point_map = {}
    points = []

    for i, p in enumerate(pts):
        if p is None or len(p) != 3 or any(v is None for v in p):
            continue

        pp = torch.tensor(p, dtype=torch.float32, device=device)

        # VR-style relative alignment, with visual-only 90 deg left rotation.
        # Rotate around the locked RealSense origin, not around world.
        rel = pp - rs_origin

        # +90 deg around Isaac Z/up axis. If this is the wrong direction,
        # change the matrix to [[0, 1, 0], [-1, 0, 0], [0, 0, 1]].
        rot_left_90 = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        rel = rot_left_90 @ rel
        world = base + rel * scale

        tup = tuple(float(v) for v in world.detach().cpu().tolist())
        point_map[i] = tup
        points.append(tup)

    if points:
        draw.draw_points(points, [(1.0, 0.35, 0.0, 1.0)] * len(points), [8.0] * len(points))

    starts = []
    ends = []
    for a, b in MP_CONNECTIONS:
        if a in point_map and b in point_map:
            starts.append(point_map[a])
            ends.append(point_map[b])

    if starts:
        draw.draw_lines(starts, ends, [(1.0, 0.35, 0.0, 1.0)] * len(starts), [2.0] * len(starts))


@configclass
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
