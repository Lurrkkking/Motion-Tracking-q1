#!/usr/bin/env python3
"""Render a Q1 stand-up visualization in MuJoCo and export mp4 with ffmpeg.

The requested q1_22dof_box.xml is a compact kinematic tree, so this script
keeps that XML as the source of truth and augments it in memory with visual
meshes, approximate inertials, a floor, and lights before rendering.
"""

import argparse
import math
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco


XML_PATH = Path(
    "/root/autodl-tmp/ASAP_official/humanoidverse/data/robots/q1/q1_22dof_box.xml"
)
OUT_PATH = Path("/root/autodl-tmp/ASAP_official/sim2sim_outputs/q1_standup/q1_fallen_standup_hold.mp4")

DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]

DEFAULT_STAND = {
    "left_hip_pitch_joint": -0.2, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.5, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.2, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.5, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
    "waist_roll_joint": 0.0, "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.3,
    "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.3,
}

JOINT_LIMITS = {
    "left_hip_pitch_joint": (-3.0543, 1.5708, 36), "left_hip_roll_joint": (-0.69813, 1.5708, 36),
    "left_hip_yaw_joint": (-1.5708, 1.5708, 36), "left_knee_joint": (0.0, 2.4435, 36),
    "left_ankle_pitch_joint": (-0.7854, 0.43633, 22), "left_ankle_roll_joint": (-0.34907, 0.34907, 22),
    "right_hip_pitch_joint": (-3.0543, 1.5708, 36), "right_hip_roll_joint": (-1.5708, 0.69813, 36),
    "right_hip_yaw_joint": (-1.5708, 1.5708, 36), "right_knee_joint": (0.0, 2.4435, 36),
    "right_ankle_pitch_joint": (-0.7854, 0.43633, 22), "right_ankle_roll_joint": (-0.34907, 0.34907, 22),
    "waist_roll_joint": (-0.2618, 0.2618, 36), "waist_yaw_joint": (-1.5708, 1.5708, 36),
    "left_shoulder_pitch_joint": (-3.1416, 1.5708, 22), "left_shoulder_roll_joint": (-0.087266, 2.7925, 22),
    "left_shoulder_yaw_joint": (-1.5708, 1.5708, 22), "left_elbow_joint": (-0.87266, 1.6581, 22),
    "right_shoulder_pitch_joint": (-3.1416, 1.5708, 22), "right_shoulder_roll_joint": (-2.7925, 0.087266, 22),
    "right_shoulder_yaw_joint": (-1.5708, 1.5708, 22), "right_elbow_joint": (-0.87266, 1.6581, 22),
}

BODY_MASS = {
    "pelvis": 1.7, "waist_roll_link": 0.12, "torso_link": 4.1,
    "left_hip_pitch_link": 0.7, "left_hip_roll_link": 0.2, "left_hip_yaw_link": 0.76,
    "left_knee_link": 1.2, "left_ankle_pitch_link": 0.4, "left_ankle_roll_link": 0.34,
    "right_hip_pitch_link": 0.7, "right_hip_roll_link": 0.2, "right_hip_yaw_link": 0.76,
    "right_knee_link": 1.2, "right_ankle_pitch_link": 0.4, "right_ankle_roll_link": 0.34,
    "left_shoulder_pitch_link": 0.37, "left_shoulder_roll_link": 0.39,
    "left_shoulder_yaw_link": 0.4, "left_elbow_link": 0.31,
    "right_shoulder_pitch_link": 0.37, "right_shoulder_roll_link": 0.39,
    "right_shoulder_yaw_link": 0.4, "right_elbow_link": 0.31,
}

BODY_BOX = {
    "pelvis": "0.05 0.06 0.04 0 0 0.03", "waist_roll_link": "0.04 0.06 0.03 0 0 0.02",
    "torso_link": "0.06 0.09 0.12 -0.01 0 0.11",
    "left_hip_pitch_link": "0.035 0.035 0.035 0 0.02 -0.02", "right_hip_pitch_link": "0.035 0.035 0.035 0 -0.02 -0.02",
    "left_hip_roll_link": "0.035 0.035 0.035 0 0 -0.03", "right_hip_roll_link": "0.035 0.035 0.035 0 0 -0.03",
    "left_hip_yaw_link": "0.04 0.04 0.06 0 0 -0.06", "right_hip_yaw_link": "0.04 0.04 0.06 0 0 -0.06",
    "left_knee_link": "0.04 0.04 0.09 0.01 0 -0.09", "right_knee_link": "0.04 0.04 0.09 0.01 0 -0.09",
    "left_ankle_pitch_link": "0.04 0.035 0.035 0 0 -0.02", "right_ankle_pitch_link": "0.04 0.035 0.035 0 0 -0.02",
    "left_ankle_roll_link": "0.10 0.035 0.025 0.03 0 -0.025", "right_ankle_roll_link": "0.10 0.035 0.025 0.03 0 -0.025",
    "left_shoulder_pitch_link": "0.035 0.035 0.035 0 0.02 0", "right_shoulder_pitch_link": "0.035 0.035 0.035 0 -0.02 0",
    "left_shoulder_roll_link": "0.035 0.03 0.06 0 0 -0.02", "right_shoulder_roll_link": "0.035 0.03 0.06 0 0 -0.02",
    "left_shoulder_yaw_link": "0.03 0.03 0.05 0 0 -0.04", "right_shoulder_yaw_link": "0.03 0.03 0.05 0 0 -0.04",
    "left_elbow_link": "0.10 0.025 0.025 0.10 0 -0.01", "right_elbow_link": "0.10 0.025 0.025 0.10 0 -0.01",
}


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def augment_xml(xml_path: Path) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_dir = xml_path.parent

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler", {"angle": "radian"})
        root.insert(0, compiler)
    compiler.set("meshdir", str(xml_dir / "meshes"))

    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.insert(1, visual)
    if visual.find("global") is None:
        ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(2, asset)
    for child in list(asset):
        asset.remove(child)

    for stl in sorted((xml_dir / "meshes").glob("*.STL")):
        ET.SubElement(asset, "mesh", {"name": stl.stem, "file": stl.name})
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient", "rgb1": "0.86 0.90 0.94", "rgb2": "0.55 0.63 0.70",
        "width": "512", "height": "512",
    })
    ET.SubElement(asset, "texture", {
        "name": "groundplane", "type": "2d", "builtin": "checker", "mark": "edge",
        "rgb1": "0.24 0.27 0.28", "rgb2": "0.18 0.20 0.21", "markrgb": "0.75 0.75 0.75",
        "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane", "texuniform": "true", "texrepeat": "6 6",
    })

    worldbody = root.find("worldbody")
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "0 0 0.05", "material": "groundplane",
        "condim": "3", "friction": "1.0 0.005 0.0001",
    })
    ET.SubElement(worldbody, "light", {"directional": "true", "pos": "2 -3 4", "dir": "-2 3 -4", "diffuse": "0.8 0.8 0.8"})
    ET.SubElement(worldbody, "light", {"pos": "-1 2 2.5", "diffuse": "0.35 0.35 0.35"})

    def enhance_body(body: ET.Element) -> None:
        name = body.get("name", "")
        if not any(child.tag == "inertial" for child in body):
            mass = BODY_MASS.get(name, 0.3)
            inertia = max(mass * 0.01, 1e-4)
            body.insert(0, ET.Element("inertial", {
                "pos": "0 0 0", "mass": f"{mass:.6f}",
                "diaginertia": f"{inertia:.6f} {inertia:.6f} {inertia:.6f}",
            }))
        if name in BODY_BOX:
            parts = BODY_BOX[name].split()
            ET.SubElement(body, "geom", {
                "type": "box", "size": " ".join(parts[:3]), "pos": " ".join(parts[3:6]),
                "contype": "1", "conaffinity": "1", "condim": "3", "friction": "1.0 0.005 0.0001",
                "rgba": "0.15 0.17 0.18 0.18",
            })
        if (xml_path.parent / "meshes" / f"{name}.STL").is_file():
            ET.SubElement(body, "geom", {
                "type": "mesh", "mesh": name, "contype": "0", "conaffinity": "0",
                "group": "1", "density": "0", "rgba": "0.72 0.72 0.70 1",
            })
        for child in body:
            if child.tag == "joint" and child.get("name") in JOINT_LIMITS:
                lo, hi, effort = JOINT_LIMITS[child.get("name")]
                child.set("range", f"{lo} {hi}")
                child.set("limited", "true")
                child.set("armature", "0.004")
                child.set("damping", "0.02")
            if child.tag == "body":
                enhance_body(child)

    for body in worldbody.findall("body"):
        enhance_body(body)

    actuator = root.find("actuator")
    if actuator is not None:
        for motor in actuator.findall("motor"):
            name = motor.get("joint") or motor.get("name")
            if name in JOINT_LIMITS:
                effort = JOINT_LIMITS[name][2]
                motor.set("ctrllimited", "true")
                motor.set("ctrlrange", f"-{effort} {effort}")
                motor.set("forcelimited", "true")
                motor.set("forcerange", f"-{effort} {effort}")

    return ET.tostring(root, encoding="unicode")


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def fallen_pose() -> dict[str, float]:
    pose = dict(DEFAULT_STAND)
    pose.update({
        "left_hip_pitch_joint": -1.1, "left_knee_joint": 1.9, "left_ankle_pitch_joint": -0.45,
        "right_hip_pitch_joint": -0.65, "right_knee_joint": 1.35, "right_ankle_pitch_joint": -0.35,
        "left_hip_roll_joint": 0.28, "right_hip_roll_joint": -0.28,
        "left_shoulder_pitch_joint": 0.9, "right_shoulder_pitch_joint": 0.65,
        "left_shoulder_roll_joint": 0.65, "right_shoulder_roll_joint": -0.65,
        "left_elbow_joint": 1.0, "right_elbow_joint": 1.0,
    })
    return pose


def pose_at(t: float, stand_time: float) -> dict[str, float]:
    a = ease(t / stand_time)
    start = fallen_pose()
    return {name: (1.0 - a) * start[name] + a * DEFAULT_STAND[name] for name in DOF_NAMES}


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    return (math.sin(theta_0 - theta) / sin_theta_0) * q0 + (math.sin(theta) / sin_theta_0) * q1


def root_pose_at(t: float, stand_time: float) -> tuple[np.ndarray, np.ndarray]:
    a = ease(t / stand_time)
    pos0 = np.array([-0.16, 0.0, 0.13], dtype=np.float64)
    pos1 = np.array([0.0, 0.0, 0.42], dtype=np.float64)
    quat0 = np.array([math.cos(math.pi / 4.0), 0.0, math.sin(math.pi / 4.0), 0.0], dtype=np.float64)
    quat1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return (1.0 - a) * pos0 + a * pos1, quat_slerp(quat0, quat1, a)


def write_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = [
        ffmpeg_exe(), "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "18", str(out_path),
    ]
    raw = b"".join(np.ascontiguousarray(frame).tobytes() for frame in frames)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(input=raw)
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=XML_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stand-time", type=float, default=3.0)
    parser.add_argument("--hold-time", type=float, default=2.0)
    args = parser.parse_args()

    xml_text = augment_xml(args.xml.resolve())
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_text)
        augmented_path = Path(f.name)

    try:
        model = mujoco.MjModel.from_xml_path(str(augmented_path))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.0, 0.0, 0.34]
        camera.distance = 1.65
        camera.azimuth = 135
        camera.elevation = -14

        qpos_ids = {name: model.jnt_qposadr[model.joint(name).id] for name in DOF_NAMES}
        total_time = args.stand_time + args.hold_time
        nframes = int(round(total_time * args.fps))
        frames = []

        for i in range(nframes):
            t = i / args.fps
            pose = pose_at(t, args.stand_time)
            root_pos, root_quat = root_pose_at(t, args.stand_time)
            data.qpos[:] = 0.0
            data.qpos[0:3] = root_pos
            data.qpos[3:7] = root_quat
            for name, value in pose.items():
                data.qpos[qpos_ids[name]] = value
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())

        write_video(frames, args.out, args.fps)
        print(f"[OK] wrote {args.out}")
        print(f"[XML] source={args.xml}")
        print(f"[XML] augmented_temp={augmented_path}")
    finally:
        try:
            augmented_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
