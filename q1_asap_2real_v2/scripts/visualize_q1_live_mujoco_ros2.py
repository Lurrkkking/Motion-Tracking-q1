#!/usr/bin/env python3
"""
Read-only Q1 live state mirror in MuJoCo.

This script subscribes to Q1 joint state and IMU topics, then mirrors the
measured robot state into a simplified MuJoCo model. It never publishes joint
commands and is intended only for visualization/debugging.
"""

import argparse
import os
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    mujoco = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
except ImportError:
    rclpy = None
    Node = object
    DurabilityPolicy = None
    HistoryPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None

try:
    from aimdk_msgs.msg import JointStateArray
except ImportError:
    JointStateArray = None

try:
    from sensor_msgs.msg import Imu
except ImportError:
    Imu = None


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
NUM_ACTIONS = 22


def default_urdf_arg() -> str:
    q1_robot_urdf = WORKSPACE_ROOT / "q1_robot" / "q1_22dof_box.urdf"
    if q1_robot_urdf.is_file():
        return str(q1_robot_urdf)
    return "robots/q1/q1_22dof_box.urdf"


def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return str(REPO_ROOT / path)


def read_conf(config_file: str) -> Dict[str, object]:
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {
        "dof_names": list(config.get("joint_names", config.get("dof_names", []))),
        "default_dof_pos": np.array(config["default_dof_pos"], dtype=np.float32),
    }


def parse_vec(value: Optional[str], length: int, default: Tuple[float, ...]) -> List[float]:
    if not value:
        return list(default)
    parts = [float(x) for x in value.split()]
    if len(parts) != length:
        raise ValueError(f"Expected {length} values, got {value!r}")
    return parts


def fmt_vec(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def quat_rotate_inverse_wxyz(q, v):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return v.copy()
    q = q / norm
    qw = q[0]
    qvec = q[1:4]
    return (
        v * (2.0 * qw * qw - 1.0)
        - np.cross(qvec, v) * qw * 2.0
        + qvec * np.dot(qvec, v) * 2.0
    )


def normalize_quat_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def geom_for_body(body_name: str) -> Tuple[str, str, str]:
    name = body_name.lower()
    if "pelvis" in name:
        return "box", "0.08 0.05 0.06", "mat_pelvis"
    if "torso" in name or "waist" in name:
        return "box", "0.08 0.045 0.09", "mat_torso"
    if "ankle_roll" in name:
        return "box", "0.10 0.035 0.025", "mat_foot"
    if "ankle" in name:
        return "capsule", "0.025 0.045", "mat_leg"
    if any(part in name for part in ("hip", "knee")):
        return "capsule", "0.032 0.09", "mat_leg"
    if any(part in name for part in ("shoulder", "elbow")):
        return "capsule", "0.025 0.075", "mat_arm"
    return "sphere", "0.035", "mat_default"


def mesh_name_for_link(link_name: str) -> str:
    return f"mesh_{link_name}"


def resolve_urdf_mesh_path(urdf_dir: Path, filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return (urdf_dir / path).resolve()


def parse_link_visual_meshes(root: ET.Element, urdf_path: str) -> Dict[str, Dict[str, object]]:
    urdf_dir = Path(urdf_path).resolve().parent
    visual_meshes = {}
    for link in root.findall("link"):
        link_name = link.get("name")
        visual = link.find("visual")
        if not link_name or visual is None:
            continue
        geometry = visual.find("geometry")
        mesh = geometry.find("mesh") if geometry is not None else None
        if mesh is None or mesh.get("filename") is None:
            continue
        origin = visual.find("origin")
        mesh_path = resolve_urdf_mesh_path(urdf_dir, mesh.get("filename"))
        visual_meshes[link_name] = {
            "file": str(mesh_path),
            "xyz": parse_vec(origin.get("xyz") if origin is not None else None, 3, (0.0, 0.0, 0.0)),
            "rpy": parse_vec(origin.get("rpy") if origin is not None else None, 3, (0.0, 0.0, 0.0)),
            "exists": mesh_path.is_file(),
        }
    return visual_meshes


def add_body_geom(
    body_elem: ET.Element,
    body_name: str,
    visual_meshes: Dict[str, Dict[str, object]],
    use_meshes: bool,
) -> None:
    visual = visual_meshes.get(body_name)
    if use_meshes and visual and visual["exists"]:
        ET.SubElement(
            body_elem,
            "geom",
            {
                "name": f"{body_name}_mesh_vis",
                "type": "mesh",
                "mesh": mesh_name_for_link(body_name),
                "pos": fmt_vec(visual["xyz"]),
                "euler": fmt_vec(visual["rpy"]),
                "rgba": "0.82 0.84 0.88 1",
                "contype": "0",
                "conaffinity": "0",
            },
        )
        return

    geom_type, size, material = geom_for_body(body_name)
    ET.SubElement(
        body_elem,
        "geom",
        {
            "name": f"{body_name}_vis",
            "type": geom_type,
            "size": size,
            "rgba": "0.75 0.78 0.86 1",
            "material": material,
            "contype": "0",
            "conaffinity": "0",
        },
    )


def parse_urdf_joints(urdf_path: str):
    root = ET.parse(urdf_path).getroot()
    visual_meshes = parse_link_visual_meshes(root, urdf_path)
    links = [link.get("name") for link in root.findall("link") if link.get("name")]
    child_links = set()
    joints_by_parent: Dict[str, List[Dict[str, object]]] = {}

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if not parent_name or not child_name:
            continue
        child_links.add(child_name)
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        info = {
            "name": joint.get("name", ""),
            "type": joint.get("type", "fixed"),
            "parent": parent_name,
            "child": child_name,
            "xyz": parse_vec(origin.get("xyz") if origin is not None else None, 3, (0.0, 0.0, 0.0)),
            "rpy": parse_vec(origin.get("rpy") if origin is not None else None, 3, (0.0, 0.0, 0.0)),
            "axis": parse_vec(axis.get("xyz") if axis is not None else None, 3, (0.0, 0.0, 1.0)),
            "lower": float(limit.get("lower")) if limit is not None and limit.get("lower") is not None else None,
            "upper": float(limit.get("upper")) if limit is not None and limit.get("upper") is not None else None,
        }
        joints_by_parent.setdefault(parent_name, []).append(info)

    root_candidates = [name for name in links if name not in child_links]
    root_link = "pelvis" if "pelvis" in root_candidates or "pelvis" in links else root_candidates[0]
    return root_link, joints_by_parent, visual_meshes


def add_joint_body(
    parent_elem: ET.Element,
    joint_info: Dict[str, object],
    joints_by_parent,
    visual_meshes: Dict[str, Dict[str, object]],
    use_meshes: bool,
) -> None:
    child_name = str(joint_info["child"])
    body_elem = ET.SubElement(
        parent_elem,
        "body",
        {
            "name": child_name,
            "pos": fmt_vec(joint_info["xyz"]),
            "euler": fmt_vec(joint_info["rpy"]),
        },
    )

    joint_type = str(joint_info["type"])
    if joint_type in ("revolute", "continuous"):
        attrs = {
            "name": str(joint_info["name"]),
            "type": "hinge",
            "axis": fmt_vec(joint_info["axis"]),
            "damping": "0.1",
        }
        lower = joint_info["lower"]
        upper = joint_info["upper"]
        if joint_type == "revolute" and lower is not None and upper is not None:
            attrs["limited"] = "true"
            attrs["range"] = f"{float(lower):.8g} {float(upper):.8g}"
        ET.SubElement(body_elem, "joint", attrs)

    add_body_geom(body_elem, child_name, visual_meshes, use_meshes)
    for child_joint in joints_by_parent.get(child_name, []):
        add_joint_body(body_elem, child_joint, joints_by_parent, visual_meshes, use_meshes)


def build_simplified_mjcf(
    urdf_path: str,
    root_height: float,
    fixed_base: bool,
    visual_geometry: str = "auto",
) -> str:
    root_link, joints_by_parent, visual_meshes = parse_urdf_joints(urdf_path)
    mesh_count = sum(1 for visual in visual_meshes.values() if visual["exists"])
    use_meshes = visual_geometry == "mesh" or (visual_geometry == "auto" and mesh_count > 0)

    mujoco_elem = ET.Element("mujoco", {"model": "q1_live_mirror"})
    ET.SubElement(mujoco_elem, "compiler", {"angle": "radian"})
    ET.SubElement(mujoco_elem, "option", {"timestep": "0.005", "gravity": "0 0 -9.81"})
    asset = ET.SubElement(mujoco_elem, "asset")
    materials = {
        "mat_pelvis": "0.35 0.44 0.68 1",
        "mat_torso": "0.45 0.50 0.58 1",
        "mat_leg": "0.72 0.74 0.76 1",
        "mat_arm": "0.58 0.64 0.72 1",
        "mat_foot": "0.25 0.28 0.32 1",
        "mat_default": "0.70 0.70 0.70 1",
        "mat_floor": "0.18 0.20 0.22 1",
    }
    for name, rgba in materials.items():
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba})
    if use_meshes:
        for link_name, visual in sorted(visual_meshes.items()):
            if not visual["exists"]:
                continue
            ET.SubElement(
                asset,
                "mesh",
                {"name": mesh_name_for_link(link_name), "file": str(visual["file"])},
            )

    worldbody = ET.SubElement(mujoco_elem, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key", "pos": "0 -3 3", "dir": "0 1 -1"})
    ET.SubElement(
        worldbody,
        "geom",
        {"name": "floor", "type": "plane", "size": "3 3 0.02", "material": "mat_floor"},
    )
    root_body = ET.SubElement(
        worldbody,
        "body",
        {"name": root_link, "pos": f"0 0 {root_height:.8g}" if fixed_base else "0 0 0"},
    )
    if not fixed_base:
        ET.SubElement(root_body, "freejoint", {"name": "root"})
    add_body_geom(root_body, root_link, visual_meshes, use_meshes)
    for child_joint in joints_by_parent.get(root_link, []):
        add_joint_body(root_body, child_joint, joints_by_parent, visual_meshes, use_meshes)

    ET.SubElement(
        worldbody,
        "camera",
        {"name": "overview", "pos": "1.4 -2.2 1.1", "xyaxes": "0.84 0.54 0 -0.25 0.39 0.89"},
    )
    return ET.tostring(mujoco_elem, encoding="unicode")


def build_model_xml(urdf_path: str, root_height: float, fixed_base: bool, visual_geometry: str):
    if visual_geometry != "auto":
        return (
            build_simplified_mjcf(urdf_path, root_height, fixed_base, visual_geometry),
            visual_geometry,
        )

    mesh_xml = build_simplified_mjcf(urdf_path, root_height, fixed_base, "mesh")
    if mujoco is None:
        return mesh_xml, "mesh"
    try:
        mujoco.MjModel.from_xml_string(mesh_xml)
        return mesh_xml, "mesh"
    except Exception as exc:
        q1_robot_urdf = WORKSPACE_ROOT / "q1_robot" / "q1_22dof_box.urdf"
        hint = ""
        if q1_robot_urdf.is_file() and Path(urdf_path).resolve() != q1_robot_urdf.resolve():
            hint = f"\n[HINT] Try the known-good mesh URDF: --urdf {q1_robot_urdf}"
        print(f"[WARN] Mesh visual model failed to load; falling back to simplified geometry: {exc}{hint}")
        return (
            build_simplified_mjcf(urdf_path, root_height, fixed_base, "simplified"),
            "simplified",
        )


class LiveStateBuffer:
    def __init__(self, dof_names: List[str], default_dof_pos: np.ndarray, fake_state: bool):
        self.lock = threading.Lock()
        self.dof_names = list(dof_names)
        self.latest_state_by_name = {}
        self.latest_dof_pos = default_dof_pos.astype(np.float64).copy()
        self.latest_dof_vel = np.zeros(len(dof_names), dtype=np.float64)
        self.base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.base_ang_vel = np.zeros(3, dtype=np.float64)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.latest_joint_msg_time = time.monotonic() if fake_state else 0.0
        self.latest_imu_msg_time = time.monotonic() if fake_state else 0.0
        self.joint_count = len(dof_names) if fake_state else 0
        self.missing_joints = [] if fake_state else list(dof_names)

    def update_joint_state(self, msg) -> None:
        now = time.monotonic()
        by_name = {joint.name: joint for joint in msg.joints}
        missing = [name for name in self.dof_names if name not in by_name]
        dof_pos = self.latest_dof_pos.copy()
        dof_vel = self.latest_dof_vel.copy()
        for idx, name in enumerate(self.dof_names):
            joint = by_name.get(name)
            if joint is None:
                continue
            dof_pos[idx] = float(joint.position)
            dof_vel[idx] = float(joint.velocity)

        with self.lock:
            self.latest_state_by_name = by_name
            self.latest_dof_pos = dof_pos
            self.latest_dof_vel = dof_vel
            self.latest_joint_msg_time = now
            self.joint_count = len(msg.joints)
            self.missing_joints = missing

    def update_imu(self, msg) -> None:
        q = msg.orientation
        quat_wxyz = normalize_quat_wxyz([q.w, q.x, q.y, q.z])
        base_ang_vel = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=np.float64,
        )
        projected_gravity = quat_rotate_inverse_wxyz(quat_wxyz, [0.0, 0.0, -1.0])
        with self.lock:
            self.base_quat_wxyz = quat_wxyz
            self.base_ang_vel = base_ang_vel
            self.projected_gravity = projected_gravity
            self.latest_imu_msg_time = time.monotonic()

    def snapshot(self):
        with self.lock:
            return {
                "dof_pos": self.latest_dof_pos.copy(),
                "dof_vel": self.latest_dof_vel.copy(),
                "base_quat_wxyz": self.base_quat_wxyz.copy(),
                "base_ang_vel": self.base_ang_vel.copy(),
                "projected_gravity": self.projected_gravity.copy(),
                "joint_time": self.latest_joint_msg_time,
                "imu_time": self.latest_imu_msg_time,
                "joint_count": self.joint_count,
                "missing_joints": list(self.missing_joints),
            }


class Q1LiveMujocoViewerNode(Node):
    def __init__(self, args, state_buffer: LiveStateBuffer):
        super().__init__("q1_live_mujoco_viewer")
        self.args = args
        self.state_buffer = state_buffer
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.joint_sub = self.create_subscription(
            JointStateArray, args.joint_state_topic, self.on_joint_state, qos
        )
        self.imu_sub = self.create_subscription(Imu, args.imu_topic, self.on_imu, qos)
        self.print_timer = self.create_timer(args.print_rate_s, self.print_status)
        self.get_logger().info(
            f"Subscribing joint_state={args.joint_state_topic}, imu={args.imu_topic}. "
            "This node is read-only and publishes no joint commands."
        )

    def on_joint_state(self, msg):
        self.state_buffer.update_joint_state(msg)

    def on_imu(self, msg):
        self.state_buffer.update_imu(msg)

    def print_status(self):
        snap = self.state_buffer.snapshot()
        now = time.monotonic()
        joint_age = now - snap["joint_time"] if snap["joint_time"] > 0.0 else float("inf")
        imu_age = now - snap["imu_time"] if snap["imu_time"] > 0.0 else float("inf")
        missing = snap["missing_joints"]
        message = (
            f"joint_age={joint_age:.3f}s imu_age={imu_age:.3f}s "
            f"joint_count={snap['joint_count']} missing={missing[:6]}"
            f"{'...' if len(missing) > 6 else ''} "
            f"projected_gravity={np.array2string(snap['projected_gravity'], precision=3)}"
        )
        if joint_age > self.args.state_timeout_s or imu_age > self.args.state_timeout_s:
            self.get_logger().warn("state timeout: " + message)
        elif missing:
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)


def build_joint_qpos_map(model, dof_names: List[str]) -> Dict[str, int]:
    joint_qpos = {}
    for name in dof_names:
        try:
            joint_qpos[name] = int(model.joint(name).qposadr[0])
        except KeyError:
            print(f"[WARN] MuJoCo model does not contain joint {name}")
    return joint_qpos


def run_viewer(args, cfg, state_buffer: LiveStateBuffer, mjcf_xml: str, stop_event: threading.Event) -> None:
    if mujoco is None:
        raise RuntimeError("mujoco is not installed. Run: python3 -m pip install mujoco")

    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)
    joint_qpos = build_joint_qpos_map(model, cfg["dof_names"])
    root_qposadr = None
    if not args.fixed_base:
        root_qposadr = int(model.joint("root").qposadr[0])

    sleep_s = 1.0 / max(float(args.viewer_hz), 1.0)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -15
        while viewer.is_running() and not stop_event.is_set():
            snap = state_buffer.snapshot()
            if root_qposadr is not None:
                data.qpos[root_qposadr:root_qposadr + 3] = np.array(
                    [0.0, 0.0, args.root_height], dtype=np.float64
                )
                data.qpos[root_qposadr + 3:root_qposadr + 7] = snap["base_quat_wxyz"]
            for idx, name in enumerate(cfg["dof_names"]):
                qposadr = joint_qpos.get(name)
                if qposadr is not None:
                    data.qpos[qposadr] = snap["dof_pos"][idx]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(sleep_s)


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only Q1 live MuJoCo mirror from ROS2 state topics")
    parser.add_argument("--config", type=str, default="config/q1_sim2real_base.yaml")
    parser.add_argument("--urdf", type=str, default=default_urdf_arg())
    parser.add_argument("--joint-state-topic", type=str, default="/aima/hal/joint/state")
    parser.add_argument("--imu-topic", type=str, default="/aima/hal/imu/state")
    parser.add_argument("--root-height", type=float, default=0.42)
    parser.add_argument("--viewer-hz", type=float, default=60.0)
    parser.add_argument("--state-timeout-s", type=float, default=0.5)
    parser.add_argument("--print-rate-s", type=float, default=1.0)
    parser.add_argument("--fixed-base", action="store_true")
    parser.add_argument("--fake-state", action="store_true")
    parser.add_argument(
        "--visual-geometry",
        choices=("auto", "mesh", "simplified"),
        default="auto",
        help="auto uses URDF meshes when available and simplified geoms otherwise",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.config = resolve_path(args.config)
    args.urdf = resolve_path(args.urdf)

    cfg = read_conf(args.config)
    if len(cfg["dof_names"]) != NUM_ACTIONS:
        raise ValueError(f"Expected {NUM_ACTIONS} dof_names, got {len(cfg['dof_names'])}")

    if args.viewer_hz <= 0.0:
        raise ValueError("--viewer-hz must be > 0")
    if args.state_timeout_s <= 0.0:
        raise ValueError("--state-timeout-s must be > 0")

    mjcf_xml, actual_visual_geometry = build_model_xml(
        args.urdf, args.root_height, args.fixed_base, args.visual_geometry
    )
    state_buffer = LiveStateBuffer(cfg["dof_names"], cfg["default_dof_pos"], args.fake_state)

    print("=" * 72)
    print("Q1 Live MuJoCo Mirror (read-only)")
    print("=" * 72)
    print(f"config={args.config}")
    print(f"urdf={args.urdf}")
    print(f"joint_state_topic={args.joint_state_topic}")
    print(f"imu_topic={args.imu_topic}")
    print(
        f"fixed_base={args.fixed_base} fake_state={args.fake_state} "
        f"visual_geometry={args.visual_geometry} actual_visual_geometry={actual_visual_geometry}"
    )
    print("This script does not publish /aima/hal/joint/command.")
    print()

    node = None
    spin_thread = None
    stop_event = threading.Event()
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_shutdown(signum, _frame):
        if stop_event.is_set():
            raise KeyboardInterrupt
        print(f"\n[INFO] Received signal {signum}; closing MuJoCo viewer gracefully...")
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    if not args.fake_state:
        if rclpy is None or JointStateArray is None or Imu is None:
            raise RuntimeError(
                "ROS2/Q1 message packages are unavailable. Source the SDK workspace first, "
                "for example: source /home/hoho/q1_wordandexample/202606091310/install/setup.bash"
            )
        rclpy.init(args=None)
        node = Q1LiveMujocoViewerNode(args, state_buffer)
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()

    try:
        run_viewer(args, cfg, state_buffer, mjcf_xml, stop_event)
    finally:
        stop_event.set()
        if node is not None:
            node.destroy_node()
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
