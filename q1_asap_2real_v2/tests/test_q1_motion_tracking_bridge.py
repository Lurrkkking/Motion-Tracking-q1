import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sim2real_q1_motion_tracking_ros2.py"
CONFIG_PATH = REPO_ROOT / "config" / "q1_sim2real_base.yaml"
FLOAT_ATOL = 1e-6


def load_bridge_module():
    spec = importlib.util.spec_from_file_location("q1_motion_bridge", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class FakeClockTime:
    def to_msg(self):
        return "fake_stamp"


class FakeClock:
    def now(self):
        return FakeClockTime()


class FakeJointState:
    def __init__(self, name, position, velocity, **extra):
        self.name = name
        self.position = position
        self.velocity = velocity
        for key, value in extra.items():
            setattr(self, key, value)


class FakeJointStateArray:
    def __init__(self, joints):
        self.joints = joints
        self.header = SimpleNamespace(frame_id="fake_frame")
        self.meas_stamp = SimpleNamespace(sec=1, nanosec=2)


class FakeJointCommand:
    def __init__(self):
        self.name = ""
        self.position = 0.0
        self.stiffness = 0.0
        self.damping = 0.0
        self.velocity = 0.0
        self.effort = 0.0


class FakeJointCommandArray:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None, sequence=0, frame_id="")
        self.meas_stamp = None
        self.joints = []


def make_node(module, cfg):
    node = module.Q1MotionTrackingNode.__new__(module.Q1MotionTrackingNode)
    node.args = SimpleNamespace(
        enable_motors=False,
        risk_confirm="",
        fake_base_state=False,
        enable_policy=False,
        max_base_ang_vel=10.0,
        max_dof_vel=30.0,
        max_motor_temp_c=80.0,
        max_coil_temp_c=80.0,
        joint_limit_margin_rad=0.02,
        emergency_damping=5.0,
        hold_uncontrolled_stiffness=0.0,
        hold_uncontrolled_damping=5.0,
    )
    node.cfg = cfg
    node.dof_names = cfg["dof_names"]
    node.policy_joint_index = {name: idx for idx, name in enumerate(node.dof_names)}
    node.state = module.ControlState.WAIT_STATE
    node.latest_joint_msg_time = 0.0
    node.latest_base_msg_time = 1.0
    node.latest_joint_state_msg = None
    node.latest_state_by_name = {}
    node.latest_dof_pos = np.zeros(module.NUM_ACTIONS, dtype=np.float32)
    node.latest_dof_vel = np.zeros(module.NUM_ACTIONS, dtype=np.float32)
    node.latest_target_dof_pos = cfg["default_dof_pos"].copy()
    node.command_target = cfg["default_dof_pos"].copy()
    node.base_ang_vel = np.zeros(3, dtype=np.float32)
    node.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    node.max_abs_dof_vel_seen = 0.0
    node.max_abs_base_ang_vel_seen = 0.0
    node.motors_enabled = False
    node.last_error = ""
    node.lower_limits = np.full(module.NUM_ACTIONS, -10.0, dtype=np.float32)
    node.upper_limits = np.full(module.NUM_ACTIONS, 10.0, dtype=np.float32)
    node.clamp_count_limits = 0
    node.command_publish_count = 0
    node.simulated_command_count = 0
    node.get_logger = lambda: FakeLogger()
    node.get_clock = lambda: FakeClock()
    return node


def test_extract_joint_state_orders_positions_and_velocities_by_config():
    module = load_bridge_module()
    cfg = module.read_conf(CONFIG_PATH)
    node = make_node(module, cfg)

    reversed_names = list(reversed(cfg["dof_names"]))
    msg = FakeJointStateArray(
        [
            FakeJointState(name, position=100.0 + idx, velocity=200.0 + idx)
            for idx, name in enumerate(reversed_names)
        ]
    )
    node.latest_state_by_name = {joint.name: joint for joint in msg.joints}

    ok, dof_pos, dof_vel = node.extract_joint_state()

    assert ok is True
    expected_pos = np.array(
        [100.0 + reversed_names.index(name) for name in cfg["dof_names"]],
        dtype=np.float32,
    )
    expected_vel = np.array(
        [200.0 + reversed_names.index(name) for name in cfg["dof_names"]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(dof_pos, expected_pos)
    np.testing.assert_allclose(dof_vel, expected_vel)


def test_on_joint_state_updates_cached_state_and_enters_hold_current_when_base_ready():
    module = load_bridge_module()
    cfg = module.read_conf(CONFIG_PATH)
    node = make_node(module, cfg)
    msg = FakeJointStateArray(
        [
            FakeJointState(name, position=0.01 * idx, velocity=-0.02 * idx)
            for idx, name in enumerate(cfg["dof_names"])
        ]
    )

    node.on_joint_state(msg)

    assert node.latest_joint_state_msg is msg
    assert node.state == module.ControlState.HOLD_CURRENT
    np.testing.assert_allclose(
        node.latest_dof_pos,
        np.array([0.01 * idx for idx in range(module.NUM_ACTIONS)], dtype=np.float32),
    )
    np.testing.assert_allclose(
        node.latest_dof_vel,
        np.array([-0.02 * idx for idx in range(module.NUM_ACTIONS)], dtype=np.float32),
    )
    np.testing.assert_allclose(node.command_target, node.latest_dof_pos)


def test_on_imu_updates_base_ang_vel_and_projected_gravity():
    module = load_bridge_module()
    cfg = module.read_conf(CONFIG_PATH)
    node = make_node(module, cfg)
    imu_msg = SimpleNamespace(
        orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
        angular_velocity=SimpleNamespace(x=0.1, y=-0.2, z=0.3),
    )

    node.on_imu(imu_msg)

    np.testing.assert_allclose(node.base_ang_vel, np.array([0.1, -0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(node.projected_gravity, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    assert node.latest_base_msg_time > 0.0


def test_get_obs_builds_expected_365_dim_observation_and_updates_history():
    module = load_bridge_module()
    cfg = module.read_conf(CONFIG_PATH)
    cfg["cycle_time"] = 2.0
    hist_dict, hist_obs = module.create_history(cfg)
    action = np.linspace(-0.2, 0.2, module.NUM_ACTIONS, dtype=np.float32)
    state = {
        "dof_pos": cfg["default_dof_pos"] + 0.1,
        "dof_vel": np.full(module.NUM_ACTIONS, 2.0, dtype=np.float32),
        "base_ang_vel": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }

    obs, hist_obs_new = module.get_obs(hist_obs, hist_dict, state, action, counter=0, cfg=cfg)

    assert obs.shape == (1, 365)
    assert hist_obs_new.shape == (1, 292)
    np.testing.assert_allclose(obs[0, 0:22], action)
    np.testing.assert_allclose(obs[0, 22:25], state["base_ang_vel"] * cfg["obs_scale_base_ang_vel"])
    np.testing.assert_allclose(
        obs[0, 25:47],
        np.full(module.NUM_ACTIONS, 0.1, dtype=np.float32),
        atol=FLOAT_ATOL,
    )
    np.testing.assert_allclose(
        obs[0, 47:69],
        np.full(module.NUM_ACTIONS, 0.1, dtype=np.float32),
        atol=FLOAT_ATOL,
    )
    np.testing.assert_allclose(obs[0, 361:364], state["projected_gravity"])
    assert obs[0, 364] == np.float32(module.compute_ref_motion_phase(0, cfg))
    np.testing.assert_allclose(hist_dict["actions"][0], action)
    np.testing.assert_allclose(hist_dict["projected_gravity"][0], state["projected_gravity"])


def test_make_command_msg_maps_targets_and_pd_gains(monkeypatch):
    module = load_bridge_module()
    cfg = module.read_conf(CONFIG_PATH)
    node = make_node(module, cfg)
    monkeypatch.setattr(module, "JointCommand", FakeJointCommand)
    monkeypatch.setattr(module, "JointCommandArray", FakeJointCommandArray)
    node.latest_joint_state_msg = FakeJointStateArray(
        [
            FakeJointState(name, position=-1.0, velocity=0.0)
            for name in cfg["dof_names"]
        ]
    )
    node.command_target = np.linspace(-0.5, 0.5, module.NUM_ACTIONS, dtype=np.float32)

    cmd = node.make_command_msg()

    assert cmd is not None
    assert len(cmd.joints) == module.NUM_ACTIONS
    assert cmd.header.sequence == 0
    assert cmd.header.frame_id == "fake_frame"
    for idx, joint in enumerate(cmd.joints):
        assert joint.name == cfg["dof_names"][idx]
        assert joint.position == float(node.command_target[idx])
        assert joint.stiffness == float(cfg["kps"][idx])
        assert joint.damping == float(cfg["kds"][idx])
        assert joint.velocity == 0.0
        assert joint.effort == 0.0


def test_make_command_msg_uses_damping_frame_for_emergency(monkeypatch):
    module = load_bridge_module()
    cfg = module.read_conf(CONFIG_PATH)
    node = make_node(module, cfg)
    monkeypatch.setattr(module, "JointCommand", FakeJointCommand)
    monkeypatch.setattr(module, "JointCommandArray", FakeJointCommandArray)
    node.latest_joint_state_msg = FakeJointStateArray(
        [
            FakeJointState(name, position=1.23, velocity=0.0)
            for name in cfg["dof_names"]
        ]
    )

    cmd = node.make_command_msg(damping_frame=True)

    for joint in cmd.joints:
        assert joint.position == 1.23
        assert joint.stiffness == 0.0
        assert joint.damping == node.args.emergency_damping
