#!/usr/bin/env python3
"""Comprehensive G1 vs Q1 comparison for jump diagnosis."""
import joblib, numpy as np, glob

# =============================================
# LOAD MOTIONS
# =============================================
# G1
g1_dir = "humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles"
g1_files = sorted(glob.glob(f"{g1_dir}/*.pkl"))

# Q1 CR7
q1_path = "humanoidverse/data/motions/q1/q1_cr7_scale045.pkl"
q1_data = joblib.load(q1_path)['q1_cr7_scale045']

print("="*80)
print("STEP 5: MOTION ANALYSIS — JOINT RANGES & FLIGHT")
print("="*80)

# =============================================
# G1: Scan all motions for jump/turn content
# =============================================
print("\n--- G1 MOTIONS SCAN (all 52 motions) ---")
print(f"{'motion':50s} {'frames':>6s} {'rz_delta(cm)':>12s} {'yaw_delta(deg)':>14s} {'rz_min':>8s} {'rz_max':>8s}")
print("-"*100)

g1_jump_motions = []
for f in g1_files:
    data = joblib.load(f)
    key = list(data.keys())[0]
    md = data[key]
    rz = md['root_trans_offset'][:, 2]
    rz_delta = (rz.max() - rz.min()) * 100  # cm

    # Check yaw from root_rot if available
    if 'root_rot' in md:
        from scipy.spatial.transform import Rotation as sRot
        root_rot = md['root_rot']  # quaternion (w,x,y,z) or (x,y,z,w)?
        # Usually from SMPL it's wxyz
        rot = sRot.from_quat(root_rot[:, [1,2,3,0]])  # xyzw
        euler = rot.as_euler('XYZ', degrees=True)
        yaw_raw = euler[:, 2]
        yaw_uw = np.rad2deg(np.unwrap(np.deg2rad(yaw_raw)))
        yaw_delta = yaw_uw[-1] - yaw_uw[0]
    else:
        yaw_delta = 0

    name = f.split('/')[-1][:48]
    print(f"{name:50s} {len(rz):>6d} {rz_delta:>12.1f} {yaw_delta:>14.1f} {rz.min():>8.3f} {rz.max():>8.3f}")

    if rz_delta > 10 or abs(yaw_delta) > 60:
        g1_jump_motions.append((f, rz_delta, yaw_delta, key))

print(f"\nG1 motions with significant root_z (>10cm) or yaw (>60deg): {len(g1_jump_motions)}")
for f, rz_d, y_d, key in g1_jump_motions:
    print(f"  {key}: rz_delta={rz_d:.1f}cm, yaw_delta={y_d:.1f}deg")

# Check for CR7 specifically
cr7_files = [f for f in g1_files if 'CR7' in f]
print(f"\nG1 CR7 motions: {len(cr7_files)}")
for f in cr7_files:
    print(f"  {f.split('/')[-1]}")

# =============================================
# Q1 CR7 analysis
# =============================================
print("\n--- Q1 CR7 MOTION ---")
rz = q1_data['root_trans_offset'][:, 2]
rz_delta_q1 = (rz.max() - rz.min()) * 100
from scipy.spatial.transform import Rotation as sRot
pelvis_aa = q1_data['pose_aa'][:, 0, :]
pelvis_rot = sRot.from_rotvec(pelvis_aa)
euler = pelvis_rot.as_euler('XYZ', degrees=True)
yaw_uw_q1 = np.rad2deg(np.unwrap(np.deg2rad(euler[:, 2])))
yaw_delta_q1 = yaw_uw_q1[-1] - yaw_uw_q1[0]

print(f"  frames: {len(rz)}, rz_delta: {rz_delta_q1:.1f}cm, yaw_delta: {yaw_delta_q1:.1f}deg")
print(f"  rz_min: {rz.min():.3f}, rz_max: {rz.max():.3f}")

# =============================================
# G1: Pick largest jump motion for joint range analysis
# =============================================
if g1_jump_motions:
    best_g1 = max(g1_jump_motions, key=lambda x: x[1])  # max rz_delta
    print(f"\n--- G1 LARGEST JUMP MOTION: {best_g1[3]} (rz_delta={best_g1[1]:.1f}cm, yaw_delta={best_g1[2]:.1f}deg) ---")
    g1_best = joblib.load(best_g1[0])[best_g1[3]]

    # G1 has 'dof' field directly
    if 'dof' in g1_best:
        g1_dof = g1_best['dof']  # (frames, 23)
        print(f"  G1 dof shape: {g1_dof.shape}")
        # G1 dof order:
        # left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
        # right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll,
        # waist_yaw, waist_roll, waist_pitch,
        # left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
        # right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow
        g1_names = ['L_hip_pitch', 'L_hip_roll', 'L_hip_yaw', 'L_knee', 'L_ankle_pitch', 'L_ankle_roll',
                    'R_hip_pitch', 'R_hip_roll', 'R_hip_yaw', 'R_knee', 'R_ankle_pitch', 'R_ankle_roll',
                    'waist_yaw', 'waist_roll', 'waist_pitch',
                    'L_shoulder_pitch', 'L_shoulder_roll', 'L_shoulder_yaw', 'L_elbow',
                    'R_shoulder_pitch', 'R_shoulder_roll', 'R_shoulder_yaw', 'R_elbow']
        for i, name in enumerate(g1_names):
            print(f"    G1 dof[{i:2d}] {name:20s}: min={g1_dof[:,i].min():8.4f}  max={g1_dof[:,i].max():8.4f}  range={g1_dof[:,i].max()-g1_dof[:,i].min():8.4f}")
    else:
        print("  G1: NO 'dof' field!")
else:
    print("\n  WARNING: No G1 motions found with rz_delta > 10cm")

# =============================================
# Q1: We need dof from pose_aa via skeleton
# Actually, we can approximate by checking pose_aa magnitudes
# =============================================
print("\n--- Q1 CR7 POSE_AA ANALYSIS ---")
pose_aa = q1_data['pose_aa']  # (178, 23, 3)
# Compute rotation magnitude (norm) per joint per frame
pose_aa_norm = np.linalg.norm(pose_aa, axis=-1)  # (178, 23)

# Q1 joint matches (from config):
# pelvis(0), L_Hip(1), L_Knee(2), L_Ankle(3), R_Hip(4), R_Knee(5), R_Ankle(6),
# L_Shoulder(7), L_Elbow(8), R_Shoulder(9), R_Elbow(10)
# Plus 12 more joints in SMPL that are unused
q1_joint_names = ['pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle',
                  'L_Shoulder', 'L_Elbow', 'R_Shoulder', 'R_Elbow']
# Remaining 12 SMPL joints
for i in range(min(11, pose_aa_norm.shape[1])):
    name = q1_joint_names[i] if i < len(q1_joint_names) else f"joint_{i}"
    print(f"    Q1 pose_aa[{i:2d}] {name:20s}: norm min={pose_aa_norm[:,i].min():7.4f}  max={pose_aa_norm[:,i].max():7.4f}")

# =============================================
# STEP 6: ACTION TARGET RANGE vs REFERENCE RANGE
# =============================================
print("\n" + "="*80)
print("STEP 6: PD TARGET RANGE vs REFERENCE RANGE COMPARISON")
print("="*80)

# G1 Control
g1_default = {'L_hip_pitch': -0.1, 'L_hip_roll': 0., 'L_hip_yaw': 0.,
              'L_knee': 0.3, 'L_ankle_pitch': -0.2, 'L_ankle_roll': 0.,
              'R_hip_pitch': -0.1, 'R_hip_roll': 0., 'R_hip_yaw': 0.,
              'R_knee': 0.3, 'R_ankle_pitch': -0.2, 'R_ankle_roll': 0.,
              'waist_yaw': 0., 'waist_roll': 0., 'waist_pitch': 0.}
g1_action_scale = 0.25

# Q1 Control (from q1_22dof.yaml)
q1_default = {'L_hip_pitch': -0.2, 'L_hip_roll': 0., 'L_hip_yaw': 0.,
              'L_knee': 0.5, 'L_ankle_pitch': -0.2, 'L_ankle_roll': 0.,
              'R_hip_pitch': -0.2, 'R_hip_roll': 0., 'R_hip_yaw': 0.,
              'R_knee': 0.5, 'R_ankle_pitch': -0.2, 'R_ankle_roll': 0.,
              'waist_roll': 0., 'waist_yaw': 0.}
q1_action_scale = 0.25

# G1 reference dof ranges (from the largest jump motion)
if 'dof' in g1_best:
    g1_ref_dof = g1_best['dof']
    g1_ref_min = g1_ref_dof.min(axis=0)
    g1_ref_max = g1_ref_dof.max(axis=0)

print("\n--- G1 PD TARGET vs REFERENCE (largest jump motion) ---")
print(f"{'dof':20s} {'default':>8s} {'targ_min':>8s} {'targ_max':>8s} {'ref_min':>8s} {'ref_max':>8s} {'covered?':>10s} {'ref_range':>8s}")
print("-"*90)

g1_dof_order = ['L_hip_pitch', 'L_hip_roll', 'L_hip_yaw', 'L_knee', 'L_ankle_pitch', 'L_ankle_roll',
                'R_hip_pitch', 'R_hip_roll', 'R_hip_yaw', 'R_knee', 'R_ankle_pitch', 'R_ankle_roll',
                'waist_yaw', 'waist_roll', 'waist_pitch']

for i, name in enumerate(g1_dof_order):
    if name in g1_default:
        d = g1_default[name]
        tmin = d - g1_action_scale
        tmax = d + g1_action_scale
        rmin = g1_ref_min[i]
        rmax = g1_ref_max[i]
        covered = (rmin >= tmin) and (rmax <= tmax)
        print(f"{name:20s} {d:>8.3f} {tmin:>8.3f} {tmax:>8.3f} {rmin:>8.3f} {rmax:>8.3f} {'YES' if covered else 'NO':>10s} {rmax-rmin:>8.3f}")

print("\n--- Q1 PD TARGET vs CR7 REFERENCE (pose_aa rotation norms) ---")
print("NOTE: Q1 doesn't have dof in PKL; pose_aa is axis-angle, not dof_pos in radians.")
print("The actual dof_pos is computed at runtime from skeleton FK.")
print("We compare PD target range vs the joint angle that the ref motion would produce.")
print(f"{'dof':20s} {'default':>8s} {'targ_min':>8s} {'targ_max':>8s}")
print("-"*50)
for name, d in q1_default.items():
    tmin = d - q1_action_scale
    tmax = d + q1_action_scale
    print(f"{name:20s} {d:>8.3f} {tmin:>8.3f} {tmax:>8.3f}")

# Check which joints are most constrained
print("\n--- KEY FINDING: Q1 KNEE TARGET RANGE ---")
knee_default = 0.5
knee_tmin = knee_default - 0.25
knee_tmax = knee_default + 0.25
print(f"Q1 knee PD target range: [{knee_tmin:.2f}, {knee_tmax:.2f}] rad")
print(f"A deep squat knee (~1.5 rad) is {1.5 - knee_tmax:.2f} rad BEYOND the max target")
print(f"A full extension knee (~0.0 rad) is {knee_tmin - 0.0:.2f} rad BELOW the min target")
print(f"The PD controller FIGHTS extension beyond {knee_tmin:.2f} rad")

print("\n--- COMPARISON: G1 KNEE ---")
g1_knee_default = 0.3
g1_knee_tmin = g1_knee_default - 0.25
g1_knee_tmax = g1_knee_default + 0.25
print(f"G1 knee PD target range: [{g1_knee_tmin:.2f}, {g1_knee_tmax:.2f}] rad")
if 'dof' in g1_best:
    g1_knee_ref = g1_ref_dof[:, 3]  # L_knee is index 3
    print(f"G1 ref knee range: [{g1_knee_ref.min():.3f}, {g1_knee_ref.max():.3f}]")
    deep_squat = g1_knee_ref.max()
    full_ext = g1_knee_ref.min()
    print(f"G1 deep squat knee ({deep_squat:.3f}) is {deep_squat - g1_knee_tmax:.3f} rad beyond tmax")
    print(f"G1 full ext knee ({full_ext:.3f}) is {g1_knee_tmin - full_ext:.3f} rad below tmin")

PYEOF
