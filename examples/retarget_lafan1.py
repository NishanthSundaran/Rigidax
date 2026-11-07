"""Retarget LAFAN1 human motion capture onto the Unitree G1 with rigidax.ik.

The robot-specific parts live here: which G1 bodies follow which human joints, how
the human frames are built, the foot calibration. Solving, hard joint limits, elastic
self-collision avoidance and batching across segments / CPU cores come from rigidax.ik.

LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0: non-commercial use only.

  python examples/retarget_lafan1.py dance1_subject1 jumps1_subject1 ...
  -> data/g1_lafan1/<clip>.npz  with qpos (T, 36) at 30 fps
"""

import os
import sys

GPU = "--gpu" in sys.argv
if not GPU:  # one XLA device per physical core
    os.environ.setdefault("XLA_FLAGS",
                          f"--xla_force_host_platform_device_count={os.cpu_count() // 2}")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import rigidax as rx  # noqa: E402
from rigidax import ik  # noqa: E402
from rigidax.ik.bvh import bvh_positions, bvh_rotations  # noqa: E402

# IK runs in float32 (fast batched linear algebra); --f64 for double precision
jax.config.update("jax_enable_x64", "--f64" in sys.argv)
if not GPU:
    jax.config.update("jax_platforms", "cpu")

ROOT = Path(__file__).resolve().parents[1]
LAFAN = ROOT / "data" / "lafan1"
OUT = ROOT / "data" / "g1_lafan1"  # what render_motions.py shows by default
OUT_HANDS = ROOT / "data" / "g1_lafan1_hands"  # G1 + Dex3 hands
OUT_PLAIN = ROOT / "data" / "g1_lafan1_nohands"  # plain G1 (rigid hands)
# G1 with Dex3-1 hands (thumb/index/middle, 7 joints per hand); --no-hands: plain G1
HANDS = "--no-hands" not in sys.argv
# The body is SOLVED on the 35-DOF G1 (the fingers follow their own targets and nothing
# depends on them, and the QP cost grows ~n^3); the OUTPUT is the G1 + Dex3 layout with
# the finger angles written in afterwards.
IK_XML = ROOT / "examples" / "unitree_g1" / "g1.xml"
HANDS_XML = ROOT / "examples" / "unitree_g1" / "g1_with_hands.xml"
G1_XML = HANDS_XML if HANDS else IK_XML  # output model
FINGERS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
FINGER_JOINTS = [f"{s}_hand_{f}_joint" for s in ("left", "right") for f in FINGERS]
FINGER_WEIGHT = 1.0
# LAFAN1 records no fingers: the grip is SYNTHESIZED from context (fist while striking
# or guarding, flat on the floor, relaxed otherwise)
GRIP_RELAXED = 0.35
STRIKE_SPEED = 1.8  # m/s, hand speed relative to the chest that counts as a strike
GUARD_DIST = 0.30  # m, hand this close to the head counts as a fighting guard
FOREARM_TWIST_WEIGHT = 1.0  # forearm pronation, about the forearm axis only
PALM_WEIGHT = 6.0  # palm height task while the human hand rests on the floor
PALM_HEIGHT = 0.02  # palm center above the floor (hand thickness) when on the floor
FINGER_SPHERE_SPACING = 0.06  # coarse finger envelope: fewer collision rows, same protection
FPS = 30

# Absolute position targets (scaled human positions): (robot body, human joint, weight).
# The G1 pelvis origin sits 10 cm above its hip joints, so the hip joints are matched.
POS_TASKS = [
    ("left_hip_pitch_link", "LeftUpLeg", 3.0), ("right_hip_pitch_link", "RightUpLeg", 3.0),
    ("left_ankle_roll_link", "LeftFoot", 4.0), ("right_ankle_roll_link", "RightFoot", 4.0),
]
# Arms follow limb directions with the G1's own segment lengths (its upper arm is ~37%
# shorter than a scaled human's): (robot body, robot base, human from, human to, weight)
# Legs too can be retargeted by direction (thigh: hip -> knee; shin: knee -> ankle)
# instead of knee positions: the G1's hips are 3.3 cm narrower than a scaled human's
LEG_DIR_TASKS = [
    ("left_knee_link", "left_hip_pitch_link", "LeftUpLeg", "LeftLeg", 2.0),
    ("right_knee_link", "right_hip_pitch_link", "RightUpLeg", "RightLeg", 2.0),
]
DIR_TASKS = [
    ("left_elbow_link", "left_shoulder_roll_link", "LeftArm", "LeftForeArm", 3.0),
    ("left_wrist_yaw_link", "left_elbow_link", "LeftForeArm", "LeftHand", 3.0),
    ("right_elbow_link", "right_shoulder_roll_link", "RightArm", "RightForeArm", 3.0),
    ("right_wrist_yaw_link", "right_elbow_link", "RightForeArm", "RightHand", 3.0),
    # legs: thigh and shin directions (knee positions would be off by the hip width)
    ("left_knee_link", "left_hip_pitch_link", "LeftUpLeg", "LeftLeg", 2.0),
    ("right_knee_link", "right_hip_pitch_link", "RightUpLeg", "RightLeg", 2.0),
    ("left_ankle_roll_link", "left_knee_link", "LeftLeg", "LeftFoot", 2.0),
    ("right_ankle_roll_link", "right_knee_link", "RightLeg", "RightFoot", 2.0),
]
ROT_TASKS = [("pelvis", "hips", 2.0), ("torso_link", "chest", 1.5),
             ("left_ankle_roll_link", "lfoot", 1.0), ("right_ankle_roll_link", "rfoot", 1.0)]
# Posture pulls toward neutral: wrists (LAFAN1 has no hand orientation) and the hip
# twist / waist, which a straight leg leaves ambiguous (hip yaw otherwise flips)
# Arm link orientations follow the human bones (fixes forearm twist and wrist gestures).
# Weights stay low: the G1 wrist range is narrower than a human's, and a strong hand
# target would rotate the whole arm (wrecking its direction) when out of wrist range.
# Offsets map LAFAN1 bone frames to G1 link frames (axis conventions, as in GMR);
# verified on this data: the pelvis matches to a median of 3 deg.
ARM_ROT_TASKS = [  # (robot body, human bone, weight, wxyz offset)
    ("left_shoulder_yaw_link", "LeftArm", 0.05, (0.5, 0.5, -0.5, -0.5)),
    ("left_elbow_link", "LeftForeArm", 0.15, (0.70710678, 0.70710678, 0.0, 0.0)),
    ("left_wrist_yaw_link", "LeftHand", 0.6, (0.70710678, 0.70710678, 0.0, 0.0)),
    ("right_shoulder_yaw_link", "RightArm", 0.05, (0.5, 0.5, -0.5, -0.5)),
    ("right_elbow_link", "RightForeArm", 0.15, (0.70710678, 0.70710678, 0.0, 0.0)),
    ("right_wrist_yaw_link", "RightHand", 0.6, (0.70710678, 0.70710678, 0.0, 0.0)),
    # leg links too (what frax tracks): thigh and shin orientations
    ("left_hip_yaw_link", "LeftUpLeg", 0.2, (-0.5, 0.5, 0.5, -0.5)),
    ("left_knee_link", "LeftLeg", 0.2, (-0.5, 0.5, 0.5, -0.5)),
    ("right_hip_yaw_link", "RightUpLeg", 0.2, (-0.5, 0.5, 0.5, -0.5)),
    ("right_knee_link", "RightLeg", 0.2, (-0.5, 0.5, 0.5, -0.5)),
]
WRIST_POSTURE = 0.05  # small residual pull to neutral (was 0.25 when hands were untracked)
POSTURE = {"wrist_roll": WRIST_POSTURE, "wrist_pitch": WRIST_POSTURE, "wrist_yaw": WRIST_POSTURE,
           "hip_yaw": 0.3, "hip_roll": 0.3, "waist": 0.3}
TRACK_HANDS = True
ARM_ROT_SMOOTH = 0.05  # s: zero-phase smoothing of the bone-rotation targets (hand noise)
# Damping toward the previous frame. Legs get much more: the legs jitter on mocap noise
# and contact transitions, while arms damped this much lose their liveliness.
# Twist joints (hip yaw/roll, waist yaw/roll) are weakly constrained by the tasks and
# would wander on residual errors: they get heavy damping.
DAMPING = {"hip_yaw": 0.3, "waist_yaw": 0.3, "waist_roll": 0.3,
           "default": 0.05, "hip": 0.02, "knee": 0.02, "ankle": 0.02}
MAX_STEP = 0.15  # rad per iteration (trust region; 2 iterations/frame): no flips
# Human-plausible hip ranges (rad, G1 sign conventions; pitch < 0 is flexion). The G1's
# mechanical hip ranges are far wider than a person's, and the IK can otherwise settle
# into contortions (e.g. 160 deg hip extension, 120 deg abduction) and stay stuck there.
# Arm ranges narrowed as in GMR's g1_mocap_29dof (also used by frax), so the arms cannot
# flip into G1's second shoulder solution (pitch and yaw ~150 deg). (Narrowing the hips
# was tested and made leg tracking worse, so the legs keep their full range.)
RETARGET_RANGES = {
    "left_shoulder_pitch_joint": (-3.089, 1.149), "right_shoulder_pitch_joint": (-3.089, 1.149),
    "left_shoulder_roll_joint": (-0.6, 2.252), "right_shoulder_roll_joint": (-2.252, 0.6),
    "left_shoulder_yaw_joint": (-1.4, 2.0), "right_shoulder_yaw_joint": (-2.0, 1.4),
    "left_elbow_joint": (-1.047, 1.7), "right_elbow_joint": (-1.047, 1.7),
}
ARM = ("shoulder_yaw", "elbow", "wrist", "hand")
BODY = ("pelvis", "torso", "waist", "hip", "knee")
FOOT_SPHERES = np.array([[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03],
                         [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]])
FOOT_RADIUS = 0.005
WHOLE_BODY_FLOOR = True  # knees, hands, torso... must not sink through the floor either
FLOOR_ROWS = 12  # floor rows per step: the spheres closest to the floor
TOE_POINT = FOOT_SPHERES[2:].mean(0) - np.array([0.0, 0.0, FOOT_RADIUS])  # sole, front
HEEL_POINT = FOOT_SPHERES[:2].mean(0) - np.array([0.0, 0.0, FOOT_RADIUS])  # sole, rear


# ---------------- human frames ----------------

def _frame(y, up):
    """Rotation with y along y, z as close as possible to up"""
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = up - np.sum(up * y, -1, keepdims=True) * y
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    return np.stack([np.cross(y, z), y, z], -1)


def stance_bone_pitch(H, side):
    """Median ankle->toe bone slope (rad, toe-down +) while that foot is planted"""
    foot, toe = H[f"{side}Foot"], H[f"{side}Toe"]
    floor = np.percentile(np.minimum(H["LeftToe"][:, 2], H["RightToe"][:, 2]), 5)
    speed = np.r_[0, np.linalg.norm(np.diff(toe, axis=0), axis=1)] * FPS
    planted = (toe[:, 2] < floor + 0.03) & (speed < 0.2)
    b = toe - foot
    pitch = np.arcsin(-b[:, 2] / np.linalg.norm(b, axis=1))
    return float(np.median(pitch[planted])) if planted.sum() > 10 else float(np.median(pitch))


def _foot_frame(foot, toe, flat_pitch, knee=None):
    """Foot frame from the ankle->toe bone, minus its slope in a flat stance.

    The lateral axis comes from the leg (shin x foot), which stays well defined in any
    pose; cross(world up, foot) degenerates whenever the foot points up or down (kicks,
    leg raises, floor moves) and then flips frame to frame, twisting the whole leg."""
    x = (toe - foot) / np.linalg.norm(toe - foot, axis=-1, keepdims=True)
    if knee is not None:
        shin = foot - knee
        y = np.cross(x, shin)
        # sign: match the usual convention on the first frame, then stay continuous in
        # time (comparing against cross(up, foot) would degenerate again)
        y_up = np.cross(np.array([0.0, 0, 1]), x[0])
        if np.dot(y[0], y_up) < 0:
            y[0] = -y[0]
        for t in range(1, len(y)):
            if np.dot(y[t], y[t - 1]) < 0:
                y[t] = -y[t]
    else:
        y = np.cross(np.array([0.0, 0, 1]), x)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    R = np.stack([x, y, np.cross(x, y)], -1)
    c, s = np.cos(flat_pitch), np.sin(flat_pitch)
    return R @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


ANKLE_HEIGHT = 0.035  # G1 ankle_roll origin above the sole when the foot is flat


def planted(H, side):
    """Frames where the human foot is planted: toe near the floor and nearly still"""
    toe = H[f"{side}Toe"]
    floor = np.percentile(np.minimum(H["LeftToe"][:, 2], H["RightToe"][:, 2]), 5)
    speed = np.r_[0, np.linalg.norm(np.diff(toe[:, :2], axis=0), axis=1)] * FPS
    return (toe[:, 2] < floor + 0.03) & (speed < 0.2)


LOCK_RAMP = 0  # frames to blend into / out of a contact lock (0: snap; ramps let feet drift)
CONTACT_WEIGHT = 3.0  # foot task weight multiplier while that foot is planted
CONTACT_RAMP = 6  # frames over which the contact weight ramps (no abrupt switching)
CONTACT_RAMP_OUTSIDE = False  # True: ramp before touchdown / after lift-off (tested: worse overall)
LOCK_FLAT_ONLY = True  # lock/pin the ankle only while the whole foot is flat
LOCK_ANKLE = False  # the contact points are pinned instead; the ankle rolls freely
CONTACT_POINT_WEIGHT = 8.0
ANKLE_IN_CONTACT = 0.25  # ankle-position weight scale while that foot is in contact
OFF_CONTACT = 0.02  # contact-point weight scale while not in contact (nearly off)
HEEL_PIN = False  # pinning the heel too locks the whole foot: knee jitter in walking
LOCK_AT = "touchdown"  # anchor a planted foot where it touched down (no target jump)


def flat_planted(H, side):
    """Frames where the human foot is flat on the ground: toe planted AND the ankle low
    and still. After heel-off the ankle rises and rolls forward over the toe, so it must
    not be held in place then (that makes the legs jump at every lift-off)."""
    ankle = H[f"{side}Foot"]
    speed = np.r_[0, np.linalg.norm(np.diff(ankle, axis=0), axis=1)] * FPS
    low = ankle[:, 2] < np.percentile(ankle[:, 2], 5) + 0.03
    return planted(H, side) & low & (speed < 0.25)


def lock_contacts(p, mask, min_len=3, ramp=None):
    """Hold a foot target fixed (its mean) over each planted interval: mocap feet slide
    a little during contact, and copying that makes the robot's feet skate. The lock is
    blended in and out over `ramp` frames, so the target never steps."""
    ramp = LOCK_RAMP if ramp is None else ramp
    out = p.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < min_len:
            continue
        n = b - a
        k = np.arange(n)
        if ramp <= 0:
            w = np.ones(n)
        else:
            r = max(1, min(ramp, n // 2))
            w = np.minimum(1.0, np.minimum(k + 1, n - k) / r)  # 0 -> 1 -> 0 at the ends
            w = 0.5 - 0.5 * np.cos(np.pi * w)  # smooth ramp
        anchor = p[a:min(b, a + 2)].mean(0) if LOCK_AT == "touchdown" else p[a:b].mean(0)
        out[a:b] = w[:, None] * anchor + (1 - w[:, None]) * p[a:b]
    return out


def contact_scale(mask, high, ramp, low=1.0):
    """Per-frame profile: `high` while planted, `low` otherwise, ramped over `ramp` frames
    (low=1: a weight multiplier; low=0, high=1: a 0..1 contact indicator)"""
    if ramp > 0 and CONTACT_RAMP_OUTSIDE:
        # ramp OUTSIDE the contact: full strength from the first to the last contact
        # frame (a symmetric blur leaves touchdown frames weakly pinned, and they slide)
        m = np.asarray(mask, bool)
        d = m.copy()
        for k in range(1, ramp + 1):
            d[k:] |= m[:-k]
            d[:-k] |= m[k:]
        s = np.where(d, high, low).astype(float)
        k = np.exp(-0.5 * (np.arange(-2 * ramp, 2 * ramp + 1) / (ramp / 2)) ** 2)
        s = np.convolve(np.pad(s, 2 * ramp, mode="edge"), k / k.sum(), mode="valid")
        return np.where(m, high, s)
    s = np.where(mask, high, low).astype(float)
    if ramp > 0:
        k = np.exp(-0.5 * (np.arange(-2 * ramp, 2 * ramp + 1) / (ramp / 2)) ** 2)
        s = np.convolve(np.pad(s, 2 * ramp, mode="edge"), k / k.sum(), mode="valid")
    return s


def gaussian_smooth(x, sigma_frames):
    """Gaussian filter along time (axis 0), edge-padded"""
    if sigma_frames <= 0:
        return x
    r = int(np.ceil(3 * sigma_frames))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, xp)


def smooth_rotations(R, sigma_frames):
    """Smooth rotation matrices: Gaussian-average, then project back onto SO(3)"""
    if sigma_frames <= 0:
        return R
    shape = R.shape
    A = gaussian_smooth(R.reshape(shape[0], -1), sigma_frames).reshape(shape)
    U, _, Vt = np.linalg.svd(A)
    D = np.ones(shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return U @ (D[..., None] * Vt)


# Zero-phase target smoothing (s): none on the arms (keeps the motion lively), some on
# the legs (their mocap noise and contacts otherwise make the legs jitter)
SMOOTH = 0.0
LEG_SMOOTH = 0.033  # (re-tuned after the foot-frame fix)
# The pelvis / chest orientation targets are built from a few joint positions; their
# noise becomes rotation noise that the hip twist and waist chase. They are built from
# a separately smoothed copy (the limb directions keep the lively raw arms).
FRAME_SMOOTH = 0.1


FEET_JOINTS = ("LeftFoot", "RightFoot", "LeftToe", "RightToe")


LEG_JOINTS = ("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
              "RightUpLeg", "RightLeg", "RightFoot", "RightToe")


LEAD = 0.25  # frames: targets are sampled this far ahead to cancel the solver's tracking lag


def time_shift(x, lead):
    """Sample a (T, ...) signal at t + lead frames (linear interpolation, edge-clamped)"""
    if not lead:
        return x
    T = len(x)
    t = np.clip(np.arange(T) + lead, 0, T - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (t - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return (1 - w) * x[i0] + w * x[i1]


def _quat_mat(wxyz):
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, np.asarray(wxyz, float))
    return M.reshape(3, 3)


def _project_so3(R):
    U, _, Vt = np.linalg.svd(R)
    D = np.ones(R.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return U @ (D[..., None] * Vt)


def finger_poses(m):
    """(open, closed) finger joint angles. One-sided joint ranges (index, middle and the
    thumb tip: one end is 0 = open) close toward their other end. The two-sided thumb
    base joints close toward the end that brings the thumb tip to the index fingertip
    (forward kinematics, link centers of mass), so mirrored hands need no hand-written
    signs."""
    d = mujoco.MjData(m)
    open_q = np.zeros(len(FINGER_JOINTS))
    closed = np.zeros(len(FINGER_JOINTS))
    for side in ("left", "right"):
        for f in FINGERS:
            name = f"{side}_hand_{f}_joint"
            j, k = m.joint(name).id, FINGER_JOINTS.index(name)
            lo, hi = m.jnt_range[j]
            open_q[k] = np.clip(0.0, lo, hi)
            if abs(lo) < 1e-6 or abs(hi) < 1e-6:  # one-sided: the nonzero end
                closed[k] = 0.85 * (lo if abs(hi) < 1e-6 else hi)
                continue
            best = None
            for end in (lo, hi):  # two-sided thumb joint: toward the index fingertip
                d.qpos[:] = m.qpos0
                d.qpos[m.jnt_qposadr[j]] = end
                mujoco.mj_kinematics(m, d)
                dist = np.linalg.norm(d.xipos[m.body(f"{side}_hand_thumb_2_link").id]
                                      - d.xipos[m.body(f"{side}_hand_index_1_link").id])
                if best is None or dist < best[0]:
                    best = (dist, end)
            closed[k] = 0.6 * best[1]
    return open_q, closed


def grip_signal(H, side, fight=True):
    """0 = open flat, GRIP_RELAXED = relaxed, 1 = fist; per frame, smooth.

    Fist when (a) in a fighting guard: hand raised between chest and head height, in
    front of the chest, near the centerline, elbow bent; or (b) striking: hand moving
    fast relative to the chest, held while the arm stays extended. Open flat while the
    hand rests on the floor. (Synthesized: LAFAN1 has no finger data.) Fists only in
    fighting clips (`fight`: from the LAFAN1 clip name); elsewhere hands stay relaxed,
    since a dancer's hands in front of the chest are not a guard."""
    hand, chest, head, shoulder = H[f"{side}Hand"], H["Spine2"], H["Head"], H[f"{side}Arm"]
    F = _frame(H["LeftArm"] - H["RightArm"], H["Neck"] - H["Hips"])  # chest frame
    rel = np.einsum("tji,tj->ti", F, hand - chest)  # hand in chest frame: x fwd, y left, z up
    reach = np.linalg.norm(hand - shoulder, axis=1)
    # guard, in the CHEST frame (the torso leans a lot in kicks): hands in front of the
    # chest at chest-to-chin height, near the centerline, elbows bent
    guard = ((rel[:, 2] > -0.12) & (rel[:, 2] < 0.30) & (rel[:, 0] > 0.12)
             & (np.abs(rel[:, 1]) < 0.25) & (reach < 0.38))
    v_rel = np.r_[0, np.linalg.norm(np.diff(hand - chest, axis=0), axis=1)] * FPS
    strike = (v_rel > STRIKE_SPEED) & (rel[:, 0] > 0.15)  # fast, out in front
    # hold the fist while the striking arm stays out in front, at most 0.6 s
    held, run = strike.copy(), 0
    for t in range(1, len(held)):
        run = run + 1 if (held[t - 1] and not strike[t]) else 0
        held[t] |= held[t - 1] and rel[t, 0] > 0.25 and reach[t] > 0.42 and run < 18
    fist = (guard | held) if fight else np.zeros(len(hand), bool)
    fist = np.convolve(fist.astype(float), np.ones(7), mode="same") > 0  # +-3 frames
    g = np.where(fist, 1.0, GRIP_RELAXED)
    g = np.where(hand_on_floor(H, side), 0.0, g)
    return gaussian_smooth(g, 3.0)


def hand_on_floor(H, side):
    floor = np.percentile(np.minimum(H["LeftToe"][:, 2], H["RightToe"][:, 2]), 5)
    hand = H[f"{side}Hand"]
    speed = np.r_[0, np.linalg.norm(np.diff(hand, axis=0), axis=1)] * FPS
    return (hand[:, 2] - floor < 0.10) & (speed < 0.3)


def human_targets(H, scale, smooth=SMOOTH, leg_smooth=LEG_SMOOTH, lead=None, Hrot=None,
                  fingers=None, clip=""):
    # Smooth every joint (raw foot noise makes the legs jitter), but detect contact
    # phases on the raw data and lock the foot targets inside them, so contacts stay
    # crisp: smooth swing, still stance.
    # offline: sample the motion slightly ahead so the solver's small tracking delay
    # cancels (contacts are detected on the shifted data too, so they stay aligned)
    lead = LEAD if lead is None else lead
    H = {k: time_shift(v, lead) for k, v in H.items()}
    raw = H
    leg_smooth = smooth if leg_smooth is None else leg_smooth
    H = {k: gaussian_smooth(v, (leg_smooth if k in LEG_JOINTS else smooth) * FPS)
         for k, v in H.items()}
    pos = np.stack([H[h] * scale for _, h, _ in POS_TASKS], 1)
    # One global height shift: planted ankles land at the G1's flat-foot ankle height
    # (the IK floor constraint then keeps every frame above the floor)
    feet = {"LeftFoot": "Left", "RightFoot": "Right"}
    planted_z = np.concatenate([raw[f][planted(raw, s), 2] * scale for f, s in feet.items()])
    pos[..., 2] -= (np.median(planted_z) if len(planted_z) > 10 else 0.0) - ANKLE_HEIGHT
    pos_scale = np.ones(pos.shape[:2])
    for k, (_, h, _) in enumerate(POS_TASKS):
        if h in feet and not LOCK_ANKLE:
            # in contact, the toe/heel points govern the foot; a full-weight ankle target
            # would fight them (scaled human and robot feet never match exactly)
            pos_scale[:, k] = 1.0 - (1.0 - ANKLE_IN_CONTACT) * contact_scale(
                planted(raw, feet[h]), 1.0, CONTACT_RAMP, low=0.0)
        if h in feet and LOCK_ANKLE:
            mask = (flat_planted if LOCK_FLAT_ONLY else planted)(raw, feet[h])
            pos[:, k] = lock_contacts(pos[:, k], mask)
            pos_scale[:, k] = contact_scale(mask, CONTACT_WEIGHT, CONTACT_RAMP)
    dirs = np.stack([H[b] - H[a] for *_, a, b, _ in DIR_TASKS], 1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    Hf = {k: gaussian_smooth(raw[k], FRAME_SMOOTH * FPS)
          for k in ("LeftUpLeg", "RightUpLeg", "Spine1", "Hips", "LeftArm", "RightArm", "Neck")}
    frames = {
        "hips": _frame(Hf["LeftUpLeg"] - Hf["RightUpLeg"], Hf["Spine1"] - Hf["Hips"]),
        # whole-trunk up: the G1 torso is one rigid block
        "chest": _frame(Hf["LeftArm"] - Hf["RightArm"], Hf["Neck"] - Hf["Hips"]),
        "lfoot": _foot_frame(H["LeftFoot"], H["LeftToe"], stance_bone_pitch(raw, "Left"),
                             H["LeftLeg"]),
        "rfoot": _foot_frame(H["RightFoot"], H["RightToe"], stance_bone_pitch(raw, "Right"),
                             H["RightLeg"]),
    }
    rot = np.stack([frames[f] for _, f, _ in ROT_TASKS], 1)
    # Contact-point targets: where the robot's toe / heel sole points would be given the
    # (scaled) human ankle and foot frame; on the floor and anchored at touchdown while
    # in contact, nearly weightless otherwise.
    ankle_k = {h: k for k, (_, h, _) in enumerate(POS_TASKS)}
    foot_rot = {"LeftFoot": frames["lfoot"], "RightFoot": frames["rfoot"]}
    toe_t, heel_t, toe_s, heel_s = [], [], [], []
    for h, side in feet.items():
        ank = pos[:, ankle_k[h]]
        for point, mask, out_t, out_s in (
                (TOE_POINT, planted(raw, side), toe_t, toe_s),
                (HEEL_POINT, flat_planted(raw, side) & HEEL_PIN, heel_t, heel_s)):
            p = ank + np.einsum("tij,j->ti", foot_rot[h], point)
            # onto the floor while in contact, blended in/out (no height step at
            # touchdown: the knee would absorb it in one frame, at every step)
            c = contact_scale(mask, 1.0, CONTACT_RAMP, low=0.0)  # 0..1 contact indicator
            p[:, 2] = (1 - c) * p[:, 2]
            out_t.append(lock_contacts(p, mask))
            out_s.append(c * (1 - OFF_CONTACT) + OFF_CONTACT)
    toe = (np.stack(toe_t, 1), np.stack(toe_s, 1))
    heel = (np.stack(heel_t, 1), np.stack(heel_s, 1))
    rot_scale = np.ones(rot.shape[:2])
    for k, (_, f, _) in enumerate(ROT_TASKS):
        if f in ("lfoot", "rfoot"):
            side = "Left" if f == "lfoot" else "Right"
            rot_scale[:, k] = contact_scale((flat_planted if LOCK_FLAT_ONLY else planted)(raw, side),
                                            CONTACT_WEIGHT, CONTACT_RAMP)
    # arm link orientations from the human bone rotations (time-shifted like the rest)
    if Hrot is None:
        arm_rot = np.broadcast_to(np.eye(3), (len(pos), len(ARM_ROT_TASKS), 3, 3)).copy()
    else:
        arm_rot = np.stack([smooth_rotations(_project_so3(time_shift(Hrot[bone], lead)),
                                             ARM_ROT_SMOOTH * FPS) @ _quat_mat(off)
                            for _, bone, _, off in ARM_ROT_TASKS], 1)
    out = [(pos, pos_scale), toe, heel, dirs, (rot, rot_scale), arm_rot]
    # forearm twist: the human forearm bone orientation (only its twist is used)
    out.append(arm_rot[:, [i for i, (b, *_) in enumerate(ARM_ROT_TASKS) if "elbow" in b]])
    if HANDS:
        # palm height: on the floor while the human hand rests there
        palm_t, palm_s = [], []
        for side in ("Left", "Right"):
            mask = hand_on_floor(raw, side)
            c = contact_scale(mask, 1.0, CONTACT_RAMP, low=0.0)
            palm_t.append(np.full(len(mask), PALM_HEIGHT))
            palm_s.append(c * (1 - OFF_CONTACT) + OFF_CONTACT)
        out.append((np.stack(palm_t, 1), np.stack(palm_s, 1)))
        # fingers (synthesized grip: LAFAN1 has no finger data)
        open_q, closed = fingers
        fight = "fight" in clip.lower()
        g = np.stack([grip_signal(raw, "Left" if f.startswith("left") else "Right", fight)
                      for f in FINGER_JOINTS], 1)
        out.append(open_q + g * (closed - open_q))
    return tuple(out)


# ---------------- solver setup ----------------

def per_dof(m, model, values: dict, default=0.0):
    """{joint-name substring: value} -> per-DOF vector (first match wins)"""
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(m.njnt)]
    out = np.full(model.nv, float(default))
    for j, n in enumerate(names):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        w = next((v for k, v in values.items() if k in n), None)
        if w is not None:
            out[m.jnt_dofadr[j]] = w
    return out


def build_solver(m, model, collision=True, cover=95.0, margin=0.01, posture=None,
                 damping=None, **kw):
    post_w = per_dof(m, model, POSTURE if posture is None else posture)
    damping = DAMPING if damping is None else damping
    kw["damping"] = per_dof(m, model, damping, default=damping.get("default", 0.05)) \
        if isinstance(damping, dict) else damping
    kw.setdefault("max_step", MAX_STEP)
    if "joint_ranges" not in kw and RETARGET_RANGES:
        kw["joint_ranges"] = {m.joint(n).id: r for n, r in RETARGET_RANGES.items()}
    kw.setdefault("collision_partners", 2)
    feet_b = ["left_ankle_roll_link", "right_ankle_roll_link"]
    tasks = [
        ik.position(model, [b for b, _, _ in POS_TASKS], [w for *_, w in POS_TASKS]),
        # contact points: toe (front) and heel (rear) under each foot, pinned only while
        # in contact (per-frame weights); what touches the floor must not slide
        ik.position(model, feet_b, CONTACT_POINT_WEIGHT, offset=np.tile(TOE_POINT, (2, 1))),
        ik.position(model, feet_b, CONTACT_POINT_WEIGHT, offset=np.tile(HEEL_POINT, (2, 1))),
        ik.direction(model, [b for b, *_ in DIR_TASKS], [b for _, b, *_ in DIR_TASKS],
                     [w for *_, w in DIR_TASKS]),
        ik.orientation(model, [b for b, _, _ in ROT_TASKS], [w for *_, w in ROT_TASKS]),
        ik.orientation(model, [b for b, *_ in ARM_ROT_TASKS], [w for _, _, w, _ in ARM_ROT_TASKS]
                       if TRACK_HANDS else [1e-6] * len(ARM_ROT_TASKS)),
    ]
    # forearm twist (both models: the forearm is the same)
    d0 = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d0)
    axes = []
    for side in ("left", "right"):
        e, w = m.body(f"{side}_elbow_link").id, m.body(f"{side}_wrist_yaw_link").id
        axes.append(d0.xmat[e].reshape(3, 3).T @ (d0.xpos[w] - d0.xpos[e]))
    tasks.append(ik.twist(model, ["left_elbow_link", "right_elbow_link"], axes,
                          FOREARM_TWIST_WEIGHT))
    if HANDS:  # palm floor contact needs the Dex3 palm geometry
        mh = mujoco.MjModel.from_xml_path(str(HANDS_XML))
        dh = mujoco.MjData(mh)
        mujoco.mj_kinematics(mh, dh)
        palms = []
        for side in ("left", "right"):
            w = mh.body(f"{side}_wrist_yaw_link").id
            palm = np.mean([dh.xpos[mh.body(f"{side}_hand_{c}_0_link").id]
                            for c in ("index", "middle", "thumb")], 0)
            palms.append(dh.xmat[w].reshape(3, 3).T @ (palm - dh.xpos[w]))
        tasks.append(ik.height(model, ["left_wrist_yaw_link", "right_wrist_yaw_link"], palms,
                               PALM_WEIGHT))
    tasks.append(ik.PostureTask(post_w))

    def kind(n):
        return "arm" if any(k in n for k in ARM) else "body" if any(k in n for k in BODY) else None

    def side(n):
        return n.split("_")[0] if n.split("_")[0] in ("left", "right") else "mid"

    def pair_filter(a, b):  # arms vs body and vs the other arm
        # (the upper arm vs torso pair is kept: the arm swings into the chest when
        # sprinting; contact in the rest pose is relaxed automatically by SphereModel)
        ka, kb = kind(a), kind(b)
        return "arm" in (ka, kb) and not (ka == kb == "arm" and side(a) == side(b))

    feet_pts = ik.FloorPoints.on(model, ["left_ankle_roll_link", "right_ankle_roll_link"],
                                 FOOT_SPHERES - np.array([0.0, 0.0, FOOT_RADIUS]))
    if WHOLE_BODY_FLOOR:  # every collision sphere of the body stays above the floor
        # (Dex3: plus the open-hand finger envelope; fingers are open on the floor)
        body_spheres = ik.SphereModel.fit(m, model, pair_filter=lambda a, b: False,
                                          extra=finger_envelope(0.0) if HANDS else None)
        floor = ik.FloorPoints.spheres(body_spheres, lowest=FLOOR_ROWS, extra=feet_pts)
    else:
        floor = feet_pts
    # (Dex3: the relaxed-hand finger envelope also collides; a fist stays inside it)
    spheres = ik.SphereModel.fit(m, model, bodies=lambda n: kind(n) is not None,
                                 pair_filter=pair_filter, cover=cover, margin=margin,
                                 extra=finger_envelope(GRIP_RELAXED) if HANDS else None
                                 ) if collision else None
    return ik.IKSolver(model, tasks, collision=spheres, floor=floor, **kw)


def to_output(qs, finger_q, m_ik, m_out):
    """Body solution (G1 layout) + finger angles -> output layout (G1 + Dex3)"""
    qs = np.asarray(qs, np.float64)
    if m_out.nq == m_ik.nq:
        return qs
    Q = np.tile(m_out.qpos0, (len(qs), 1))
    Q[:, :7] = qs[:, :7]
    for j in range(1, m_ik.njnt):
        n = mujoco.mj_id2name(m_ik, mujoco.mjtObj.mjOBJ_JOINT, j)
        Q[:, m_out.jnt_qposadr[m_out.joint(n).id]] = qs[:, m_ik.jnt_qposadr[j]]
    for k, n in enumerate(FINGER_JOINTS):
        Q[:, m_out.jnt_qposadr[m_out.joint(n).id]] = finger_q[:, k]
    return Q


def finger_envelope(grip):
    """Dex3 finger collision geometry at a grip (0 open .. 1 fist), as spheres attached
    rigidly to each wrist_yaw_link: the solver does not move the fingers, but it must
    keep them out of the floor and the body. -> [(body name, centers, radii)]"""
    mh = mujoco.MjModel.from_xml_path(str(HANDS_XML))
    from rigidax import from_mujoco
    from rigidax.ik.collision import _body_points, _fit_spheres
    open_q, closed = finger_poses(mh)
    d = mujoco.MjData(mh)
    d.qpos[:] = mh.qpos0
    for k, n in enumerate(FINGER_JOINTS):
        d.qpos[mh.jnt_qposadr[mh.joint(n).id]] = open_q[k] + grip * (closed[k] - open_q[k])
    mujoco.mj_kinematics(mh, d)
    out = []
    for side in ("left", "right"):
        w = mh.body(f"{side}_wrist_yaw_link").id
        Rw, pw = d.xmat[w].reshape(3, 3), d.xpos[w]
        C, Rr = [], []
        for f in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"):
            b = mh.body(f"{side}_hand_{f}_link").id
            V = _body_points(mh, b, 3)
            if len(V) < 4:
                continue
            Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
            Vw = V @ Rb.T + pb  # world
            c, r = _fit_spheres((Vw - pw) @ Rw, FINGER_SPHERE_SPACING, 95.0)  # wrist frame
            C.append(c), Rr.append(r)
        out.append((f"{side}_wrist_yaw_link", np.concatenate(C), np.concatenate(Rr)))
    return out


def put_on_floor(model, qs):  # legacy post-processing; the IK floor constraint replaces it
    """Shift the root so feet touch the floor during contact and never sink below it"""
    lf, rf = model.body_id("left_ankle_roll_link"), model.body_id("right_ankle_roll_link")

    @jax.jit
    @jax.vmap
    def lowest(q):
        kin = rx.forward_kinematics(model, q)
        pts = [kin.body_p[b] + FOOT_SPHERES @ kin.body_R[b].T for b in (lf, rf)]
        return jnp.min(jnp.concatenate(pts)[:, 2]) - FOOT_RADIUS

    qs = np.array(qs)
    qs[:, 2] -= np.percentile(np.asarray(lowest(jnp.asarray(qs))), 10)
    qs[:, 2] += np.maximum(0.0, -np.asarray(lowest(jnp.asarray(qs))))
    return qs


def leg_scale(m, model, H):
    kin0 = rx.forward_kinematics(model, m.qpos0)
    bp = lambda n: np.asarray(kin0.body_p[model.body_id(n)])  # noqa: E731
    robot = (np.linalg.norm(bp("left_hip_pitch_link") - bp("left_knee_link"))
             + np.linalg.norm(bp("left_knee_link") - bp("left_ankle_roll_link")))
    human = np.median(np.linalg.norm(H["LeftUpLeg"] - H["LeftLeg"], axis=-1)
                      + np.linalg.norm(H["LeftLeg"] - H["LeftFoot"], axis=-1))
    return robot / human


# Multi-start arm seeds for each segment's first frame (shoulder pitch, roll, yaw, elbow):
# Gauss-Newton from a single arms-down pose can settle on the wrong arm configuration.
ARM_SEEDS = [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.2), (-1.4, 0.0, 0.0, 0.3),
             (0.0, 1.3, 0.0, 0.3), (-2.6, 0.3, 0.0, 0.3), (0.5, 0.2, 0.0, 1.5)]


def make_seed(m, n_seeds=len(ARM_SEEDS)):
    def adr(n):
        return m.jnt_qposadr[m.joint(n).id]

    cands = []
    for pitch, roll, yaw, elbow in ARM_SEEDS[:max(1, n_seeds)]:
        q = m.qpos0.copy()
        for side, sgn in (("left", 1.0), ("right", -1.0)):
            q[adr(f"{side}_shoulder_pitch_joint")] = pitch
            q[adr(f"{side}_shoulder_roll_joint")] = sgn * roll
            q[adr(f"{side}_shoulder_yaw_joint")] = sgn * yaw
            q[adr(f"{side}_elbow_joint")] = elbow
        cands.append(q)
    cands = jnp.asarray(np.stack(cands))

    def seed(targets):  # place the root between the target hip joints
        pos = targets[0][0] if isinstance(targets[0], tuple) else targets[0]
        hips = 0.5 * (pos[0] + pos[1])
        return cands.at[:, :3].set(hips + jnp.array([0.0, 0.0, 0.1]))
    return seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--iters", type=int, default=2, help="IK iterations per frame")
    ap.add_argument("--segment", type=int, default=256)
    ap.add_argument("--no-predict", action="store_true")
    ap.add_argument("--seeds", type=int, default=3, help="multi-start candidates")
    ap.add_argument("--init-iters", type=int, default=15)
    ap.add_argument("--overlap", type=int, default=8, help="warm-up / seam-blend frames")
    ap.add_argument("--max-step", type=float, default=MAX_STEP,
                    help="trust region (rad per iteration); use ~0.3 with --iters 1")
    ap.add_argument("--qp-iters", type=int, default=8)
    ap.add_argument("--no-collision", action="store_true")
    ap.add_argument("--f64", action="store_true", help="double precision (slower)")
    ap.add_argument("--gpu", action="store_true", help="run on the GPU (needs CUDA jaxlib)")
    ap.add_argument("--no-hands", action="store_true", help="plain G1 (no Dex3 fingers)")
    ap.add_argument("--out", default=str(OUT_HANDS if HANDS else OUT_PLAIN))
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(IK_XML))  # solve on the body model
    m_out = mujoco.MjModel.from_xml_path(str(G1_XML))
    model = rx.from_mujoco(m)
    solver = build_solver(m, model, collision=not args.no_collision, qp_iters=args.qp_iters,
                          max_step=args.max_step)
    seed = make_seed(m, args.seeds)
    print(f"{len(jax.devices())} x {jax.devices()[0].platform} device(s); collision spheres: "
          f"{0 if solver.collision is None else solver.collision.num_spheres}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scales, targets, fingers_all = [], [], []
    for clip in args.clips:
        H = bvh_positions(LAFAN / f"{clip}.bvh")
        scales.append(leg_scale(m, model, H))
        Hr = bvh_rotations(LAFAN / f"{clip}.bvh")
        t = human_targets(H, scales[-1], Hrot=Hr,
                          fingers=finger_poses(m_out) if HANDS else None, clip=clip)
        if HANDS:  # the finger angles bypass the solver
            fingers_all.append(t[-1])
            t = t[:-1]
        targets.append(jax.tree_util.tree_map(jnp.asarray, t))
    total = sum(jax.tree_util.tree_leaves(t)[0].shape[0] for t in targets)

    def run():  # all clips in one batch
        return solver.solve_trajectory(seed, targets, segment=args.segment, iters=args.iters,
                                       init_iters=args.init_iters, overlap=args.overlap,
                                       predict=not args.no_predict)

    t0 = time.perf_counter()
    jax.block_until_ready(run())
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    qs_all, costs_all = jax.block_until_ready(run())  # compiled: steady-state throughput
    dt = time.perf_counter() - t0
    print(f"{len(args.clips)} clips, {total} frames ({total / FPS / 60:.1f} min of motion): "
          f"{total / dt:,.0f} frames/s  (first call incl. compile {t_first:.1f}s)")
    for i, (clip, qs, costs, scale) in enumerate(zip(args.clips, qs_all, costs_all, scales)):
        q_out = to_output(qs, fingers_all[i] if HANDS else None, m, m_out)
        np.savez(out / f"{clip}.npz", qpos=q_out, fps=FPS, scale=scale, model=str(G1_XML.name))
        print(f"  {clip}: {len(qs)} frames, mean cost {float(np.mean(costs)):.4f}")


if __name__ == "__main__":
    main()
