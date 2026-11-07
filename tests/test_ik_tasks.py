"""Tests for the newer rigidax.ik pieces: twist / height / joint tasks, whole-body floor
spheres, trajectory smoothing and the BVH reader"""

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

import rigidax as rx
from rigidax import ik
from rigidax.ik.bvh import bvh_positions, bvh_rotations

G1 = Path(__file__).parents[1] / "examples" / "unitree_g1" / "g1.xml"


@pytest.fixture(scope="module")
def g1():
    m = mujoco.MjModel.from_xml_path(str(G1))
    return m, rx.from_mujoco(m)


def _hold_body(model, m):
    """tasks + targets keeping the floating base still (so only limbs move)"""
    kin = rx.forward_kinematics(model, m.qpos0)
    b = np.array([model.body_id("pelvis"), model.body_id("torso_link")])
    tasks = [ik.position(model, ["pelvis", "torso_link"], 10.0),
             ik.orientation(model, ["pelvis", "torso_link"], 10.0)]
    return tasks, [kin.body_p[b], kin.body_R[b]]


def _solve(solver, q0, targets, iters=60):
    (q, _), _ = jax.jit(lambda q: solver.solve(q, tuple(targets), iters))(jnp.asarray(q0))
    return q


def test_joint_task_reaches_targets(g1):
    m, model = g1
    names = ["left_elbow_joint", "right_knee_joint", "right_shoulder_pitch_joint"]
    tgt = jnp.array([0.9, 1.1, -0.3])
    tasks, targets = _hold_body(model, m)
    solver = ik.IKSolver(model, tasks + [ik.joints(model, m, names, 5.0)])
    q = _solve(solver, m.qpos0, targets + [tgt])
    got = np.array([q[m.jnt_qposadr[m.joint(n).id]] for n in names])
    np.testing.assert_allclose(got, tgt, atol=2e-3)


def test_twist_task_only_rotates_about_its_axis(g1):
    """On a link with a pure roll joint (G1 wrist_roll_link, axis along the forearm), a
    twist target is reached through that roll alone: the forearm direction is unchanged"""
    m, model = g1
    link = "left_wrist_roll_link"
    b = model.body_id(link)
    j = m.joint("left_wrist_roll_joint").id
    axis_local = np.array(m.jnt_axis[j])
    kin0 = rx.forward_kinematics(model, m.qpos0)
    R0 = np.asarray(kin0.body_R[b])
    aw = R0 @ axis_local
    K = np.array([[0, -aw[2], aw[1]], [aw[2], 0, -aw[0]], [-aw[1], aw[0], 0]])
    roll = np.eye(3) + np.sin(0.6) * K + (1 - np.cos(0.6)) * K @ K
    target = jnp.asarray(roll @ R0)[None]
    tasks, targets = _hold_body(model, m)
    e, wy = model.body_id("left_elbow_link"), model.body_id("left_wrist_yaw_link")
    sh = model.body_id("left_shoulder_roll_link")
    kp = kin0.body_p
    d_now = jnp.stack([kp[e] - kp[sh], kp[wy] - kp[e]])
    d_now = d_now / jnp.linalg.norm(d_now, axis=1, keepdims=True)
    dirs = ik.direction(model, ["left_elbow_link", "left_wrist_yaw_link"],
                        ["left_shoulder_roll_link", "left_elbow_link"], 3.0)
    solver = ik.IKSolver(model, tasks + [dirs, ik.twist(model, [link], axis_local, 2.0)])
    q = _solve(solver, m.qpos0, targets + [d_now, target], iters=100)
    kin = rx.forward_kinematics(model, q)
    rel = np.asarray(kin.body_R[b]) @ R0.T
    twist_angle = float(np.asarray(ik.rotation_error(jnp.eye(3)[None], jnp.asarray(rel)[None]))[0] @ aw)
    assert abs(twist_angle - 0.6) < 0.03
    assert abs(float(q[m.jnt_qposadr[j]]) - 0.6) < 0.05  # done by the roll joint itself
    d = np.asarray(kin.body_p[wy] - kin.body_p[e])
    assert np.degrees(np.arccos(d @ np.asarray(d_now[1]) / np.linalg.norm(d))) < 1.0


def test_height_task_and_per_frame_scale(g1):
    m, model = g1
    tasks, targets = _hold_body(model, m)
    h = ik.height(model, ["left_wrist_yaw_link"], [0.0, 0.0, 0.0], 5.0)
    solver = ik.IKSolver(model, tasks + [h])
    q = _solve(solver, m.qpos0, targets + [jnp.array([0.9])])
    assert abs(float(rx.forward_kinematics(model, q).body_p[model.body_id("left_wrist_yaw_link"), 2]) - 0.9) < 5e-3
    # with a (target, scale ~ 0) pair the task is switched off
    q_off = _solve(solver, m.qpos0, targets + [(jnp.array([0.9]), jnp.array([1e-6]))])
    z0 = float(rx.forward_kinematics(model, m.qpos0).body_p[model.body_id("left_wrist_yaw_link"), 2])
    z1 = float(rx.forward_kinematics(model, q_off).body_p[model.body_id("left_wrist_yaw_link"), 2])
    assert abs(z1 - z0) < 0.01


def test_whole_body_floor_spheres_stay_above_floor(g1):
    """Pull the pelvis to the floor: with sphere floor rows no sphere goes below z = 0"""
    m, model = g1
    spheres = ik.SphereModel.fit(m, model, pair_filter=lambda a, b: False)
    floor = ik.FloorPoints.spheres(spheres, lowest=12)
    task = ik.position(model, ["pelvis"], 5.0)
    solver = ik.IKSolver(model, [task], floor=floor)
    q = _solve(solver, m.qpos0, [jnp.array([[0.0, 0.0, -0.3]])], iters=80)
    kin = rx.forward_kinematics(model, q)
    z = np.asarray(spheres.centers(kin))[:, 2] - spheres.radius
    assert z.min() > -2e-3


def test_smooth_trajectory_is_zero_phase(g1):
    m, model = g1
    T = 200
    q = np.tile(m.qpos0, (T, 1))
    k = m.jnt_qposadr[m.joint("left_knee_joint").id]
    q[:, k] = 0.5 + 0.4 * np.sin(np.linspace(0, 6 * np.pi, T))
    noisy = q.copy()
    noisy[:, k] += np.random.default_rng(0).normal(0, 0.05, T)
    sm = ik.smooth_trajectory(model, noisy, 2.0)
    # removes noise ...
    assert np.std(sm[:, k] - q[:, k]) < 0.5 * np.std(noisy[:, k] - q[:, k])
    # ... without shifting the signal in time (cross-correlation peaks at lag 0)
    a, b = q[20:-20, k] - q[:, k].mean(), sm[20:-20, k] - sm[:, k].mean()
    lags = range(-5, 6)
    assert max(lags, key=lambda L: float(np.dot(a, np.roll(b, L)))) == 0
    # quaternion of the free base stays unit length
    np.testing.assert_allclose(np.linalg.norm(sm[:, 3:7], axis=1), 1.0, atol=1e-9)


def test_bvh_reader(tmp_path):
    """A 2-joint BVH: root translates, child offset rotated 90 deg about Z"""
    bvh = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Leg
  {
    OFFSET 10 0 0
    CHANNELS 3 Zrotation Yrotation Xrotation
    End Site
    {
      OFFSET 0 0 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
0 100 0 0 0 0 0 0 0
0 100 0 90 0 0 0 0 0
"""
    f = tmp_path / "t.bvh"
    f.write_text(bvh)
    P = bvh_positions(f)  # cm -> m, Y-up -> Z-up: (x, y, z)_bvh -> (x, -z, y)
    np.testing.assert_allclose(P["Hips"][0], [0, 0, 1.0], atol=1e-9)
    np.testing.assert_allclose(P["Leg"][0], [0.1, 0, 1.0], atol=1e-9)
    np.testing.assert_allclose(P["Leg"][1], [0, 0, 1.1], atol=1e-9)  # +90 deg about Z_bvh
    R = bvh_rotations(f)
    np.testing.assert_allclose(R["Leg"][1] @ R["Leg"][1].T, np.eye(3), atol=1e-9)
