"""Tests for rigidax.ik: the QP solver, IK convergence, constraints, trajectories"""

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

import rigidax as rx
from rigidax import ik
from rigidax.ik import INF, solve_qp

G1 = Path(__file__).parents[1] / "examples" / "unitree_g1" / "g1.xml"


# ---------------- QP ----------------

def test_qp_box_closed_form():
    """Diagonal H with box bounds: x* = clip(g / h, lo, hi)"""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 12
        h = rng.uniform(0.5, 3.0, n)
        g = rng.normal(size=n) * 3
        lo, hi = -rng.uniform(0.1, 1, n), rng.uniform(0.1, 1, n)
        res = solve_qp(jnp.diag(h), jnp.asarray(g), jnp.eye(n), jnp.asarray(lo), jnp.asarray(hi),
                       iters=300)
        np.testing.assert_allclose(res.x, np.clip(g / h, lo, hi), atol=1e-6)


def test_qp_kkt_general_inequalities():
    """Random dense QP: the solution is feasible and satisfies KKT stationarity"""
    rng = np.random.default_rng(1)
    for _ in range(10):
        n, m = 10, 8
        M = rng.normal(size=(20, n))
        H = M.T @ M + 0.1 * np.eye(n)
        g = rng.normal(size=n) * 5
        A = rng.normal(size=(m, n))
        l, u = rng.normal(size=m) - 1.0, np.full(m, INF)
        res = solve_qp(*map(jnp.asarray, (H, g, A, l, u)), iters=2000, polish=True)
        x, y = np.asarray(res.x), np.asarray(res.y)
        assert np.all(A @ x >= l - 1e-5)
        # stationarity: H x - g + A^T y = 0 (y <= 0 on active lower bounds)
        np.testing.assert_allclose(H @ x - g + A.T @ y, 0, atol=1e-4)


def test_qp_elastic_is_always_feasible():
    """Contradictory elastic rows (x >= 1 and x <= -1) give a finite L1 compromise"""
    H, g = jnp.eye(1), jnp.zeros(1)
    A = jnp.array([[1.0], [1.0]])
    l, u = jnp.array([1.0, -INF]), jnp.array([INF, -1.0])
    res = solve_qp(H, g, A, l, u, elastic_mu=jnp.array([10.0, 10.0]), iters=400)
    assert np.isfinite(res.x).all() and abs(float(res.x[0])) < 1e-3  # symmetric penalties


# ---------------- IK ----------------

@pytest.fixture(scope="module")
def g1():
    m = mujoco.MjModel.from_xml_path(str(G1))
    return m, rx.from_mujoco(m)


def _targets(model, q, bodies):
    kin = rx.forward_kinematics(model, q)
    ids = [model.body_id(b) for b in bodies]
    return kin.body_p[np.array(ids)], kin.body_R[np.array(ids)]


def test_ik_recovers_reachable_pose(g1):
    m, model = g1
    bodies = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link",
              "left_wrist_yaw_link", "right_wrist_yaw_link", "left_knee_link", "right_knee_link"]
    solver = ik.IKSolver(model, [ik.position(model, bodies), ik.orientation(model, bodies)],
                         damping=1e-6, max_step=None)
    rng = np.random.default_rng(0)
    q_true = m.qpos0.copy()
    q_true[7:] = np.clip(q_true[7:] + rng.normal(size=29) * 0.3, m.jnt_range[1:, 0], m.jnt_range[1:, 1])
    p, R = _targets(model, q_true, bodies)
    (q, _), _ = jax.jit(lambda q0: solver.solve(q0, (p, R), iters=60))(jnp.asarray(m.qpos0))
    p2, R2 = _targets(model, q, bodies)
    assert float(jnp.max(jnp.abs(p2 - p))) < 1e-4
    assert float(jnp.max(jnp.abs(R2 - R))) < 1e-3


def test_ik_respects_joint_limits(g1):
    m, model = g1
    solver = ik.IKSolver(model, [ik.position(model, ["left_wrist_yaw_link"])])
    far = jnp.array([[3.0, 3.0, 3.0]])  # unreachable: arm stretches into its limits
    (q, _), _ = jax.jit(lambda q0: solver.solve(q0, (far,), iters=40))(jnp.asarray(m.qpos0))
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    assert np.all(np.asarray(q[7:]) >= lo - 1e-6) and np.all(np.asarray(q[7:]) <= hi + 1e-6)


def test_ik_self_collision_keeps_spheres_apart(g1):
    m, model = g1
    arm = ("elbow", "wrist")
    spheres = ik.SphereModel.fit(
        m, model, bodies=lambda n: any(k in n for k in arm + ("torso", "pelvis")),
        pair_filter=lambda a, b: (any(k in a for k in arm)) != (any(k in b for k in arm)))
    kin0 = rx.forward_kinematics(model, m.qpos0)
    C = np.asarray(spheres.centers(kin0))
    # Target: move the elbow 4 cm past contact with the nearest torso sphere that has
    # the normal margin (pairs touching in the rest pose get a relaxed allowance)
    # (the G1 hand link has no collision geometry; the elbow does)
    eb, tb = model.body_id("left_elbow_link"), model.body_id("torso_link")
    gaps0 = np.asarray(spheres.gaps(kin0))
    cand = [k for k, (a, b) in enumerate(spheres.pairs)
            if {spheres.body[a], spheres.body[b]} == {eb, tb} and spheres.min_dist[k] > 0]
    k = cand[int(np.argmin(gaps0[cand]))]
    a, b = spheres.pairs[k]
    e_s, t_s = (a, b) if spheres.body[a] == eb else (b, a)
    d = (C[t_s] - C[e_s]) / np.linalg.norm(C[t_s] - C[e_s])
    inside = jnp.asarray(C[e_s] + d * (gaps0[k] + 0.04))
    # drive that elbow sphere's center itself (a body-frame offset on the elbow link)
    task = ik.position(model, ["left_elbow_link"], 1.0, offset=spheres.center[e_s])
    # hold the body still, so only the arm can reach (else the floating base just moves)
    hold = ["pelvis", "torso_link"]
    hp, hR = _targets(model, m.qpos0, hold)
    tasks = [task, ik.position(model, hold, 10.0), ik.orientation(model, hold, 10.0)]
    targets = (inside[None], hp, hR)
    free = ik.IKSolver(model, tasks)
    safe = ik.IKSolver(model, tasks, collision=spheres, collision_partners=3)
    run = lambda s: jax.jit(lambda q0: s.solve(q0, targets, iters=40))(  # noqa: E731
        jnp.asarray(m.qpos0))[0][0]
    gap_free = float(jnp.min(spheres.gaps(rx.forward_kinematics(model, run(free)))))
    gap_safe = float(jnp.min(spheres.gaps(rx.forward_kinematics(model, run(safe)))))
    assert gap_free < -0.02  # without avoidance the elbow sinks into the torso spheres
    assert gap_safe > -0.005


def test_trajectory_is_continuous_and_bounded(g1):
    """Batched segments + seam blending: no frame-to-frame step above the trust region"""
    m, model = g1
    T = 200
    t = np.linspace(0, 2 * np.pi, T)
    kin0 = rx.forward_kinematics(model, m.qpos0)
    hand0 = np.asarray(kin0.body_p[model.body_id("left_wrist_yaw_link")])
    path = hand0 + 0.12 * np.stack([np.sin(t), np.zeros(T), np.cos(t) - 1], 1)
    solver = ik.IKSolver(model, [ik.position(model, ["left_wrist_yaw_link"])], max_step=0.2)
    qs, _ = solver.solve_trajectory(jnp.asarray(m.qpos0), (jnp.asarray(path)[:, None],),
                                    segment=48, overlap=8, iters=2, devices=jax.devices()[:1])
    qs = np.asarray(qs)
    assert qs.shape == (T, m.nq)
    steps = np.abs(np.diff(qs[:, 7:], axis=0)).max(1)
    assert steps.max() < 2 * 0.2 + 1e-6  # two iterations per frame, 0.2 rad each
    hand = np.asarray(jax.vmap(lambda q: rx.forward_kinematics(model, q).body_p)(jnp.asarray(qs))
                      [:, model.body_id("left_wrist_yaw_link")])
    assert np.median(np.linalg.norm(hand - path, axis=1)) < 0.01
