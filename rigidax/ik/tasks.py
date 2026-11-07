"""IK tasks. Each task is vectorized over a fixed set of bodies; its targets are arrays
passed per frame, so a whole trajectory is a batch of targets.

Every task returns (rows (k, nv), errors (k,), weights (k,)) for the linearization
    rows @ dq ~= errors

A target may also be a pair (target, scale) with scale of shape (n,): per-frame weight
multipliers for each body, e.g. to pin a foot harder while it is planted.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from rigidax.kinematics import points_jacobians
from rigidax.spatial import einsum


def _split(target, n):
    """target or (target, per-body weight scale) -> (target, scale (n,))"""
    if isinstance(target, tuple):
        return target[0], target[1]
    return target, jnp.ones(n)


def rotation_error(R, Rd):
    """World-frame rotation vector taking R to Rd: log(Rd R^T) (batched (n, 3, 3))"""
    M = einsum("nij,nkj->nik", Rd, R)
    v = 0.5 * jnp.stack([M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0],
                         M[:, 1, 0] - M[:, 0, 1]], -1)
    s = jnp.linalg.norm(v, axis=-1)
    c = 0.5 * (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1.0)
    theta = jnp.arctan2(s, c)
    small = s < 1e-7
    scale = jnp.where(small, 1.0 + theta**2 / 6.0, theta / jnp.where(small, 1.0, s))
    return v * scale[:, None]


@dataclass(frozen=True, eq=False)
class PositionTask:
    """Body origins (plus optional local offsets) track world positions; targets (n, 3)"""
    bodies: np.ndarray
    weight: np.ndarray
    offset: np.ndarray | None = None

    def residual(self, model, kin, target):
        b = self.bodies
        target, scale = _split(target, len(b))
        p = kin.body_p[b]
        if self.offset is not None:
            p = p + einsum("nij,nj->ni", kin.body_R[b], self.offset)
        Jp, _ = points_jacobians(model, kin, b, p)
        return (Jp.reshape(-1, Jp.shape[-1]), (target - p).reshape(-1),
                jnp.repeat(jnp.asarray(self.weight) * scale, 3))


@dataclass(frozen=True, eq=False)
class OrientationTask:
    """Body orientations track world rotations; targets (n, 3, 3)"""
    bodies: np.ndarray
    weight: np.ndarray

    def residual(self, model, kin, target):
        b = self.bodies
        target, scale = _split(target, len(b))
        _, Jr = points_jacobians(model, kin, b, kin.body_p[b])
        e = rotation_error(kin.body_R[b], target)
        return (Jr.reshape(-1, Jr.shape[-1]), e.reshape(-1),
                jnp.repeat(jnp.asarray(self.weight) * scale, 3))


@dataclass(frozen=True, eq=False)
class DirectionTask:
    """Limb direction tracking: p(body) - p(base) -> length * target unit direction.

    Retargets limbs between skeletons of different proportions: the robot keeps its own
    segment lengths and copies the source's limb angles. targets (n, 3) unit vectors.
    """
    bodies: np.ndarray
    bases: np.ndarray
    lengths: np.ndarray
    weight: np.ndarray

    def residual(self, model, kin, target):
        p, pb = kin.body_p[self.bodies], kin.body_p[self.bases]
        Jb, _ = points_jacobians(model, kin, self.bodies, p)
        Jbase, _ = points_jacobians(model, kin, self.bases, pb)
        e = pb + jnp.asarray(self.lengths)[:, None] * target - p
        J = Jb - Jbase
        return J.reshape(-1, J.shape[-1]), e.reshape(-1), jnp.repeat(jnp.asarray(self.weight), 3)


@dataclass(frozen=True, eq=False)
class PostureTask:
    """Pull toward a nominal configuration, per-DOF weights (nv,). No per-frame target."""
    weight: np.ndarray
    nominal: np.ndarray = field(default=None)


def _as(x, dtype=float):
    return np.atleast_1d(np.asarray(x, dtype))


def position(model, bodies, weight=1.0, offset=None):
    b = _as([model.body_id(n) for n in bodies], int)
    return PositionTask(b, np.broadcast_to(_as(weight), b.shape).copy(),
                        None if offset is None else np.asarray(offset, float).reshape(-1, 3))


def orientation(model, bodies, weight=1.0):
    b = _as([model.body_id(n) for n in bodies], int)
    return OrientationTask(b, np.broadcast_to(_as(weight), b.shape).copy())


def direction(model, bodies, bases, weight=1.0, q_ref=None):
    """Lengths are measured on the robot in q_ref (default qpos0)"""
    from rigidax.kinematics import forward_kinematics

    b = _as([model.body_id(n) for n in bodies], int)
    base = _as([model.body_id(n) for n in bases], int)
    kin = forward_kinematics(model, model.q0 if q_ref is None else q_ref)
    lengths = np.linalg.norm(np.asarray(kin.body_p[b] - kin.body_p[base]), axis=1)
    return DirectionTask(b, base, lengths, np.broadcast_to(_as(weight), b.shape).copy())


@dataclass(frozen=True, eq=False)
class TwistTask:
    """Track only the rotation about a body-fixed axis (e.g. forearm pronation).

    The residual is the component of the rotation error along the axis, so it cannot
    pull the limb's direction (a full orientation task would compete with direction
    tasks whenever the target frames are not exactly consistent). targets (n, 3, 3).
    """
    bodies: np.ndarray
    axes: np.ndarray  # (n, 3) unit axis in each body frame
    weight: np.ndarray

    def residual(self, model, kin, target):
        b = self.bodies
        target, scale = _split(target, len(b))
        R = kin.body_R[b]
        a = einsum("nij,nj->ni", R, jnp.asarray(self.axes))  # world axes
        _, Jr = points_jacobians(model, kin, b, kin.body_p[b])  # (n, 3, nv)
        e = jnp.sum(rotation_error(R, target) * a, -1)
        J = einsum("nk,nkv->nv", a, Jr)
        return J, e, jnp.asarray(self.weight) * scale


@dataclass(frozen=True, eq=False)
class JointTask:
    """Per-frame joint-angle targets for scalar (hinge/slide) joints. targets (n,)"""
    qadr: np.ndarray
    dofs: np.ndarray
    weight: np.ndarray

    def residual(self, model, kin, target, q=None):
        raise NotImplementedError("JointTask needs q: it is handled by IKSolver directly")


def twist(model, bodies, axes, weight=1.0):
    b = _as([model.body_id(n) for n in bodies], int)
    ax = np.asarray(axes, float).reshape(-1, 3)
    ax = ax / np.linalg.norm(ax, axis=1, keepdims=True)
    return TwistTask(b, np.broadcast_to(ax, (len(b), 3)).copy(),
                     np.broadcast_to(_as(weight), b.shape).copy())


def joints(model, mj_model, names, weight=1.0):
    ids = [mj_model.joint(n).id for n in names]
    return JointTask(_as([mj_model.jnt_qposadr[j] for j in ids], int),
                     _as([mj_model.jnt_dofadr[j] for j in ids], int),
                     np.broadcast_to(_as(weight), (len(ids),)).copy())


@dataclass(frozen=True, eq=False)
class HeightTask:
    """Per-frame target height (world z) of body-fixed points; one row per point.
    targets (n,), or (heights, per-frame weight scale) to switch it on only in contact."""
    bodies: np.ndarray
    offset: np.ndarray  # (n, 3) points in the body frames
    weight: np.ndarray

    def residual(self, model, kin, target):
        b = self.bodies
        target, scale = _split(target, len(b))
        p = kin.body_p[b] + einsum("nij,nj->ni", kin.body_R[b], self.offset)
        Jp, _ = points_jacobians(model, kin, b, p)
        return Jp[:, 2], target - p[:, 2], jnp.asarray(self.weight) * scale


def height(model, bodies, offset, weight=1.0):
    b = _as([model.body_id(n) for n in bodies], int)
    off = np.broadcast_to(np.asarray(offset, float).reshape(-1, 3), (len(b), 3)).copy()
    return HeightTask(b, off, np.broadcast_to(_as(weight), b.shape).copy())
