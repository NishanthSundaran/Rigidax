"""Forward kinematics, frames, Jacobians, center of mass"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from rigidax.model import Model, derived
from rigidax.spatial import axis_angle_to_mat, einsum, mm, quat_normalize, quat_to_mat


class Kinematics(NamedTuple):
    body_R: Array  # (nbody, 3, 3) world orientation
    body_p: Array  # (nbody, 3) world position
    S: Array  # (nv, 6) DOF motion axes [angular; linear], world frame, about origin


def _homog(R, p):
    n = R.shape[:-2]
    top = jnp.concatenate([R, p[..., None]], -1)
    bot = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0, 1.0], R.dtype), n + (1, 4))
    return jnp.concatenate([top, bot], -2)


def forward_kinematics(model: Model, q: Array) -> Kinematics:
    t, P, D = model.topo, model.params, derived(model.topo)
    q = jnp.asarray(q)
    eye = jnp.eye(3, dtype=q.dtype)

    # Parameter-only quantities fold to constants when the model is not traced
    with jax.ensure_compile_time_eval():
        body_T = _homog(quat_to_mat(jnp.asarray(P.body_quat)), jnp.asarray(P.body_pos))
        jnt_pos, jnt_axis = jnp.asarray(P.jnt_pos), jnp.asarray(P.jnt_axis)

    # Joint transforms relative to the frame just before each joint, built per type
    # and permuted into joint order (plus an identity pad at index njnt)
    Rs, ts = [], []
    if len(D.hinge_j):
        j = D.hinge_j
        R = axis_angle_to_mat(jnt_axis[j], q[D.qadr[j]] - P.qpos0[D.qadr[j]])
        a = jnt_pos[j]
        Rs.append(R), ts.append(a - einsum("nij,nj->ni", R, a))
    if len(D.slide_j):
        j = D.slide_j
        disp = q[D.qadr[j]] - P.qpos0[D.qadr[j]]
        Rs.append(jnp.broadcast_to(eye, (len(j), 3, 3)))
        ts.append(jnt_axis[j] * disp[:, None])
    if len(D.ball_j):
        j = D.ball_j
        R = quat_to_mat(quat_normalize(q[D.qadr[j][:, None] + jnp.arange(4)]))
        a = jnt_pos[j]
        Rs.append(R), ts.append(a - einsum("nij,nj->ni", R, a))
    if len(D.free_j):
        j = D.free_j
        Rs.append(quat_to_mat(quat_normalize(q[D.qadr[j][:, None] + 3 + jnp.arange(4)])))
        ts.append(q[D.qadr[j][:, None] + jnp.arange(3)])
    Rs.append(eye[None]), ts.append(jnp.zeros((1, 3), q.dtype))
    jT = _homog(jnp.concatenate(Rs), jnp.concatenate(ts))[D.jperm]

    # Each body's transform relative to its parent, all bodies at once:
    # L = body offset * joint_1 * joint_2 * ...  (slots padded with identity)
    L = body_T
    recs = []  # local frame just before each joint slot
    for js in D.slot_jnts:
        recs.append(L)
        L = mm(L, jT[js])

    # World transforms: one batched 4x4 product per tree depth (gathers only)
    Tw = jnp.eye(4, dtype=q.dtype)[None]
    for bodies, parent_pos in D.fk_levels:
        Tw = jnp.concatenate([Tw, mm(Tw[parent_pos], L[bodies])])
    Tw = Tw[D.body_from_pos]
    Rw, pw = Tw[:, :3, :3], Tw[:, :3, 3]
    if not t.nv:
        return Kinematics(Rw, pw, jnp.zeros((0, 6), q.dtype))

    # World frame just before each joint, then the DOF motion axes per kind
    F = mm(Tw[D.jnt_parent_body], jnp.concatenate(recs)[D.rec_of_jnt])
    fR, fp = F[:, :3, :3], F[:, :3, 3]
    axis_w = einsum("nij,nj->ni", fR, jnt_axis)
    anchor_w = fp + einsum("nij,nj->ni", fR, jnt_pos)
    blocks = []
    if len(D.hinge_j):
        j = D.hinge_j
        w = axis_w[j]
        blocks.append(jnp.concatenate([w, jnp.cross(anchor_w[j], w)], -1))
    if len(D.slide_j):
        u = axis_w[D.slide_j]
        blocks.append(jnp.concatenate([jnp.zeros_like(u), u], -1))
    if len(D.rot3_dofs):
        w = Rw[D.rot3_body, :, D.rot3_k]
        # ball: joint anchor; free: body origin
        is_free = np.asarray(t.jnt_type)[D.rot3_jnt] == 0
        a = jnp.where(is_free[:, None], pw[D.rot3_body], anchor_w[D.rot3_jnt])
        blocks.append(jnp.concatenate([w, jnp.cross(a, w)], -1))
    if len(D.lin_dofs):
        blocks.append(np.concatenate([np.zeros((len(D.lin_dofs), 3)),
                                      np.eye(3)[D.lin_k]], -1).astype(q.dtype))
    S = jnp.concatenate(blocks)[D.sperm]
    return Kinematics(Rw, pw, S)


def _kin(model, q, kin):
    return forward_kinematics(model, q) if kin is None else kin


def site_pose(model: Model, q: Array, site: int, kin: Kinematics | None = None):
    """(R, p) of a site in the world frame"""
    kin = _kin(model, q, kin)
    P = model.params
    b = model.topo.site_body[site]
    R = mm(kin.body_R[b], quat_to_mat(jnp.asarray(P.site_quat[site])))
    p = kin.body_p[b] + mm(kin.body_R[b], jnp.asarray(P.site_pos[site]))
    return R, p


def body_pose(model: Model, q: Array, body: int, kin: Kinematics | None = None):
    kin = _kin(model, q, kin)
    return kin.body_R[body], kin.body_p[body]


def point_jacobian(model: Model, q: Array, body: int, point: Array,
                   kin: Kinematics | None = None):
    """Jacobians (jacp, jacr), each (3, nv), of a world-frame point fixed to `body`"""
    kin = _kin(model, q, kin)
    mask = jnp.asarray(derived(model.topo).body_mask[body])[:, None]
    w, u = kin.S[:, :3], kin.S[:, 3:]
    jacp = (u + jnp.cross(w, point)) * mask
    jacr = w * mask
    return jacp.T, jacr.T


def site_jacobian(model: Model, q: Array, site: int, kin: Kinematics | None = None):
    kin = _kin(model, q, kin)
    _, p = site_pose(model, q, site, kin)
    return point_jacobian(model, q, model.topo.site_body[site], p, kin)


def body_jacobian(model: Model, q: Array, body: int, kin: Kinematics | None = None):
    kin = _kin(model, q, kin)
    return point_jacobian(model, q, body, kin.body_p[body], kin)


def body_coms(model: Model, q: Array, kin: Kinematics | None = None) -> Array:
    """World positions of every body's COM, (nbody, 3)"""
    kin = _kin(model, q, kin)
    return kin.body_p + einsum("bij,bj->bi", kin.body_R, model.params.body_ipos)


def center_of_mass(model: Model, q: Array, kin: Kinematics | None = None) -> Array:
    m = jnp.asarray(model.params.body_mass)
    return m @ body_coms(model, q, kin) / jnp.sum(m)


def center_of_mass_jacobian(model: Model, q: Array, kin: Kinematics | None = None) -> Array:
    """(3, nv) Jacobian of the whole-body COM"""
    kin = _kin(model, q, kin)
    c = body_coms(model, q, kin)
    m = jnp.asarray(model.params.body_mass)
    mask = jnp.asarray(derived(model.topo).body_mask)  # (nbody, nv)
    w, u = kin.S[:, :3], kin.S[:, 3:]
    # sum_b m_b * mask_b,k * (u_k + w_k x c_b) = (sum_b m_b mask_b,k) u_k + w_k x (sum_b ...)
    mm_k = m @ mask  # (nv,)
    mc_k = einsum("bk,b,bi->ki", mask, m, c)  # (nv, 3)
    J = u * mm_k[:, None] + jnp.cross(w, mc_k)
    return J.T / jnp.sum(m)


def points_jacobians(model: Model, kin: Kinematics, bodies, points: Array):
    """Batched Jacobians of world points fixed to bodies.

    bodies: static int array (n,); points: (n, 3) world positions.
    Returns (jacp, jacr), each (n, 3, nv).
    """
    mask = jnp.asarray(derived(model.topo).body_mask)[np.asarray(bodies)][..., None]  # (n, nv, 1)
    w, u = kin.S[:, :3], kin.S[:, 3:]
    jacp = (u[None] + jnp.cross(w[None], points[:, None])) * mask
    jacr = jnp.broadcast_to(w[None], jacp.shape) * mask
    return jnp.swapaxes(jacp, 1, 2), jnp.swapaxes(jacr, 1, 2)
