"""Inverse/forward dynamics and configuration-space integration.

All algorithms are written as dense, vectorized passes over the DOF ancestor mask,
so there are no sequential loops over joints.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax import Array

from rigidax.kinematics import Kinematics, _kin, body_coms
from rigidax.model import Model, derived
from rigidax.spatial import (
    cross_force,
    cross_motion,
    einsum,
    mm,
    quat_conj,
    quat_integrate,
    quat_log,
    quat_mul,
    quat_normalize,
    quat_to_mat,
    skew,
)


def body_spatial_inertias(model: Model, kin: Kinematics, bodies=None) -> Array:
    """(nbody, 6, 6) spatial inertias in the world frame, about the origin"""
    P = model.params
    b = slice(None) if bodies is None else bodies
    with jax.ensure_compile_time_eval():  # parameter-only: constant if not traced
        Ri = quat_to_mat(jnp.asarray(P.body_iquat)[b])
        I_local = einsum("bij,bj,bkj->bik", Ri, jnp.asarray(P.body_inertia)[b], Ri)
        ipos = jnp.asarray(P.body_ipos)[b]
        m = jnp.asarray(P.body_mass)[b][:, None, None]
        m_eye = m * jnp.eye(3)
    R, p = kin.body_R[b], kin.body_p[b]
    Ic = mm(mm(R, I_local), jnp.swapaxes(R, -1, -2))
    C = skew(p + einsum("bij,bj->bi", R, ipos))
    mC = m * C
    top = jnp.concatenate([Ic - mm(mC, C), mC], -1)
    bot = jnp.concatenate([-mC, jnp.broadcast_to(m_eye, mC.shape)], -1)
    return jnp.concatenate([top, bot], -2)


def _moving_inertias(model, kin):
    """Spatial inertias of the bodies that some DOF moves (static subset)"""
    return body_spatial_inertias(model, kin, derived(model.topo).owned)


def dof_inertias(model: Model, kin: Kinematics) -> Array:
    """(nv, 6, 6) inertia carried by each DOF (welded bodies fused, virtual DOFs zero)"""
    D = derived(model.topo)
    own = np.zeros((model.nv, len(D.owned)))
    own[D.owned_dof, np.arange(len(D.owned))] = 1.0
    return einsum("ki,iab->kab", own, _moving_inertias(model, kin))


def _ext_wrenches(model: Model, kin: Kinematics, xfrc: Array) -> Array:
    """Per-body [force, torque] at the COM (world axes, MuJoCo's xfrc_applied layout)
    -> spatial forces about the origin, for each moving body"""
    D = derived(model.topo)
    c = body_coms(model, None, kin)[D.owned]
    f, tq = xfrc[D.owned, :3], xfrc[D.owned, 3:]
    return jnp.concatenate([tq + jnp.cross(c, f), f], -1)


def _rnea(model, kin, Ib, v, a, gravity, F_ext):
    D = derived(model.topo)
    A, W = jnp.asarray(D.A), jnp.asarray(D.W)
    S = kin.S
    acc = None
    if v is not None:
        Sv = S * v[:, None]
        V = mm(A, Sv)
        if np.array_equal(D.cross, np.arange(model.nv)):
            Vc = V  # hinge/slide only: each axis moves with its own frame
        else:
            Vc = jnp.concatenate([V, jnp.zeros((1, 6), V.dtype)])[D.cross]
        acc = cross_motion(Vc, Sv)
    if a is not None:
        Sa = S * a[:, None]
        acc = Sa if acc is None else acc + Sa
    Acc = jnp.zeros((len(D.owned), 6), S.dtype) if acc is None else mm(A, acc)[D.owned_dof]
    if gravity:
        g = jnp.asarray(model.params.gravity, S.dtype)
        Acc = Acc + jnp.concatenate([jnp.zeros(3, S.dtype), -g])
    F = einsum("iab,ib->ia", Ib, Acc)
    if v is not None:
        Vb = V[D.owned_dof]
        F = F + cross_force(Vb, einsum("iab,ib->ia", Ib, Vb))
    if F_ext is not None:
        F = F - F_ext
    tau = jnp.sum(S * mm(W, F), -1)
    if a is not None:
        tau = tau + jnp.asarray(model.params.dof_armature) * a
    return tau


def rnea(model: Model, q: Array, v: Array | None, a: Array | None, *,
         xfrc: Array | None = None, gravity: bool = True,
         kin: Kinematics | None = None) -> Array:
    """Inverse dynamics: tau = M(q) a + C(q, v) v + g(q) - J^T f_ext.

    v / a may be None (treated as zero). `xfrc` is (nbody, 6) [force, torque] applied
    at each body's COM in world axes, like MuJoCo's xfrc_applied.
    Armature is included in the M(q) a term.
    """
    kin = _kin(model, q, kin)
    F_ext = None if xfrc is None else _ext_wrenches(model, kin, xfrc)
    return _rnea(model, kin, _moving_inertias(model, kin), v, a, gravity, F_ext)


def _crba(model, kin, Ib):
    D = derived(model.topo)
    A = jnp.asarray(D.A)
    Ic = einsum("ki,iab->kab", D.W, Ib)  # composite inertia of each DOF's subtree
    F = einsum("kab,kb->ka", Ic, kin.S)
    M_all = mm(F, kin.S.T)  # M_all[k, j] = s_j . Ic_k s_k
    L = A * M_all  # keep (k, ancestor j) pairs
    return L + jnp.tril(L, -1).T + jnp.diag(jnp.asarray(model.params.dof_armature))


def mass_matrix(model: Model, q: Array, kin: Kinematics | None = None) -> Array:
    """Joint-space inertia matrix (CRBA), including armature, (nv, nv)"""
    kin = _kin(model, q, kin)
    return _crba(model, kin, _moving_inertias(model, kin))


def bias_forces(model: Model, q: Array, v: Array, kin: Kinematics | None = None) -> Array:
    """C(q, v) v + g(q), (nv,)"""
    return rnea(model, q, v, None, kin=kin)


def gravity_forces(model: Model, q: Array, kin: Kinematics | None = None) -> Array:
    return rnea(model, q, None, None, kin=kin)


def coriolis_forces(model: Model, q: Array, v: Array, kin: Kinematics | None = None) -> Array:
    return rnea(model, q, v, None, gravity=False, kin=kin)


def dynamics_terms(model: Model, q: Array, v: Array, kin: Kinematics | None = None):
    """(M, bias) from a single kinematics and inertia pass"""
    kin = _kin(model, q, kin)
    Ib = _moving_inertias(model, kin)
    return _crba(model, kin, Ib), _rnea(model, kin, Ib, v, None, True, None)


def forward_dynamics(model: Model, q: Array, v: Array, tau: Array, *,
                     xfrc: Array | None = None, kin: Kinematics | None = None) -> Array:
    """qacc = M^-1 (tau + J^T f_ext - bias), via one Cholesky factorization"""
    kin = _kin(model, q, kin)
    Ib = _moving_inertias(model, kin)
    M = _crba(model, kin, Ib)
    F_ext = None if xfrc is None else _ext_wrenches(model, kin, xfrc)
    bias = _rnea(model, kin, Ib, v, None, True, F_ext)
    return jsl.cho_solve(jsl.cho_factor(M, lower=True), tau - bias)


def mass_matrix_inverse(M: Array) -> Array:
    c = jsl.cho_factor(M, lower=True)
    return jsl.cho_solve(c, jnp.eye(M.shape[0], dtype=M.dtype))


# ----- configuration space -----


def integrate(model: Model, q: Array, v: Array, dt: float) -> Array:
    """q (+) v*dt, respecting quaternions (MuJoCo's mj_integratePos)"""
    D = derived(model.topo)
    q = jnp.asarray(q)
    if len(D.scalar_j):
        j = D.scalar_j
        q = q.at[D.qadr[j]].add(v[D.dadr[j]] * dt)
    if len(D.free_j):
        qa, da = D.qadr[D.free_j][:, None], D.dadr[D.free_j][:, None]
        q = q.at[qa + jnp.arange(3)].add(v[da + jnp.arange(3)] * dt)
        quat = quat_normalize(q[qa + 3 + jnp.arange(4)])
        q = q.at[qa + 3 + jnp.arange(4)].set(quat_integrate(quat, v[da + 3 + jnp.arange(3)], dt))
    if len(D.ball_j):
        qa, da = D.qadr[D.ball_j][:, None], D.dadr[D.ball_j][:, None]
        quat = quat_normalize(q[qa + jnp.arange(4)])
        q = q.at[qa + jnp.arange(4)].set(quat_integrate(quat, v[da + jnp.arange(3)], dt))
    return q


def difference(model: Model, q0: Array, q1: Array) -> Array:
    """v such that integrate(q0, v, 1) == q1 (MuJoCo's mj_differentiatePos, dt=1)"""
    D = derived(model.topo)
    q0, q1 = jnp.asarray(q0), jnp.asarray(q1)
    v = jnp.zeros(model.nv, jnp.result_type(q0, q1))
    if len(D.scalar_j):
        j = D.scalar_j
        v = v.at[D.dadr[j]].set(q1[D.qadr[j]] - q0[D.qadr[j]])

    def rot(qa):
        a = quat_normalize(q0[qa + jnp.arange(4)])
        b = quat_normalize(q1[qa + jnp.arange(4)])
        return quat_log(quat_mul(quat_conj(a), b))

    if len(D.free_j):
        qa, da = D.qadr[D.free_j][:, None], D.dadr[D.free_j][:, None]
        v = v.at[da + jnp.arange(3)].set(q1[qa + jnp.arange(3)] - q0[qa + jnp.arange(3)])
        v = v.at[da + 3 + jnp.arange(3)].set(rot(qa + 3))
    if len(D.ball_j):
        qa, da = D.qadr[D.ball_j][:, None], D.dadr[D.ball_j][:, None]
        v = v.at[da + jnp.arange(3)].set(rot(qa))
    return v
