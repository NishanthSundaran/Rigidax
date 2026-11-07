"""Automatic sphere collision models for any MuJoCo model.

Each body's collision geometry (meshes, boxes, capsules, cylinders, spheres, ellipsoids)
is sampled into points. The points are sliced along their principal axis, and each
slice gets a sphere at its centroid covering most of its cross-section. Pairs are
generated automatically: different bodies, not parent/child, optionally filtered.
Pairs that already overlap in the reference pose get a relaxed minimum distance, so
the robot never fights its own default posture.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from rigidax.kinematics import Kinematics, forward_kinematics
from rigidax.model import Model, derived


@dataclass(frozen=True)
class SphereModel:
    body: np.ndarray  # (S,) body index of each sphere
    center: np.ndarray  # (S, 3) center in the body frame
    radius: np.ndarray  # (S,)
    pairs: np.ndarray  # (P, 2) sphere indices
    min_dist: np.ndarray  # (P,) required clearance between sphere surfaces
    # Row structure for the solver: each pair is owned by one "row" sphere, and each
    # row constrains its closest partner. Fixed shapes, no sorting, stable meanings.
    rows: np.ndarray = None  # (R,) sphere indices that own pairs
    partner: np.ndarray = None  # (R, S) bool: partner spheres of each row
    row_min_dist: np.ndarray = None  # (R, S) clearance per (row, partner)

    @property
    def num_spheres(self) -> int:
        return len(self.radius)

    def centers(self, kin: Kinematics) -> Array:
        b = self.body
        return kin.body_p[b] + jnp.einsum("sij,sj->si", kin.body_R[b], self.center)

    def gaps(self, kin: Kinematics) -> Array:
        """Surface distance minus required clearance per pair (< 0: violated)"""
        C = self.centers(kin)
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        return (jnp.linalg.norm(C[i] - C[j], axis=1) - self.radius[i] - self.radius[j]
                - self.min_dist)

    @staticmethod
    def fit(m, model: Model, bodies: Callable[[str], bool] | None = None,
            pair_filter: Callable[[str, str], bool] | None = None,
            spacing: float = 0.05, cover: float = 95.0, margin: float = 0.01,
            geom_group: int | None = 3, q_ref=None, extra=None) -> "SphereModel":
        """Fit spheres to a compiled mujoco.MjModel `m` (the same model as `model`).

        bodies: which bodies to include (by name); default: all with geometry.
        pair_filter: which body pairs to check (by names); default: all eligible.
        geom_group: geom group holding collision geometry (None: all geoms).
        q_ref: reference pose for relaxing pre-existing overlaps (default qpos0).
        extra: optional [(body name, centers (n, 3) in that body's frame, radii (n,))] of
            spheres to attach rigidly, e.g. the envelope of parts not in the model
            (fingers whose joints are not solved).
        """
        import mujoco

        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        sb, sc, sr = [], [], []
        for b in range(1, m.nbody):
            if bodies is not None and not bodies(names[b]):
                continue
            V = _body_points(m, b, geom_group)
            if len(V) < 4:
                continue
            c, r = _fit_spheres(V, spacing, cover)
            sb += [b] * len(r)
            sc.append(c)
            sr.append(r)
        for name, c, r in (extra or ()):
            b = m.body(name).id
            if bodies is not None and not bodies(names[b]):
                continue
            sb += [b] * len(r)
            sc.append(np.asarray(c, float).reshape(-1, 3))
            sr.append(np.asarray(r, float).reshape(-1))
        sb, sc, sr = np.array(sb, int), np.concatenate(sc), np.concatenate(sr)
        parent = m.body_parentid
        pairs = [(i, j) for i in range(len(sb)) for j in range(i + 1, len(sb))
                 if sb[i] != sb[j] and parent[sb[i]] != sb[j] and parent[sb[j]] != sb[i]
                 and (pair_filter is None or pair_filter(names[sb[i]], names[sb[j]]))]
        pairs = np.array(pairs, int).reshape(-1, 2)
        model_ = SphereModel(sb, sc, sr, pairs, np.zeros(len(pairs)))
        kin = forward_kinematics(model, m.qpos0 if q_ref is None else q_ref)
        rest = np.asarray(model_.gaps(kin))
        min_dist = np.minimum(margin, rest - 0.005)
        return SphereModel.with_rows(sb, sc, sr, pairs, min_dist)

    @staticmethod
    def with_rows(body, center, radius, pairs, min_dist) -> "SphereModel":
        """Assign each pair to the endpoint with more partners (fewest rows overall)"""
        S = len(radius)
        degree = np.bincount(pairs.ravel(), minlength=S)
        owner = np.where(degree[pairs[:, 0]] >= degree[pairs[:, 1]], pairs[:, 0], pairs[:, 1])
        other = np.where(owner == pairs[:, 0], pairs[:, 1], pairs[:, 0])
        rows = np.unique(owner)
        r_of = {r: k for k, r in enumerate(rows)}
        partner = np.zeros((len(rows), S), bool)
        row_min = np.zeros((len(rows), S))
        for o, t, md in zip(owner, other, min_dist):
            partner[r_of[o], t] = True
            row_min[r_of[o], t] = md
        return SphereModel(body, center, radius, pairs, min_dist, rows, partner, row_min)

    def num_rows(self, k: int = 1) -> int:
        return len(self.rows) * min(k, self.partner.shape[1])

    def constraint_rows(self, model: Model, kin: Kinematics, k: int = 1):
        """Linearized constraints for each row sphere against its k closest partners:
        rows a (R*k, nv) and bounds b (R*k,) such that  a @ dq >= b  (b = -gap)."""
        import jax

        C = self.centers(kin)
        Cr = C[self.rows]
        D = (jnp.linalg.norm(Cr[:, None] - C[None], axis=-1)
             - self.radius[self.rows][:, None] - self.radius[None] - self.row_min_dist)
        D = jnp.where(self.partner, D, jnp.inf)
        k = min(k, D.shape[1])
        if k == 1:
            j = jnp.argmin(D, axis=1)[:, None]
        else:
            j = jax.lax.top_k(-D, k)[1]  # (R, k) closest partners
        gap = jnp.take_along_axis(D, j, axis=1)  # (R, k)
        gap = jnp.where(jnp.isfinite(gap), gap, 1.0)  # rows with < k partners: inactive
        Cj = C[j]  # (R, k, 3)
        delta = Cr[:, None] - Cj
        n = delta / jnp.maximum(jnp.linalg.norm(delta, axis=-1, keepdims=True), 1e-9)
        # a = n^T (J(c_i) - J(c_j)), with J(c) = (u + w x c) masked by the body's DOFs.
        # Per DOF: n.(u + w x c) = u.n + w.(c x n): two small mat-vecs, no (S, 3, nv)
        # Jacobian tensor.
        w, u = kin.S[:, :3], kin.S[:, 3:]
        mask = jnp.asarray(derived(model.topo).body_mask)  # (nbody, nv)
        un = jnp.einsum("vk,rqk->rqv", u, n)
        ti = jnp.einsum("vk,rqk->rqv", w, jnp.cross(Cr[:, None], n))
        tj = jnp.einsum("vk,rqk->rqv", w, jnp.cross(Cj, n))
        mi = mask[self.body[self.rows]][:, None]  # (R, 1, nv)
        mj = mask[jnp.asarray(self.body)[j]]  # (R, k, nv)
        a = mi * (un + ti) - mj * (un + tj)
        return a.reshape(-1, a.shape[-1]), -gap.reshape(-1)


def _quat_mat(q):
    import mujoco

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def _primitive_points(gtype, size, n=400):
    import mujoco

    rng = np.random.default_rng(0)
    G = mujoco.mjtGeom
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    if gtype == G.mjGEOM_SPHERE:
        return u * size[0]
    if gtype == G.mjGEOM_ELLIPSOID:
        return u * size[:3]
    if gtype == G.mjGEOM_BOX:
        return rng.uniform(-1, 1, (n, 3)) * size[:3]
    if gtype in (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER):
        z = rng.uniform(-size[1], size[1], n)
        ring = u[:, :2] / np.linalg.norm(u[:, :2], axis=1, keepdims=True) * size[0]
        pts = np.c_[ring, z]
        if gtype == G.mjGEOM_CAPSULE:
            pts = np.r_[pts, u * size[0] + [0, 0, size[1]], u * size[0] - [0, 0, size[1]]]
        return pts
    return np.zeros((0, 3))


def _body_points(m, b, group):
    import mujoco

    out = []
    for g in np.where(m.geom_bodyid == b)[0]:
        if group is not None and m.geom_group[g] != group:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        else:
            v = _primitive_points(m.geom_type[g], m.geom_size[g])
        if len(v):
            out.append(v @ _quat_mat(m.geom_quat[g]).T + m.geom_pos[g])
    return np.concatenate(out) if out else np.zeros((0, 3))


def _fit_spheres(V, spacing, cover):
    """Slice along the principal axis; split wide (anisotropic) slices along their
    second axis so flat bodies like a torso get several tight spheres, not one fat one"""
    mu = V.mean(0)
    _, _, Vt = np.linalg.svd(V - mu, full_matrices=False)
    axis = Vt[0]
    t = (V - mu) @ axis
    k = max(1, int(np.ceil((t.max() - t.min()) / spacing)))
    edges = np.linspace(t.min(), t.max(), k + 1)
    centers, radii = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (t >= a) & (t <= b)
        if sel.sum() < 4:
            continue
        P = V[sel]
        c0 = P.mean(0)
        flat = P - c0
        flat -= np.outer(flat @ axis, axis)  # cross-section points
        _, sv, Wt = np.linalg.svd(flat, full_matrices=False)
        w_axis = Wt[0]
        s_w = flat @ w_axis
        ext_w = s_w.max() - s_w.min()
        ext_d = np.ptp(flat @ Wt[1]) if len(Wt) > 1 else ext_w
        # number of sub-spheres across the wide direction
        n_sub = int(np.clip(np.round(ext_w / max(ext_d, 1e-6)), 1, 4))
        sub_edges = np.linspace(s_w.min(), s_w.max(), n_sub + 1)
        for sa, sb in zip(sub_edges[:-1], sub_edges[1:]):
            ss = (s_w >= sa) & (s_w <= sb)
            if ss.sum() < 4:
                continue
            c = P[ss].mean(0)
            radial = P[ss] - c
            radial -= np.outer(radial @ axis, axis)
            r = max(np.percentile(np.linalg.norm(radial, axis=1), cover), 0.5 * (b - a))
            centers.append(c)
            radii.append(r)
    return np.array(centers), np.array(radii)
