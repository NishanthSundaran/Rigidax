"""Automatic sphere model of the G1 collision meshes, for differentiable self-collision.

Each body's collision-mesh vertices are sliced along their principal axis; each slice
gets a sphere at its centroid covering most of its cross-section (like frax's
"bubblify" models, but fitted automatically).
"""

import mujoco
import numpy as np

ARM_LINKS = ("shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
BODY_LINKS = ("pelvis", "torso", "waist_yaw", "waist_roll", "hip_pitch", "hip_roll",
              "hip_yaw", "knee")
SPACING = 0.05  # target distance between sphere centers (m)
COVER = 95  # percentile of slice radial distance covered by the sphere
MARGIN = 0.01  # desired clearance between arm and body spheres (m)


def _quat_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def body_vertices(m, b):
    """Collision-mesh vertices of body b, in the body frame"""
    out = []
    for g in np.where((m.geom_bodyid == b) & (m.geom_group == 3)
                      & (m.geom_type == mujoco.mjtGeom.mjGEOM_MESH))[0]:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        out.append(v @ _quat_mat(m.geom_quat[g]).T + m.geom_pos[g])
    return np.concatenate(out) if out else np.zeros((0, 3))


def fit_body(V):
    """-> centers (k, 3), radii (k,) covering vertices V"""
    mu = V.mean(0)
    _, _, Vt = np.linalg.svd(V - mu, full_matrices=False)
    axis = Vt[0]
    t = (V - mu) @ axis
    k = max(1, int(np.ceil((t.max() - t.min()) / SPACING)))
    edges = np.linspace(t.min(), t.max(), k + 1)
    centers, radii = [], []
    for i in range(k):
        sel = (t >= edges[i]) & (t <= edges[i + 1])
        if sel.sum() < 4:
            continue
        c = V[sel].mean(0)
        radial = V[sel] - c
        radial -= np.outer(radial @ axis, axis)
        r = max(np.percentile(np.linalg.norm(radial, axis=1), COVER),
                0.5 * (edges[1] - edges[0]))
        centers.append(c), radii.append(r)
    return np.array(centers), np.array(radii)


def link_kind(name):
    if any(k in name for k in ARM_LINKS):
        return "arm"
    if any(k in name for k in BODY_LINKS):
        return "body"
    return None


def fit_g1(m):
    """-> sphere body ids (S,), local centers (S, 3), radii (S,), and the collision
    pairs (P, 2) of sphere indices: each arm vs body links and vs the other arm"""
    sb, sc, sr, kind, side = [], [], [], [], []
    for b in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        k = link_kind(name)
        V = body_vertices(m, b)
        if k is None or len(V) == 0:
            continue
        c, r = fit_body(V)
        sb += [b] * len(r)
        sc.append(c)
        sr.append(r)
        kind += [k] * len(r)
        side += ["left" if name.startswith("left") else
                 "right" if name.startswith("right") else "mid"] * len(r)
    sb, sc, sr = np.array(sb), np.concatenate(sc), np.concatenate(sr)
    pairs = []
    for i in range(len(sb)):
        for j in range(i + 1, len(sb)):
            if kind[i] != "arm" and kind[j] != "arm":
                continue
            if kind[i] == kind[j] == "arm" and side[i] == side[j]:
                continue  # same arm
            if m.body_parentid[sb[i]] == sb[j] or m.body_parentid[sb[j]] == sb[i]:
                continue  # adjacent links
            ni = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, sb[i])
            nj = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, sb[j])
            if {"shoulder" in ni, "shoulder" in nj} == {True} or (
                    ("shoulder" in ni and "torso" in nj) or ("shoulder" in nj and "torso" in ni)):
                continue  # the shoulder sits in the torso by design
            pairs.append((i, j))
    pairs = np.array(pairs)
    # Per-pair minimum distance: MARGIN, relaxed only where the standing pose is closer
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    W = d.xpos[sb] + np.einsum("sij,sj->si", d.xmat[sb].reshape(-1, 3, 3), sc)
    rest = np.linalg.norm(W[pairs[:, 0]] - W[pairs[:, 1]], axis=1) - sr[pairs[:, 0]] - sr[pairs[:, 1]]
    min_dist = np.minimum(MARGIN, rest - 0.005)
    return sb, sc, sr, pairs, min_dist
