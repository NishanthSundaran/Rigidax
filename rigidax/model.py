"""Robot model: differentiable physical parameters + static (hashable) topology.

Models are compiled by MuJoCo, so anything MuJoCo can load (MJCF or URDF) works.
The resulting `Model` is a JAX pytree: its numeric parameters (masses, inertias,
offsets, gravity, ...) are leaves, so they can be differentiated, batched with vmap,
or swapped without recompiling. The topology is static aux data.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import numpy as np

FREE, BALL, SLIDE, HINGE = 0, 1, 2, 3


class Params(NamedTuple):
    body_pos: np.ndarray  # (nbody, 3) offset in parent frame
    body_quat: np.ndarray  # (nbody, 4) wxyz, orientation in parent frame
    body_mass: np.ndarray  # (nbody,)
    body_inertia: np.ndarray  # (nbody, 3) principal moments
    body_ipos: np.ndarray  # (nbody, 3) COM in body frame
    body_iquat: np.ndarray  # (nbody, 4) principal axes in body frame
    jnt_pos: np.ndarray  # (njnt, 3) anchor in body frame
    jnt_axis: np.ndarray  # (njnt, 3) unit axis in body frame
    jnt_range: np.ndarray  # (njnt, 2)
    qpos0: np.ndarray  # (nq,) reference configuration
    dof_armature: np.ndarray  # (nv,)
    gravity: np.ndarray  # (3,)
    site_pos: np.ndarray  # (nsite, 3)
    site_quat: np.ndarray  # (nsite, 4)


@dataclass(frozen=True)
class Topology:
    nq: int
    nv: int
    nbody: int
    njnt: int
    nsite: int
    body_parent: tuple
    body_jntadr: tuple
    body_jntnum: tuple
    jnt_type: tuple
    jnt_body: tuple
    jnt_qposadr: tuple
    jnt_dofadr: tuple
    jnt_limited: tuple
    dof_parent: tuple
    dof_jnt: tuple
    site_body: tuple
    body_names: tuple
    jnt_names: tuple
    site_names: tuple


@functools.lru_cache(maxsize=None)
def derived(t: Topology) -> SimpleNamespace:
    """Index tables precomputed from the topology (numpy, static under jit)"""
    bp = np.array(t.body_parent)
    jadr, jnum = np.array(t.body_jntadr), np.array(t.body_jntnum)
    jtype, jbody = np.array(t.jnt_type, dtype=int), np.array(t.jnt_body, dtype=int)
    qadr, dadr = np.array(t.jnt_qposadr, dtype=int), np.array(t.jnt_dofadr, dtype=int)
    dpar = np.array(t.dof_parent, dtype=int)
    dof_body = jbody[np.array(t.dof_jnt, dtype=int)] if t.nv else np.zeros(0, int)

    # Bodies grouped by tree depth: FK takes depth sequential steps, not nbody
    depth = np.zeros(t.nbody, int)
    for b in range(1, t.nbody):
        depth[b] = depth[bp[b]] + 1
    levels = []
    for d in range(1, depth.max(initial=0) + 1):
        bodies = np.where(depth == d)[0]
        slots = []
        for s in range(jnum[bodies].max(initial=0)):
            pos = np.where(jnum[bodies] > s)[0]
            slots.append((pos, jadr[bodies[pos]] + s))
        levels.append((bodies, bp[bodies], tuple(slots)))

    # Ancestor-or-self mask over DOFs: A[k, j] = 1 iff dof j moves dof k's frame
    A = np.zeros((t.nv, t.nv))
    for k in range(t.nv):
        if dpar[k] >= 0:
            A[k] = A[dpar[k]]
        A[k, k] = 1.0

    # Each body's inertia is carried by the last DOF that moves it (-1: world-fixed)
    owner = -np.ones(t.nbody, int)
    for b in range(1, t.nbody):
        dofs = np.where(dof_body == b)[0]
        owner[b] = dofs.max() if len(dofs) else owner[bp[b]]
    body_mask = np.where(owner[:, None] >= 0, A[np.maximum(owner, 0)], 0.0)

    # Velocity used for d/dt of each DOF axis: s_dot = v_cross x s.
    # Hinge/slide: own cumulative velocity. Ball / free-rotation: velocity before the
    # rotation (equivalent to the child velocity once summed over the 3 axes).
    cross = np.arange(t.nv)
    rot3_dofs, rot3_body, rot3_k, rot3_jnt = [], [], [], []
    lin_dofs, lin_k = [], []
    for j in np.where(jtype == BALL)[0]:
        d0 = dadr[j]
        cross[d0 : d0 + 3] = dpar[d0]
        for k in range(3):
            rot3_dofs.append(d0 + k), rot3_body.append(jbody[j]), rot3_k.append(k)
            rot3_jnt.append(j)
    for j in np.where(jtype == FREE)[0]:
        d0 = dadr[j]
        cross[d0 : d0 + 3] = -1
        cross[d0 + 3 : d0 + 6] = d0 + 2
        for k in range(3):
            lin_dofs.append(d0 + k), lin_k.append(k)
            rot3_dofs.append(d0 + 3 + k), rot3_body.append(jbody[j]), rot3_k.append(k)
            rot3_jnt.append(j)
    cross = np.where(cross < 0, t.nv, cross)  # index nv -> zero row

    hinge_j = np.where(jtype == HINGE)[0]
    slide_j = np.where(jtype == SLIDE)[0]
    ball_j = np.where(jtype == BALL)[0]
    free_j = np.where(jtype == FREE)[0]
    scalar_j = np.where((jtype == HINGE) | (jtype == SLIDE))[0]
    rot3_dofs, lin_dofs = np.array(rot3_dofs, int), np.array(lin_dofs, int)

    # ---- gather-only FK schedule (scatters are slow on XLA:CPU) ----
    # Joint transforms are built per type then permuted; index njnt is an identity pad
    jcat = np.concatenate([hinge_j, slide_j, ball_j, free_j]).astype(int)
    jperm = np.empty(t.njnt + 1, int)
    jperm[jcat] = np.arange(len(jcat))
    jperm[t.njnt] = len(jcat)
    # Per-slot joint index for every body (njnt = identity pad); joint j sits in
    # slot j - body_jntadr[body], and its pre-joint local frame is recorded there
    slot_jnts = []
    for s in range(jnum.max(initial=0)):
        slot_jnts.append(np.where(jnum > s, jadr + s, t.njnt))
    rec_of_jnt = np.array([(j - jadr[jbody[j]]) * t.nbody + jbody[j] for j in range(t.njnt)], int)
    # Bodies are stored in level order during FK: position 0 is the world
    pos_of = np.zeros(t.nbody, int)
    fk_levels, npos = [], 1
    for bodies, parents, _ in levels:
        pos_of[bodies] = npos + np.arange(len(bodies))
        npos += len(bodies)
        fk_levels.append((bodies, pos_of[parents]))
    # DOF axes are built per kind then permuted
    scat = np.concatenate([dadr[hinge_j], dadr[slide_j], rot3_dofs, lin_dofs]).astype(int)
    sperm = np.empty(t.nv, int)
    sperm[scat] = np.arange(len(scat))
    # Dynamics act on bodies that move (have an owner DOF)
    owned = np.where(owner >= 0)[0]
    W = A[owner[owned]].T  # W[k, i] = 1 iff dof k moves owned body i

    return SimpleNamespace(
        jperm=jperm,
        fk_levels=tuple(fk_levels),
        body_from_pos=pos_of,
        slot_jnts=tuple(slot_jnts),
        rec_of_jnt=rec_of_jnt,
        jnt_parent_body=bp[jbody] if t.njnt else np.zeros(0, int),
        sperm=sperm,
        W=W,
        levels=tuple(levels),
        A=A,
        owner=owner,
        owned=owned,
        owned_dof=owner[owned],
        body_mask=body_mask,
        dof_body=dof_body,
        cross=cross,
        hinge_j=hinge_j,
        slide_j=slide_j,
        ball_j=ball_j,
        free_j=free_j,
        scalar_j=scalar_j,
        rot3_dofs=rot3_dofs,
        rot3_body=np.array(rot3_body, int),
        rot3_k=np.array(rot3_k, int),
        rot3_jnt=np.array(rot3_jnt, int),
        lin_dofs=lin_dofs,
        lin_k=np.array(lin_k, int),
        qadr=qadr,
        dadr=dadr,
        jbody=jbody,
    )


@jax.tree_util.register_pytree_node_class
class Model:
    """Kinematic tree with differentiable parameters"""

    def __init__(self, params: Params, topo: Topology):
        self.params = params
        self.topo = topo

    def tree_flatten(self):
        return (self.params,), self.topo

    @classmethod
    def tree_unflatten(cls, topo, children):
        return cls(children[0], topo)

    @property
    def nq(self) -> int:
        return self.topo.nq

    @property
    def nv(self) -> int:
        return self.topo.nv

    @property
    def nbody(self) -> int:
        return self.topo.nbody

    @property
    def q0(self):
        return self.params.qpos0

    def replace(self, **kwargs) -> Model:
        """New model with some parameters replaced, e.g. model.replace(gravity=g)"""
        return Model(self.params._replace(**kwargs), self.topo)

    def body_id(self, name: str) -> int:
        return self.topo.body_names.index(name)

    def site_id(self, name: str) -> int:
        return self.topo.site_names.index(name)

    def __repr__(self):
        t = self.topo
        return f"Model(nq={t.nq}, nv={t.nv}, nbody={t.nbody}, njnt={t.njnt}, nsite={t.nsite})"


def from_mujoco(m) -> Model:
    """Build a Model from a compiled mujoco.MjModel"""
    import mujoco

    def name(obj, i):
        return mujoco.mj_id2name(m, obj, i) or ""

    body_pos = np.array(m.body_pos, float)
    body_quat = np.array(m.body_quat, float)
    for b in range(1, m.nbody):
        adr, num = m.body_jntadr[b], m.body_jntnum[b]
        types = list(m.jnt_type[adr : adr + num])
        if FREE in types:
            if num != 1 or m.body_parentid[b] != 0:
                raise ValueError(f"body {name(mujoco.mjtObj.mjOBJ_BODY, b)!r}: a free "
                                 "joint must be the only joint of a top-level body")
            # MuJoCo sets free bodies' world pose directly from qpos
            body_pos[b] = 0.0
            body_quat[b] = (1.0, 0.0, 0.0, 0.0)
        if BALL in types[:-1]:
            raise ValueError(f"body {name(mujoco.mjtObj.mjOBJ_BODY, b)!r}: a ball "
                             "joint must be the last joint of its body")

    f = lambda x: np.array(x, float)  # noqa: E731
    params = Params(
        body_pos=body_pos,
        body_quat=body_quat,
        body_mass=f(m.body_mass),
        body_inertia=f(m.body_inertia),
        body_ipos=f(m.body_ipos),
        body_iquat=f(m.body_iquat),
        jnt_pos=f(m.jnt_pos).reshape(-1, 3),
        jnt_axis=f(m.jnt_axis).reshape(-1, 3),
        jnt_range=f(m.jnt_range).reshape(-1, 2),
        qpos0=f(m.qpos0),
        dof_armature=f(m.dof_armature),
        gravity=f(m.opt.gravity),
        site_pos=f(m.site_pos).reshape(-1, 3),
        site_quat=f(m.site_quat).reshape(-1, 4),
    )
    ti = lambda x: tuple(int(v) for v in np.asarray(x).ravel())  # noqa: E731
    O = mujoco.mjtObj
    topo = Topology(
        nq=m.nq,
        nv=m.nv,
        nbody=m.nbody,
        njnt=m.njnt,
        nsite=m.nsite,
        body_parent=ti(m.body_parentid),
        body_jntadr=ti(m.body_jntadr),
        body_jntnum=ti(m.body_jntnum),
        jnt_type=ti(m.jnt_type),
        jnt_body=ti(m.jnt_bodyid),
        jnt_qposadr=ti(m.jnt_qposadr),
        jnt_dofadr=ti(m.jnt_dofadr),
        jnt_limited=ti(m.jnt_limited),
        dof_parent=ti(m.dof_parentid),
        dof_jnt=ti(m.dof_jntid),
        site_body=ti(m.site_bodyid),
        body_names=tuple(name(O.mjOBJ_BODY, i) for i in range(m.nbody)),
        jnt_names=tuple(name(O.mjOBJ_JOINT, i) for i in range(m.njnt)),
        site_names=tuple(name(O.mjOBJ_SITE, i) for i in range(m.nsite)),
    )
    return Model(params, topo)


def _strip_meshes(xml: str, is_urdf: bool) -> str:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    if is_urdf:
        for parent in root.iter():
            for child in list(parent):
                if child.tag in ("visual", "collision"):
                    parent.remove(child)
    else:
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "mesh" or (child.tag == "geom" and (
                        child.get("mesh") or child.get("type") == "mesh")):
                    parent.remove(child)
    return ET.tostring(root, encoding="unicode")


def load(path: str | Path, strip_meshes: bool = False) -> Model:
    """Load an MJCF (.xml) or URDF (.urdf) file.

    strip_meshes: drop mesh geometry (and URDF visual/collision) before compiling, so
    a model loads without its mesh files. Bodies must then declare their inertia.
    """
    import mujoco

    path = Path(path)
    if not strip_meshes:
        return from_mujoco(mujoco.MjModel.from_xml_path(str(path)))
    xml = _strip_meshes(path.read_text(), path.suffix.lower() == ".urdf")
    if path.suffix.lower() == ".urdf":
        return from_xml_string(xml)
    return from_mujoco(mujoco.MjModel.from_xml_string(xml))


def from_xml_string(xml: str) -> Model:
    import mujoco

    return from_mujoco(mujoco.MjModel.from_xml_string(xml))
