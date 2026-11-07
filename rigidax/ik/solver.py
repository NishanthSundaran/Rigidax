"""Differential IK solver: each Gauss-Newton step is a QP

    minimize    sum_tasks || w (J dq - e) ||^2 + damping ||dq - (q_prev - q)||^2
                + || p (dq - (q_nom - q)) ||^2
    subject to  q_lo - q <= dq <= q_hi - q                (hard joint limits)
                gap_k(q) + a_k dq >= 0                     (elastic self-collision)

solved with the fixed-iteration ADMM in rigidax.ik.qp. Everything is branch-free, so
trajectories are solved as one batched program: they are split into overlapping
segments that run in parallel (vmap within a device, pmap across devices).
"""

import functools
from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from rigidax.dynamics import difference, integrate
from rigidax.ik.collision import SphereModel
from rigidax.ik.qp import INF, QPState, solve_qp
from rigidax.ik.tasks import JointTask, PostureTask
from rigidax.kinematics import forward_kinematics, points_jacobians
from rigidax.model import HINGE, SLIDE, Model
from rigidax.spatial import einsum


class FloorPoints(NamedTuple):
    """Points (or spheres) fixed to bodies that must stay at or above z = height (hard QP
    rows). With `lowest`, only the `lowest` points closest to the floor get a row each
    step, so a whole-body sphere set stays cheap."""
    bodies: np.ndarray  # (P,) body index of each point
    points: np.ndarray  # (P, 3) point in the body frame
    height: float = 0.0
    radius: np.ndarray | None = None  # (P,) sphere radii (None: points)
    lowest: int | None = None

    @staticmethod
    def on(model: Model, bodies: Sequence[str], points, height: float = 0.0) -> "FloorPoints":
        """The same body-frame points on each named body"""
        pts = np.asarray(points, float).reshape(-1, 3)
        b = np.repeat([model.body_id(n) for n in bodies], len(pts))
        return FloorPoints(b, np.tile(pts, (len(bodies), 1)), height)

    @staticmethod
    def spheres(sphere_model, height: float = 0.0, lowest: int | None = 12,
                extra: "FloorPoints | None" = None) -> "FloorPoints":
        """Every sphere of a SphereModel (plus optional extra points) above the floor"""
        b, c, r = sphere_model.body, sphere_model.center, sphere_model.radius
        if extra is not None:
            er = np.zeros(len(extra.bodies)) if extra.radius is None else extra.radius
            b, c, r = np.r_[b, extra.bodies], np.r_[c, extra.points], np.r_[r, er]
        return FloorPoints(b, c, height, r, None if lowest is None else min(lowest, len(b)))

    @property
    def num_rows(self) -> int:
        return len(self.bodies) if self.lowest is None else self.lowest


class IKSolver:
    """Multi-task differential IK with hard limits and elastic collision constraints.

    tasks: task objects (rigidax.ik.tasks), each with targets supplied per frame in
        the same order (PostureTask takes no target).
    collision: optional SphereModel; one row per row-sphere (its closest partner) is
        linearized each step. collision_mode "constraint": elastic QP inequality with
        L1 penalty `collision_mu` (always feasible); "penalty": one-sided quadratic
        penalty of weight `collision_weight`, active only while closer than allowed.
    """

    def __init__(self, model: Model, tasks: Sequence, *, collision: SphereModel | None = None,
                 floor: FloorPoints | None = None,
                 collision_mu: float = 1e3, joint_limits: bool = True,
                 damping: float = 1e-3, qp_iters: int = 15, warm_collision: bool = True,
                 collision_mode: str = "constraint", collision_weight: float = 20.0,
                 collision_partners: int = 2, reuse_inverse: bool = False,
                 max_step: float | None = 0.5, joint_ranges: dict | None = None):
        self.model = model
        self.tracking = [t for t in tasks if not isinstance(t, PostureTask)]
        postures = [t for t in tasks if isinstance(t, PostureTask)]
        self.posture_w = (np.asarray(postures[0].weight, float) if postures
                          else np.zeros(model.nv))
        nominal = postures[0].nominal if postures and postures[0].nominal is not None else None
        self.nominal = np.asarray(model.q0 if nominal is None else nominal, float)
        self.collision = collision
        self.floor = floor
        # free root joint (floating base) whose height the floor projection may adjust
        root = [j for j in range(model.topo.njnt) if model.topo.jnt_type[j] == 0
                and model.topo.body_parent[model.topo.jnt_body[j]] == 0]
        self.free_root_qadr = model.topo.jnt_qposadr[root[0]] if root else None
        self.warm_collision = warm_collision
        assert collision_mode in ("constraint", "penalty")
        self.collision_mode = collision_mode
        self.collision_weight = collision_weight
        self.collision_partners = collision_partners
        # Reusing the first iteration's QP inverse (with refinement) is ~13% faster but
        # can diverge when constraint rows change a lot between iterations: off.
        self.reuse_inverse = reuse_inverse
        # Trust region: |dq| <= max_step (rad) per iteration on limited joints. Bounds
        # the step near singular poses (e.g. a straight leg), so a joint can never
        # flip across its range in one frame.
        self.max_step = max_step
        self.collision_mu = collision_mu
        # scalar or per-DOF (nv,): pulls each step toward q_prev (temporal smoothness)
        self.damping = np.broadcast_to(np.asarray(damping, float), (model.nv,)).copy()
        self.qp_iters = qp_iters

        t = model.topo
        lim = [j for j in range(t.njnt) if t.jnt_limited[j] and t.jnt_type[j] in (HINGE, SLIDE)]
        self.lim_qadr = np.array([t.jnt_qposadr[j] for j in lim], int)
        self.lim_dof = np.array([t.jnt_dofadr[j] for j in lim], int)
        rng = np.array(model.params.jnt_range, float)
        # joint_ranges: {joint index: (lo, hi)} narrower than the mechanical range, e.g.
        # human-plausible limits for retargeting (wide robot ranges admit contortions
        # the IK can get stuck in)
        for j, (lo_j, hi_j) in (joint_ranges or {}).items():
            rng[j] = (max(rng[j, 0], lo_j), min(rng[j, 1], hi_j))
        rng = rng[lim] if lim else np.zeros((0, 2))
        self.lim_lo, self.lim_hi = rng[:, 0].copy(), rng[:, 1].copy()
        if not joint_limits:
            self.lim_qadr = self.lim_qadr[:0]
        self.A_lim = np.eye(model.nv)[self.lim_dof[: len(self.lim_qadr)]]

    # ---------- one Gauss-Newton / QP step ----------

    def linearize(self, q, q_prev, targets):
        model = self.model
        kin = forward_kinematics(model, q)
        rows, errs, ws = [], [], []
        for task, target in zip(self.tracking, targets):
            if isinstance(task, JointTask):  # per-frame joint targets: identity rows
                tgt, scale = target if isinstance(target, tuple) else (target, 1.0)
                rows.append(jnp.asarray(np.eye(model.nv)[task.dofs], q.dtype))
                errs.append(tgt - q[task.qadr])
                ws.append(jnp.asarray(task.weight, q.dtype) * scale)
                continue
            r, e, w = task.residual(model, kin, target)
            rows.append(r), errs.append(e), ws.append(w)
        if self.collision is not None and self.collision_mode == "penalty":
            a, b = self.collision.constraint_rows(model, kin, self.collision_partners)
            active = (b > 0).astype(a.dtype)  # b = -gap > 0: closer than allowed
            rows.append(a * active[:, None]), errs.append(b * active)
            ws.append(jnp.full(b.shape, self.collision_weight, a.dtype))
        J, e, w = jnp.concatenate(rows), jnp.concatenate(errs), jnp.concatenate(ws)
        W = w**2
        pw = jnp.asarray(self.posture_w, J.dtype) ** 2
        damp = jnp.asarray(self.damping, J.dtype)
        # Damping and posture are folded in as extra identity rows so H is ONE matmul:
        # XLA:CPU otherwise fuses "matmul + diagonal" into a slow naive loop (~40x).
        d = damp + pw
        J_aug = jnp.concatenate([J, jnp.eye(model.nv, dtype=J.dtype)])
        W_aug = jnp.concatenate([W, d])
        e_aug = jnp.concatenate([
            e, (damp * difference(model, q, q_prev)
                + pw * difference(model, q, self.nominal)) / d])
        H = J_aug.T @ (W_aug[:, None] * J_aug)
        g = J_aug.T @ (W_aug * e_aug)

        A, l, u, mu = [], [], [], []
        if len(self.lim_qadr):
            qa = q[self.lim_qadr]
            A.append(jnp.asarray(self.A_lim))
            lo, hi = self.lim_lo - qa, self.lim_hi - qa
            if self.max_step is not None:
                lo, hi = jnp.maximum(lo, -self.max_step), jnp.minimum(hi, self.max_step)
            l.append(lo), u.append(hi)
            mu.append(jnp.full(len(qa), INF))
        if self.floor is not None:  # z(p) - r + Jz dq >= height
            f = self.floor
            P = kin.body_p[f.bodies] + einsum("nij,nj->ni", kin.body_R[f.bodies], f.points)
            r = 0.0 if f.radius is None else jnp.asarray(f.radius, P.dtype)
            gap = P[:, 2] - r - f.height
            Jp, _ = points_jacobians(model, kin, f.bodies, P)  # (P, 3, nv)
            Jz = Jp[:, 2]
            if f.lowest is not None:  # rows only for the points closest to the floor
                _, idx = jax.lax.top_k(-gap, f.lowest)
                Jz, gap = Jz[idx], gap[idx]
            A.append(Jz), l.append(-gap)
            u.append(jnp.full(gap.shape, INF)), mu.append(jnp.full(gap.shape, INF))
        col_violation = jnp.zeros((), q.dtype)
        if self.collision is not None and self.collision_mode == "constraint":
            a, b = self.collision.constraint_rows(model, kin, self.collision_partners)
            A.append(a), l.append(b), u.append(jnp.full(b.shape, INF))
            mu.append(jnp.full(b.shape, self.collision_mu))
            col_violation = jnp.sum(jnp.maximum(b, 0.0) ** 2)
        cost = jnp.sum(W * e**2) + self.collision_mu * col_violation
        return H, g, A, l, u, mu, cost

    def init_state(self, q):
        """Initial (q, qp_state) carried through iterations and frames"""
        n_rows = len(self.lim_qadr) + (0 if self.floor is None else self.floor.num_rows) + (
            self.collision.num_rows(self.collision_partners)
                                       if self.collision is not None
                                       and self.collision_mode == "constraint" else 0)
        z = jnp.zeros(n_rows, q.dtype)
        return q, QPState(jnp.zeros(self.model.nv, q.dtype), z, z)

    def step(self, state, q_prev, targets, precond=None):
        """One Gauss-Newton/QP step. state = (q, qp warm start). precond: a QP matrix
        inverse from an earlier step of the same frame, reused with refinement."""
        q, qp_state = state
        H, g, A, l, u, mu, cost = self.linearize(q, q_prev, targets)
        Kinv = None
        if A:
            res = solve_qp(H, g, jnp.concatenate(A), jnp.concatenate(l), jnp.concatenate(u),
                           elastic_mu=jnp.concatenate(mu), iters=self.qp_iters,
                           warm_start=qp_state, precond=precond)
            dq, Kinv = res.x, res.Kinv
            # dq restarts at 0; joint-limit duals carry over (stable meanings). Collision
            # rows can switch partner between steps; warm_collision keeps their duals.
            y = res.state.y
            if not self.warm_collision and self.collision is not None and \
                    self.collision_mode == "constraint":
                nf = 0 if self.floor is None else self.floor.num_rows
                y = y.at[len(self.lim_qadr) + nf:].set(0.0)
            qp_state = res.state._replace(x=jnp.zeros_like(res.state.x), y=y)
        else:
            dq = jnp.linalg.solve(H, g)
        q = integrate(self.model, q, dq, 1.0)
        if len(self.lim_qadr):
            q = q.at[self.lim_qadr].set(jnp.clip(q[self.lim_qadr], self.lim_lo, self.lim_hi))
        q = self._floor_projection(q)
        return (q, qp_state), cost, Kinv

    def _floor_projection(self, q):
        """Exact floor guarantee for a floating base: after the (approximate) QP step,
        lift the root by the deepest remaining floor penetration, if any."""
        f = self.floor
        if f is None or self.free_root_qadr is None:
            return q
        kin = forward_kinematics(self.model, q)
        P = kin.body_p[f.bodies] + einsum("nij,nj->ni", kin.body_R[f.bodies], f.points)
        r = 0.0 if f.radius is None else jnp.asarray(f.radius, P.dtype)
        depth = jnp.maximum(0.0, jnp.max(f.height - (P[:, 2] - r)))
        return q.at[self.free_root_qadr + 2].add(depth)

    def solve(self, state, targets, iters: int = 30, q_ref=None):
        """Solve one frame from state = (q, qp_state) (or a bare q).

        q_ref: configuration the damping term pulls toward (default: the start q);
        e.g. a constant-velocity prediction while tracking.
        Up to 4 iterations are unrolled (no device loop).
        """
        if not isinstance(state, tuple):
            state = self.init_state(state)
        q0 = state[0] if q_ref is None else q_ref
        state, cost, Kinv = self.step(state, q0, targets)
        reuse = Kinv if (self.reuse_inverse and Kinv is not None) else None
        if iters <= 4:
            for _ in range(iters - 1):
                state, cost, _ = self.step(state, q0, targets, precond=reuse)
            return state, cost

        def body(_, c):
            st, _ = c
            st, cost, _ = self.step(st, q0, targets, precond=reuse)
            return st, cost

        return jax.lax.fori_loop(0, iters - 1, body, (state, cost))

    def track(self, state, targets_seq, iters: int = 2, predict: bool = True):
        """Sequentially solve a trajectory (targets with a leading time axis).

        predict: start each frame from a constant-velocity prediction
        q_t (+) (q_t (-) q_{t-1}); mocap is smooth, so it lands close to the solution.
        """
        if not isinstance(state, tuple):
            state = self.init_state(state)
        model = self.model

        def scan_fn(carry, t):
            st, q_last = carry
            q = st[0]
            if predict:
                q_guess = integrate(model, q, difference(model, q_last, q), 1.0)
                if len(self.lim_qadr):
                    q_guess = q_guess.at[self.lim_qadr].set(
                        jnp.clip(q_guess[self.lim_qadr], self.lim_lo, self.lim_hi))
                st = (q_guess, st[1])
            st, cost = self.solve(st, t, iters, q_ref=st[0])
            return (st, q), (st[0], cost)

        _, (qs, costs) = jax.lax.scan(scan_fn, (state, state[0]), targets_seq)
        return qs, costs

    # ---------- batched trajectories ----------

    def solve_trajectory(self, q_init: Array | Callable, targets_seq, *, segment: int = 128,
                         overlap: int = 16, iters: int = 2, init_iters: int = 15,
                         predict: bool = True, devices: Sequence | None = None):
        """Solve one trajectory, or a list of them, as parallel overlapping segments.

        targets_seq: a targets pytree with a leading time axis, or a list of them (one
            per clip). All segments of all clips run in one batch; segments never
            cross clip boundaries.
        q_init: (nq,) seed, or a function targets_frame -> seed (e.g. to place the root).
            A seed of shape (K, nq) is a multi-start: all K are solved, the best kept.
        Each segment starts `overlap` frames early from a solved seed; those warm-up
        frames are discarded, so seams are continuous.
        Returns (qs (T, nq), costs (T,)), or lists of them for a list input.
        """
        single = not isinstance(targets_seq, list)
        clips = [targets_seq] if single else targets_seq
        lengths = [jax.tree_util.tree_leaves(c)[0].shape[0] for c in clips]
        L = segment + overlap
        # frame index into the concatenated clips for every (segment, step)
        offsets = np.cumsum([0] + lengths[:-1])
        seg_idx, seg_clip = [], []
        L = segment + 2 * overlap  # warm-up before + blend window after
        for c, (T, off) in enumerate(zip(lengths, offsets)):
            nseg = -(-T // segment)
            idx = np.arange(nseg)[:, None] * segment - overlap + np.arange(L)[None]
            seg_idx.append(off + np.clip(idx, 0, T - 1))
            seg_clip += [c] * nseg
        idx = np.concatenate(seg_idx)
        B = len(idx)
        devices = list(jax.devices()) if devices is None else list(devices)
        D = max(1, min(len(devices), B))
        pad = (-B) % D
        idx = np.concatenate([idx, np.repeat(idx[-1:], pad, 0)])
        allt = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *clips)
        seg_targets = jax.tree_util.tree_map(lambda x: x[idx], allt)  # (B + pad, L, ...)

        # 1) seeds: every (segment, candidate) pair as ONE flat batch (nested vmaps
        #    trip an XLA GPU compiler bug), 2) keep the best per segment, 3) track.
        Bp = B + pad
        firsts = jax.tree_util.tree_map(lambda x: x[:, 0], seg_targets)  # (Bp, ...)
        seeds = jax.vmap(q_init)(firsts) if callable(q_init) else jnp.broadcast_to(
            jnp.asarray(q_init), (Bp,) + jnp.shape(q_init))
        multi = seeds.ndim == 3
        K = seeds.shape[1] if multi else 1
        flat_seeds = seeds.reshape(Bp * K, -1)
        flat_first = jax.tree_util.tree_map(lambda x: jnp.repeat(x, K, axis=0), firsts)

        def seed_solve(q0, first):
            st, _ = self.solve(q0, first, init_iters)
            return st, self.linearize(st[0], st[0], first)[-1]

        def track(st, tg):
            return self.track(st, tg, iters, predict=predict)

        if D > 1:
            def shard(x):
                return x.reshape((D, x.shape[0] // D) + x.shape[1:])

            def unshard(x):
                return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])

            pm_seed = _pmapped(self, "seed", seed_solve, tuple(devices[:D]), init_iters)
            pm_track = _pmapped(self, "track", track, tuple(devices[:D]), iters, predict)
            states, costs0 = jax.tree_util.tree_map(
                unshard, pm_seed(shard(flat_seeds), jax.tree_util.tree_map(shard, flat_first)))
        else:
            states, costs0 = jax.jit(jax.vmap(seed_solve))(flat_seeds, flat_first)
        best = jnp.argmin(costs0.reshape(Bp, K), axis=1)
        states = jax.tree_util.tree_map(
            lambda x: x.reshape((Bp, K) + x.shape[1:])[jnp.arange(Bp), best], states)
        if D > 1:
            qs, costs = jax.tree_util.tree_map(
                unshard, pm_track(jax.tree_util.tree_map(shard, states),
                                  jax.tree_util.tree_map(shard, seg_targets)))
        else:
            qs, costs = jax.jit(jax.vmap(track))(states, seg_targets)
        qs, costs = qs[:B, overlap:], costs[:B, overlap:]  # (B, segment + overlap)
        # Crossfade each segment's tail (frames past its end) into the next segment's
        # head: seams become a smooth `overlap`-frame transition, never a jump.
        w = (np.arange(1, overlap + 1) / (overlap + 1)) if overlap else np.zeros(0)
        out_q, out_c, k = [], [], 0
        for T in lengths:
            nseg = -(-T // segment)
            seg_q = qs[k:k + nseg]
            body = seg_q[:, :segment]
            if overlap and nseg > 1:
                body = self._blend_seams(seg_q, jnp.asarray(w, seg_q.dtype), segment, overlap)
            out_q.append(body.reshape(nseg * segment, -1)[:T])
            out_c.append(costs[k:k + nseg, :segment].reshape(-1)[:T])
            k += nseg
        return (out_q[0], out_c[0]) if single else (out_q, out_c)

    @functools.partial(jax.jit, static_argnums=(0, 3, 4))
    def _blend_seams(self, seg_q, w, segment, overlap):
        """Compiled crossfade of each segment's tail into the next segment's head"""
        body = seg_q[:, :segment]
        tail, head = seg_q[:-1, segment:], seg_q[1:, :overlap]  # the same frames
        blend = jax.vmap(jax.vmap(self._interpolate, in_axes=(0, 0, 0)),
                         in_axes=(0, 0, None))(tail, head, w)
        return body.at[1:, :overlap].set(blend)

    def _interpolate(self, qa, qb, t):
        """Geodesic interpolation qa -> qb (quaternions slerped via difference/integrate)"""
        return integrate(self.model, qa, difference(self.model, qa, qb), t)


_PMAP_CACHE = {}


def _pmapped(solver, name, fn, devices, *key):
    """Cache pmapped batch functions so repeated calls reuse the compilation"""
    k = (id(solver), name, devices) + key
    if k not in _PMAP_CACHE:
        _PMAP_CACHE[k] = jax.pmap(jax.vmap(fn), devices=list(devices))
    return _PMAP_CACHE[k]
