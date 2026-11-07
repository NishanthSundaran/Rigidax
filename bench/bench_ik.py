"""Per-component cost of one IK step, batched over N frames on one CPU device (float32)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import jax, jax.numpy as jnp, mujoco, numpy as np
jax.config.update("jax_enable_x64", False)
import rigidax as rx
from rigidax.ik import solve_qp, spd_inverse
from rigidax.ik.bvh import bvh_positions
import retarget_lafan1 as R

N = 4096
m = mujoco.MjModel.from_xml_path(str(R.G1_XML)); model = rx.from_mujoco(m)
H = bvh_positions(R.LAFAN / "walk1_subject1.bvh")
tg = tuple(jnp.asarray(t[:N], jnp.float32) for t in R.human_targets(H, R.leg_scale(m, model, H)))
q = jnp.broadcast_to(jnp.asarray(m.qpos0, jnp.float32), (N, m.nq))


def timeit(name, f, *a):
    f = jax.jit(f); jax.block_until_ready(f(*a))
    t = time.perf_counter(); [jax.block_until_ready(f(*a)) for _ in range(3)]
    us = (time.perf_counter() - t) / 3 / N * 1e6
    print(f"  {name:46s} {us:7.2f} us/frame")
    return us


for k, qp_iters in ((2, 15), (2, 8)):
    s = R.build_solver(m, model, collision_partners=k, qp_iters=qp_iters)
    print(f"--- {k} partners, QP {qp_iters} iters, {s.collision.num_rows(k)} collision rows")
    timeit("forward kinematics", jax.vmap(lambda q: rx.forward_kinematics(model, q)), q)
    timeit("tasks + H, g", jax.vmap(lambda q, t: s.linearize(q, q, t)[:2]), q, tg)
    timeit("collision rows", jax.vmap(lambda q: s.collision.constraint_rows(model, rx.forward_kinematics(model, q), k)), q)
    Hm, g, A, l, u, mu, _ = jax.jit(jax.vmap(lambda q, t: s.linearize(q, q, t)))(q, tg)
    A, l, u, mu = (jnp.concatenate(x, axis=-1 if x[0].ndim == 2 else -2) if False else jnp.concatenate(x, axis=1) for x in (A, l, u, mu))
    timeit("QP matrix inverse only", jax.vmap(lambda H, A: spd_inverse(H + A.T @ A)), Hm, A)
    timeit(f"QP solve ({qp_iters} iters)", jax.vmap(lambda H, g, A, l, u, mu: solve_qp(H, g, A, l, u, elastic_mu=mu, iters=qp_iters).x), Hm, g, A, l, u, mu)
    timeit("full step", jax.vmap(lambda q, t: s.step(s.init_state(q), q, t)[:2]), q, tg)
    timeit("frame, 3 iterations (1 inverse + 2 reuses)", jax.vmap(lambda q, t: s.solve(q, t, 3)), q, tg)
