"""rigidax vs frax on the same robots, CPU, float64, frax's recommended JAX settings.

Run:
  XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' \
  OPENBLAS_NUM_THREADS=1 JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python bench/bench_vs_frax.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import frax
import rigidax as rx

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).resolve().parent.parent / "tests" / "models"
BATCH = 1024


def timeit(f, *args, n=2000):
    t0 = time.perf_counter()
    jax.block_until_ready(f(*args))
    compile_s = time.perf_counter() - t0
    for _ in range(50):
        jax.block_until_ready(f(*args))
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t)
    return np.median(ts) * 1e6, compile_s


def ops_frax(r):
    return {
        "mass_matrix": lambda q, v, tau: r.mass_matrix(q),
        "bias": lambda q, v, tau: r.nonlinear_bias(q, v),
        "forward_dyn": lambda q, v, tau: r.forward_dynamics(q, v, tau, None),
        "M + bias": lambda q, v, tau: (r.mass_matrix(q), r.nonlinear_bias(q, v)),
    }


def ops_rx(m):
    return {
        "mass_matrix": lambda q, v, tau: rx.mass_matrix(m, q),
        "bias": lambda q, v, tau: rx.bias_forces(m, q, v),
        "forward_dyn": lambda q, v, tau: rx.forward_dynamics(m, q, v, tau),
        "M + bias": lambda q, v, tau: rx.dynamics_terms(m, q, v),
    }


def state(nq_q0, nv, rng, batch=None):
    q = np.asarray(nq_q0, float)
    shape = () if batch is None else (batch,)
    Q = np.broadcast_to(q, shape + q.shape).copy()
    V = rng.normal(size=shape + (nv,))
    T = rng.normal(size=shape + (nv,))
    return jnp.asarray(Q), jnp.asarray(V), jnp.asarray(T)


def main():
    rng = np.random.default_rng(0)
    robots = {
        "Panda (7 dof, chain)": (frax.load_panda(), rx.load(
            Path(frax.__file__).parent / "assets/franka_panda/panda.urdf", strip_meshes=True)),
        "G1 (35 dof, tree + free base)": (frax.load_g1("quaternion"),
                                          rx.load(HERE / "g1_29dof_rev_1_0.xml", strip_meshes=True)),
    }
    print(f"jax {jax.__version__}, device {jax.devices()[0]}, batch {BATCH}\n")
    for name, (fr, rm) in robots.items():
        assert fr.nv == rm.nv, (fr.nv, rm.nv)
        print(f"== {name}")
        print(f"{'op':<12} {'frax us':>9} {'rigidax us':>11} {'speedup':>8} | "
              f"{'frax batch us':>14} {'rigidax batch us':>17} {'speedup':>8} | compile s (fx/rx)")
        fo, ro = ops_frax(fr), ops_rx(rm)
        args_f = state(fr.default_configuration, fr.nv, rng)
        args_r = state(rm.q0, rm.nv, rng)
        bargs_f = state(fr.default_configuration, fr.nv, rng, BATCH)
        bargs_r = state(rm.q0, rm.nv, rng, BATCH)
        for op in fo:
            tf, cf = timeit(jax.jit(fo[op]), *args_f)
            tr, cr = timeit(jax.jit(ro[op]), *args_r)
            bf, _ = timeit(jax.jit(jax.vmap(fo[op])), *bargs_f, n=200)
            br, _ = timeit(jax.jit(jax.vmap(ro[op])), *bargs_r, n=200)
            print(f"{op:<12} {tf:9.2f} {tr:11.2f} {tf / tr:7.2f}x | {bf:14.1f} {br:17.1f} "
                  f"{bf / br:7.2f}x | {cf:.2f}/{cr:.2f}")
        print()


if __name__ == "__main__":
    main()
