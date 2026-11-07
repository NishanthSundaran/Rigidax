"""Fixed-iteration ADMM QP solver for small, dense, batched problems.

    minimize    1/2 x^T H x - g^T x
    subject to  l <= A x <= u          (hard rows)
                l <= A x <= u          (elastic rows: violation costs mu * |.|_1)

OSQP-style splitting, specialized for thousands of tiny problems at once:
  * row equilibration (constraint rows normalized),
  * the ADMM system matrix is inverted once (Cholesky + triangular solve) and every
    iteration is a matrix-vector product: batched triangular solves are slow on CPU,
    batched mat-vecs are not,
  * warm starts (x, z, y) from a previous, similar problem (e.g. the previous frame),
  * optional residual-balanced rho update and optional active-set polishing.
No data-dependent branches: it jits, vmaps, and takes the same time on every input.
Elastic rows are always feasible: an exact L1 penalty on their violation, i.e. their
duals are bounded |y| <= mu.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

INF = 1e10


class QPState(NamedTuple):
    """ADMM iterates in the solver's internal (row-scaled) units, for warm starts"""
    x: Array
    z: Array
    y: Array


class QPResult(NamedTuple):
    x: Array
    y: Array  # constraint duals (original row scaling)
    primal_residual: Array  # max violation of the hard rows
    state: QPState
    Kinv: Array  # inverse (or preconditioner) of the ADMM system matrix, for reuse


def spd_inverse(K: Array) -> Array:
    """Inverse of a symmetric positive definite matrix via Cholesky"""
    L = jnp.linalg.cholesky(K)
    eye = jnp.broadcast_to(jnp.eye(K.shape[-1], dtype=K.dtype), K.shape)
    Li = jax.lax.linalg.triangular_solve(L, eye, left_side=True, lower=True)
    return jnp.swapaxes(Li, -1, -2) @ Li


def _admm(H, g, A, l, u, mu, rho_v, state, iters, sigma, alpha, P=None, refine=1):
    """P: preconditioner (inverse of a nearby system matrix) to reuse instead of
    inverting; each x-update then applies `refine` steps of iterative refinement, so
    it still solves the current system."""
    n = H.shape[0]
    K = H + sigma * jnp.eye(n, dtype=H.dtype) + (A.T * rho_v) @ A
    exact = P is None
    Kinv = spd_inverse(K) if exact else P

    def solve_x(rhs, x_prev):
        if exact:
            return Kinv @ rhs
        xs = x_prev + Kinv @ (rhs - K @ x_prev)
        for _ in range(refine):
            xs = xs + Kinv @ (rhs - K @ xs)
        return xs

    def body(s):
        x, z, y = s
        xt = solve_x(sigma * x + g + A.T @ (rho_v * z - y), x)
        zt = A @ xt
        x = alpha * xt + (1 - alpha) * x
        zr = alpha * zt + (1 - alpha) * z
        # z-update = prox of the box indicator (hard rows) or of mu * L1 distance to
        # the box (elastic rows): project, then let z stay outside by up to mu / rho
        v = zr + y / rho_v
        box = jnp.clip(v, l, u)
        d = v - box
        z_new = box + jnp.sign(d) * jnp.maximum(jnp.abs(d) - mu / rho_v, 0.0)
        y = y + rho_v * (zr - z_new)
        return QPState(x, z_new, y)

    if iters <= 16:  # unrolled: straight-line fused code (no device loop overhead)
        for _ in range(iters):
            state = body(state)
        return state, Kinv
    return jax.lax.fori_loop(0, iters, lambda _, st: body(st), state), Kinv


def solve_qp(H: Array, g: Array, A: Array, l: Array, u: Array, *,
             elastic_mu: Array | None = None, iters: int = 25, rho: float = 0.1,
             sigma: float = 1e-6, alpha: float = 1.6, warm_start: QPState | None = None,
             adaptive_rho: bool = False, polish: bool = False,
             precond: Array | None = None, refine: int = 1, hard_rho: float = 1.0) -> QPResult:
    """Solve the QP above.

    H: (n, n) PSD; g: (n,); A: (m, n); l, u: (m,) (use +-INF for unbounded sides).
    elastic_mu: (m,) L1 penalty per row; INF (or None) makes a row hard.
    warm_start: the `state` of a previous result with the same shapes.
    adaptive_rho: one residual-balanced rho update halfway (one extra inverse).
    polish: exact solve on the detected active set (accurate, but one dense solve).
    precond: `Kinv` of a previous, similar solve: skips the inversion and uses iterative
        refinement (`refine` steps per iteration) against the current system instead.
    """
    n, m = H.shape[0], A.shape[0]
    dt = H.dtype
    mu = jnp.full(m, INF, dt) if elastic_mu is None else jnp.asarray(elastic_mu, dt)
    s = 1.0 / jnp.maximum(jnp.linalg.norm(A, axis=1), 1e-12)
    As = A * s[:, None]
    ls = jnp.where(l > -INF, l * s, l)
    us = jnp.where(u < INF, u * s, u)
    mus = jnp.where(mu < INF, mu / s, INF)
    eq = (us - ls) < 1e-6
    # stiffer penalty on hard rows (as OSQP does for equalities): within a fixed
    # iteration budget they are then enforced much more tightly than elastic ones
    hard_scale = jnp.where(eq, 1e3, jnp.where(mus >= INF, hard_rho, 1.0))
    rho_v = hard_scale * rho * jnp.mean(jnp.diag(H)) * jnp.ones(m, dt)
    z0 = jnp.clip(jnp.zeros(m, dt), ls, us)
    if warm_start is None:
        state = QPState(jnp.zeros(n, dt), z0, jnp.zeros(m, dt))
    else:
        # warm start x and the duals y; the slack restarts consistently with x
        x0 = warm_start.x
        state = QPState(x0, jnp.clip(As @ x0, ls, us), warm_start.y)

    if adaptive_rho:
        k1 = iters // 2
        state, _ = _admm(H, g, As, ls, us, mus, rho_v, state, k1, sigma, alpha)
        x, z, y = state
        Ax = As @ x
        r_p = jnp.max(jnp.abs(Ax - z)) / (
            jnp.maximum(jnp.max(jnp.abs(Ax)), jnp.max(jnp.abs(z))) + 1e-12)
        grad = H @ x - g
        r_d = jnp.max(jnp.abs(grad + As.T @ y)) / (
            jnp.maximum(jnp.max(jnp.abs(grad + g)), jnp.max(jnp.abs(As.T @ y)))
            + jnp.max(jnp.abs(g)) + 1e-12)
        rho_v = rho_v * jnp.clip(jnp.sqrt(r_p / (r_d + 1e-12)), 1e-3, 1e3)
        state, Kinv = _admm(H, g, As, ls, us, mus, rho_v, state, iters - k1, sigma, alpha)
    else:
        state, Kinv = _admm(H, g, As, ls, us, mus, rho_v, state, iters, sigma, alpha,
                            P=precond, refine=refine)

    x = state.x
    if polish:
        x = _polish(H, g, As, ls, us, mus, x, state.y)
    Ax = As @ x
    viol = jnp.where(mus >= INF, jnp.maximum(jnp.maximum(ls - Ax, Ax - us), 0.0) / s, 0.0)
    return QPResult(x, state.y * s, jnp.max(viol, initial=0.0), state, Kinv)


def _polish(H, g, A, l, u, mu, x_admm, y):
    """Exact solve on ADMM's active set; keep it only if feasible and not worse"""
    n = H.shape[0]
    dt = H.dtype
    tol = 1e-7 * (1.0 + jnp.max(jnp.abs(y)))
    saturated = (mu < INF) & (jnp.abs(y) >= mu - tol)  # elastic rows at their max force
    lower = (y < -tol) & ~saturated
    upper = (y > tol) & ~saturated
    active = lower | upper
    b = jnp.where(lower, l, u)
    g_eff = g - A.T @ jnp.where(saturated, y, 0.0)
    Aa = A * active[:, None]
    K = jnp.block([[H + 1e-10 * jnp.eye(n, dtype=dt), Aa.T],
                   [Aa, jnp.diag(jnp.where(active, -1e-9, -1.0))]])
    sol = jnp.linalg.solve(K, jnp.concatenate([g_eff, jnp.where(active, b, 0.0)]))
    xp = sol[:n]
    Axp = A @ xp
    hard = mu >= INF
    feasible = jnp.all(~hard | ((Axp >= l - 1e-6) & (Axp <= u + 1e-6)))

    def obj(xv):
        Av = A @ xv
        out = jnp.maximum(l - Av, 0.0) + jnp.maximum(Av - u, 0.0)
        return (0.5 * xv @ H @ xv - g @ xv + jnp.sum(jnp.where(hard, 1e6, mu) * out))

    better = feasible & (obj(xp) <= obj(x_admm) + 1e-12 * (1 + jnp.abs(obj(x_admm))))
    return jnp.where(better & jnp.all(jnp.isfinite(xp)), xp, x_admm)
