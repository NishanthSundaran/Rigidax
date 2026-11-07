"""Rotation and spatial-vector helpers.

Spatial vectors use the [angular; linear] ordering and are expressed in the world
frame, about the world origin.
"""

import jax.numpy as jnp
from jax import lax

HI = lax.Precision.HIGHEST


def mm(a, b):
    return jnp.matmul(a, b, precision=HI)


def einsum(spec, *ops):
    return jnp.einsum(spec, *ops, precision=HI)


def skew(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = jnp.zeros_like(x)
    return jnp.stack(
        [jnp.stack([o, -z, y], -1), jnp.stack([z, o, -x], -1), jnp.stack([-y, x, o], -1)],
        -2,
    )


def quat_normalize(q):
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_mat(q):
    """WXYZ quaternion(s) -> rotation matrix(es)"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack(
        [
            jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def axis_angle_to_mat(axis, angle):
    """Rodrigues' formula; axis must be unit length"""
    K = skew(axis)
    s = jnp.sin(angle)[..., None, None]
    c = jnp.cos(angle)[..., None, None]
    return jnp.eye(3) + s * K + (1 - c) * mm(K, K)


def quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    w = aw * bw - jnp.sum(av * bv, -1, keepdims=True)
    v = aw * bv + bw * av + jnp.cross(av, bv)
    return jnp.concatenate([w, v], -1)


def quat_conj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def _safe_norm(x):
    n2 = jnp.sum(x * x, -1, keepdims=True)
    small = n2 < 1e-24
    return jnp.sqrt(jnp.where(small, 1.0, n2)), small


def quat_exp(w):
    """Rotation vector -> unit quaternion (gradient-safe at zero)"""
    n, small = _safe_norm(w)
    half = 0.5 * n
    k = jnp.where(small, 0.5, jnp.sin(half) / n)
    c = jnp.where(small, 1.0, jnp.cos(half))
    return jnp.concatenate([c, k * w], -1)


def quat_log(q):
    """Unit quaternion -> rotation vector (shortest path)"""
    q = jnp.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    n, small = _safe_norm(v)
    angle = 2.0 * jnp.arctan2(n, q[..., :1])
    return jnp.where(small, 2.0 * v, v * angle / n)


def quat_integrate(q, w_body, dt):
    """q <- q * exp(w_body * dt), with w in the body frame (MuJoCo convention)"""
    return quat_normalize(quat_mul(q, quat_exp(w_body * dt)))


def cross_motion(v, m):
    """Spatial motion cross product v x m"""
    w, u = v[..., :3], v[..., 3:]
    a, b = m[..., :3], m[..., 3:]
    return jnp.concatenate([jnp.cross(w, a), jnp.cross(w, b) + jnp.cross(u, a)], -1)


def cross_force(v, f):
    """Spatial force cross product v x* f"""
    w, u = v[..., :3], v[..., 3:]
    n, g = f[..., :3], f[..., 3:]
    return jnp.concatenate([jnp.cross(w, n) + jnp.cross(u, g), jnp.cross(w, g)], -1)
