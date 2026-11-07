"""Offline trajectory post-processing"""

import numpy as np

from rigidax.model import BALL, FREE


def gaussian_kernel(sigma_frames: float) -> np.ndarray:
    r = int(np.ceil(3 * sigma_frames))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    return k / k.sum()


def _filter(x, k):
    r = len(k) // 2
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, xp)


def smooth_trajectory(model, qs, sigma_frames: float, joints=None) -> np.ndarray:
    """Zero-phase Gaussian smoothing of a configuration trajectory (T, nq).

    Unlike damping toward the previous frame (a causal filter, which lags), filtering
    the whole trajectory forward and backward in time adds no delay. Scalar joints and
    free-joint positions are filtered directly; quaternions are sign-aligned, filtered
    and renormalized. joints: optional boolean mask over joints to smooth (default all).
    """
    if sigma_frames <= 0:
        return np.asarray(qs)
    qs = np.array(qs, dtype=float)
    t = model.topo
    k = gaussian_kernel(sigma_frames)
    sel = np.ones(t.njnt, bool) if joints is None else np.asarray(joints, bool)
    for j in range(t.njnt):
        if not sel[j]:
            continue
        a = t.jnt_qposadr[j]
        if t.jnt_type[j] == FREE:
            qs[:, a:a + 3] = _filter(qs[:, a:a + 3], k)
            quat = qs[:, a + 3:a + 7]
        elif t.jnt_type[j] == BALL:
            quat = qs[:, a:a + 4]
        else:
            qs[:, a] = _filter(qs[:, a:a + 1], k)[:, 0]
            continue
        # quaternions: align hemispheres frame to frame, filter, renormalize
        quat = quat.copy()
        for i in range(1, len(quat)):
            if np.dot(quat[i], quat[i - 1]) < 0:
                quat[i] = -quat[i]
        quat = _filter(quat, k)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        if t.jnt_type[j] == FREE:
            qs[:, a + 3:a + 7] = quat
        else:
            qs[:, a:a + 4] = quat
    return qs
