"""Minimal BVH motion-capture reader (Y-up files converted to Z-up)"""

from pathlib import Path

import numpy as np


def load_bvh(path):
    """-> joint names, parents, offsets (J, 3), channels per joint, motion (T, C)"""
    names, parents, offsets, channels, stack = [], [], [], [], []
    lines = iter(Path(path).read_text().split("\n"))
    for line in lines:
        tok = line.split()
        if not tok:
            continue
        if tok[0] in ("ROOT", "JOINT"):
            names.append(tok[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append(None)
            channels.append([])
        elif tok[0] == "End":
            names.append(None)  # end site: parsed for nesting, dropped later
            parents.append(stack[-1])
            offsets.append(None)
            channels.append([])
        elif tok[0] == "{":
            stack.append(len(names) - 1)
        elif tok[0] == "}":
            stack.pop()
        elif tok[0] == "OFFSET":
            offsets[stack[-1]] = [float(x) for x in tok[1:4]]
        elif tok[0] == "CHANNELS":
            channels[stack[-1]] = tok[2:]
        elif tok[0] == "MOTION":
            break
    next(lines)  # Frames:
    next(lines)  # Frame Time:
    motion = np.array([[float(x) for x in ln.split()] for ln in lines if ln.strip()])
    keep = [i for i, n in enumerate(names) if n is not None]
    remap = {old: new for new, old in enumerate(keep)}
    return ([names[i] for i in keep], [remap.get(parents[i], -1) for i in keep],
            np.array([offsets[i] for i in keep]), [channels[i] for i in keep], motion)


def _axis_rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.moveaxis(np.array(m), [0, 1], [-2, -1])


def bvh_rotations(path):
    """Global joint orientations {name: (T, 3, 3)}, Z-up (same frame as bvh_positions)"""
    names, parents, offsets, channels, motion = load_bvh(path)
    T, col = len(motion), 0
    R = [None] * len(names)
    for j, ch in enumerate(channels):
        Rl = np.broadcast_to(np.eye(3), (T, 3, 3))
        for c in ch:
            if not c.endswith("position"):
                Rl = Rl @ _axis_rot(c[0], motion[:, col])
            col += 1
        R[j] = Rl if parents[j] < 0 else R[parents[j]] @ Rl
    C = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # Y-up -> Z-up
    return {n: C @ R[i] for i, n in enumerate(names)}


def bvh_positions(path, unit: float = 0.01):
    """Global joint positions {name: (T, 3)}, Z-up, in meters (unit: BVH units -> m)"""
    names, parents, offsets, channels, motion = load_bvh(path)
    T, col = len(motion), 0
    R = [None] * len(names)
    P = [None] * len(names)
    for j, ch in enumerate(channels):
        Rl = np.broadcast_to(np.eye(3), (T, 3, 3))
        pos = np.broadcast_to(offsets[j], (T, 3))
        for c in ch:
            if c.endswith("position"):
                pos = pos.copy() if pos.base is not None else pos
                pos = np.array(pos)
                pos[:, "XYZ".index(c[0])] = motion[:, col]
            else:
                Rl = Rl @ _axis_rot(c[0], motion[:, col])
            col += 1
        p = parents[j]
        if p < 0:
            R[j], P[j] = Rl, pos
        else:
            R[j] = R[p] @ Rl
            P[j] = P[p] + np.einsum("tij,j->ti", R[p], offsets[j])
    pos = np.stack(P, 1) * unit
    C = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # Y-up -> Z-up
    return {n: pos[:, i] @ C.T for i, n in enumerate(names)}
