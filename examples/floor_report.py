"""Ground truth: how far any part of the robot (not just the feet) sinks through the
floor, from MuJoCo's mesh collision against the floor plane."""
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def floor_penetration(m, q, stride=2):
    d = mujoco.MjData(m)
    floor = m.geom("floor").id
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    depth = np.zeros(len(q[::stride]))
    worst_body = {}
    for i, k in enumerate(range(0, len(q), stride)):
        d.qpos[:] = q[k]
        mujoco.mj_forward(m, d)
        for c in d.contact[: d.ncon]:
            if floor not in (c.geom1, c.geom2):
                continue
            g = c.geom2 if c.geom1 == floor else c.geom1
            if -c.dist > depth[i]:
                depth[i] = -c.dist
                worst_body[i] = names[m.geom_bodyid[g]]
    return depth, worst_body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--data", default=str(ROOT / "data" / "g1_lafan1"))
    args = ap.parse_args()
    for clip in args.clips:
        q = np.load(Path(args.data) / f"{clip}.npz")["qpos"]
        m = mujoco.MjModel.from_xml_path(str(ROOT / "examples" / "unitree_g1" / (
            "scene_hands.xml" if q.shape[1] == 50 else "scene.xml")))
        depth, wb = floor_penetration(m, q)
        bad = depth > 0.02
        bodies = {}
        for i in np.flatnonzero(bad):
            b = wb[i].replace("_link", "")
            bodies[b] = bodies.get(b, 0) + 1
        top = ", ".join(f"{b} {n}" for b, n in sorted(bodies.items(), key=lambda x: -x[1])[:4])
        print(f"{clip:26s} through floor > 2 cm: {bad.mean() * 100:5.1f}% of frames, deepest "
              f"{depth.max() * 100:5.1f} cm | {top}")


if __name__ == "__main__":
    main()
