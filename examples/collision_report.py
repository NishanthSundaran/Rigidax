"""Ground-truth self-collision report for retargeted G1 clips, from MuJoCo's own
collision detection on the G1 collision meshes (convex hulls).

Counts frames where an arm (upper arm / forearm / hand) penetrates the torso,
pelvis, legs or the other arm by more than --depth.

  python examples/collision_report.py dance1_subject1 fightAndSports1_subject1 ...
"""

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G1_XML = ROOT / "examples" / "unitree_g1" / "g1.xml"
DATA = Path(os.environ.get("DATA", ROOT / "data" / "g1_lafan1"))
ARM = ("shoulder_yaw", "elbow", "wrist", "hand")


def side(name):
    return "left" if name.startswith("left") else "right" if name.startswith("right") else None


def report(m, q, depth, stride):
    d = mujoco.MjData(m)
    bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    hits, worst, pairs = 0, 0.0, {}
    frames = range(0, len(q), stride)
    for k in frames:
        d.qpos[:] = q[k]
        mujoco.mj_forward(m, d)
        hit = False
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == 0 or b2 == 0 or c.dist > -depth:
                continue  # floor contact, or too shallow
            n1, n2 = bname[b1], bname[b2]
            a1, a2 = any(k in n1 for k in ARM), any(k in n2 for k in ARM)
            if not (a1 or a2):
                continue
            if a1 and a2 and side(n1) == side(n2):
                continue  # same-arm links touching
            hit = True
            worst = max(worst, -c.dist)
            key = tuple(sorted((n1.replace("_link", ""), n2.replace("_link", ""))))
            pairs[key] = pairs.get(key, 0) + 1
        hits += hit
    return hits / len(frames), worst, sorted(pairs.items(), key=lambda kv: -kv[1])[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--depth", type=float, default=0.01, help="penetration threshold (m)")
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()
    for clip in args.clips:
        q = np.load(DATA / f"{clip}.npz")["qpos"]
        m = mujoco.MjModel.from_xml_path(str(G1_XML.parent / (
            "g1_with_hands.xml" if q.shape[1] == 50 else "g1.xml")))
        frac, worst, top = report(m, q, args.depth, args.stride)
        tops = ", ".join(f"{a}/{b} {n}" for (a, b), n in top)
        print(f"{clip:26s} arm penetration > {args.depth * 100:.0f} cm in {frac * 100:5.1f}% of "
              f"frames, deepest {worst * 100:4.1f} cm | top pairs: {tops}")


if __name__ == "__main__":
    main()
