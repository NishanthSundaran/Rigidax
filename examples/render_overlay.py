"""Diagnostic render: the scaled human mocap skeleton (red) drawn over the retargeted
robot, so tracking errors, lag and foot sliding are visible frame by frame.

  MUJOCO_GL=egl python examples/render_overlay.py dance1_subject1 --start 113 --seconds 4 \
      --out overlay.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import retarget_lafan1 as R  # noqa: E402
from rigidax.ik.bvh import bvh_positions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BONES = [("Hips", "Spine1"), ("Spine1", "Neck"), ("Neck", "Head"),
         ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToe"),
         ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"), ("RightFoot", "RightToe"),
         ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
         ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
         ("LeftUpLeg", "RightUpLeg"), ("LeftArm", "RightArm")]


def add_capsule(scene, a, b, rgba, r=0.008):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r, np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--data", default=str(ROOT / "data" / "g1_lafan1"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--out", default="overlay.mp4")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--mode", choices=("skeleton", "targets"), default="targets",
                    help="targets: each human limb direction drawn from the robot's own joint "
                         "with the robot's limb length (any visible red = real error)")
    args = ap.parse_args()

    q = np.load(Path(args.data) / f"{args.clip}.npz")["qpos"]
    scene = "scene_light_hands.xml" if q.shape[1] == 50 else "scene_light.xml"
    m = mujoco.MjModel.from_xml_path(str(ROOT / "examples" / "unitree_g1" / scene))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(640, args.width), max(480, args.height)
    model_m = mujoco.MjModel.from_xml_path(str(ROOT / "examples" / "unitree_g1" / (
        "g1_with_hands.xml" if q.shape[1] == 50 else "g1.xml")))
    import rigidax as rx
    model = rx.from_mujoco(model_m)
    H = bvh_positions(R.LAFAN / f"{args.clip}.bvh")
    scale = R.leg_scale(model_m, model, H)
    # the same vertical shift the retargeter applies (planted ankles at ankle height)
    feet = {"LeftFoot": "Left", "RightFoot": "Right"}
    pz = np.concatenate([H[f][R.planted(H, s), 2] * scale for f, s in feet.items()])
    dz = (np.median(pz) if len(pz) > 10 else 0.0) - R.ANKLE_HEIGHT
    Hs = {k: v * scale - np.array([0, 0, dz]) for k, v in H.items()}
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, args.height, args.width, max_geom=2000)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.4, 120.0, -8.0
    s0, n = int(args.start * 30), int(args.seconds * 30)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{args.width}x{args.height}", "-r", "30", "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "18", args.out], stdin=subprocess.PIPE)
    for k in range(s0, min(s0 + n, len(q))):
        d.qpos[:] = q[k]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = (d.qpos[0], d.qpos[1], 0.5)
        r.update_scene(d, cam)
        if args.mode == "skeleton":
            for a, b in BONES:  # human skeleton, scaled like the retargeting targets
                add_capsule(r.scene, Hs[a][k], Hs[b][k], (0.9, 0.1, 0.1, 0.9))
        else:
            for body, base, ha, hb, _ in R.DIR_TASKS:  # the targets the solver follows
                pb = d.xpos[m.body(base).id]
                L = np.linalg.norm(d.xpos[m.body(body).id] - pb)
                u = H[hb][k] - H[ha][k]
                add_capsule(r.scene, pb, pb + L * u / np.linalg.norm(u), (0.9, 0.1, 0.1, 0.9), 0.01)
            for f, body in (("LeftFoot", "left_ankle_roll_link"), ("RightFoot", "right_ankle_roll_link")):
                toe = H[f.replace("Foot", "Toe")][k] - H[f][k]
                pa = d.xpos[m.body(body).id]
                add_capsule(r.scene, pa, pa + 0.15 * toe / np.linalg.norm(toe), (0.1, 0.3, 0.9, 0.9), 0.008)
        for f in ("LeftToe", "RightToe"):
            planted = R.planted(H, "Left" if "Left" in f else "Right")[k]
            add_capsule(r.scene, Hs[f][k] - [0, 0, 0.002], Hs[f][k] + [0, 0, 0.002],
                        (0.1, 0.8, 0.1, 1.0) if planted else (0.9, 0.6, 0.0, 1.0), 0.02)
        ff.stdin.write(r.render().tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
