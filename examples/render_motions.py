"""Render retargeted G1 clips side by side (kinematic playback, frax-style strip).

  MUJOCO_GL=egl python examples/render_motions.py --out examples/g1_lafan1.gif \
      dance1_subject1:DANCE jumps1_subject1:JUMPS ...

Each clip plays its most active window of --seconds (or clip@start_s:LABEL to choose).
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "g1_lafan1_hands"
SCENE = ROOT / "examples" / "unitree_g1" / "scene_light.xml"
SCENE_HANDS = ROOT / "examples" / "unitree_g1" / "scene_light_hands.xml"  # G1 + Dex3 (nq 50)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def active_window(q, fps, seconds):
    """Start frame of the window with the most joint + root motion"""
    n = int(seconds * fps)
    if len(q) <= n:
        return 0
    speed = np.linalg.norm(np.diff(q[:, 7:], axis=0), axis=1) + 3 * np.linalg.norm(
        np.diff(q[:, :3], axis=0), axis=1)
    c = np.concatenate([[0], np.cumsum(speed)])
    return int(np.argmax(c[n:] - c[:-n]))


def render_clip(m, q, start, n, W, H, path):
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.7, 120.0, -6.0
    look = q[start, :2].copy()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
                          stdin=subprocess.PIPE)
    for k in range(start, start + n):
        d.qpos[:] = q[min(k, len(q) - 1)]
        mujoco.mj_forward(m, d)
        look += 0.35 * (d.qpos[:2] - look)  # smooth follow camera
        cam.lookat[:] = (look[0], look[1], 0.5)
        r.update_scene(d, cam)
        ff.stdin.write(r.render().tobytes())
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", help="clip[@start_s][:LABEL]")
    ap.add_argument("--out", default="g1_lafan1.gif")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--gif-width", type=int, default=1440)
    ap.add_argument("--data", default=str(DATA), help="folder of retargeted clips")
    args = ap.parse_args()

    data = Path(args.data)
    first = np.load(data / f"{args.clips[0].partition(':')[0].partition('@')[0]}.npz")["qpos"]
    m = mujoco.MjModel.from_xml_path(str(SCENE_HANDS if first.shape[1] == 50 else SCENE))
    m.vis.global_.offwidth = max(640, args.width)
    m.vis.global_.offheight = max(480, args.height)
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "data"))
    parts, labels = [], []
    for spec in args.clips:
        spec, _, label = spec.partition(":")
        clip, _, start_s = spec.partition("@")
        z = np.load(data / f"{clip}.npz")
        q, fps = z["qpos"], int(z["fps"])
        n = int(args.seconds * fps)
        start = int(float(start_s) * fps) if start_s else active_window(q, fps, args.seconds)
        path = tmp / f"{len(parts)}.mp4"
        render_clip(m, q, start, n, args.width, args.height, path)
        parts.append(path)
        labels.append(label)
        print(f"{clip}: frames {start}-{start + n} ({start / fps:.1f}s)")

    inputs = sum([["-i", str(p)] for p in parts], [])
    fl = "".join(
        f"[{i}:v]drawtext=fontfile={FONT}:text='{lab}':fontcolor=0x666666:fontsize=20:"
        f"x=(w-tw)/2:y=h-36[v{i}];" if lab else f"[{i}:v]null[v{i}];"
        for i, lab in enumerate(labels))
    stack = ("".join(f"[v{i}]" for i in range(len(parts))) + f"hstack=inputs={len(parts)}[s]"
             if len(parts) > 1 else "[v0]null[s]")
    mp4 = Path(args.out).with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", fl + stack,
                    "-map", "[s]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    f"fps=15,scale={args.gif_width}:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=max_colors=96:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    str(Path(args.out).with_suffix(".gif"))], check=True)
    for p in parts:
        p.unlink()
    tmp.rmdir()
    print(f"wrote {mp4} and {Path(args.out).with_suffix('.gif')}")


if __name__ == "__main__":
    main()
