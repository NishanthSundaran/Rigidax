"""Run frax's own LAFAN1 -> G1 retargeting (examples/ik/lafan1, unchanged settings) on
the same clips and machine, time it, and save results in MuJoCo G1 qpos layout so
rigidax and frax are scored with identical metrics.

Run in a venv with frax[ik] (jax >= 0.9.1):
  .venv-fraxik/bin/python bench/bench_frax_ik.py --frax-repo <frax clone> clip1 clip2 ...
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED")
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import jax  # noqa: E402

jax.config.update("jax_num_cpu_devices", os.cpu_count())  # frax's setting: one per core

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--frax-repo", required=True)
    ap.add_argument("--self-collision", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.frax_repo) / "examples" / "ik" / "lafan1"))
    from lafan1_g1 import LAFAN1_FPS, make_retargeter  # noqa: E402
    from retarget import load_clip  # noqa: E402

    from frax import load_g1  # noqa: E402
    from frax.ik import OfflineRetargeter  # noqa: E402

    clips = [load_clip(ROOT / "data" / "lafan1" / f"{c}.bvh") for c in args.clips]
    robot = load_g1()
    retarget = OfflineRetargeter(make_retargeter(robot, self_collision=args.self_collision),
                                 1.0 / LAFAN1_FPS)
    total = sum(map(len, clips))
    t0 = time.perf_counter()
    qs = retarget(clips)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    qs = retarget(clips)  # compiled: steady state
    dt = time.perf_counter() - t0
    tag = "with self-collision" if args.self_collision else "no self-collision"
    print(f"frax ({tag}): {len(clips)} clips, {total} frames on "
          f"{min(len(clips), len(retarget.devices))} of {len(retarget.devices)} devices: "
          f"{total / dt:,.0f} frames/s (first call {t_first:.1f}s)")

    # frax q = [pos, wxyz quat, actuated joints in robot.joint_names order] -> MuJoCo G1
    m = mujoco.MjModel.from_xml_path(str(ROOT / "examples" / "unitree_g1" / "g1.xml"))
    names = [n for n in robot.joint_names][robot.nv_floating:]
    out = ROOT / "data" / ("frax_lafan1_sc" if args.self_collision else "frax_lafan1")
    out.mkdir(parents=True, exist_ok=True)
    for clip, q in zip(args.clips, qs):
        q = np.asarray(q)
        Q = np.tile(m.qpos0, (len(q), 1))
        Q[:, :7] = q[:, :7]
        for k, n in enumerate(names):
            Q[:, m.jnt_qposadr[m.joint(n).id]] = q[:, 7 + k]
        np.savez(out / f"{clip}.npz", qpos=Q, fps=LAFAN1_FPS)


if __name__ == "__main__":
    main()
