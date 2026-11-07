"""Operational space control of a Franka Panda; all dynamics terms from rigidax.

MuJoCo only simulates and renders. Each control step, rigidax computes (jitted):
forward kinematics, the end-effector Jacobian, the mass matrix and the bias forces.

  MUJOCO_GL=egl python examples/osc_demo.py [--out osc_demo.mp4] [--seconds 12]
"""

import argparse
import subprocess
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

import rigidax as rx

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

XML = Path(__file__).parent / "xml" / "demo_scene.xml"
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
TCP = np.array([0.0, 0.0, 0.1034])  # tool point in the hand frame
NA = 7  # arm joints

KP, KD = 400.0, 40.0  # position
KP_R, KD_R = 250.0, 30.0  # orientation
KP_N, KD_N = 25.0, 10.0  # null-space posture


def make_controller(model, damping):
    hand = model.body_id("hand")

    @jax.jit
    def control(q, v, p_des, v_des, a_des, R_des):
        kin = rx.forward_kinematics(model, q)
        M, bias = rx.dynamics_terms(model, q, v, kin=kin)
        R, p_h = kin.body_R[hand], kin.body_p[hand]
        p = p_h + R @ TCP
        Jp, Jr = rx.point_jacobian(model, q, hand, p, kin)
        J = jnp.concatenate([Jp, Jr])[:, :NA]
        Ma, va = M[:NA, :NA], v[:NA]
        Minv = jnp.linalg.inv(Ma)
        Lam = jnp.linalg.inv(J @ Minv @ J.T + 1e-6 * jnp.eye(6))

        twist = J @ va
        e_r = 0.5 * jnp.sum(jnp.cross(R.T, R_des.T), axis=0)  # orientation error
        acc = jnp.concatenate([a_des + KP * (p_des - p) + KD * (v_des - twist[:3]),
                               KP_R * e_r - KD_R * twist[3:]])
        tau_task = J.T @ (Lam @ acc)

        Jbar = Minv @ J.T @ Lam  # dynamically consistent null space
        N = jnp.eye(NA) - J.T @ Jbar.T
        tau_null = N @ (Ma @ (KP_N * (Q_HOME[:NA] - q[:NA]) - KD_N * va))
        tau = tau_task + tau_null + bias[:NA] + damping * va
        return tau, p

    return control


def target(t):
    """Figure-eight in the y-z plane: position, velocity, acceleration"""
    w = 2 * np.pi / 4.0
    c = np.array([0.5, 0.0, 0.45])
    p = c + np.array([0.0, 0.22 * np.sin(w * t), 0.14 * np.sin(2 * w * t)])
    v = np.array([0.0, 0.22 * w * np.cos(w * t), 0.28 * w * np.cos(2 * w * t)])
    a = np.array([0.0, -0.22 * w**2 * np.sin(w * t), -0.56 * w**2 * np.sin(2 * w * t)])
    return p, v, a


def add_trail(scene, points, rgba, size):
    for i, pt in enumerate(points):
        if scene.ngeom >= scene.maxgeom:
            return
        alpha = rgba[3] * (i + 1) / len(points)
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([size, 0, 0]), pt, np.eye(3).ravel(),
                            np.array([*rgba[:3], alpha], np.float32))
        scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="osc_demo.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    model = rx.from_mujoco(m)
    control = make_controller(model, np.asarray(m.dof_damping[:NA]))
    target_mocap = m.body_mocapid[m.body("target").id]

    d.qpos[:] = Q_HOME
    mujoco.mj_forward(m, d)
    hand = m.body("hand").id
    R_des = d.xmat[hand].reshape(3, 3).copy()  # keep the home orientation (pointing down)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.35, 0.0, 0.4)
    cam.distance, cam.azimuth, cam.elevation = 1.6, 135.0, -18.0
    W, H = 1280, 720
    renderer = mujoco.Renderer(m, H, W, max_geom=4000)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", args.out], stdin=subprocess.PIPE)

    steps = int(args.seconds / m.opt.timestep)
    every = int(round(1 / (args.fps * m.opt.timestep)))
    ee_trail, tgt_trail = deque(maxlen=120), deque(maxlen=120)
    times, errs = [], []
    control(d.qpos, d.qvel, *target(0.0), R_des)[0].block_until_ready()  # compile

    for k in range(steps):
        t = k * m.opt.timestep
        p_des, v_des, a_des = target(t)
        d.mocap_pos[target_mocap] = p_des
        t0 = time.perf_counter()
        tau, p = control(d.qpos, d.qvel, p_des, v_des, a_des, R_des)
        tau = np.asarray(tau)
        times.append(time.perf_counter() - t0)
        d.ctrl[:NA] = tau
        d.ctrl[NA] = 255.0  # gripper open
        mujoco.mj_step(m, d)
        if t > 1.0:
            errs.append(np.linalg.norm(np.asarray(p) - p_des))
        if k % every == 0:
            ee_trail.append(np.asarray(p).copy())
            tgt_trail.append(p_des.copy())
            renderer.update_scene(d, cam)
            add_trail(renderer.scene, tgt_trail, (1.0, 0.3, 0.3, 0.5), 0.004)
            add_trail(renderer.scene, ee_trail, (0.2, 1.0, 0.4, 0.9), 0.006)
            ffmpeg.stdin.write(renderer.render().tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"wrote {args.out}: {args.seconds:.0f}s @ {args.fps} fps")
    print(f"controller (rigidax FK + Jacobian + M + bias + OSC, jitted): "
          f"median {np.median(times) * 1e6:.1f} us/step incl. Python dispatch")
    print(f"tracking error after 1s: mean {np.mean(errs) * 1000:.2f} mm, max {np.max(errs) * 1000:.2f} mm")


if __name__ == "__main__":
    main()
