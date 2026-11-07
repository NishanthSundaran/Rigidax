"""Unitree G1 whole-body inverse dynamics; every dynamics term from rigidax.

The free-floating G1 squats, waves both arms and twists its waist while balancing on
its own feet. MuJoCo simulates contact and renders, and its actuators are switched to
pure torque motors. Each control step, rigidax (jitted) computes:

  * mass matrix M and bias forces h for all 35 DOFs (6 floating base + 29 joints)
  * 12x35 foot contact Jacobians Jc, and Jc_dot v by autodiff through the kinematics
  * contact-consistent reference accelerations (KKT: feet stay planted exactly)
  * contact wrenches that balance the unactuated base rows: Jcb^T f = M_b a + h_b
  * feedforward torques tau_ff = M_a a + h_a - Jca^T f, plus joint PD feedback

  MUJOCO_GL=egl python examples/g1_demo.py [--out g1_demo.mp4] [--seconds 12]
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

SCENES = {"dark": Path(__file__).parent / "unitree_g1" / "scene.xml",
          "light": Path(__file__).parent / "unitree_g1" / "scene_light.xml"}  # frax-like studio
XML = SCENES["dark"]
FEET = ("left_foot", "right_foot")
HANDS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
# Foot drift stabilization inside the plan. Off: with MuJoCo's soft contacts the
# feet comply slightly, and damping that motion in the plan destabilizes the loop.
KP_C, KD_C = 0.0, 0.0
CROUCH = 0.2  # initial hip/knee/ankle bend


MOTIONS = ("squat", "twist", "wave")


def reference(names, motions=MOTIONS):
    """Per-joint q = A (1 - cos(w t)) / 2 + B sin(w t): starts at rest.
    motions: any of "squat" (legs + waist pitch), "twist" (waist yaw), "wave" (arms)"""
    n = len(names)
    A, B, w = np.zeros(n), np.zeros(n), np.ones(n)
    i = {nm: k for k, nm in enumerate(names)}
    ws = 2 * np.pi / 3.0  # squat period 3 s
    th = 0.45  # extra squat depth (hip -th, knee 2 th, ankle -th keeps feet flat)
    for side in ("left", "right"):
        if "squat" not in motions:
            break
        A[i[f"{side}_hip_pitch_joint"]], w[i[f"{side}_hip_pitch_joint"]] = -th, ws
        A[i[f"{side}_knee_joint"]], w[i[f"{side}_knee_joint"]] = 2 * th, ws
        A[i[f"{side}_ankle_pitch_joint"]], w[i[f"{side}_ankle_pitch_joint"]] = -th, ws
    for side in ("left", "right"):
        if "wave" not in motions:
            break
        s = 1 if side == "left" else -1
        # arms out to the side and waving at the elbow
        A[i[f"{side}_shoulder_roll_joint"]], w[i[f"{side}_shoulder_roll_joint"]] = s * 1.2, ws / 2
        A[i[f"{side}_shoulder_pitch_joint"]], w[i[f"{side}_shoulder_pitch_joint"]] = -0.5, ws / 2
        A[i[f"{side}_elbow_joint"]], w[i[f"{side}_elbow_joint"]] = 1.2, 2 * ws
        A[i[f"{side}_wrist_roll_joint"]], w[i[f"{side}_wrist_roll_joint"]] = s * 0.8, 2 * ws
    if "squat" in motions:
        A[i["waist_pitch_joint"]], w[i["waist_pitch_joint"]] = 0.25, ws  # lean into the squat
    if "twist" in motions:
        B[i["waist_yaw_joint"]], w[i["waist_yaw_joint"]] = 0.45, ws / 2
    return A, B, w


def make_controller(model, m, q0, motions=MOTIONS):
    feet = [model.site_id(f) for f in FEET]
    kin0 = rx.forward_kinematics(model, q0)
    foot_p0 = [np.asarray(rx.site_pose(model, q0, s, kin0)[1]) for s in feet]
    foot_R0 = [np.asarray(rx.site_pose(model, q0, s, kin0)[0]) for s in feet]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    A, B, w = reference(names, motions)
    qa0 = q0[7:]
    # Joint PD gains (torque level): legs stiff, waist medium, arms soft
    KP_J = np.array([200.0 if "hip" in n or "knee" in n or "ankle" in n else
                     120.0 if "waist" in n else 40.0 for n in names])
    KD_J = np.array([5.0 if k >= 120 else 1.5 for k in KP_J])

    def foot_jac(q, kin):
        return jnp.concatenate([jnp.concatenate(rx.site_jacobian(model, q, s, kin)) for s in feet])

    def foot_err(kin, q):
        errs = []
        for s, p0, R0 in zip(feet, foot_p0, foot_R0):
            R, p = rx.site_pose(model, q, s, kin)
            errs += [p0 - p, 0.5 * jnp.sum(jnp.cross(R.T, R0.T), axis=0)]
        return jnp.concatenate(errs)

    @jax.jit
    def control(q, v, t):
        kin = rx.forward_kinematics(model, q)
        M, h = rx.dynamics_terms(model, q, v, kin=kin)
        Jc = foot_jac(q, kin)

        # Jc_dot v = d/de [Jc(q (+) e v) v] at e = 0, by forward-mode autodiff
        _, Jdv = jax.jvp(lambda e: foot_jac(rx.integrate(model, q, v, e), None) @ v,
                         (0.0,), (1.0,))

        # Reference motion (analytic): joint positions, velocities, accelerations
        c, s = jnp.cos(w * t), jnp.sin(w * t)
        q_ref = qa0 + A * (1 - c) / 2 + B * s
        qd_ref = A * w * s / 2 + B * w * c
        qdd_ref = A * w**2 * c / 2 - B * w**2 * s

        # Feedforward: all 35 accelerations, tracking qdd_ref while keeping the feet
        # planted exactly (equality-constrained least squares via KKT):
        #   min ||a_a - qdd_ref||^2 + eps ||a_b||^2   s.t.  Jc a = -Jc_dot v + stabilization
        rhs = -Jdv - KD_C * (Jc @ v) + KP_C * foot_err(kin, q)
        nv, nc = model.nv, Jc.shape[0]
        Wd = jnp.concatenate([jnp.full(6, 1e-3), jnp.ones(nv - 6)])
        K = jnp.block([[jnp.diag(Wd), Jc.T], [Jc, -1e-9 * jnp.eye(nc)]])
        a_des = jnp.concatenate([jnp.zeros(6), qdd_ref])
        a = jnp.linalg.solve(K, jnp.concatenate([Wd * a_des, rhs]))[:nv]

        # Contact wrenches balance the (unactuated) base rows, then joint torques:
        # rigidax inverse dynamics as feedforward + joint PD as feedback
        f = jnp.linalg.pinv(Jc[:, :6].T) @ (M[:6] @ a + h[:6])
        tau_ff = M[6:] @ a + h[6:] - Jc[:, 6:].T @ f
        tau = tau_ff + KP_J * (q_ref - q[7:]) + KD_J * (qd_ref - v[6:])
        return tau, f

    return control


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
    ap.add_argument("--out", default="g1_demo.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--motion", default="all", help="all, or a comma list of " + ", ".join(MOTIONS))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--style", choices=SCENES, default="dark",
                    help="light: plain grey studio, side-on camera, no trails")
    args = ap.parse_args()
    motions = MOTIONS if args.motion == "all" else tuple(args.motion.split(","))
    assert set(motions) <= set(MOTIONS), f"unknown motion in {motions}"

    m = mujoco.MjModel.from_xml_path(str(SCENES[args.style]))
    # Position servos -> pure torque motors: force = ctrl (still clipped to force limits)
    m.actuator_gainprm[:, :] = 0
    m.actuator_gainprm[:, 0] = 1
    m.actuator_biasprm[:, :] = 0
    m.actuator_ctrllimited[:] = 0
    m.vis.global_.offwidth = max(args.width, 640)
    m.vis.global_.offheight = max(args.height, 480)
    d = mujoco.MjData(m)

    # Start in a slight crouch with the feet just touching the floor
    q0 = m.qpos0.copy()
    for side in ("left", "right"):
        for jn, val in (("hip_pitch", -CROUCH), ("knee", 2 * CROUCH), ("ankle_pitch", -CROUCH)):
            q0[m.jnt_qposadr[m.joint(f"{side}_{jn}_joint").id]] = val
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    q0[2] -= min(d.site_xpos[m.site(f).id][2] for f in FEET) - 0.03
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)

    model = rx.from_mujoco(m)
    control = make_controller(model, m, q0, motions)
    act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] - 6 for i in range(m.nu)])
    damping = m.dof_damping[6:]

    W, H = args.width, args.height
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, 0.0, 0.6)
    cam.distance, cam.azimuth, cam.elevation = 1.9 * max(1.0, H / W) ** 0.5, 150.0, -10.0
    if args.style == "light":
        cam.lookat[:] = (0.0, 0.0, 0.5)
        cam.distance, cam.azimuth, cam.elevation = 2.6, 120.0, -6.0
    renderer = mujoco.Renderer(m, H, W, max_geom=4000)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", args.out], stdin=subprocess.PIPE)

    steps = int(args.seconds / m.opt.timestep)
    every = int(round(1 / (args.fps * m.opt.timestep)))
    trails = [deque(maxlen=90) for _ in HANDS]
    hand_ids = [m.body(b).id for b in HANDS]
    times, min_fz = [], np.inf
    foot0 = np.array([d.site_xpos[m.site(f).id].copy() for f in FEET])
    control(d.qpos, d.qvel, 0.0)[0].block_until_ready()  # compile

    for k in range(steps):
        t0 = time.perf_counter()
        tau, f = control(d.qpos, d.qvel, k * m.opt.timestep)
        tau = np.asarray(tau)
        times.append(time.perf_counter() - t0)
        min_fz = min(min_fz, float(f[2]), float(f[8]))
        d.ctrl[:] = tau[act_dof] + damping[act_dof] * d.qvel[6:][act_dof]
        mujoco.mj_step(m, d)
        if d.qpos[2] < 0.35:
            print(f"fell at t={k * m.opt.timestep:.2f}s")
            break
        if k % every == 0:
            for tr, b in zip(trails, hand_ids):
                tr.append(d.xpos[b].copy())
            renderer.update_scene(d, cam)
            if args.style == "dark":
                add_trail(renderer.scene, trails[0], (0.2, 1.0, 0.4, 0.9), 0.008)
                add_trail(renderer.scene, trails[1], (0.3, 0.6, 1.0, 0.9), 0.008)
            ffmpeg.stdin.write(renderer.render().tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    slip = max(np.linalg.norm(d.site_xpos[m.site(f).id][:2] - foot0[i][:2]) for i, f in enumerate(FEET))
    print(f"wrote {args.out}")
    print(f"controller (rigidax: FK, M, h, foot Jacobians + Jdot*v, contact wrench, torques; jitted): "
          f"median {np.median(times) * 1e6:.1f} us/step incl. Python dispatch")
    print(f"final foot slip {slip * 1000:.1f} mm, min planned foot normal force {min_fz:.1f} N, "
          f"pelvis height now {d.qpos[2]:.3f} m")


if __name__ == "__main__":
    main()
