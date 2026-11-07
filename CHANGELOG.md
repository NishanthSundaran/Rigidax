# Changelog

## 1.0.0 (2026-11-07)

First stable release.

### Core (`rigidax`)
- Differentiable rigid-body kinematics and dynamics in JAX for any MuJoCo-compiled model
  (MJCF or URDF): forward kinematics, frames, point/site/body Jacobians, center of mass
  and its Jacobian, inverse dynamics (RNEA), mass matrix (CRBA), bias / gravity /
  Coriolis forces, forward dynamics, quaternion-aware `integrate` / `difference`.
- Hinge, slide, ball and free joints, several joints per body, armature, configurable
  gravity. Physical parameters are pytree leaves, so gradients and `vmap` over masses,
  inertias, offsets and gravity work, with no recompilation when they change.
- Validated against MuJoCo (to 1e-8) on random trees and the G1, A2 and Panda models,
  and against Pinocchio.

### IK and retargeting (`rigidax.ik`)
- Fixed-iteration ADMM QP solver (`solve_qp`) for batches of small dense problems:
  equilibration, explicit inverse plus mat-vec iterations, warm starts, elastic (L1,
  always-feasible) rows, optional polishing.
- `IKSolver` with Gauss-Newton/QP steps and these pieces:
  - hard joint limits, with optional narrowed retargeting ranges;
  - a per-step trust region;
  - an exact floor guarantee for floating bases;
  - elastic self-collision on an automatic sphere model (`SphereModel.fit`).
- Tasks: `position`, `orientation`, `direction` (limb directions with the robot's own
  segment lengths), `twist` (rotation about a body axis only), `height`, `joints`
  (per-frame joint targets), and `PostureTask`. Every task accepts per-frame weight scales.
- `FloorPoints`: points or whole-body spheres kept above the floor, using only the
  lowest k rows per step.
- `solve_trajectory`: many clips at once, split into overlapping segments, parallel
  across devices, multi-start seeds, crossfaded seams.
- `smooth_trajectory`: zero-phase, quaternion-aware smoothing. `ik.bvh`: BVH reader.

### Examples
- LAFAN1 → Unitree G1 retargeting (`examples/retarget_lafan1.py`), with two outputs:
  - G1 + Dex3 hands (the default). The fingers are synthesized, since LAFAN1 has no
    finger data.
  - Plain G1 (`--no-hands`).
- Diagnostics: human/target overlays, self-collision and floor penetration reports.
- Control demos: Panda operational space control, G1 whole-body balance.
- Benchmarks against frax (`bench/`).
