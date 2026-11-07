# Third-party notices

rigidax itself is MIT licensed (see `LICENSE`). This repository also contains, or is
derived from, the following third-party material. Their licences apply to those files.

## frax (MIT)

https://github.com/StanfordASL/frax

rigidax's dynamics formulation (dense, loop-free dynamics over a DOF ancestor mask) follows
frax. These files are copied from frax:

- `tests/models/panda.urdf`, `tests/models/g1_29dof_rev_1_0.xml`, `tests/models/a2.xml`
- `examples/xml/panda.xml`, `examples/xml/scene.xml`, `examples/xml/obstacle_scene.xml`
  and `examples/xml/assets/` (the Panda model is in turn from MuJoCo Menagerie; see below)

```
MIT License

Copyright (c) 2026 Daniel Morton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MuJoCo Menagerie: Franka Emika Panda (Apache-2.0)

https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda

The Panda MJCF model and meshes in `examples/xml/` originate from MuJoCo Menagerie (via
frax). They are licensed under the Apache License, Version 2.0:
https://www.apache.org/licenses/LICENSE-2.0

## MuJoCo Menagerie: Unitree G1 (BSD-3-Clause)

https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1

`examples/unitree_g1/` (G1 and G1 with Dex3 hands, meshes, scenes) is from MuJoCo
Menagerie. Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics").
The full licence is in `examples/unitree_g1/LICENSE`. The extra scene files
(`scene_light*.xml`, `scene_hands.xml`) are rigidax additions.

## GMR: General Motion Retargeting

https://github.com/YanjieZe/GMR

`examples/retarget_lafan1.py` uses GMR's LAFAN1-to-G1 link-frame offsets and its
narrowed G1 arm ranges (`g1_mocap_29dof`), which frax also uses.

## LAFAN1 (CC BY-NC-ND 4.0)

https://github.com/ubisoft/ubisoft-laforge-animation-dataset

The motion data is not included in this repository. The example scripts download it.
The demo GIFs and videos are renderings of retargeted LAFAN1 motions. They are for
non-commercial use only.
