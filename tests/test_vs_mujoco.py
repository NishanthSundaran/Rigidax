"""Every quantity is checked against MuJoCo on random trees and real robots"""

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

import rigidax as rx

MODELS = Path(__file__).parent / "models"
ROBOTS = ["panda.urdf", "g1_29dof_rev_1_0.xml", "a2.xml"]
TOL = dict(rtol=1e-8, atol=1e-8)


def _rand_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def random_mjcf(seed: int, nbody: int = 12, free_root: bool = True) -> str:
    rng = np.random.default_rng(seed)
    f = lambda x: " ".join(f"{v:.6f}" for v in np.ravel(x))  # noqa: E731
    options = [[], ["hinge"], ["slide"], ["ball"], ["hinge", "hinge"],
               ["slide", "hinge"], ["hinge", "ball"]]
    probs = [0.12, 0.38, 0.1, 0.12, 0.1, 0.08, 0.1]
    children = {i: [] for i in range(-1, nbody)}
    for i in range(nbody):
        parent = -1 if i == 0 or (i == 1 and seed % 2) else int(rng.integers(0, i))
        children[parent].append(i)

    def body(i, top):
        joints = ["free"] if (top and free_root and i == 0) else options[rng.choice(len(options), p=probs)]
        s = [f'<body name="b{i}" pos="{f(rng.uniform(-0.3, 0.3, 3))}" quat="{f(_rand_quat(rng))}">']
        s.append(f'<inertial pos="{f(rng.uniform(-0.1, 0.1, 3))}" quat="{f(_rand_quat(rng))}" '
                 f'mass="{rng.uniform(0.5, 3):.4f}" diaginertia="{f(rng.uniform(0.06, 0.1, 3))}"/>')
        for k, jt in enumerate(joints):
            attrs = f'name="j{i}_{k}" type="{jt}" armature="{rng.uniform(0, 0.1):.4f}"'
            if jt in ("hinge", "slide"):
                attrs += f' axis="{f(rng.normal(size=3))}" pos="{f(rng.uniform(-0.1, 0.1, 3))}"'
                attrs += f' ref="{rng.uniform(-0.5, 0.5):.4f}"'
            if jt == "ball":
                attrs += f' pos="{f(rng.uniform(-0.1, 0.1, 3))}"'
            s.append(f"<joint {attrs}/>")
        s.append(f'<site name="s{i}" pos="{f(rng.uniform(-0.2, 0.2, 3))}" quat="{f(_rand_quat(rng))}"/>')
        s += [body(c, False) for c in children[i]]
        s.append("</body>")
        return "\n".join(s)

    g = rng.normal(size=3) * 5
    world = "\n".join(body(c, True) for c in children[-1])
    # a massive body welded to the world, which must not affect dynamics
    world += '<body name="fixed" pos="0 0 1"><inertial pos="0 0 0" mass="5" diaginertia="1 1 1"/></body>'
    return f"""<mujoco><compiler angle="radian"/>
      <option gravity="{f(g)}"><flag constraint="disable" contact="disable"/></option>
      <worldbody>{world}</worldbody></mujoco>"""


def random_state(m, rng):
    q = m.qpos0.copy()
    for j in range(m.njnt):
        a, t = m.jnt_qposadr[j], m.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            q[a : a + 3] = rng.normal(size=3)
            q[a + 3 : a + 7] = _rand_quat(rng)
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            q[a : a + 4] = _rand_quat(rng)
        else:
            q[a] += rng.uniform(-1.5, 1.5)
    return q, rng.normal(size=m.nv), rng.normal(size=m.nv), rng.normal(size=(m.nbody, 6))


def mj_model(name_or_seed):
    if isinstance(name_or_seed, int):
        return mujoco.MjModel.from_xml_string(random_mjcf(name_or_seed))
    path = MODELS / name_or_seed
    m = mujoco.MjModel.from_xml_string(rx.model._strip_meshes(path.read_text(), path.suffix == ".urdf"))
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_CONTACT
    m.dof_damping[:] = 0
    m.jnt_stiffness[:] = 0
    m.dof_frictionloss[:] = 0
    m.actuator_gainprm[:] = 0
    m.actuator_biasprm[:] = 0
    return m


CASES = list(range(8)) + ROBOTS


@pytest.mark.parametrize("case", CASES)
def test_against_mujoco(case):
    m = mj_model(case)
    model = rx.from_mujoco(m)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(123)
    for _ in range(3):
        q, v, tau, xfrc = random_state(m, rng)
        xfrc[0] = 0
        d.qpos, d.qvel, d.qfrc_applied, d.xfrc_applied = q, v, tau, xfrc
        mujoco.mj_forward(m, d)

        kin = jax.jit(rx.forward_kinematics)(model, q)
        np.testing.assert_allclose(kin.body_p, d.xpos, **TOL)
        np.testing.assert_allclose(kin.body_R, d.xmat.reshape(-1, 3, 3), **TOL)

        for s in range(m.nsite):
            R, p = rx.site_pose(model, q, s, kin)
            np.testing.assert_allclose(p, d.site_xpos[s], **TOL)
            np.testing.assert_allclose(R, d.site_xmat[s].reshape(3, 3), **TOL)

        M_ref = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M_ref)
        M = jax.jit(rx.mass_matrix)(model, q)
        np.testing.assert_allclose(M, M_ref, **TOL)
        np.testing.assert_allclose(jax.jit(rx.bias_forces)(model, q, v), d.qfrc_bias, **TOL)

        qacc = jax.jit(rx.forward_dynamics)(model, q, v, tau, xfrc=xfrc)
        np.testing.assert_allclose(qacc, d.qacc, rtol=1e-6, atol=1e-6)

        # inverse dynamics with a random acceleration
        a = rng.normal(size=m.nv)
        d.qacc[:] = a
        mujoco.mj_inverse(m, d)
        np.testing.assert_allclose(jax.jit(rx.rnea)(model, q, v, a), d.qfrc_inverse, **TOL)

        # point Jacobians on every body
        for b in range(1, m.nbody):
            pt = d.xpos[b] + rng.normal(size=3) * 0.1
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, pt, b)
            Jp, Jr = rx.point_jacobian(model, q, b, jnp.asarray(pt), kin)
            np.testing.assert_allclose(Jp, jp, **TOL)
            np.testing.assert_allclose(Jr, jr, **TOL)

        np.testing.assert_allclose(rx.center_of_mass(model, q, kin), d.subtree_com[0], **TOL)
        jc = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jc, 0)
        np.testing.assert_allclose(rx.center_of_mass_jacobian(model, q, kin), jc, **TOL)

        # configuration-space integration / difference
        dt = 0.37
        q_ref = q.copy()
        mujoco.mj_integratePos(m, q_ref, v, dt)
        q_new = rx.integrate(model, q, v, dt)
        np.testing.assert_allclose(q_new, q_ref, **TOL)
        np.testing.assert_allclose(rx.difference(model, q, q_new), v * dt, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("case", [0, 3])
def test_gravity_is_a_parameter(case):
    m = mj_model(case)
    model = rx.from_mujoco(m)
    q, v, _, _ = random_state(m, np.random.default_rng(0))
    f = jax.jit(lambda mdl, q, v: rx.bias_forces(mdl, q, v))
    b1 = f(model, q, v)
    b2 = f(model.replace(gravity=np.array([0.0, 0.0, -1.62])), q, v)  # moon, no recompile
    m.opt.gravity[:] = [0, 0, -1.62]
    d = mujoco.MjData(m)
    d.qpos, d.qvel = q, v
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(b2, d.qfrc_bias, **TOL)
    assert not np.allclose(b1, b2)
    assert f._cache_size() == 1


def test_gradient_wrt_physical_parameters():
    model = rx.from_mujoco(mj_model(2))
    q, v, tau, _ = random_state(mj_model(2), np.random.default_rng(1))

    def loss(mass):
        qacc = rx.forward_dynamics(model.replace(body_mass=mass), q, v, tau)
        return jnp.sum(qacc**2)

    mass = jnp.asarray(model.params.body_mass)
    g = jax.grad(loss)(mass)
    eps = 1e-6
    for b in [1, 4, 7]:
        e = jnp.zeros_like(mass).at[b].set(eps)
        fd = (loss(mass + e) - loss(mass - e)) / (2 * eps)
        np.testing.assert_allclose(g[b], fd, rtol=1e-5, atol=1e-7)


def test_vmap_batch():
    m = mj_model("g1_29dof_rev_1_0.xml")
    model = rx.from_mujoco(m)
    rng = np.random.default_rng(0)
    states = [random_state(m, rng) for _ in range(5)]
    Q = np.stack([s[0] for s in states])
    V = np.stack([s[1] for s in states])
    T = np.stack([s[2] for s in states])
    batched = jax.jit(jax.vmap(rx.forward_dynamics, in_axes=(None, 0, 0, 0)))(model, Q, V, T)
    for i in range(5):
        np.testing.assert_allclose(batched[i], rx.forward_dynamics(model, Q[i], V[i], T[i]), **TOL)


def test_against_pinocchio():
    pin = pytest.importorskip("pinocchio")
    path = str(MODELS / "panda.urdf")
    # This URDF gives two massless links an inertia of 0.1 (physically invalid):
    # Pinocchio keeps it, MuJoCo drops it. Zero it so both see the same robot.
    xml = rx.model._strip_meshes(open(path).read(), True)
    xml = xml.replace('ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"',
                      'ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"')
    pm = pin.buildModelFromXML(xml)
    pd = pm.createData()
    model = rx.from_xml_string(xml)
    rng = np.random.default_rng(0)
    for _ in range(3):
        q, v = rng.uniform(-1, 1, 7), rng.normal(size=7)
        M_pin = pin.crba(pm, pd, q)
        M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T
        np.testing.assert_allclose(rx.mass_matrix(model, q), M_pin, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(rx.bias_forces(model, q, v), pin.nle(pm, pd, q, v), rtol=1e-8, atol=1e-8)
