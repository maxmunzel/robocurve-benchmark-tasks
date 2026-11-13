"""On the station table: a round robot vacuum (34 cm across, 7.4 cm tall, 3 kg) with its status ring lit red and a
start button in the middle of its lid, and a TV remote control lying flat, partly under the vacuum's front edge and
sticking out to its right. The vacuum sits tilted.

Instruction: "The robot vacuum has stopped. Get it cleaning again."

VACUUM_DEBUG=N prints the state every N control steps.

Run the scripted demo:  inspect-robots run --embodiment yam_arena --policy yam_scripted \\
    -E task=tasks/robot_vacuum/robot_vacuum.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 \\
    -E cam_height=360 --instruction "..." --scorer task_score --max-steps 1800 --store-frames
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from yam_arena import demo as _demo
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

# Load yampatch.py from this folder (task files are imported by path, so their folder is not on sys.path).
if "yampatch" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("yampatch", Path(__file__).with_name("yampatch.py"))
    sys.modules["yampatch"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["yampatch"])
yampatch = sys.modules["yampatch"]

SCENE = "station"
TOP = TABLE_TOP[SCENE]
_DEBUG = int(os.environ.get("VACUUM_DEBUG", "0") or 0)


def _quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def _matrix_quat(m):
    """Rotation matrix -> quaternion (x, y, z, w), by the largest-component (Shepperd) method."""
    m = np.asarray(m, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        k = 2 * math.sqrt(1 + tr)
        q = ((m[2, 1] - m[1, 2]) / k, (m[0, 2] - m[2, 0]) / k, (m[1, 0] - m[0, 1]) / k, k / 4)
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        k = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = (k / 4, (m[0, 1] + m[1, 0]) / k, (m[0, 2] + m[2, 0]) / k, (m[2, 1] - m[1, 2]) / k)
    elif m[1, 1] >= m[2, 2]:
        k = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 1] + m[1, 0]) / k, k / 4, (m[1, 2] + m[2, 1]) / k, (m[0, 2] - m[2, 0]) / k)
    else:
        k = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[0, 2] + m[2, 0]) / k, (m[1, 2] + m[2, 1]) / k, k / 4, (m[1, 0] - m[0, 1]) / k)
    return tuple(float(v) for v in q)


def _rot(axis, a):
    """Rotation matrix about a unit axis (Rodrigues)."""
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * kx @ kx


def _rz(a):
    return _rot((0, 0, 1), a)


def _ry(a):
    return _rot((0, 1, 0), a)


# -- the vacuum -----------------------------------------------------------------------------------------------------
# Its own frame: origin at the centre of its underside, z up. A chassis cylinder (the collider), a black lid, a status
# ring and a start button on a sprung prismatic joint (0 up, negative pressed).
VAC_R, VAC_H = 0.17, 0.07  # chassis radius and height
LID_H = 0.004  # black lid on top (visual), so the vacuum is 7.4 cm tall
VAC_MASS = 3.0
VAC_FRICTION = 0.5  # plastic on the table
BTN_R, BTN_H, BTN_TRAVEL = 0.018, 0.008, 0.004  # the button stands 4 mm proud of the lid and travels 4 mm
BTN_K, BTN_PRELOAD = 600.0, 0.001  # N/m; the drive's target is 1 mm above the top stop: 0.6 N at rest, 3 N pressed
PRESSED_Q = -0.003  # the button counts as pressed once 3 mm down
RELEASED_Q = -0.001  # and released once back within 1 mm of the top
RING_R = 0.026  # status ring around the button
BTN_TOP = VAC_H + LID_H + BTN_H / 2  # button top above the underside, at rest: 7.8 cm
VAC_XY = (0.37, 0.0)  # where it stands (its centre)

# -- the remote ------------------------------------------------------------------------------------------------------
# Arena's library remote control (a scan, 16.4 x 3.6 x 2.5 cm, 0.2 kg, collider a convex decomposition). In its own
# frame its length is along z and its buttons face -y; laid flat, buttons up, it is rotated by REMOTE_Q0, then yawed.
REMOTE_ASSET = "remote_control_hot3d_robolab"
REMOTE_LO, REMOTE_HI = np.array([-0.0182, -0.0377, -0.0819]), np.array([0.0174, -0.0124, 0.0824])
REMOTE_MASS = 0.2
REMOTE_U = np.array([0.0, -1.0])  # its long axis on the table (along y), towards the end that sticks out (right)
REMOTE_D = np.array([-0.155, -0.06])  # its centre from the vacuum's centre: under the front edge, 7 cm sticking out
_R0 = _quat_matrix((0.0, math.sqrt(0.5), -math.sqrt(0.5), 0.0))  # native z -> -y, native -y (buttons) -> up
REMOTE_YAW = math.atan2(REMOTE_U[0], -REMOTE_U[1])  # turns -y onto REMOTE_U
REMOTE_R = _rz(REMOTE_YAW) @ _R0


def _remote_corners(pos, rmat):
    """The remote's bounding box corners in the world."""
    lo, hi = REMOTE_LO, REMOTE_HI
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return c @ np.asarray(rmat).T + np.asarray(pos)


_rc = REMOTE_R @ ((REMOTE_LO + REMOTE_HI) / 2)
REMOTE_POS = (float(VAC_XY[0] + REMOTE_D[0] - _rc[0]), float(VAC_XY[1] + REMOTE_D[1] - _rc[1]),
              float(TOP + 0.0005 - _remote_corners((0, 0, 0), REMOTE_R)[:, 2].min()))  # fmt: skip
REMOTE_THICK = float(np.ptp(_remote_corners((0, 0, 0), REMOTE_R)[:, 2]))  # 2.5 cm


# The vacuum's start pose: resting tilted on the remote, its far edge on the table. Of all tilt directions it takes
# the one needing the least tilt to clear the remote's top (where it would settle); it settles in the first moments.
def _start_pose():
    u = np.array([*REMOTE_U, 0.0])
    w = np.array([-REMOTE_U[1], REMOTE_U[0], 0.0])
    c = np.array([*REMOTE_D, 0.0])
    half_l, half_w = 0.082, 0.017
    pts = [c + u * a + w * b for a in np.linspace(-half_l, half_l, 33) for b in np.linspace(-half_w, half_w, 5)]
    pts = [p[:2] for p in pts if np.linalg.norm(p[:2]) < VAC_R]
    best = None
    for k in range(360):
        n = np.array([math.cos(math.radians(k)), math.sin(math.radians(k))])
        need = max(math.asin(min(1.0, (REMOTE_THICK + 0.005) / (VAC_R + float(p @ n)))) for p in pts)
        if best is None or need < best[0]:
            best = (need, n)
    a, n = best
    axis = np.array([n[1], -n[0], 0.0])
    rm = _rot(axis, a)
    if (rm @ np.array([n[0] * VAC_R, n[1] * VAC_R, 0.0]))[2] < 0:
        rm = _rot(axis, -a)
    back = rm @ np.array([-n[0] * VAC_R, -n[1] * VAC_R, 0.0])
    pos = (float(VAC_XY[0]), float(VAC_XY[1]), float(TOP + 0.001 - back[2]))
    return pos, tuple(float(v) for v in _matrix_quat(rm)), math.degrees(a), n


VAC_START_POS, VAC_START_Q, VAC_START_TILT, VAC_TILT_DIR = _start_pose()


# -- generated USD ----------------------------------------------------------------------------------------------------
def _write_usd(name: str, build) -> str:
    from pxr import Usd, UsdGeom, UsdPhysics

    path = Path(tempfile.gettempdir()) / "yam_tasks" / f"{name}.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    build(stage, f"/{name}")
    stage.GetRootLayer().Export(str(path))
    return str(path)


def _material(stage, path, mu):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    api.CreateRestitutionAttr(0.0)
    return mat


def _mesh_cyl(stage, path, pos, r, h, color, n=96):
    """A smooth z-axis cylinder for rendering only (the renderer draws UsdGeom.Cylinder with few facets)."""
    from pxr import Gf, UsdGeom

    m = UsdGeom.Mesh.Define(stage, path)
    pts, counts, idx = [], [], []
    for k in range(n):
        a = 2 * math.pi * k / n
        pts += [Gf.Vec3f(r * math.cos(a), r * math.sin(a), -h / 2), Gf.Vec3f(r * math.cos(a), r * math.sin(a), h / 2)]
    for k in range(n):
        j = (k + 1) % n
        counts.append(4)
        idx += [2 * k, 2 * j, 2 * j + 1, 2 * k + 1]
    counts += [n, n]
    idx += [2 * k + 1 for k in range(n)] + [2 * k for k in reversed(range(n))]
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateSubdivisionSchemeAttr("none")
    m.AddTranslateOp().Set(Gf.Vec3d(*pos))
    m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return m


def _cyl(stage, path, pos, r, h, color, *, mat=None, collide=True):
    """A z-axis cylinder: drawn as a smooth mesh; with ``collide``, also a (hidden) cylinder collider."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    if not collide:
        return _mesh_cyl(stage, path, pos, r, h, color)
    _mesh_cyl(stage, f"{path}_look", pos, r, h, color)
    g = UsdGeom.Cylinder.Define(stage, path)
    g.CreateAxisAttr("Z")
    g.CreateRadiusAttr(float(r))
    g.CreateHeightAttr(float(h))
    g.AddTranslateOp().Set(Gf.Vec3d(*pos))
    g.CreatePurposeAttr("guide")
    if collide:
        prim = g.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        coll = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        coll.CreateContactOffsetAttr(0.002)
        coll.CreateRestOffsetAttr(0.0)
        if mat is not None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return g


def _build_vacuum():
    """A floating articulation: the chassis (root link) and the start button on a prismatic joint ``button``."""

    def build(stage, root):
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root))
        PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root)).CreateEnabledSelfCollisionsAttr(False)
        shell = _material(stage, f"{root}/shell", VAC_FRICTION)
        body = UsdGeom.Xform.Define(stage, f"{root}/body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(VAC_MASS)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.03))
        b = f"{root}/body"
        _cyl(stage, f"{b}/chassis", (0, 0, VAC_H / 2), VAC_R, VAC_H, (0.24, 0.25, 0.27), mat=shell)
        _cyl(stage, f"{b}/bumper", (0, 0, 0.03), VAC_R + 0.002, 0.03, (0.12, 0.12, 0.13), collide=False)
        _cyl(stage, f"{b}/lid", (0, 0, VAC_H + LID_H / 2), VAC_R - 0.012, LID_H, (0.06, 0.06, 0.07), collide=False)
        ring_z = VAC_H + LID_H + 0.0006
        _cyl(stage, f"{b}/ring_off", (0, 0, ring_z), RING_R, 0.001, (0.30, 0.30, 0.32), collide=False)
        for c, rgb in (("red", (1.0, 0.06, 0.04)), ("green", (0.08, 1.0, 0.15))):
            g = _cyl(stage, f"{b}/ring_{c}", (0, 0, ring_z + 0.0003), RING_R, 0.001, rgb, collide=False)
            if c == "green":
                g.CreateVisibilityAttr("invisible")

        btn = UsdGeom.Xform.Define(stage, f"{root}/button")
        btn.AddTranslateOp().Set(Gf.Vec3d(0, 0, VAC_H + LID_H + BTN_H / 2 - BTN_H / 2))
        UsdPhysics.RigidBodyAPI.Apply(btn.GetPrim())
        UsdPhysics.MassAPI.Apply(btn.GetPrim()).CreateMassAttr(0.01)
        _cyl(stage, f"{root}/button/cap", (0, 0, BTN_H / 2), BTN_R, BTN_H, (0.80, 0.81, 0.83), mat=shell)
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/joints/button")
        j.CreateBody0Rel().SetTargets([Sdf.Path(b)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f"{root}/button")])
        j.CreateAxisAttr("Z")
        j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, VAC_H + LID_H))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLowerLimitAttr(-BTN_TRAVEL)
        j.CreateUpperLimitAttr(0.0)
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(BTN_K)
        d.CreateDampingAttr(2.0)
        d.CreateTargetPositionAttr(BTN_PRELOAD)

    return lambda: _write_usd("robot_vacuum", build)


def _remote_usd():
    from isaaclab_arena.assets.object_library import RemoteControlHot3DRobolab

    return RemoteControlHot3DRobolab.usd_path


OBJECTS = [
    yampatch.usd_object("vacuum", VAC_START_POS, kind="articulation", build=_build_vacuum(), rot_xyzw=VAC_START_Q,
                        extent=2 * VAC_R),
    yampatch.usd_object("remote", REMOTE_POS, kind="rigid", build=_remote_usd, rot_xyzw=tuple(float(v) for v in _matrix_quat(REMOTE_R)),
                        mass=REMOTE_MASS, extent=0.164),
]  # fmt: skip

# -- state ------------------------------------------------------------------------------------------------------------
FLAT_DEG = 2.0  # flat: tilted at most 2 degrees, its underside within 3 mm of the table
FLAT_Z = 0.003
CLEAR = 0.005  # the remote is clear once no part of it is within 5 mm of the vacuum's footprint (seen from above)
DRIVE_V, DRIVE_S = 0.05, 3.0  # once started, the vacuum backs away at 5 cm/s for 3 s
_STATE: dict = {}
_RINGS: dict = {}
_LIVE: dict = {}
_calls = [0]


def _fresh():
    return {"t": 0.0, "btn": 0.0, "pressed": False, "presses": 0, "started": False, "drive_t": 0.0, "drive0": None,
            "flat": False, "clear": False, "tilt": 0.0, "ring": "?"}


_STATE.update(_fresh())


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def remote_clearance(vac_pos, rem_pos, rem_rmat) -> float:
    """Horizontal distance from the vacuum's footprint (a disc of VAC_R) to the remote's outline (negative: under)."""
    c = _remote_corners(rem_pos, rem_rmat)[:, :2]
    # the outline seen from above: the convex hull of the corners (a rectangle when the remote lies flat)
    from math import atan2

    centre = c.mean(axis=0)
    hull = sorted({tuple(np.round(p, 6)) for p in c}, key=lambda p: atan2(p[1] - centre[1], p[0] - centre[0]))
    hull = [np.array(p) for p in hull]
    q = np.asarray(vac_pos[:2], dtype=float)
    inside = all(np.cross(hull[(i + 1) % len(hull)] - hull[i], q - hull[i]) >= 0 for i in range(len(hull)))
    if inside:
        return -VAC_R
    d = min(_seg_dist(q, hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull)))
    return d - VAC_R


def _set_ring(color) -> None:
    if color == _STATE["ring"]:
        return
    try:
        import omni.usd
        from pxr import UsdGeom

        if not _RINGS:
            stage = omni.usd.get_context().get_stage()
            for prim in stage.Traverse():
                p = str(prim.GetPath())
                for c in ("red", "green"):
                    if p.endswith(f"/ring_{c}") and c not in _RINGS:
                        _RINGS[c] = UsdGeom.Imageable(prim)
        for c, img in _RINGS.items():
            img.GetVisibilityAttr().Set("inherited" if c == color else "invisible")
    except Exception as e:  # noqa: BLE001 - the ring is cosmetic for the scorer, but say why it failed
        print(f"[vacuum] ring: {e}", flush=True)
    _STATE["ring"] = color


def _drive(env) -> None:
    """While started: move the vacuum straight back (+x) at DRIVE_V for DRIVE_S, as if it were driving away."""
    import torch

    art = env.unwrapped.scene["vacuum"]
    p0, q0 = _STATE["drive0"]
    s = min(_STATE["drive_t"], DRIVE_S) * DRIVE_V
    origin = env.unwrapped.scene.env_origins[0].detach().cpu().numpy()
    pose = torch.tensor([[p0[0] + s + origin[0], p0[1] + origin[1], p0[2] + origin[2], *q0]], dtype=torch.float32,
                        device=art.device)  # fmt: skip
    art.write_root_pose_to_sim_index(root_pose=pose)
    art.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=art.device))


def on_reset(env) -> None:
    _STATE.update(_fresh())
    _LIVE.clear()
    _set_ring("red")
    _calls[0] = 0


def on_step(env, states) -> None:
    _calls[0] += 1
    dt = float(env.unwrapped.step_dt)
    _STATE["t"] += dt
    vac = yampatch._root_state(env, "vacuum")
    rem = yampatch._root_state(env, "remote")
    vr, rr = _quat_matrix(vac.quat_xyzw), _quat_matrix(rem.quat_xyzw)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, vr[2, 2]))))
    _STATE["tilt"] = tilt
    _STATE["flat"] = tilt <= FLAT_DEG and vac.pos[2] - TOP <= FLAT_Z
    _STATE["clear"] = remote_clearance(vac.pos, rem.pos, rr) > CLEAR
    q = yampatch.joint_positions(env, "vacuum")["button"]
    _STATE["btn"] = q
    if not _STATE["pressed"] and q <= PRESSED_Q:
        _STATE["pressed"] = True
    elif _STATE["pressed"] and q >= RELEASED_Q:  # pressed and released: it starts if it can
        _STATE["pressed"] = False
        _STATE["presses"] += 1
        if not _STATE["started"] and _STATE["flat"] and _STATE["clear"]:
            _STATE["started"] = True
            _STATE["drive0"] = (np.asarray(vac.pos, float).copy(), tuple(float(v) for v in vac.quat_xyzw))
            if _DEBUG:
                print(f"[vacuum step {_calls[0]}] started", flush=True)
    if _STATE["started"] and _STATE["drive_t"] < DRIVE_S:
        _STATE["drive_t"] += dt
        _drive(env)
    _set_ring("green" if _STATE["started"] else "red")
    _LIVE.update(vac=(np.asarray(vac.pos, float), vr), remote=(np.asarray(rem.pos, float), rr))
    if _DEBUG and _calls[0] % _DEBUG == 0:
        print(f"[vacuum step {_calls[0]} t {_STATE['t']:.1f}] vac {np.round((vac.pos - (0, 0, TOP)) * 100, 1)} cm "
              f"tilt {tilt:.1f} deg flat {_STATE['flat']} | remote {np.round((rem.pos - (0, 0, TOP)) * 100, 1)} cm "
              f"clearance {remote_clearance(vac.pos, rem.pos, rr) * 100:.1f} cm | button {q * 1000:.1f} mm "
              f"presses {_STATE['presses']} started {_STATE['started']} | score {score(states)}", flush=True)
        if os.environ.get("VACUUM_DEBUG_HANDS"):
            from yam_arena import ik, station

            out = [f"   rim(-x) {np.round((_rim(0.0) - (0, 0, TOP)) * 1000, 1)} mm"]
            for arm in ("left", "right"):
                jq = yampatch.joint_positions(env, f"{arm}_robot")
                tp, tr = ik.tcp_pose(np.array([jq[f"joint{i}"] for i in range(1, 7)]), station.BASES[arm])
                out.append(f"{arm} tcp {np.round((tp - (0, 0, TOP)) * 1000, 1)} appr {np.round(tr[:, 2], 2)} "
                           f"close {np.round(tr[:, 0], 2)} fing {jq['joint7'] * 1000:.1f}")
            print("; ".join(out), flush=True)


yampatch.install(on_reset=on_reset, on_step=on_step)


# -- scoring ----------------------------------------------------------------------------------------------------------
def score(objects):
    """1.0 once the vacuum has been started (and stands flat); else 0.5 if it stands flat with the remote clear of
    it; 0.25 if the remote is clear of it; 0 otherwise."""
    if _STATE["started"] and _STATE["flat"]:
        return 1.0
    if _STATE["clear"]:
        return 0.5 if _STATE["flat"] else 0.25
    return 0.0


SUCCESS_HOLD_S = 1.0


def success(objects):
    return score(objects) == 1.0 and _STATE["drive_t"] >= DRIVE_S


def distance(objects):
    """Steps left: free the remote, set the vacuum flat, start it."""
    return float(3 - _STATE["clear"] - (_STATE["clear"] and _STATE["flat"]) - _STATE["started"])


# -- demo -------------------------------------------------------------------------------------------------------------
# 1. The left hand, pointing forward with its fingers one above the other, slides its lower finger under the vacuum's
#    raised front edge and pinches the edge top and bottom; it lifts the edge 3.5 cm, freeing the remote.
# 2. The right hand grips the end of the remote that sticks out and pulls the remote out along its length.
# 3. The left hand lowers the edge until it is 1 cm above the table, lets go and draws its finger out, so the vacuum
#    drops flat onto the table.
# 4. The left hand, pointing down with its fingers closed, presses the start button and lets go of it.
TIP = 0.013  # pad tips past the TCP
OPEN, CLOSED = -0.047, 0.0
PARK = {"left": (0.12, 0.28, TOP + 0.17), "right": (0.12, -0.28, TOP + 0.17)}
_STOCK_GRIP = getattr(_demo.grip_rotation, "_stock", _demo.grip_rotation)


def _grip_rotation(yaw=0.0):
    """A rotation matrix given as a waypoint's ``yaw`` passes through; else the stock downward grip."""
    if isinstance(yaw, np.ndarray):
        return yaw
    return _STOCK_GRIP(yaw)


_grip_rotation._stock = _STOCK_GRIP
_demo.grip_rotation = _grip_rotation
DOWN = _STOCK_GRIP(0.0)  # pointing down, fingers closing along y
# The left hand pinches the vacuum's rim where it faces the left arm's base, pointing at the vacuum's centre with its
# fingers one above the other (the arm reaches straight out there, clear of its wrist's limits).
_LB = np.asarray(__import__("yam_arena.station", fromlist=["BASES"]).BASES["left"][:2], dtype=float)
PINCH_M = (_LB - VAC_XY) / np.linalg.norm(_LB - VAC_XY)  # from the vacuum's centre towards the left base
PINCH_A = -PINCH_M  # the approach
PINCH_PITCH = math.radians(60)  # tilted 30 degrees below horizontal (pointing level is out of reach so low)
PINCH_ROT = _rz(math.atan2(PINCH_A[1], PINCH_A[0])) @ _ry(-PINCH_PITCH) @ _rz(math.pi / 2) @ DOWN
_PA = PINCH_ROT[:, 2]  # approach
_PC = PINCH_ROT[:, 0] if PINCH_ROT[2, 0] > 0 else -PINCH_ROT[:, 0]  # closing axis, towards the upper finger
PAD = 0.047  # each pad this far from the TCP along the closing axis when open
# The right hand grips the remote's free end from above, fingers closing across it, leaning 30 degrees away from the
# vacuum so its wrist stays clear of the vacuum's raised edge.
REMOTE_ROT = (_rot((1, 0, 0), math.radians(30)) @ _rz(math.atan2(-REMOTE_U[1], -REMOTE_U[0])) @ DOWN)
INSERT = 0.030  # the lower finger's tip this far in under the vacuum's rim
LIFT = 0.030
PULL = 0.135  # how far the remote is pulled out along its length
GRIP_T = 0.06  # the right hand grips the remote this far from its centre, towards the end that sticks out
REMOTE_PINCH = -(0.018 - 0.005)  # 5 mm into each side of the 3.6 cm-wide remote


def _wp(d, pos, finger, label, rot=DOWN):
    return Waypoint(float(d), tuple(float(v) for v in pos), float(finger), label, rot)


class _Track:
    """A waypoint aimed from the live poses when its move starts (``fn()``; ``fallback`` while planning); fixed once
    the move begins. ``like`` repeats another waypoint's position."""

    def __init__(self, duration, fn, finger, label, rot=DOWN, fallback=(0.0, 0.0, 0.0), like=None):
        self.duration, self.fn, self.finger, self.label, self.yaw = float(duration), fn, float(finger), label, rot
        self.fallback, self.like, self.t0, self._pos = tuple(fallback), like, 0.0, None

    @property
    def pos(self):
        if self.like is not None:
            return self.like.pos
        if self._pos is None or _STATE["t"] < self.t0:
            try:
                p = tuple(float(v) for v in self.fn())
            except KeyError:  # no live poses yet (building the plan)
                return self.fallback
            if _STATE["t"] < self.t0:
                return p
            self._pos = p
        return self._pos


def _rim(dz_local=VAC_H / 2):
    """The vacuum's rim point facing the left arm's base, at height dz_local in its own frame, in the world now."""
    p, r = _LIVE["vac"]
    return p + r @ np.array([*(PINCH_M * VAC_R), dz_local])


def _pinch(dx=0.0, dz=0.0):
    """The left TCP (open) that puts the lower finger's tip INSERT (+ dx) in under the rim, 8 mm below the rim's
    underside (at least 1 cm above the table); the upper finger then reaches over the vacuum's top."""
    r = _rim(0.0)
    tip = np.array([*(r[:2] + PINCH_A * (INSERT + dx)), max(r[2] - 0.008, TOP + 0.010)])
    return tip + PAD * _PC - TIP * _PA + (0.0, 0.0, dz)


def _remote_grip(ds=0.0, dz=0.0):
    p, r = _LIVE["remote"]
    c = p + r @ ((REMOTE_LO + REMOTE_HI) / 2)
    u = np.array([*REMOTE_U, 0.0])
    under = _remote_corners(p, r)[:, 2].min()  # its underside
    g = c + u * (GRIP_T + ds)
    tip = np.array([g[0], g[1], under + 0.006])
    return tip - TIP * REMOTE_ROT[:, 2] + (0.0, 0.0, dz)


PRESS_ROT = _ry(math.radians(-25)) @ DOWN  # tilted 25 degrees forward: pointing straight down is out of reach there


def _button(dz=0.0):
    """The TCP that puts the closed fingertips dz above the middle of the button's top."""
    p, r = _LIVE["vac"]
    return p + r @ np.array([0.0, 0.0, BTN_TOP]) + np.array([0.0, 0.0, dz]) - TIP * PRESS_ROT[:, 2]


class _Plan:
    def __init__(self):
        self.w = {arm: [_wp(1.5, PARK[arm], OPEN, "park")] for arm in ("left", "right")}

    def t(self, arm):
        return sum(w.duration for w in self.w[arm])

    def add(self, arm, *wps):
        t = self.t(arm)
        for w in wps:
            if isinstance(w, _Track):
                w.t0 = t
            t += w.duration
        self.w[arm] += wps

    def hold(self, arm, until, label="wait"):
        dt = until - self.t(arm)
        if dt > 1e-6:
            last = self.w[arm][-1]
            if isinstance(last, _Track):
                self.w[arm].append(_Track(dt, None, last.finger, label, last.yaw, like=last))
            else:
                self.w[arm].append(Waypoint(dt, last.pos, last.finger, label, last.yaw))

    def sync(self, pause=0.0):
        t = max(self.t("left"), self.t("right")) + pause
        for arm in self.w:
            self.hold(arm, t)


def demo():
    plan = _Plan()
    L, R = "left", "right"
    fz = TOP + 0.056  # fallbacks while planning
    pin = _Track(1.5, lambda: _pinch(), OPEN, "slide the lower finger under the edge", PINCH_ROT,
                 fallback=(0.252, 0.097, fz))
    out = INSERT + 0.015  # the lower tip 15 mm outside the rim
    plan.add(L,
             _Track(2.0, lambda: _pinch(dx=-out, dz=0.09), OPEN, "in front of the vacuum's edge", PINCH_ROT,
                    fallback=(0.22, 0.124, TOP + 0.15)),
             _Track(1.5, lambda: _pinch(dx=-out), OPEN, "down, level with its edge", PINCH_ROT,
                    fallback=(0.22, 0.124, fz)),
             pin,
             _Track(1.0, None, CLOSED, "pinch the edge", PINCH_ROT, like=pin))
    lift = _Track(2.5, lambda: np.asarray(pin.pos) + (0.0, 0.0, LIFT), CLOSED, "lift the edge", PINCH_ROT,
                  fallback=(0.252, 0.097, fz + LIFT))
    plan.add(L, lift, _Track(0.5, None, CLOSED, "hold", PINCH_ROT, like=lift))
    plan.sync()
    grip = _Track(1.2, lambda: _remote_grip(), OPEN, "onto the remote's end", REMOTE_ROT,
                  fallback=(0.215, -0.11, TOP + 0.018))
    plan.add(R,
             _Track(2.0, lambda: _remote_grip(dz=0.06), OPEN, "above the remote's end", REMOTE_ROT,
                    fallback=(0.215, -0.11, TOP + 0.08)),
             grip,
             _Track(0.8, None, REMOTE_PINCH, "grip it", REMOTE_ROT, like=grip))
    pull = _Track(2.5, lambda: np.asarray(grip.pos) + (*(REMOTE_U * PULL), 0.004), REMOTE_PINCH,
                  "pull it out along its length", REMOTE_ROT, fallback=(0.215, -0.245, TOP + 0.022))
    plan.add(R, pull,
             _Track(0.6, None, OPEN, "let go", REMOTE_ROT, like=pull),
             _Track(1.0, lambda: np.asarray(pull.pos) + (0.0, 0.0, 0.08), OPEN, "up", REMOTE_ROT,
                    fallback=(0.215, -0.245, TOP + 0.10)),
             _wp(1.5, PARK[R], OPEN, "park"))
    plan.hold(L, plan.t(R) - 2.5)
    def _lowered():  # the held edge's underside 12 mm above the table
        held = np.asarray(lift.pos)
        return held - (0.0, 0.0, _rim(0.0)[2] - (TOP + 0.012))

    low = _Track(2.5, _lowered, CLOSED, "lower the edge", PINCH_ROT, fallback=(0.252, 0.097, TOP + 0.06))
    plan.add(L, low,
             _Track(0.8, None, OPEN, "let go", PINCH_ROT, like=low),
             _Track(1.2, lambda: np.asarray(low.pos) - (*(PINCH_A * (INSERT + 0.012)), 0.0), OPEN,
                    "draw the finger out", PINCH_ROT, fallback=(0.225, 0.12, TOP + 0.06)),
             _Track(1.0, lambda: np.asarray(low.pos) - (*(PINCH_A * (INSERT + 0.012)), -0.08), OPEN, "up",
                    PINCH_ROT, fallback=(0.225, 0.12, TOP + 0.14)),
             _wp(1.2, (0.24, 0.0, TOP + 0.14), CLOSED, "turn to point down", PRESS_ROT),
             _Track(1.8, lambda: _button(dz=0.04), CLOSED, "above the button", PRESS_ROT,
                    fallback=(VAC_XY[0], 0.0, TOP + 0.13)),
             _Track(1.0, lambda: _button(dz=-0.007), CLOSED, "press the button", PRESS_ROT,
                    fallback=(VAC_XY[0], 0.0, TOP + 0.084)),
             _Track(0.6, lambda: _button(dz=-0.007), CLOSED, "hold it", PRESS_ROT, fallback=(VAC_XY[0], 0.0, TOP + 0.084)),
             _Track(0.8, lambda: _button(dz=0.03), CLOSED, "let go of it", PRESS_ROT,
                    fallback=(VAC_XY[0], 0.0, TOP + 0.14)),
             _wp(1.5, PARK[L], OPEN, "park"))
    plan.sync(4.0)
    return plan.w
