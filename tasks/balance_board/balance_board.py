"""Balance a roller on a lifted board. On the station table a long wooden board rests on a low stand (two posts, so
the fingers can reach under its ends); a wooden roller (a cylinder lying across the board) sits on the board's
middle. Lift the board with both hands and keep it up for 10 s without the roller falling off or touching the robot.

The roller rolls along the board when the board tilts; its rolling is damped, like a roller on a felt-covered board.
The scripted demo grips the board's ends, lifts it 10 cm and balances it with a feedback controller: every control
step it reads where the roller is and tilts the board (one hand up, the other down) to roll it back to the middle.
BALANCE_DEBUG=N prints the state every N control steps; a lost trial prints why. BALANCE_PUSH=v (m/s) gives the
roller a push along the board once the board is up, to show the controller at work; BALANCE_CONTROLLER=0 holds
the board level instead.

Run the scripted demo:  inspect-robots run --embodiment yam_arena --policy yam_scripted \\
    -E task=tasks/balance_board/balance_board.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 \\
    -E cam_height=176 --instruction "..." --scorer task_score --max-steps 1200 --store-frames
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box

# Load yampatch.py from this folder (task files are imported by path, so their folder is not on sys.path).
if "yampatch" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("yampatch", Path(__file__).with_name("yampatch.py"))
    sys.modules["yampatch"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["yampatch"])
yampatch = sys.modules["yampatch"]

SCENE = "station"
TOP = TABLE_TOP[SCENE]
_DEBUG = int(os.environ.get("BALANCE_DEBUG", "0") or 0)
_PUSH = float(os.environ.get("BALANCE_PUSH", "0") or 0)
_CONTROLLER = os.environ.get("BALANCE_CONTROLLER", "1") != "0"

# -- layout ---------------------------------------------------------------------------------------------------------
BOARD_SIZE = (0.06, 0.54, 0.02)  # x by y by z: 6 cm deep, 54 cm long (across the table), 2 cm thick
BOARD_X = 0.38
POST_SIZE = (0.04, 0.04, 0.04)  # two posts, 4 cm cubes, under the board 12 cm either side of its middle
POST_Y = 0.12
STAND_TOP = TOP + POST_SIZE[2]
BOARD_Z = STAND_TOP + BOARD_SIZE[2] / 2
ROLLER_R, ROLLER_L = 0.02, 0.08  # 4 cm across, 8 cm long, lying across the board (axis x)
ROLLER_MASS = 0.07  # light wood
ROLLER_DAMPING = 1.0  # PhysX angular damping (1/s): slows its rolling, as on felt
WOOD, ROLLER_COLOUR, POST_COLOUR = (0.62, 0.45, 0.28), (0.95, 0.55, 0.10), (0.35, 0.36, 0.40)


def _make_roller():
    """The roller as an Arena object: a cylinder lying along x with damped spin. Called after Isaac Sim starts."""
    import isaaclab.sim as sim_utils

    from yam_arena import scene
    from yam_arena.station import linear_rgb

    cfg = sim_utils.CylinderCfg(
        radius=ROLLER_R, height=ROLLER_L, axis="X",
        rigid_props=scene.PhysxRigidBodyPropertiesCfg(angular_damping=ROLLER_DAMPING),
        mass_props=sim_utils.MassPropertiesCfg(mass=ROLLER_MASS),
        collision_props=scene._contact_props(), physics_material=scene._grippy(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(ROLLER_COLOUR), roughness=0.4),
    )  # fmt: skip
    return scene.Object(name="roller", object_type=scene.ObjectType.RIGID, spawner_cfg=cfg,
                        initial_pose=scene.Pose(position_xyz=ROLLER_START, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


ROLLER_START = (BOARD_X, 0.0, STAND_TOP + BOARD_SIZE[2] + ROLLER_R)
OBJECTS = [
    Box("post_left", pos=(BOARD_X, POST_Y, TOP + POST_SIZE[2] / 2), size=POST_SIZE, color=POST_COLOUR),
    Box("post_right", pos=(BOARD_X, -POST_Y, TOP + POST_SIZE[2] / 2), size=POST_SIZE, color=POST_COLOUR),
    Box("board", pos=(BOARD_X, 0.0, BOARD_Z), size=BOARD_SIZE, mass=0.30, color=WOOD),
    yampatch.usd_object("roller", ROLLER_START, kind="rigid", make=_make_roller, mass=ROLLER_MASS, extent=ROLLER_L),
]  # posts: no mass, fixed in place (glued to the table)

# -- rules ----------------------------------------------------------------------------------------------------------
LIFT = 0.05  # lifted: the board's lowest point at least 5 cm above the stand's top
HOLD_S = 10.0
TOUCH = 0.002  # surfaces within 2 mm count as touching
ON_BOARD_TOL = 0.005  # the roller's bottom at most 5 mm off the board's top
_STATE = {"lost": "", "run_s": 0.0, "best_s": 0.0, "t": 0.0}


def _quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def roller_on_board(states) -> tuple[bool, np.ndarray]:
    """The roller rests on the board: its centre over the board's top face and one radius (within 5 mm) above it.
    Also returns the roller's centre in the board's frame."""
    b, r = states["board"], states["roller"]
    local = (r.pos - b.pos) @ _quat_matrix(b.quat_xyzw)
    over = abs(local[0]) <= BOARD_SIZE[0] / 2 and abs(local[1]) <= BOARD_SIZE[1] / 2
    height = local[2] - BOARD_SIZE[2] / 2 - ROLLER_R
    return bool(over and abs(height) <= ON_BOARD_TOL), local


def board_bottom(states) -> float:
    return float(states["board"].low[2])


def lifted(states) -> bool:
    return board_bottom(states) >= STAND_TOP + LIFT


def _roller_points() -> np.ndarray:
    """The roller's surface, about every 2 mm, in its own frame (axis along x)."""
    pts = []
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for x in np.linspace(-ROLLER_L / 2, ROLLER_L / 2, 26):
        pts.append(np.c_[np.full_like(a, x), ROLLER_R * np.cos(a), ROLLER_R * np.sin(a)])
    for x in (-ROLLER_L / 2, ROLLER_L / 2):
        for rr in np.linspace(0, ROLLER_R, 10):
            k = max(int(2 * np.pi * rr / 0.002), 1)
            b = np.linspace(0, 2 * np.pi, k, endpoint=False)
            pts.append(np.c_[np.full_like(b, x), rr * np.cos(b), rr * np.sin(b)])
    return np.concatenate(pts)


def _robot_points() -> dict[str, np.ndarray]:
    """Surface points (link frame) of every moving link's collision meshes, about one per 4 mm^2."""
    import yam_arena
    from yam_arena.urdf import rpy_matrix

    urdf = Path(yam_arena.YAM_URDF)
    rng = np.random.default_rng(0)
    points = {}
    for link in ET.parse(urdf).getroot().findall("link"):
        if link.get("name") == "base":
            continue
        tris = []
        for c in link.findall("collision"):
            o = c.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
            raw = (urdf.parent / c.find("geometry/mesh").get("filename")).read_bytes()
            n = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
            v = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)["v"]
            tris.append(v.astype(float) @ rpy_matrix(rpy).T + xyz)
        if not tris:
            continue
        t = np.concatenate(tris)
        area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        k = max(int(area.sum() / 4e-6), 1)  # about one point per 2 x 2 mm
        idx = rng.choice(len(t), k, p=area / area.sum())
        a, b = rng.random((2, k, 1))
        flip = (a + b) > 1
        a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
        pts = t[idx, 0] + a * (t[idx, 1] - t[idx, 0]) + b * (t[idx, 2] - t[idx, 0])
        points[link.get("name")] = np.concatenate([pts, t.reshape(-1, 3)])  # samples and every vertex
    return points


_POINTS: dict[str, np.ndarray] = {}


def _robot_surface(env) -> np.ndarray:
    """Both robots' collision surface points in the world now, from their joint positions (URDF kinematics)."""
    import yam_arena
    from yam_arena import station
    from yam_arena.urdf import forward_kinematics

    if not _POINTS:
        _POINTS.update(_robot_points())
    out = []
    for arm in ("left", "right"):
        fk = forward_kinematics(yam_arena.YAM_URDF, yampatch.joint_positions(env, f"{arm}_robot"))
        base = np.asarray(station.BASES[arm])
        out += [p @ fk[name][:3, :3].T + fk[name][:3, 3] + base for name, p in _POINTS.items()]
    return np.concatenate(out)


def robot_touches_roller(points, r) -> bool:
    near = points[np.all(np.abs(points - r.pos) < ROLLER_L, axis=1)]
    local = (near - r.pos) @ _quat_matrix(r.quat_xyzw)
    return bool(np.any((np.abs(local[:, 0]) <= ROLLER_L / 2 + TOUCH)
                       & (np.hypot(local[:, 1], local[:, 2]) <= ROLLER_R + TOUCH)))  # fmt: skip


def violation(states, robot_points=None) -> str:
    """Why this state loses the trial ('' if it does not)."""
    on, _ = roller_on_board(states)
    if not on:
        return "the roller fell off the board"
    if board_bottom(states) < TOP + 0.002:
        return "the board was dropped onto the table"
    if robot_points is not None and robot_touches_roller(robot_points, states["roller"]):
        return "the robot touched the roller"
    return ""


# -- per-step tracking (and the demo's controller) ------------------------------------------------------------------
_calls = [0]
# The demo controller's output: the board's commanded tilt (rad, + raises the left end) and its inputs.
_CTRL = {"tilt": 0.0, "e": 0.0, "v": 0.0, "active": False, "pushed": False}


def on_reset(env) -> None:
    _STATE.update(lost="", run_s=0.0, best_s=0.0, t=0.0)
    _CTRL.update(tilt=0.0, e=0.0, v=0.0, active=False, pushed=False)
    _calls[0] = 0


def on_step(env, states) -> None:
    _calls[0] += 1
    dt = float(env.unwrapped.step_dt)
    _STATE["t"] += dt
    if not _STATE["lost"]:
        why = violation(states, _robot_surface(env))
        if why:
            _STATE["lost"] = why
            _STATE["run_s"] = 0.0
            print(f"[balance step {_calls[0]}] lost: {why}", flush=True)
        elif lifted(states):
            _STATE["run_s"] += dt
            _STATE["best_s"] = max(_STATE["best_s"], _STATE["run_s"])
        else:
            _STATE["run_s"] = 0.0
    _control(env, states, dt)
    if _DEBUG and _calls[0] % _DEBUG == 0:
        _, local = roller_on_board(states)
        print(f"[balance step {_calls[0]}] board bottom +{(board_bottom(states) - STAND_TOP) * 100:.1f} cm over the "
              f"stand, roller at {local[1] * 100:+.1f} cm along it, "
              f"tilt cmd {math.degrees(_CTRL['tilt']):+.2f} deg, "
              f"balanced {_STATE['run_s']:.1f} s (best {_STATE['best_s']:.1f}) lost {_STATE['lost']!r} "
              f"score {score(states)}", flush=True)  # fmt: skip


yampatch.install(on_reset=on_reset, on_step=on_step)


# -- scoring -------------------------------------------------------------------------------------------------------
def score(objects):
    """0 once lost; otherwise 0.1 per full second of the longest stretch balanced with the board lifted (1.0 at
    10 s)."""
    if _STATE["lost"]:
        return 0.0
    return round(min(math.floor(_STATE["best_s"] + 1e-6), HOLD_S) / HOLD_S, 2)


SUCCESS_HOLD_S = 0.0  # the 10 s of balancing are the hold


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """Seconds of balancing still to go."""
    return float(max(HOLD_S - _STATE["best_s"], 0.0))


# -- demo -----------------------------------------------------------------------------------------------------------
# Both arms grip the board's ends (fingers closing across its 6 cm depth, 4 mm into it each side), lift it 10 cm
# over 3 s and hold it there while a PD controller tilts it: tilt = KP * offset + KD * speed of the roller along the
# board (from its middle, + towards the left), so the raised end pushes the roller back to the middle. The pinch is a
# hinge about x, so one hand rising and the other sinking tilts the board in the grip.
GRIP_Y = 0.21  # each hand 21 cm from the board's middle; 6 cm of board beyond it
GRIP_YAW = np.pi / 2  # fingers close along x
PINCH = -(BOARD_SIZE[0] / 2 - 0.004)
Z_GRIP = STAND_TOP + 0.004 + 0.013  # fingertips 4 mm above the stand's top: around the board's lower half
Z_UP = Z_GRIP + 0.10
KP, KD = 0.6, 0.4  # rad per m, rad per m/s
MAX_TILT = math.radians(4.0)
MAX_TILT_RATE = math.radians(20.0)  # rad/s
BALANCE_S = 13.0


def _control(env, states, dt) -> None:
    """The demo's balance controller (runs every control step; the demo's waypoints read its output)."""
    _, local = roller_on_board(states)
    rot = _quat_matrix(states["board"].quat_xyzw)
    v = float((states["roller"].lin_vel - states["board"].lin_vel) @ rot[:, 1])
    e = float(local[1])
    want = float(np.clip(KP * e + KD * v, -MAX_TILT, MAX_TILT)) if _CTRL["active"] and _CONTROLLER else 0.0
    step = MAX_TILT_RATE * dt
    _CTRL.update(e=e, v=v, tilt=_CTRL["tilt"] + float(np.clip(want - _CTRL["tilt"], -step, step)))
    if _PUSH and _CTRL["active"] and not _CTRL["pushed"] and lifted(states):
        _CTRL["pushed"] = True
        r = env.unwrapped.scene["roller"]
        vel = r.data.root_vel_w
        vel = getattr(vel, "torch", vel).clone()
        vel[:, :3] += vel.new_tensor([0.0, _PUSH, 0.0])
        vel[:, 3:] += vel.new_tensor([-_PUSH / ROLLER_R, 0.0, 0.0])  # rolling, not sliding
        r.write_root_velocity_to_sim_index(root_velocity=vel)


class _Balance:
    """A waypoint whose target is the controller's: the hand's lifted pose, raised or lowered by the commanded tilt.
    Two in a row interpolate between the same live target, so the hand follows the controller."""

    def __init__(self, arm, duration, label, start=False):
        self.arm, self.duration, self.label, self.start = arm, duration, label, start
        self.finger, self.yaw = PINCH, GRIP_YAW

    @property
    def pos(self):
        if self.start:
            _CTRL["active"] = True
        y = GRIP_Y if self.arm == "left" else -GRIP_Y
        return (BOARD_X, y, Z_UP + y * math.tan(_CTRL["tilt"]))


def _wp(t, pos, finger, label, yaw=GRIP_YAW):
    return Waypoint(t, tuple(float(v) for v in pos), finger, label, float(yaw))


def hold_board(arm):
    y = GRIP_Y if arm == "left" else -GRIP_Y
    return [
        _wp(2.5, (BOARD_X, y, Z_UP), FINGER_OPEN, "above the board's end"),
        _wp(1.5, (BOARD_X, y, Z_GRIP), FINGER_OPEN, "descend"),
        _wp(1.0, (BOARD_X, y, Z_GRIP), PINCH, "grip"),
        _wp(3.0, (BOARD_X, y, Z_UP), PINCH, "lift 10 cm"),
        _Balance(arm, 0.3, "balance", start=True),
        _Balance(arm, BALANCE_S, "balance"),
    ]


def demo():
    return {arm: hold_board(arm) for arm in ("left", "right")}
