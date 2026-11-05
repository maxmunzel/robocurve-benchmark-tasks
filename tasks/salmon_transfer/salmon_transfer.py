"""On the station table: a heavy kitchen counter block (30 x 28 x 10 cm) in front of the left arm, and a plate on the
table to its right. On the counter lie a wooden cutting board (18 x 7 x 1.5 cm, long side forward) near the counter's
right edge, with five raw salmon cubes (1.5 cm) in a row along its middle, and a knife (a 1 x 1 x 20 cm steel bar,
long side forward) near the counter's left edge. The plate is a flat white disc, 18 cm across and 1.5 cm tall.

Move all five salmon pieces onto the plate without touching the salmon with the robot: only the knife, the board
and the plate may touch it.

Run the scripted demo:  inspect-robots run --embodiment yam_arena --policy yam_scripted \\
    -E task=tasks/salmon_transfer/salmon_transfer.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 \\
    -E cam_height=176 --instruction "..." --scorer task_score --max-steps 2400 --store-frames
"""

from __future__ import annotations

import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box, Cylinder

# Load yampatch.py from this folder (task files are imported by path, so their folder is not on sys.path).
if "yampatch" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("yampatch", Path(__file__).with_name("yampatch.py"))
    sys.modules["yampatch"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["yampatch"])
yampatch = sys.modules["yampatch"]

SCENE = "station"
TOP = TABLE_TOP[SCENE]
_DEBUG = int(os.environ.get("SALMON_DEBUG", "0") or 0)  # print the state every N steps

# -- the scene -----------------------------------------------------------------------------------------------------
COUNTER_SIZE = (0.30, 0.28, 0.10)
COUNTER_XY = (0.29, 0.20)  # x 14..44 cm, y 6..34 cm
COUNTER_TOP = TOP + COUNTER_SIZE[2]
BOARD_SIZE = (0.18, 0.07, 0.015)
BOARD_XY = (0.29, 0.105)  # 1 cm in from the counter's right edge
PIECE = 0.015
PIECE_DX = (-0.02, 0.0, 0.02, 0.04, 0.06)  # along the board from its centre (its ends are at -9 and +9 cm)
PLATE_R, PLATE_H = 0.09, 0.015
PLATE_XY = (0.21, -0.035)  # its edge 5 mm right of the counter
KNIFE_SIZE = (0.20, 0.01, 0.01)
KNIFE_XY = (0.29, 0.30)

SALMON = [f"salmon_{i}" for i in range(1, len(PIECE_DX) + 1)]
OBJECTS = [
    Box("counter", pos=(*COUNTER_XY, TOP + COUNTER_SIZE[2] / 2), size=COUNTER_SIZE, mass=30.0, color=(0.30, 0.32, 0.35)),
    Box("cutting_board", pos=(*BOARD_XY, COUNTER_TOP + BOARD_SIZE[2] / 2), size=BOARD_SIZE, mass=0.11,
        color=(0.72, 0.50, 0.28)),
    Box("knife", pos=(*KNIFE_XY, COUNTER_TOP + KNIFE_SIZE[2] / 2), size=KNIFE_SIZE, mass=0.10, color=(0.78, 0.80, 0.83)),
    Cylinder("plate", pos=(*PLATE_XY, TOP + PLATE_H / 2), radius=PLATE_R, height=PLATE_H, mass=0.40,
             color=(0.96, 0.96, 0.94)),
] + [
    Box(name, pos=(BOARD_XY[0] + dx, BOARD_XY[1], COUNTER_TOP + BOARD_SIZE[2] + PIECE / 2), size=(PIECE,) * 3,
        mass=0.0035, color=(0.98, 0.50, 0.38))
    for name, dx in zip(SALMON, PIECE_DX, strict=True)
]  # fmt: skip

# -- touching the salmon: the robots' collision meshes, sampled every ~2 mm, against each piece ---------------------
TOUCH = 0.003  # a robot surface point within 3 mm of a piece counts as touching it
_STATE = {"touched": False, "by": ""}


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
    if "labels" not in _STATE:  # which arm and link each point belongs to, for the touch report
        _STATE["labels"] = np.concatenate([np.full(len(p), f"{arm} {name}") for arm in ("left", "right")
                                           for name, p in _POINTS.items()])  # fmt: skip
    return np.concatenate(out)


def _quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def touching(points, s) -> np.ndarray:
    """Indices of ``points`` within TOUCH of the piece ``s`` (in its own, rotated frame); empty if none."""
    near = np.flatnonzero(np.all(np.abs(points - s.pos) < PIECE, axis=1))  # pre-filter: within 1.5 cm per axis
    local = (points[near] - s.pos) @ _quat_matrix(s.quat_xyzw)
    return near[np.all(np.abs(local) <= PIECE / 2 + TOUCH, axis=1)]


_calls = [0]


def on_reset(env) -> None:
    _STATE.update(touched=False, by="")
    _calls[0] = 0


def on_step(env, states) -> None:
    _calls[0] += 1
    if not _STATE["touched"]:
        points = _robot_surface(env)
        for n in SALMON:
            hit = touching(points, states[n])
            if len(hit):
                _STATE.update(touched=True, by=n)
                where = ", ".join(sorted(set(_STATE["labels"][hit])))
                print(f"[salmon step {_calls[0]}] the robot ({where}) touched {n} at "
                      f"{np.round(states[n].pos * 100, 1)} cm, point {np.round(points[hit[0]] * 100, 1)}", flush=True)
                break
    if _DEBUG and _calls[0] % _DEBUG == 0:
        on = [n for n in SALMON if on_plate(states[n], states["plate"])]
        b = states["cutting_board"]
        rpy = np.degrees([np.arctan2(*_quat_matrix(b.quat_xyzw)[[2, 2], [1, 2]]), -np.arcsin(_quat_matrix(b.quat_xyzw)[2, 0]),
                          np.arctan2(*_quat_matrix(b.quat_xyzw)[[1, 0], [0, 0]])])  # fmt: skip
        pieces = " ".join(f"{np.round((states[n].pos - b.pos) * 100, 1)}" for n in SALMON)
        print(f"[salmon step {_calls[0]}] on plate {len(on)}/{len(SALMON)} touched {_STATE['touched']} "
              f"board {np.round(b.pos * 100, 1)} cm rpy {np.round(rpy, 1)} deg; pieces - board (cm) {pieces}", flush=True)


yampatch.install(on_reset=on_reset, on_step=on_step)


# -- scoring -------------------------------------------------------------------------------------------------------
def on_plate(s, plate) -> bool:
    """The piece's centre is over the plate and its bottom on it (or on another piece lying on it)."""
    over = np.linalg.norm(s.pos[:2] - plate.pos[:2]) <= PLATE_R
    return bool(over and plate.high[2] - 0.005 <= s.low[2] <= plate.high[2] + 0.02)


def score(objects):
    """0.2 per salmon piece on the plate; 0 once the robot has touched any piece."""
    if _STATE["touched"]:
        return 0.0
    return sum(on_plate(objects[n], objects["plate"]) for n in SALMON) / len(SALMON)


SUCCESS_HOLD_S = 6.0  # the trial ends early once every piece has lain on the plate, still, for 6 s


def success(objects):
    return score(objects) == 1.0 and all(objects[n].resting() for n in SALMON)


def distance(objects):
    """Summed horizontal distance (m) of the pieces from the plate's centre."""
    plate = objects["plate"].pos[:2]
    return float(sum(np.linalg.norm(objects[n].pos[:2] - plate) for n in SALMON))


# -- demo -----------------------------------------------------------------------------------------------------------
# 1. The left arm pushes the board 10 cm towards the robots, over the counter's near edge (4 cm overhang).
# 2. The right arm grips the overhanging end across the board's width and drags the board 5.5 cm to the right, so that
#    it sticks out 4.5 cm over the counter's right edge, above the plate. It holds it there, level on the counter.
# 3. The left arm picks up the knife by its far end, sets it across the board behind the salmon and sweeps the pieces
#    off the board's right side; they drop onto the plate. Its fingers stay beyond the board's far end, the right
#    hand's before its near end.
# 4. The right hand drags the board back until it rests on the counter and lets go; both arms park (the left one
#    still holding the knife), clear of the top camera's view of the plate.
PUSH = 0.10
PUSH_GIVE = 0.014  # the arm gives this much under the sliding board: the push is commanded this much further
TIP_HALF = 0.006  # closed fingertips' half thickness across the closing direction
BOARD_GRIP_X = 0.12  # after the push the board spans x 10..28 cm; the right hand holds it 2 cm from its near end
HOLD_Y = COUNTER_XY[1] - COUNTER_SIZE[1] / 2 + BOARD_SIZE[1] / 2 - 0.045  # board centre: 4.5 cm over the right edge
KNIFE_GRIP_DX = 0.07  # the left hand holds the knife 3 cm from its far end
KNIFE_X = 0.33  # TCP x while sweeping: the knife spans x 16..36 cm, the hand 3 cm beyond the board's far end
BOARD_PINCH = -(BOARD_SIZE[1] / 2 - 0.002)  # finger target on the board: 2 mm into it each side, a gentle squeeze


def _wp(t, pos, finger, label, yaw=0.0):
    return Waypoint(t, tuple(float(v) for v in pos), finger, label, float(yaw))


def push_board():
    bx, by = BOARD_XY
    z = COUNTER_TOP + 0.005 + 0.013  # fingertips 5 mm above the counter, against the board's far end
    start = bx + BOARD_SIZE[0] / 2 + TIP_HALF + 0.01
    end = bx - PUSH + BOARD_SIZE[0] / 2 + TIP_HALF - PUSH_GIVE
    yaw = -np.pi / 2  # the fingertips' wide side (3 cm) against the board
    return [
        _wp(1.5, (start, by, TOP + 0.13), FINGER_CLOSED, "board: behind it", yaw),
        _wp(1.2, (start, by, z), FINGER_CLOSED, "down", yaw),
        _wp(3.5, (end, by, z), FINGER_CLOSED, "push it over the near edge", yaw),
        _wp(0.6, (end + 0.01, by, z), FINGER_CLOSED, "back off", yaw),
        _wp(1.0, (end + 0.01, by, TOP + 0.16), FINGER_CLOSED, "up, clear of the salmon", yaw),
    ]


def hold_board():
    by = BOARD_XY[1]
    z_on = COUNTER_TOP + BOARD_SIZE[2] / 2
    return [
        _wp(1.5, (BOARD_GRIP_X, by, TOP + 0.16), FINGER_OPEN, "board: above its overhang"),
        _wp(1.5, (BOARD_GRIP_X, by, z_on), FINGER_OPEN, "fingers either side"),
        _wp(1.2, (BOARD_GRIP_X, by, z_on), BOARD_PINCH, "grip"),
        _wp(3.0, (BOARD_GRIP_X, HOLD_Y, z_on), BOARD_PINCH, "drag it over the right edge"),
    ]


def return_board():
    by = COUNTER_XY[1] - COUNTER_SIZE[1] / 2 + 0.015  # its centre 1.5 cm inside the right edge: it rests on the counter
    z_on = COUNTER_TOP + BOARD_SIZE[2] / 2
    return [
        _wp(1.0, (BOARD_GRIP_X, by, z_on), BOARD_PINCH, "board: drag it back onto the counter"),
        _wp(0.5, (BOARD_GRIP_X, by, z_on), FINGER_OPEN, "let go"),
        _wp(0.8, (BOARD_GRIP_X, by, TOP + 0.16), FINGER_OPEN, "up"),
    ]


def sweep():
    kx, ky = KNIFE_XY
    gx = kx + KNIFE_GRIP_DX
    z_pick = COUNTER_TOP + 0.015  # fingertips 2 mm above the counter, pads on the knife's sides
    board_top = COUNTER_TOP + BOARD_SIZE[2]
    z_cut = board_top + 0.002 + KNIFE_SIZE[2] / 2  # the knife's edge 2 mm above the board
    y0 = HOLD_Y + BOARD_SIZE[1] / 2 - KNIFE_SIZE[1] / 2 - 0.001  # behind the salmon, on the board's left edge
    y1 = HOLD_Y - BOARD_SIZE[1] / 2 - 0.025  # past the board's right side
    return [
        _wp(1.5, (gx, ky, TOP + 0.16), FINGER_OPEN, "knife: above its far end"),
        _wp(1.5, (gx, ky, z_pick), FINGER_OPEN, "descend"),
        _wp(0.8, (gx, ky, z_pick), FINGER_CLOSED, "grip"),
        _wp(1.2, (gx, ky, TOP + 0.16), FINGER_CLOSED, "lift"),
        _wp(2.0, (KNIFE_X, y0, TOP + 0.16), FINGER_CLOSED, "over the board"),
        _wp(1.5, (KNIFE_X, y0, z_cut), FINGER_CLOSED, "onto the board, behind the salmon"),
        _wp(3.5, (KNIFE_X, y1, z_cut), FINGER_CLOSED, "sweep the salmon onto the plate"),
        _wp(1.0, (KNIFE_X, y1, TOP + 0.14), FINGER_CLOSED, "lift"),
    ]


def park(arm, seconds=1.5, finger=FINGER_OPEN):
    return [_wp(seconds, (0.15, 0.28 if arm == "left" else -0.28, TOP + 0.16), finger, "park")]


def _duration(waypoints):
    return sum(w.duration for w in waypoints)


def demo():
    """Left pushes; right grips and drags; left sweeps while right holds; then both clear the camera's view of the
    plate (the left hand parks with the knife)."""
    left_push = push_board() + park("left")
    right_take = hold_board()
    left_sweep = sweep()
    return {
        "left": left_push + park("left", _duration(right_take)) + left_sweep + park("left", 2.0, FINGER_CLOSED),
        "right": park("right", _duration(left_push)) + right_take
        + [_wp(_duration(left_sweep), right_take[-1].pos, BOARD_PINCH, "hold the board")]
        + return_board() + park("right"),
    }  # fmt: skip
