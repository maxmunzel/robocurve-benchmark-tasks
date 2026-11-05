"""On the station table, from far to near: a long wooden board lying flat; a row of ten flat indicator pads (white
coasters, one of them red); a row of ten flat grey target pads, one in front of each indicator; and a dark start pad
with a yellow puck standing on it. All pads are glued down.

Memorize which indicator is red, cover all ten indicators with the board, then put the puck on the target pad in
front of the red indicator and let go. The puck may touch only the robot, its start pad and that target pad, and the
robot must not touch it before the indicators are covered; once covered, they must stay covered.

The red indicator is number 7 of 10 (counted from the left, as seen from the robots); set MEMORY_MARKED=1..10 to
change it. MEMORY_DEBUG=N prints the state every N control steps.

Run the scripted demo:  inspect-robots run --embodiment yam_arena --policy yam_scripted \\
    -E task=tasks/memory_puck/memory_puck.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 \\
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
_DEBUG = int(os.environ.get("MEMORY_DEBUG", "0") or 0)

# -- layout ---------------------------------------------------------------------------------------------------------
N = 10
MARKED = int(os.environ.get("MEMORY_MARKED", "7"))  # 1..10, counted from the left (+y) as seen from the robots
assert 1 <= MARKED <= N, f"MEMORY_MARKED must be 1..{N}, got {MARKED}"
PITCH = 0.055  # pad centre to centre, across the table
PAD = 0.04  # indicator and target pads: 4 x 4 cm coasters
PAD_H = 0.005  # 5 mm thick
START_PAD = 0.05  # 5 x 5 cm
ROW_Y = [(N - 1) / 2 * PITCH - i * PITCH for i in range(N)]  # +24.75 .. -24.75 cm, left to right
INDICATOR_X, TARGET_X, START_XY = 0.36, 0.28, (0.17, 0.0)
BOARD_SIZE = (0.065, 0.60, 0.025)  # x by y by z: 6.5 cm deep, 60 cm long, 2.5 cm thick
BOARD_XY = (0.45, 0.0)  # behind the indicators
PUCK_R, PUCK_H = 0.015, 0.04  # 3 cm across, 4 cm tall

WHITE, RED, GREY, DARK = (0.93, 0.93, 0.91), (0.85, 0.08, 0.08), (0.50, 0.52, 0.55), (0.18, 0.18, 0.20)
INDICATORS = [f"indicator_{i + 1}" for i in range(N)]
TARGETS = [f"target_{i + 1}" for i in range(N)]
CORRECT = TARGETS[MARKED - 1]


def _pad(name, xy, side, color):  # no mass: fixed in place (glued to the table)
    return Box(name, pos=(xy[0], xy[1], TOP + PAD_H / 2), size=(side, side, PAD_H), color=color)


OBJECTS = (
    [_pad(n, (INDICATOR_X, y), PAD, RED if i == MARKED - 1 else WHITE) for i, (n, y) in enumerate(zip(INDICATORS, ROW_Y))]
    + [_pad(n, (TARGET_X, y), PAD, GREY) for n, y in zip(TARGETS, ROW_Y)]
    + [
        _pad("start_pad", START_XY, START_PAD, DARK),
        Box("board", pos=(*BOARD_XY, TOP + BOARD_SIZE[2] / 2), size=BOARD_SIZE, mass=0.30, color=(0.55, 0.38, 0.22)),
        Cylinder("puck", pos=(*START_XY, TOP + PAD_H + PUCK_H / 2), radius=PUCK_R, height=PUCK_H, mass=0.03,
                 color=(0.97, 0.80, 0.10)),
    ]
)  # fmt: skip
PADS = {o.name: (np.array(o.pos), np.array(o.size)) for o in OBJECTS if o.mass is None}  # centre, size
ALLOWED = {"start_pad", CORRECT}  # the only things the puck may touch, besides the robot

# -- touch detection ------------------------------------------------------------------------------------------------
TOUCH = 0.002  # surfaces within 2 mm count as touching
GRIP_OPEN = 0.02  # each finger at least 2 cm open (pads 4 cm apart, wider than the 3 cm puck): not holding it
_STATE = {"covered": False, "lost": "", "grippers": {"left": 0.0, "right": 0.0}}


def _quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def _puck_points() -> np.ndarray:
    """The puck's surface, about every 2 mm, in its own frame (axis along z)."""
    pts = []
    for z in np.linspace(-PUCK_H / 2, PUCK_H / 2, 21):  # side
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        pts.append(np.c_[PUCK_R * np.cos(a), PUCK_R * np.sin(a), np.full_like(a, z)])
    for z in (-PUCK_H / 2, PUCK_H / 2):  # bottom and top discs
        for r in np.linspace(0, PUCK_R, 8):
            k = max(int(2 * np.pi * r / 0.002), 1)
            a = np.linspace(0, 2 * np.pi, k, endpoint=False)
            pts.append(np.c_[r * np.cos(a), r * np.sin(a), np.full_like(a, z)])
    return np.concatenate(pts)


_PUCK = _puck_points()


def _in_box(points, centre, rot, size, margin):
    local = (points - centre) @ rot
    return np.all(np.abs(local) <= np.asarray(size) / 2 + margin, axis=1)


def puck_contacts(states) -> list[str]:
    """What the puck touches (other than the robot): pads, the board, the table."""
    p = states["puck"]
    pts = _PUCK @ _quat_matrix(p.quat_xyzw).T + p.pos
    hits = [n for n, (c, s) in PADS.items() if _in_box(pts, c, np.eye(3), s, TOUCH).any()]
    b = states["board"]
    if _in_box(pts, b.pos, _quat_matrix(b.quat_xyzw), BOARD_SIZE, TOUCH).any():
        hits.append("board")
    low = pts[pts[:, 2] < TOP + TOUCH]
    if len(low) and not all(_in_box(low, c, np.eye(3), s, 0.0).any() for c, s in PADS.values()):
        hits.append("table")
    return hits


def covered(states) -> bool:
    """Every indicator pad's top lies under the board's footprint, the board's bottom at most 1.5 cm above it."""
    b = states["board"]
    rot = _quat_matrix(b.quat_xyzw)
    for n in INDICATORS:
        c = PADS[n][0]
        corners = np.array([[c[0] + sx * PAD / 2, c[1] + sy * PAD / 2, TOP + PAD_H] for sx in (-1, 1) for sy in (-1, 1)])
        local = (corners - b.pos) @ rot
        if not (np.all(np.abs(local[:, :2]) <= np.array(BOARD_SIZE[:2]) / 2)
                and np.all(local[:, 2] >= -BOARD_SIZE[2] / 2 - 0.015) and np.all(local[:, 2] <= 0)):  # fmt: skip
            return False
    return True


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


def robot_touches_puck(points, p) -> bool:
    near = points[np.all(np.abs(points - p.pos) < PUCK_H, axis=1)]
    local = (near - p.pos) @ _quat_matrix(p.quat_xyzw)
    return bool(np.any((np.hypot(local[:, 0], local[:, 1]) <= PUCK_R + TOUCH) & (np.abs(local[:, 2]) <= PUCK_H / 2 + TOUCH)))


_calls = [0]


def _lose(why):
    if not _STATE["lost"]:
        _STATE["lost"] = why
        print(f"[memory step {_calls[0]}] lost: {why}", flush=True)


def on_reset(env) -> None:
    _STATE.update(covered=False, lost="", grippers={"left": 0.0, "right": 0.0})
    _calls[0] = 0


def on_step(env, states) -> None:
    _calls[0] += 1
    for arm in ("left", "right"):
        q = yampatch.joint_positions(env, f"{arm}_robot")
        _STATE["grippers"][arm] = min(abs(q["joint7"]), abs(q["joint8"]))
    is_covered = covered(states)
    if _STATE["covered"] and not is_covered:
        _lose("an indicator was uncovered again")
    if not _STATE["covered"] and robot_touches_puck(_robot_surface(env), states["puck"]):
        _lose("the robot touched the puck before the indicators were covered")
    _STATE["covered"] |= is_covered
    bad = [n for n in puck_contacts(states) if n not in ALLOWED]
    if bad:
        _lose(f"the puck touched {', '.join(bad)}")
    if _DEBUG and _calls[0] % _DEBUG == 0:
        g = _STATE["grippers"]
        print(f"[memory step {_calls[0]}] covered {is_covered} (ever {_STATE['covered']}) lost {_STATE['lost']!r} "
              f"puck {np.round(states['puck'].pos * 100, 1)} cm board {np.round(states['board'].pos * 100, 1)} cm "
              f"grippers {g['left'] * 1000:.0f}/{g['right'] * 1000:.0f} mm score {score(states)}", flush=True)  # fmt: skip


yampatch.install(on_reset=on_reset, on_step=on_step)


# -- scoring -------------------------------------------------------------------------------------------------------
def on_target(p) -> bool:
    """The puck's centre is over the correct target pad and its bottom rests on the pad's top."""
    c, s = PADS[CORRECT]
    over = np.all(np.abs(p.pos[:2] - c[:2]) <= s[:2] / 2)
    return bool(over and abs(p.low[2] - (TOP + PAD_H)) <= 0.003)


def grippers_open() -> bool:
    return all(v >= GRIP_OPEN for v in _STATE["grippers"].values())


def score(objects):
    """0 once lost; 0.25 with the indicators covered; 0.5 with the puck on the correct target too; 1.0 also with both
    grippers open."""
    if _STATE["lost"] or not covered(objects):
        return 0.0
    if not on_target(objects["puck"]):
        return 0.25
    return 1.0 if grippers_open() else 0.5


SUCCESS_HOLD_S = 3.0  # the trial ends 3 s after success, once the arm has cleared the view


def success(objects):
    return score(objects) == 1.0 and objects["puck"].resting() and objects["board"].resting()


def distance(objects):
    """Horizontal distance (m) of the puck from the correct target pad's centre."""
    return float(np.linalg.norm(objects["puck"].pos[:2] - PADS[CORRECT][0][:2]))


# -- demo -----------------------------------------------------------------------------------------------------------
# 1. Both arms grip the board near its ends (fingers closing across its 6.5 cm depth), lift it 3 cm, carry it 9 cm
#    towards the robots and set it down on the indicator pads; they let go.
# 2. The left arm parks. The right arm picks the puck off the start pad, carries it 10 cm above the table to the
#    target in front of the red indicator, sets it down and lets go; it parks.
GRIP_Y = 0.20  # each hand 20 cm from the board's centre
BOARD_PINCH = -(BOARD_SIZE[0] / 2 - 0.004)  # 4 mm into the board each side
HOVER = TOP + 0.10


def _wp(t, pos, finger, label, yaw=0.0):
    return Waypoint(t, tuple(float(v) for v in pos), finger, label, float(yaw))


def carry_board(side):
    y = GRIP_Y if side == "left" else -GRIP_Y
    yaw = np.pi / 2  # fingers close along x, across the board's depth
    z_grip = TOP + 0.008 + 0.013  # fingertips 8 mm above the table
    z_set = TOP + PAD_H + 0.001 + 0.008 + 0.013  # the board's bottom 1 mm above the pads
    bx, ix = BOARD_XY[0], INDICATOR_X
    return [
        _wp(2.5, (bx, y, TOP + 0.08), FINGER_OPEN, "board: above its end", yaw),
        _wp(1.5, (bx, y, z_grip), FINGER_OPEN, "descend", yaw),
        _wp(1.0, (bx, y, z_grip), BOARD_PINCH, "grip", yaw),
        _wp(1.0, (bx, y, z_grip + 0.03), BOARD_PINCH, "lift", yaw),
        _wp(2.5, (ix, y, z_grip + 0.03), BOARD_PINCH, "over the indicators", yaw),
        _wp(1.0, (ix, y, z_set), BOARD_PINCH, "set it down on them", yaw),
        _wp(0.8, (ix, y, z_set), FINGER_OPEN, "let go", yaw),
        _wp(1.0, (ix, y, HOVER), FINGER_OPEN, "up", yaw),
    ]


def move_puck():
    tx, ty = TARGET_X, ROW_Y[MARKED - 1]
    z_pick = TOP + PAD_H + 0.015 + 0.013  # fingertips 1.5 cm above the puck's bottom
    z_place = z_pick + 0.002
    return [
        _wp(2.0, (*START_XY, HOVER), FINGER_OPEN, "puck: above it"),
        _wp(1.2, (*START_XY, z_pick), FINGER_OPEN, "descend"),
        _wp(0.8, (*START_XY, z_pick), FINGER_CLOSED, "grip"),
        _wp(1.0, (*START_XY, HOVER), FINGER_CLOSED, "lift"),
        _wp(2.0, (tx, ty, HOVER), FINGER_CLOSED, "over the remembered target"),
        _wp(1.2, (tx, ty, z_place), FINGER_CLOSED, "set it down"),
        _wp(0.8, (tx, ty, z_place), FINGER_OPEN, "let go"),
        _wp(0.8, (tx, ty, HOVER), FINGER_OPEN, "up"),
    ]


def park(arm, seconds=1.5):
    return [_wp(seconds, (0.15, 0.28 if arm == "left" else -0.28, TOP + 0.16), FINGER_OPEN, "park")]


def demo():
    return {
        "left": carry_board("left") + park("left"),
        "right": carry_board("right") + move_puck() + park("right"),
    }
