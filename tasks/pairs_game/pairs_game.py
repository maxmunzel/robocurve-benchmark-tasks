"""A matching-pairs game (concentration) on the station table. Six round coloured discs (two red, two blue, two
green) lie on round grey coasters 20 cm apart (a far row of four, a near row of two in its gaps), each hidden under
an opaque cylindrical cover (an upturned cup). In front of them lies a long mat. All discs, coasters and the mat are
glued down.

Play by the usual pairs rules: each turn, lift two covers. If the two discs match, set both covers aside on the mat;
if not, put both covers back on their discs. Never have more than two discs uncovered (not counting pairs already
cleared). Done when all six covers stand on the mat and both grippers are open.

The scripted demo knows the colours; it plays five turns with both arms (two misses, three matches). PAIRS_DEBUG=N
prints the state every N control steps; a lost trial prints why.

Run the scripted demo:  inspect-robots run --embodiment yam_arena --policy yam_scripted \\
    -E task=tasks/pairs_game/pairs_game.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 \\
    -E cam_height=176 --instruction "..." --scorer task_score --max-steps 2400 --store-frames
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from yam_arena.constants import FINGER_OPEN
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
_DEBUG = int(os.environ.get("PAIRS_DEBUG", "0") or 0)

# -- layout ---------------------------------------------------------------------------------------------------------
COASTER_R, COASTER_H = 0.035, 0.003  # grey coaster: 7 cm across, its rim shows around a centred cover
DISC_R, DISC_H = 0.017, 0.002  # coloured disc on the coaster: 3.4 cm across
COVER_R, COVER_H = 0.030, 0.050  # cover: 6 cm across, 5 cm tall
DISC_TOP = TOP + COASTER_H + DISC_H
SPACING = 0.20  # every pad 20 cm from its nearest neighbours (a triangular grid)
FAR_X, NEAR_X = 0.403, round(0.403 - SPACING * 3**0.5 / 2, 4)  # 40.3 and 23.0 cm
# Pads 1-4: far row, left to right as seen from the robots; 5-6: near row, in the far row's gaps.
PAD_XY = [(FAR_X, y) for y in (0.30, 0.10, -0.10, -0.30)] + [(NEAR_X, y) for y in (0.20, -0.20)]
COLOURS = {"red": (0.85, 0.10, 0.10), "blue": (0.10, 0.30, 0.85), "green": (0.10, 0.65, 0.20)}
PAD_COLOUR = ["red", "green", "blue", "green", "blue", "red"]
MAT_SIZE, MAT_X = (0.09, 0.56, 0.004), 0.12  # 9 x 56 cm, long side across the table, in front of the grid
MAT_TOP = TOP + MAT_SIZE[2]
N = len(PAD_XY)
PADS = range(N)
COVERS = [f"cover_{i + 1}" for i in PADS]

OBJECTS = (
    [Cylinder(f"coaster_{i + 1}", pos=(x, y, TOP + COASTER_H / 2), radius=COASTER_R, height=COASTER_H,
              color=(0.72, 0.72, 0.70)) for i, (x, y) in enumerate(PAD_XY)]
    + [Cylinder(f"disc_{i + 1}", pos=(x, y, TOP + COASTER_H + DISC_H / 2), radius=DISC_R, height=DISC_H,
                color=COLOURS[PAD_COLOUR[i]]) for i, (x, y) in enumerate(PAD_XY)]
    + [Box("mat", pos=(MAT_X, 0.0, TOP + MAT_SIZE[2] / 2), size=MAT_SIZE, color=(0.93, 0.93, 0.90))]
    + [Cylinder(c, pos=(x, y, DISC_TOP + COVER_H / 2), radius=COVER_R, height=COVER_H, mass=0.05,
                color=(0.20, 0.21, 0.24)) for c, (x, y) in zip(COVERS, PAD_XY)]
)  # fmt: skip  (no mass: fixed in place, glued to the table)

# -- the game's state -------------------------------------------------------------------------------------------------
COVER_MARGIN = COVER_R - DISC_R  # 1.3 cm: a cover centred within this of a disc hides all of it
LIFT_TOL = 0.005  # the cover's bottom at most 5 mm above the disc's top
TILT_COVERING = np.radians(10)  # and tilted at most 10 deg (no peeking under a tipped cover)
TILT_FALLEN = np.radians(45)  # a cover tilted further than this has been knocked over
GRIP_OPEN = 0.04  # each finger at least 4 cm open: wider than a cover, so nothing is held
_STATE = {"lost": "", "grippers": {"left": 0.0, "right": 0.0}}


def _axis(s) -> np.ndarray:
    """The cover's axis (unit, world) from its quaternion."""
    x, y, z, w = np.asarray(s.quat_xyzw, dtype=float) / np.linalg.norm(s.quat_xyzw)
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def tilt(s) -> float:
    return float(np.arccos(min(abs(_axis(s)[2]), 1.0)))


def _bottom(s) -> np.ndarray:
    a = _axis(s)
    return s.pos - np.sign(a[2] or 1.0) * a * COVER_H / 2  # centre of the lower end face


def covered(i, states) -> bool:
    """Disc i is hidden: some cover stands on it, centred within 1.3 cm, nearly upright."""
    for c in COVERS:
        s = states[c]
        b = _bottom(s)
        if (np.linalg.norm(b[:2] - PAD_XY[i]) <= COVER_MARGIN and b[2] <= DISC_TOP + LIFT_TOL
                and tilt(s) <= TILT_COVERING):  # fmt: skip
            return True
    return False


def on_mat(s) -> bool:
    """The cover stands upright on the mat: its bottom on the mat's top, its centre over the mat."""
    b = _bottom(s)
    inside = abs(b[0] - MAT_X) <= MAT_SIZE[0] / 2 and abs(b[1]) <= MAT_SIZE[1] / 2
    return bool(inside and abs(b[2] - MAT_TOP) <= LIFT_TOL and tilt(s) <= TILT_COVERING)


def game(states) -> dict:
    """Uncovered discs, matched pairs (both discs of a colour uncovered), singles (uncovered, partner covered)."""
    up = [i for i in PADS if not covered(i, states)]
    matched = sorted({PAD_COLOUR[i] for i in up if sum(PAD_COLOUR[j] == PAD_COLOUR[i] for j in up) == 2})
    singles = [i for i in up if PAD_COLOUR[i] not in matched]
    return {"up": up, "matched": matched, "singles": singles, "on_mat": sum(on_mat(states[c]) for c in COVERS)}


def violation(states) -> str:
    """Why this state breaks the rules ('' if it does not)."""
    g = game(states)
    cleared = min(len(g["matched"]), g["on_mat"] // 2)
    open_discs = len(g["up"]) - 2 * cleared  # uncovered discs, not counting pairs already cleared to the mat
    if open_discs > 2:
        return f"{open_discs} discs uncovered at once (pads {', '.join(str(i + 1) for i in g['up'])})"
    if g["on_mat"] > 2 * len(g["matched"]):
        return "a cover was set aside on the mat without its disc's match uncovered"
    for c in COVERS:
        s = states[c]
        if tilt(s) > TILT_FALLEN:
            return f"{c} was knocked over"
        if s.low[2] < TOP - 0.01:
            return f"{c} fell off the table"
    return ""


_calls = [0]


def on_reset(env) -> None:
    _STATE.update(lost="", grippers={"left": 0.0, "right": 0.0})
    _calls[0] = 0


def on_step(env, states) -> None:
    _calls[0] += 1
    for arm in ("left", "right"):
        q = yampatch.joint_positions(env, f"{arm}_robot")
        _STATE["grippers"][arm] = min(abs(q["joint7"]), abs(q["joint8"]))
    why = violation(states)
    if why and not _STATE["lost"]:
        _STATE["lost"] = why
        print(f"[pairs step {_calls[0]}] lost: {why}", flush=True)
    if _DEBUG and _calls[0] % _DEBUG == 0:
        g, gr = game(states), _STATE["grippers"]
        print(f"[pairs step {_calls[0]}] uncovered {[i + 1 for i in g['up']]} matched {g['matched']} on mat "
              f"{g['on_mat']} grippers {gr['left'] * 1000:.0f}/{gr['right'] * 1000:.0f} mm lost {_STATE['lost']!r} "
              f"score {score(states)}", flush=True)  # fmt: skip


yampatch.install(on_reset=on_reset, on_step=on_step)


# -- scoring -------------------------------------------------------------------------------------------------------
def pairs_done(objects) -> int:
    """Matched pairs whose covers are on the mat (2 covers on the mat per pair)."""
    g = game(objects)
    return min(len(g["matched"]), g["on_mat"] // 2)


def grippers_open() -> bool:
    return all(v >= GRIP_OPEN for v in _STATE["grippers"].values())


def score(objects):
    """0 once lost; 0.3 per pair cleared; 1.0 with all three cleared and both grippers open."""
    if _STATE["lost"]:
        return 0.0
    done = pairs_done(objects)
    return 1.0 if done == 3 and grippers_open() else round(0.3 * done, 2)


SUCCESS_HOLD_S = 3.0  # ends 3 s after success, once the arms have moved aside


def success(objects):
    return score(objects) == 1.0 and all(objects[c].resting() for c in COVERS)


def distance(objects):
    """Covers still to put on the mat."""
    return float(N - game(objects)["on_mat"])


# -- demo -----------------------------------------------------------------------------------------------------------
# Five turns; in each, the left arm lifts a cover on its side (pads 1, 2, 5) and the right arm one on its side (3, 4,
# 6) at the same time, 4 cm up. The first turn lifts the two neighbouring inner pads of the far row. A miss: both set
# them back. A match: the left arm carries its cover to its next slot on the mat and moves aside, then the right arm
# does (near the mat both forearms would collide); the fingers close along x throughout, clear of the neighbouring
# covers.
Z_GRIP = DISC_TOP + 0.025 + 0.013  # fingertips 2.5 cm above the cover's bottom: level with its middle
Z_LOOK = Z_GRIP + 0.04
Z_HOVER = TOP + 0.09  # fingertips 2.2 cm above the covers
Z_CARRY = TOP + 0.13  # a carried cover's bottom 4.4 cm above the standing covers' tops
PINCH = -(COVER_R - 0.006)  # 6 mm into the cover each side: a firm grip
SLOTS = {"left": [0.09, 0.16, 0.23], "right": [-0.09, -0.16, -0.23]}
YAW = {"left": -np.pi / 2, "right": np.pi / 2}  # fingers close along x, on the grid and on the mat
PARK = {"left": (0.13, 0.27, TOP + 0.15), "right": (0.13, -0.27, TOP + 0.15)}  # also where an arm waits
TURNS = [(1, 2), (4, 3), (4, 2), (1, 3), (0, 5)]  # (left pad, right pad): green/blue, blue/green, then the pairs


def _wp(t, pos, finger, label, yaw=0.0):
    return Waypoint(t, tuple(float(v) for v in pos), finger, label, float(yaw))


def lift(arm, i):
    x, y = PAD_XY[i]
    yaw = YAW[arm]
    return [
        _wp(1.6, (x, y, Z_HOVER), FINGER_OPEN, f"pad {i + 1}: above its cover", yaw),
        _wp(0.8, (x, y, Z_GRIP), FINGER_OPEN, "descend", yaw),
        _wp(0.5, (x, y, Z_GRIP), PINCH, "grip", yaw),
        _wp(0.6, (x, y, Z_LOOK), PINCH, "lift: look", yaw),
        _wp(0.8, (x, y, Z_LOOK), PINCH, "hold", yaw),
    ]


def put_back(arm, i):
    x, y = PAD_XY[i]
    yaw = YAW[arm]
    return [
        _wp(0.6, (x, y, Z_GRIP + 0.0005), PINCH, "no match: set it back", yaw),
        _wp(0.4, (x, y, Z_GRIP), FINGER_OPEN, "let go", yaw),
        _wp(0.6, (x, y, Z_HOVER), FINGER_OPEN, "up", yaw),
    ]


def set_aside(arm, i, slot_y):
    z_set = Z_GRIP + (MAT_TOP - DISC_TOP) + 0.001
    yaw = YAW[arm]
    x, y = PAD_XY[i]
    return [
        _wp(0.6, (x, y, Z_HOVER), PINCH, "match: lift", yaw),
        _wp(0.9, ((x + MAT_X) / 2 + 0.03, (y + slot_y) / 2, Z_CARRY), PINCH, "up over the other covers", yaw),
        _wp(1.0, (MAT_X, slot_y, Z_CARRY), PINCH, "over the mat", yaw),
        _wp(0.8, (MAT_X, slot_y, z_set), PINCH, "set it down", yaw),
        _wp(0.4, (MAT_X, slot_y, z_set), FINGER_OPEN, "let go", yaw),
        _wp(0.6, (MAT_X, slot_y, Z_HOVER), FINGER_OPEN, "up", yaw),
        _wp(1.0, PARK[arm], FINGER_OPEN, "aside"),
    ]


def _duration(wps):
    return sum(w.duration for w in wps)


def demo():
    plans = {"left": [], "right": []}
    used = {"left": 0, "right": 0}
    for li, ri in TURNS:
        lifts = {"left": lift("left", li), "right": lift("right", ri)}
        for arm in plans:
            plans[arm] += lifts[arm]
        if PAD_COLOUR[li] == PAD_COLOUR[ri]:  # a match: left, then right, to the mat
            aside = {arm: set_aside(arm, i, SLOTS[arm][used[arm]]) for arm, i in (("left", li), ("right", ri))}
            used = {arm: used[arm] + 1 for arm in used}
            hold = plans["right"][-1]
            plans["left"] += aside["left"] + [_wp(_duration(aside["right"]), PARK["left"], FINGER_OPEN, "wait")]
            plans["right"] += [_wp(_duration(aside["left"]), hold.pos, PINCH, "hold the match up", hold.yaw)]
            plans["right"] += aside["right"]
        else:
            for arm, i in (("left", li), ("right", ri)):
                plans[arm] += put_back(arm, i)
    for arm in plans:
        plans[arm] += [_wp(1.5, PARK[arm], FINGER_OPEN, "park")]
    return plans
