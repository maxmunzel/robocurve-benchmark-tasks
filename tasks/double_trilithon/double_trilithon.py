"""On the station table: six identical wooden blocks, 1 x 2 x 8 cm each. Four stand upright on their 1 x 2 cm ends,
two on each side of the table; two lie flat on their 2 x 8 cm faces, further away, long side across the table.

Build two Stonehenge-style trilithons, one on top of the other: two upright blocks standing on the table with an
opening of at least 2 cm between them and one block lying flat across their tops (the lintel); then, on that lintel,
two more upright blocks with an opening of at least 2 cm and a second flat lintel across their tops. The finished
structure is 18 cm tall (8 + 1 + 8 + 1 cm). Let go of every block at the end: both grippers open.

Run the scripted demo:  inspect-robots run --embodiment yam_arena --policy yam_scripted \\
    -E task=tasks/double_trilithon/double_trilithon.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 \\
    -E cam_height=176 --instruction "..." --scorer task_score --max-steps 2400 --store-frames
"""

from __future__ import annotations

import importlib.util
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
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
_DEBUG = int(os.environ.get("TRILITHON_DEBUG", "0") or 0)  # print the structure every N steps

# -- the blocks: 1 x 2 x 8 cm, wood --------------------------------------------------------------------------------
THIN, WIDE, LONG = 0.01, 0.02, 0.08
MASS = 0.010  # kg: 16 cm^3 of pine
WOOD = (0.45, 0.46, 0.50)  # painted stone grey: contrasts with the light table
UPRIGHT = (WIDE, THIN, LONG)  # x, y, z: standing on its 1 x 2 cm end, its 1 cm side across the table
FLAT = (WIDE, LONG, THIN)  # lying on its 2 x 8 cm face, long side across the table (along y)


def _block(name, xy, size):
    return Box(name, pos=(xy[0], xy[1], TOP + size[2] / 2), size=size, mass=MASS, color=WOOD)


START_UPRIGHT = {
    "block_1": (0.20, 0.19), "block_2": (0.30, 0.19),  # left of centre, in front of the left arm
    "block_3": (0.20, -0.19), "block_4": (0.30, -0.19),  # right of centre
}  # fmt: skip
START_FLAT = {"block_5": (0.31, 0.09), "block_6": (0.31, -0.09)}
OBJECTS = [_block(n, xy, UPRIGHT) for n, xy in START_UPRIGHT.items()] + [
    _block(n, xy, FLAT) for n, xy in START_FLAT.items()
]
BLOCKS = [o.name for o in OBJECTS]
START = {o.name: np.array(o.pos) for o in OBJECTS}

# -- scoring ------------------------------------------------------------------------------------------------------
GOAL_HEIGHT = 2 * (LONG + THIN)  # 18 cm: the top of the upper lintel above the table
TOL = 0.005  # height tolerances (m)
MIN_OPENING = 0.02  # clear gap between the two uprights of a trilithon
STANDS = LONG - 0.005  # bounding box at least this tall: an upright block (within about 7 deg of vertical)
LIES = 0.015  # bounding box at most this tall, and at least STANDS long: a flat block (within about 3 deg)
GRIP_OPEN = 0.02  # each finger at least this far open (pads 4 cm apart, twice a block's width): not holding a block
_GRIPPERS = {"left": 0.0, "right": 0.0}  # finger opening (m) per arm, updated every step


def _ext(s):
    return s.high - s.low


def upright(s) -> bool:
    return bool(_ext(s)[2] >= STANDS)


def flat(s) -> bool:
    e = _ext(s)
    return bool(e[2] <= LIES and max(e[0], e[1]) >= STANDS)


def overlaps_xy(a, b) -> bool:
    return bool(np.all(a.low[:2] < b.high[:2]) and np.all(b.low[:2] < a.high[:2]))


def rests_on(a, b) -> bool:
    """``a``'s bottom within TOL of ``b``'s top, directly above it."""
    return bool(abs(a.low[2] - b.high[2]) < TOL and overlaps_xy(a, b))


def opening(a, b) -> float:
    """Clear horizontal gap between two bounding boxes (m); <= 0 if their footprints overlap."""
    gap = np.maximum(a.low[:2] - b.high[:2], b.low[:2] - a.high[:2])
    return float(np.max(gap))


def _pairs(objects, names, base):
    """Pairs of upright blocks among ``names`` standing on ``base`` (a height, or a block they both rest on), with an
    opening of at least MIN_OPENING."""
    def on(n):
        s = objects[n]
        if not upright(s):
            return False
        return abs(s.low[2] - base) < TOL if isinstance(base, float) else rests_on(s, base)

    stand = [n for n in names if on(n)]
    return [(a, b) for a, b in combinations(stand, 2) if opening(objects[a], objects[b]) >= MIN_OPENING]


def stage(objects) -> int:
    """How far the structure got: 0 nothing; 1 the lower trilithon; 2 plus one upright on its lintel; 3 plus a second
    upright with a 2 cm opening; 4 plus the upper lintel, its top at 18 cm."""
    best = 0
    names = set(objects)
    for a, b in _pairs(objects, names, float(TOP)):
        for low in names - {a, b}:
            lintel = objects[low]
            if not (flat(lintel) and rests_on(lintel, objects[a]) and rests_on(lintel, objects[b])):
                continue
            best = max(best, 1)
            rest = names - {a, b, low}
            if any(upright(objects[n]) and rests_on(objects[n], lintel) for n in rest):
                best = max(best, 2)
            for c, d in _pairs(objects, rest, lintel):
                best = max(best, 3)
                for up in rest - {c, d}:
                    s = objects[up]
                    if (flat(s) and rests_on(s, objects[c]) and rests_on(s, objects[d])
                            and abs(s.high[2] - (TOP + GOAL_HEIGHT)) < TOL):  # fmt: skip
                        return 4
    return best


def grippers_open() -> bool:
    return all(v >= GRIP_OPEN for v in _GRIPPERS.values())


def topmost_height(objects) -> float:
    """Height of the highest block's top above the table (m)."""
    return float(max(objects[n].high[2] for n in BLOCKS) - TOP)


_calls = [0]


def on_reset(env) -> None:
    _GRIPPERS.update(left=0.0, right=0.0)
    _calls[0] = 0


def on_step(env, states) -> None:
    for arm in ("left", "right"):
        q = yampatch.joint_positions(env, f"{arm}_robot")
        _GRIPPERS[arm] = min(abs(q["joint7"]), abs(q["joint8"]))
    _calls[0] += 1
    if _DEBUG and _calls[0] % _DEBUG == 0:
        objs = {n: states[n] for n in BLOCKS}
        print(f"[trilithon step {_calls[0]}] stage {stage(objs)} top {topmost_height(objs) * 100:.1f} cm "
              f"grippers {_GRIPPERS['left'] * 1000:.0f}/{_GRIPPERS['right'] * 1000:.0f} mm", flush=True)  # fmt: skip


yampatch.install(on_reset=on_reset, on_step=on_step)


def success(objects):
    """The topmost block is the upper lintel, its top at 18 cm, every block still, both grippers open."""
    objs = {n: objects[n] for n in BLOCKS}
    return (stage(objs) == 4 and abs(topmost_height(objs) - GOAL_HEIGHT) < TOL and grippers_open()
            and all(s.resting() for s in objs.values()))  # fmt: skip


def score(objects):
    """0.25 per stage reached, with both grippers open; 0 while either gripper is closed."""
    if not grippers_open():
        return 0.0
    objs = {n: objects[n] for n in BLOCKS}
    s = stage(objs)
    if s == 4 and abs(topmost_height(objs) - GOAL_HEIGHT) >= TOL:  # a block above the upper lintel
        s = 3
    return s / 4


def distance(objects):
    """How far the topmost block's top is from 18 cm (m)."""
    return abs(topmost_height({n: objects[n] for n in BLOCKS}) - GOAL_HEIGHT)


# -- demo: one arm at a time; the left arm builds the left uprights, the right arm the right ones ---------------------
C = (0.22, 0.0)  # the structure's centre: where both arms reach the 18 cm top straight down
LEG_Y = (LONG - THIN) / 2  # uprights under the lintel's ends: 3.5 cm either side, a 6 cm opening
TRAVEL_Z = TOP + 0.20
UP_GRIP = 0.06  # TCP above an upright's bottom when holding it (the pad tips 4.7 cm above it)
FLAT_GRIP = 0.016  # TCP above a flat block's bottom (pad tips 3 mm above it)
DROP = 0.002  # let go this far above the support
RELEASE = -0.015  # finger target while letting go: pads 3 cm apart, clear of the block and its neighbours
YAW = {"left": -np.pi / 2, "right": np.pi / 2}  # flat blocks: fingers close across the 2 cm width (along x)


def _wp(t, pos, finger, label, yaw=0.0):
    return Waypoint(t, tuple(float(v) for v in pos), finger, label, float(yaw))


def move(name, goal_xy, bottom, grip, yaw=0.0, travel=TRAVEL_Z):
    """Pick ``name`` straight down, carry it at ``travel`` and set its bottom on ``bottom`` (world z) at ``goal_xy``."""
    (px, py, pz), (gx, gy) = START[name], goal_xy
    p_low = pz - (LONG if grip == UP_GRIP else THIN) / 2 + grip
    g_low = bottom + grip + DROP
    return [
        _wp(1.5, (px, py, travel), FINGER_OPEN, f"{name}: to it", yaw),
        _wp(1.5, (px, py, p_low), FINGER_OPEN, "descend", yaw),
        _wp(0.8, (px, py, p_low), FINGER_CLOSED, "grip", yaw),
        _wp(1.5, (px, py, travel), FINGER_CLOSED, "lift", yaw),
        _wp(2.0, (gx, gy, travel), FINGER_CLOSED, "carry", yaw),
        _wp(2.0, (gx, gy, g_low), FINGER_CLOSED, "lower", yaw),
        _wp(0.6, (gx, gy, g_low), FINGER_CLOSED, "settle", yaw),
        _wp(0.8, (gx, gy, g_low), RELEASE, "let go", yaw),
        _wp(1.2, (gx, gy, travel), RELEASE, "up", yaw),
    ]


def park(arm, seconds=2.0):
    y = 0.28 if arm == "left" else -0.28
    return [_wp(seconds, (0.15, y, TOP + 0.16), FINGER_OPEN, "park")]


def _duration(waypoints):
    return sum(w.duration for w in waypoints)


def demo():
    cx, cy = C
    base, upper = TOP, TOP + LONG + THIN
    steps = [  # (arm, waypoints), played one after the other
        ("left", move("block_1", (cx, cy + LEG_Y), base, UP_GRIP) + park("left")),
        ("right", move("block_3", (cx, cy - LEG_Y), base, UP_GRIP) + park("right")),
        ("left", move("block_5", (cx, cy), base + LONG, FLAT_GRIP, YAW["left"]) + park("left")),
        ("left", move("block_2", (cx, cy + LEG_Y), upper, UP_GRIP) + park("left")),
        ("right", move("block_4", (cx, cy - LEG_Y), upper, UP_GRIP) + park("right")),
        ("right", move("block_6", (cx, cy), upper + LONG, FLAT_GRIP, YAW["right"]) + park("right")),
    ]
    plan = {"left": park("left", 1.0), "right": park("right", 1.0)}
    for arm, wps in steps:
        other = "right" if arm == "left" else "left"
        plan[arm] += wps
        plan[other] += park(other, _duration(wps))
    return plan
