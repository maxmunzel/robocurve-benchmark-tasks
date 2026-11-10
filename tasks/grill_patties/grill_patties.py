"""Grill two patties. On the station table: a flat grill plate in the middle; to its left a pink raw-meat plate with
two raw patties; to its right a white plate for done patties; in front, two spatulas: a flat turner (right arm) and
an upright scraper standing in a holder (left arm). A patty is four short cylinders welded side by side, so its
rounded sides face the arms and a spatula blade slides under it from the side.

Sear each side of each patty on the grill for 20-40 s, then serve it on the white plate; a patty with more than 40 s
on a side is burnt and must not be served. Two minutes in all. The scorer also enforces safety and hygiene rules
that the agent is not told (see the README): only the spatulas may touch the patties, the robot may not touch the
grill or the plates, patties may not touch the table, a patty that has been on the grill may not touch the raw-meat
plate or a raw patty, and nothing raw or burnt may be served.

The scripted demo knows the layout and the clock. The right arm does the turner work from the patty's right side:
it scoops each patty from the raw plate, sets it on the grill, tips it over after 28 s and serves it after another
28 s. The left arm stands the scraper beside the patty as a stop: on its far side while the turner slides under it
or tips it up (it then lays the patty down on its other side), on its near side while the turner is pulled out from
under it. GRILL_DEBUG=N prints the cooking state every N control steps (GRILL_DEBUG_TOOLS=1 adds the spatulas);
a lost trial prints why. GRILL_TEST=load or flip runs a shortened demo with one patty.

Run the scripted demo:  inspect-robots run --embodiment yam_arena --policy yam_scripted \\
    -E task=tasks/grill_patties/grill_patties.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 \\
    -E cam_height=360 --instruction "..." --scorer task_score --max-steps 2400 --store-frames
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from yam_arena import demo as _demo
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
_DEBUG = int(os.environ.get("GRILL_DEBUG", "0") or 0)

# -- layout ---------------------------------------------------------------------------------------------------------
# The arms work from the sides: the right arm slides its turner under a patty from the right (+y direction), the
# left arm stands its scraper upright beside the patty as a stop. So the patties' rounded sides face the arms. The
# layout is packed toward the middle, where both arms reach.
SLAB_H = 0.01  # grill plate and plates: 1 cm thick
GRILL = ((0.25, 0.1415), (0.22, 0.179))  # centre xy, size xy: x 14-36 cm, y +5.2..+23.1 cm
RAW_PLATE = ((0.21, 0.29), (0.18, 0.10))  # left of the grill: x 12-30 cm, y +24..+34 cm
DONE_PLATE = ((0.215, -0.005), (0.18, 0.11))  # right of the grill: x 12.5-30.5 cm, y -6..+5 cm
SURFACE = TOP + SLAB_H  # top of the grill and the plates

# A patty is four cylinders welded side by side. In its own frame the cylinders lie along y, side by side along x;
# it is placed turned 90 deg, so on the table the cylinders run along x (away from the robot) and their rounded
# sides face the arms (+-y).
PATTY_R, PATTY_W = 0.009, 0.06  # each cylinder 1.8 cm across, 6 cm long
PATTY_N = 4
PATTY_D = 2 * PATTY_R * PATTY_N  # 7.2 cm across the cylinders (along y on the table)
PATTY_CX = [(-PATTY_D / 2 + PATTY_R + 2 * PATTY_R * i) for i in range(PATTY_N)]  # cylinder centres, own frame
PATTY_MASS = 0.07
PATTY_FRICTION = 1.0  # sticky raw meat: 1.1 against the grill and plates (averaged with their 1.2)
PATTIES = [f"patty_{i + 1}" for i in range(2)]
PATTY_START = [(0.17, 0.29), (0.25, 0.29)]  # on the raw plate, one behind the other
YAW90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # quaternion xyzw: turned 90 deg about z

# Two spatulas. The right one (a turner, used flat): a flat blade, a low wall on its back edge and a raised neck to a
# vertical handle 13 cm further back, so the hand stays away from the patty. Body frame: origin at the bottom of the
# blade's back edge, x along the blade; it is placed turned 90 deg, blade pointing to the left (+y), its handle resting
# on a block. The left one (a scraper, used upright as a stop): a tall flat blade with a handle straight above it, so
# the left hand holds it with its normal downward grip, blade hanging down. Body frame: origin at the middle of the
# blade's bottom edge; it stands in a slotted holder, blade along x.
BLADE = (0.10, 0.066, 0.0015)  # turner: x 0..10 cm, 6.6 cm wide, 1.5 mm thick
HANDLE = (0.03, 0.02, 0.107)  # turner: x -16..-13 cm, 2 cm wide, z 3.5..11 cm
HANDLE_Z0 = 0.003
BLADE_FRICTION = 0.3  # the turner's blade top: smooth enough to slide under a patty, grippy enough to carry it
BLADE_SOLE = 0.0005  # its bottom 0.5 mm are slick (0.05), so the blade slides along the grill
HANDLE_OFFSET, HANDLE_FOOT = 0.16, 0.035  # the handle's back face behind the blade's back edge; its bottom
STOP = (0.08, 0.003, 0.11)  # scraper: 8 cm along x, 3 mm thick, 11 cm tall (a patty on its edge is 7.2 cm)
STOP_HANDLE = (0.04, 0.02, 0.03)  # a grippy block over the scraper's top 3 cm, where the hand holds it
SPATULAS = {"left": "spatula_left", "right": "spatula_right"}
REST = (0.04, 0.04, HANDLE_FOOT)  # a block the turner's handle rests on at the start, so it lies level
HOLDER = (0.09, 0.01, 0.03)  # each of the holder's two jaws; 2 mm gap either side of the scraper's blade
SPATULA_START = {"left": (0.08, 0.12), "right": (0.08, -0.12)}  # scraper: blade middle; turner: blade back edge

COLORS = {"grill": (0.16, 0.16, 0.17), "raw_plate": (0.93, 0.78, 0.78), "done_plate": (0.95, 0.95, 0.93),
          "patty": (0.80, 0.38, 0.40), "blade": (0.75, 0.76, 0.78),
          "handle": (0.12, 0.12, 0.12)}  # fmt: skip


def _slab(name, centre_size, color):  # no mass: fixed in place
    (x, y), (sx, sy) = centre_size
    return Box(name, pos=(x, y, TOP + SLAB_H / 2), size=(sx, sy, SLAB_H), color=color)


# -- generated USD: patties and spatulas (one rigid body each, several colliders with their own friction) ------------
def _write_body(name: str, parts: list[dict]) -> str:
    """A rigid body of boxes and y-axis cylinders. Each part: kind, pos, size (box) or r/h (cylinder), color,
    density (kg/m^3), friction. Returns the USD path. Runs after Isaac Sim has started."""
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    path = Path(tempfile.gettempdir()) / "yam_tasks" / f"{name}.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    materials = {}
    for i, p in enumerate(parts):
        mu = float(p["friction"])
        if mu not in materials:
            mat = UsdShade.Material.Define(stage, f"/{name}/physics_material_{len(materials)}")
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(mu)
            api.CreateDynamicFrictionAttr(mu)
            api.CreateRestitutionAttr(0.0)
            if p.get("combine"):
                PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr(p["combine"])
            materials[mu] = mat
        prim_path = f"/{name}/part_{i}"
        if p["kind"] == "box":
            geom = UsdGeom.Cube.Define(stage, prim_path)
            geom.CreateSizeAttr(1.0)
            geom.AddTranslateOp().Set(Gf.Vec3d(*p["pos"]))
            geom.AddScaleOp().Set(Gf.Vec3f(*p["size"]))
        else:
            geom = UsdGeom.Cylinder.Define(stage, prim_path)
            geom.CreateAxisAttr("Y")
            geom.CreateRadiusAttr(float(p["r"]))
            geom.CreateHeightAttr(float(p["h"]))
            geom.AddTranslateOp().Set(Gf.Vec3d(*p["pos"]))
        geom.CreateDisplayColorAttr([Gf.Vec3f(*p["color"])])
        prim = geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(float(p["density"]))
        coll = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        coll.CreateContactOffsetAttr(0.002)
        coll.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            materials[mu], UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Export(str(path))
    return str(path)


def _build_patty(name):
    vol = PATTY_N * math.pi * PATTY_R**2 * PATTY_W
    parts = [dict(kind="cyl", pos=(cx, 0.0, 0.0), r=PATTY_R, h=PATTY_W, color=COLORS["patty"],
                  density=PATTY_MASS / vol, friction=PATTY_FRICTION) for cx in PATTY_CX]  # fmt: skip
    return lambda: _write_body(name, parts)


def _build_spatula(name, arm):
    if arm == "left":
        parts = [
            dict(kind="box", pos=(0.0, 0.0, (STOP[2] - STOP_HANDLE[2]) / 2),
                 size=(STOP[0], STOP[1], STOP[2] - STOP_HANDLE[2]), color=COLORS["blade"], density=2700.0,
                 friction=0.1, combine="min"),
            dict(kind="box", pos=(0.0, 0.0, STOP[2] - STOP_HANDLE[2] / 2), size=STOP_HANDLE, color=COLORS["handle"],
                 density=450.0, friction=2.0),
        ]  # fmt: skip
        return lambda: _write_body(name, parts)
    bx, by, bz = BLADE
    hx, hy, hz = HANDLE
    off, z0 = HANDLE_OFFSET, HANDLE_FOOT
    parts = [
        dict(kind="box", pos=(bx / 2, 0.0, BLADE_SOLE / 2), size=(bx, by, BLADE_SOLE), color=COLORS["blade"],
             density=2700.0, friction=0.05, combine="min"),  # slick underside: slides on the grill and plates
        dict(kind="box", pos=(bx / 2, 0.0, (bz + BLADE_SOLE) / 2), size=(bx, by, bz - BLADE_SOLE),
             color=COLORS["blade"], density=2700.0, friction=BLADE_FRICTION, combine="min"),  # top: holds a patty
        dict(kind="box", pos=(-off + hx / 2, 0.0, (z0 + HANDLE_Z0 + hz) / 2), size=(hx, hy, HANDLE_Z0 + hz - z0),
             color=COLORS["handle"], density=450.0, friction=2.0),
        dict(kind="box", pos=(-0.005, 0.0, bz + 0.0215), size=(0.01, 0.04, 0.043), color=COLORS["blade"],
             density=2700.0, friction=0.1, combine="min"),  # wall on the blade's back edge
        dict(kind="box", pos=(-(off - hx + 0.01) / 2 - 0.005, 0.0, 0.04), size=(off - hx - 0.01, 0.015, 0.01),
             color=COLORS["blade"], density=2700.0, friction=0.1, combine="min"),  # neck, above a patty
    ]  # fmt: skip
    return lambda: _write_body(name, parts)


_HX, _HY = SPATULA_START["left"]
_JAW_Y = STOP[1] / 2 + 0.002 + HOLDER[1] / 2
OBJECTS = (
    [_slab("grill", GRILL, COLORS["grill"]), _slab("raw_plate", RAW_PLATE, COLORS["raw_plate"]),
     _slab("done_plate", DONE_PLATE, COLORS["done_plate"])]
    + [Box("spatula_rest", pos=(SPATULA_START["right"][0], SPATULA_START["right"][1] - HANDLE_OFFSET + HANDLE[0] / 2,
                                TOP + REST[2] / 2), size=REST, color=(0.35, 0.36, 0.40))]
    + [Box(f"holder_{i}", pos=(_HX, _HY + s * _JAW_Y, TOP + HOLDER[2] / 2), size=HOLDER, color=(0.35, 0.36, 0.40))
       for i, s in enumerate((-1, 1))]
    + [yampatch.usd_object(n, (x, y, SURFACE + PATTY_R), kind="rigid", build=_build_patty(n), mass=PATTY_MASS,
                           extent=PATTY_D, rot_xyzw=YAW90) for n, (x, y) in zip(PATTIES, PATTY_START)]
    + [yampatch.usd_object(SPATULAS["right"], (*SPATULA_START["right"], TOP), kind="rigid",
                           build=_build_spatula(SPATULAS["right"], "right"), mass=0.07, extent=0.13, rot_xyzw=YAW90),
       yampatch.usd_object(SPATULAS["left"], (_HX, _HY, TOP + 0.0005), kind="rigid",
                           build=_build_spatula(SPATULAS["left"], "left"), mass=0.07, extent=0.11)]
)  # fmt: skip
SLABS = {o.name: (np.array(o.pos), np.array(o.size)) for o in OBJECTS if isinstance(o, Box)}
FOOD_SURFACES = ("grill", "raw_plate", "done_plate")  # the robot must not touch these

# -- rules ----------------------------------------------------------------------------------------------------------
COOK_MIN, COOK_MAX = 20.0, 40.0  # s per side
TOUCH = 0.002  # surfaces within 2 mm count as touching
ON_TOL = 0.004  # a patty lies on a surface when its lowest point is within 4 mm of it
FLAT = math.radians(15)  # and it is tilted at most 15 deg


def _quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def _patty_points() -> np.ndarray:
    """A patty's surface, about every 2 mm, in its own frame."""
    pts = []
    a = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    for cx in PATTY_CX:
        for y in np.linspace(-PATTY_W / 2, PATTY_W / 2, 31):
            pts.append(np.c_[cx + PATTY_R * np.cos(a), np.full_like(a, y), PATTY_R * np.sin(a)])
        for y in (-PATTY_W / 2, PATTY_W / 2):
            for r in np.linspace(0, PATTY_R, 6):
                k = max(int(2 * np.pi * r / 0.002), 1)
                b = np.linspace(0, 2 * np.pi, k, endpoint=False)
                pts.append(np.c_[cx + r * np.cos(b), np.full_like(b, y), r * np.sin(b)])
    return np.concatenate(pts)


_PATTY = _patty_points()


def _in_patty(points_local, tol=TOUCH) -> np.ndarray:
    inside = np.zeros(len(points_local), dtype=bool)
    ok_y = np.abs(points_local[:, 1]) <= PATTY_W / 2 + tol
    for cx in PATTY_CX:
        inside |= ok_y & (np.hypot(points_local[:, 0] - cx, points_local[:, 2]) <= PATTY_R + tol)
    return inside


def _in_slab(points, name, tol=TOUCH) -> np.ndarray:
    c, s = SLABS[name]
    return np.all(np.abs(points - c) <= s / 2 + tol, axis=1)


class Patty:
    """One patty's pose (world) and surface points."""

    def __init__(self, s):
        self.pos = np.asarray(s.pos, dtype=float)
        self.rot = _quat_matrix(s.quat_xyzw)
        self.points = _PATTY @ self.rot.T + self.pos
        self.low = float(self.points[:, 2].min())
        self.up = self.rot[:, 2]  # the patty's local z axis
        self.tilt = float(math.acos(min(abs(self.up[2]), 1.0)))
        self.state = s

    def local(self, points):
        return (points - self.pos) @ self.rot

    def flat_on(self, name) -> bool:
        c, s = SLABS[name]
        over = bool(np.all(np.abs(self.pos[:2] - c[:2]) <= s[:2] / 2))
        return over and abs(self.low - (c[2] + s[2] / 2)) <= ON_TOL and self.tilt <= FLAT

    def down_side(self) -> int:
        """0: the side facing down at the start is down; 1: the other side is."""
        return 0 if self.up[2] > 0 else 1



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
        k = max(int(area.sum() / 4e-6), 1)
        idx = rng.choice(len(t), k, p=area / area.sum())
        a, b = rng.random((2, k, 1))
        flip = (a + b) > 1
        a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
        pts = t[idx, 0] + a * (t[idx, 1] - t[idx, 0]) + b * (t[idx, 2] - t[idx, 0])
        points[link.get("name")] = np.concatenate([pts, t.reshape(-1, 3)])
    return points


_POINTS: dict[str, np.ndarray] = {}


def _robot_surface(env) -> np.ndarray:
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


# -- per-step tracking ----------------------------------------------------------------------------------------------
_STATE: dict = {}


def _fresh_state():
    return {"lost": "", "cook": {n: [0.0, 0.0] for n in PATTIES}, "grilled": {n: False for n in PATTIES}, "t": 0.0}


_STATE.update(_fresh_state())
_calls = [0]


def cooked(name) -> str:
    """'raw' (a side under 20 s), 'done' (both sides 20-40 s) or 'burnt' (a side over 40 s)."""
    a, b = _STATE["cook"][name]
    if max(a, b) > COOK_MAX:
        return "burnt"
    return "done" if min(a, b) >= COOK_MIN else "raw"


def violation(patties: dict[str, Patty], robot) -> str:
    """Why this state loses the trial ('' if it does not)."""
    if robot is not None:
        for name, p in patties.items():
            near = robot[np.all(np.abs(robot - p.pos) < PATTY_D, axis=1)]
            if len(near) and _in_patty(p.local(near)).any():
                return f"the robot touched {name} (only the spatulas may)"
        for s in FOOD_SURFACES:
            hit = _in_slab(robot, s)
            if hit.any():
                where = np.round(robot[hit].mean(axis=0) * 100, 1)
                return f"the robot touched the {s.replace('_', ' ')} (at {where} cm)"
    for name, p in patties.items():
        if p.low < TOP + TOUCH:
            return f"{name} touched the table"
        if _STATE["grilled"][name] and _in_slab(p.points, "raw_plate").any():
            return f"{name}, after grilling, touched the raw-meat plate"
        if _STATE["grilled"][name]:
            for other, q in patties.items():
                if other != name and not _STATE["grilled"][other]:
                    near = p.points[np.all(np.abs(p.points - q.pos) < PATTY_D, axis=1)]
                    if len(near) and _in_patty(q.local(near)).any():
                        return f"{name}, after grilling, touched raw {other}"
        if p.flat_on("done_plate") and cooked(name) != "done":
            a, b = _STATE["cook"][name]
            return f"{name} was served {cooked(name)} ({a:.0f} s and {b:.0f} s on its sides)"
    return ""


def on_reset(env) -> None:
    _STATE.update(_fresh_state())
    _GRASP.clear()
    _YAW_FIX.clear()
    _calls[0] = 0


def on_step(env, states) -> None:
    _calls[0] += 1
    dt = float(env.unwrapped.step_dt)
    _STATE["t"] += dt
    patties = {n: Patty(states[n]) for n in PATTIES}
    for n, p in patties.items():
        if p.flat_on("grill"):
            _STATE["cook"][n][p.down_side()] += dt
            _STATE["grilled"][n] = True
    _track_grasps(env)
    if not _STATE["lost"]:
        why = violation(patties, _robot_surface(env))
        if why:
            _STATE["lost"] = why
            print(f"[grill step {_calls[0]} t {_STATE['t']:.1f}] lost: {why}", flush=True)
    if _DEBUG and _calls[0] % _DEBUG == 0:
        desc = ", ".join(f"{n[-1]}: {_STATE['cook'][n][0]:.0f}/{_STATE['cook'][n][1]:.0f} s "
                         f"{'grill' if p.flat_on('grill') else 'done' if p.flat_on('done_plate') else 'raw plate' if p.flat_on('raw_plate') else 'moving'}"
                         for n, p in patties.items())  # fmt: skip
        print(f"[grill step {_calls[0]} t {_STATE['t']:.1f}] {desc}; lost {_STATE['lost']!r} score {score(states)}",
              flush=True)
        for n, p in patties.items():
            yaw = math.degrees(math.atan2(p.rot[1, 0], p.rot[0, 0]))
            print(f"   {n} at {np.round(p.pos[:2] * 100, 1)} cm, {(p.low - SURFACE) * 1000:+.0f} mm over the surface, "
                  f"yaw {yaw:+.0f} tilt {math.degrees(p.tilt):.0f} deg, side {p.down_side() + 1} down", flush=True)
        if os.environ.get("GRILL_DEBUG_TOOLS"):
            for arm, (p_g, r_g) in _GRASP.items():
                g = _STOCK_GRIP(0.0)
                d = g @ r_g  # deviation from the nominal grasp (identity if nominal)
                ang = np.degrees([math.atan2(d[2, 1], d[2, 2]), math.asin(max(-1, min(1, -d[2, 0]))),
                                  math.atan2(d[1, 0], d[0, 0])])
                print(f"   {arm} grasp: origin {np.round((g @ p_g + GRIP[arm]) * 1000, 1)} mm off nominal, "
                      f"rotation rpy {np.round(ang, 1)} deg", flush=True)
            for arm, n in SPATULAS.items():
                pose = env.unwrapped.arena_world.get_pose_w(n)[0].detach().cpu().numpy()
                pos = pose[:3] - env.unwrapped.scene.env_origins[0].detach().cpu().numpy()
                rot = _quat_matrix(pose[3:])
                tip = pos + rot @ TIP[arm]
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, -rot[2, 0]))))  # blade's x axis: + is tip down
                print(f"   {arm} spatula tip {np.round((tip - (0, 0, SURFACE)) * 100, 1)} cm (z over surface) pitch "
                      f"{pitch:+.1f} deg", flush=True)


yampatch.install(on_reset=on_reset, on_step=on_step)


# -- scoring -------------------------------------------------------------------------------------------------------
def _patty_points_score(name, p: Patty) -> float:
    state = cooked(name)
    if state == "done" and p.flat_on("done_plate"):
        return 1 / len(PATTIES)
    if state == "burnt":
        return 0.0
    return 0.1 * sum(COOK_MIN <= t <= COOK_MAX for t in _STATE["cook"][name])


def score(objects):
    """0 once lost; otherwise 1/2 per done patty on the done plate, 0.1 per well-seared side of a patty not yet
    served (0 for a burnt patty), clipped to 0..1."""
    if _STATE["lost"]:
        return 0.0
    total = sum(_patty_points_score(n, Patty(objects[n])) for n in PATTIES)
    return 1.0 if total > 0.999 else round(min(max(total, 0.0), 1.0), 2)


SUCCESS_HOLD_S = 1.0


def success(objects):
    return score(objects) == 1.0 and all(objects[n].resting() for n in PATTIES)


def distance(objects):
    """Patties not yet served done."""
    return float(sum(not (cooked(n) == "done" and Patty(objects[n]).flat_on("done_plate")) for n in PATTIES))


# -- demo -----------------------------------------------------------------------------------------------------------
# The right arm does the turner work, always from the patty's right side (blade pointing +y). The left arm holds the
# scraper upright, blade hanging down, as a stop: on the patty's far (left) side while the turner slides under it
# or tips it over, and on its near side while the turner is pulled out from under it to set it down. Waypoints give
# the spatula's reference point (the turner's tip, the middle of the scraper's bottom edge) and the turner's tilt
# (tip down); the hand's pose follows from how the spatula actually sits in the hand, measured every control step
# while the spatula is in the air, so it goes where it is meant to.
GRIP = {"right": np.array([-HANDLE_OFFSET + HANDLE[0] / 2, 0.0, 0.075]),
        "left": np.array([0.0, 0.0, STOP[2] - 0.012])}  # TCP in the spatula frame; the left pads reach 13 mm lower
TIP = {"right": np.array([BLADE[0], 0.0, 0.0]), "left": np.zeros(3)}
PINCH = {"right": -(HANDLE[0] / 2 - 0.005), "left": -(STOP_HANDLE[1] / 2 - 0.005)}  # 5 mm into each side
SP_YAW = {"right": math.pi / 2, "left": 0.0}  # turner blade points +y; scraper blade along x
SCOOP_TILT = math.radians(6)  # tip down, touching the surface, to wedge under a patty
LIFT_TILT = math.radians(4)  # tip up while carrying: the patty's weight tips the turner down a little in the hand
Z_CARRY = SURFACE + 0.05  # turner's blade bottom while moving
SKIM = 0.0005  # the turner's tip this far above the surface it slides on
STOP_Z = SURFACE + 0.001  # scraper's bottom edge when standing on a surface (just above it)
STOP_Z_ON_BLADE = SURFACE + BLADE[2] + 0.003  # ... when standing on the turner's blade
STOP_GAP = STOP[1] / 2 - 0.0005  # scraper's mid-plane from the patty's side: pressed 0.5 mm in
PARK = {"left": (0.12, 0.31, TOP + 0.17), "right": (0.14, -0.30, TOP + 0.16)}  # TCP, out to the sides
GRILL_X = [0.20, 0.30]  # the patties' places on the grill, along x
PLACE_Y = 0.112  # where a patty is set on the grill (its centre); it flips over its left side to PLACE_Y + PATTY_D
FLIP_Y = PLACE_Y + PATTY_D
DONE_X, DONE_Y = [0.24, 0.165], DONE_PLATE[0][1]
SIDE_S = 28.0  # aim per side
_GRASP: dict = {}  # arm -> (spatula origin in the TCP frame, spatula rotation in the TCP frame)
_YAW_FIX: dict = {}  # arm -> how far (rad) the hand is turned against its heading error (the arm sags under load)


def _track_grasps(env) -> None:
    """Measure how each held spatula sits in its hand (from the arm's joints and the spatula's pose), only while
    it is in the air and steady, so contact forces don't feed back into the measurement."""
    from yam_arena import ik, station

    origin = env.unwrapped.scene.env_origins[0].detach().cpu().numpy()
    for arm, name in SPATULAS.items():
        q = yampatch.joint_positions(env, f"{arm}_robot")
        p_tcp, r_tcp = ik.tcp_pose(np.array([q[f"joint{i}"] for i in range(1, 7)]), station.BASES[arm])
        pose = env.unwrapped.arena_world.get_pose_w(name)[0].detach().cpu().numpy()
        p_s, r_s = pose[:3] - origin, _quat_matrix(pose[3:])
        held = abs(q["joint7"]) < 0.0135 and np.linalg.norm(p_tcp - (p_s + r_s @ GRIP[arm])) < 0.03
        tip_z = (p_s + r_s @ TIP[arm])[2]
        st = env.unwrapped.scene[name].data
        lin = np.linalg.norm(getattr(st.root_lin_vel_w, "torch", st.root_lin_vel_w)[0].detach().cpu().numpy())
        ang = np.linalg.norm(getattr(st.root_ang_vel_w, "torch", st.root_ang_vel_w)[0].detach().cpu().numpy())
        if os.environ.get("GRILL_DEBUG_GRASP") and _calls[0] % 20 == 0:
            print(f"   [grasp {arm}] tcp {np.round(p_tcp, 3)} tip {np.round(p_s + r_s @ TIP[arm], 3)} yaw_s {math.degrees(math.atan2(r_s[1, 0], r_s[0, 0])):.1f} yaw_tcp_x {math.degrees(math.atan2(r_tcp[1, 0], r_tcp[0, 0])):.1f} r_tcp_z {np.round(r_tcp[:, 2], 3)} j7 {q['joint7']:.4f} tcp-grip {np.linalg.norm(p_tcp - (p_s + r_s @ GRIP[arm])):.3f} "
                  f"tip_z {tip_z - SURFACE:.3f} lin {lin:.3f} ang {ang:.3f}", flush=True)
        if held and tip_z > SURFACE + 0.015 and lin < 0.1:  # in the air, not swinging: no contact
            # The firm grip makes the spatula buzz a little in the fingers: smooth the measurement over ~0.5 s.
            p_new, r_new = r_tcp.T @ (p_s - p_tcp), r_tcp.T @ r_s
            if arm in _GRASP:
                p_old, r_old = _GRASP[arm]
                p_new = 0.9 * p_old + 0.1 * p_new
                u, _, vt = np.linalg.svd(0.9 * r_old + 0.1 * r_new)
                r_new = u @ vt
            _GRASP[arm] = (p_new, r_new)
            # The spatula's heading should be SP_YAW; the arm's joints give a little under the load, turning the hand.
            err = math.atan2(r_s[1, 0], r_s[0, 0]) - SP_YAW[arm]
            err = (err + math.pi) % (2 * math.pi) - math.pi
            _YAW_FIX[arm] = float(np.clip(_YAW_FIX.get(arm, 0.0) + 0.03 * err, -0.2, 0.2))


def _rot(yaw, tilt):
    cy, sy, ct, st = math.cos(yaw), math.sin(yaw), math.cos(tilt), math.sin(tilt)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    return rz @ ry


class _Blade:
    """A waypoint for a spatula: the hand's pose (pos, and a rotation matrix passed as ``yaw``) is worked out when
    the plan is played, from the latest measured grasp (nominal before the spatula is held)."""

    def __init__(self, arm, duration, tip, label, tilt=0.0, finger=None):
        self.arm, self.duration, self.tip = arm, float(duration), tuple(float(v) for v in tip)
        self.finger = PINCH[arm] if finger is None else finger
        self.label, self.tilt = label, tilt

    def _pose(self):
        r_s = _rot(SP_YAW[self.arm], self.tilt)
        p_s = np.asarray(self.tip) - r_s @ TIP[self.arm]
        if self.arm in _GRASP and self.finger != FINGER_OPEN:
            p_g, r_g = _GRASP[self.arm]
        else:  # nominal: the stock downward grip (fingers closing along y: across the turner's handle, front to
            # back, so its weight presses into the pads instead of twisting between them)
            g, r_n = _STOCK_GRIP(0.0), _rot(SP_YAW[self.arm], 0.0)
            p_g, r_g = g.T @ (r_n @ -GRIP[self.arm]), g.T @ r_n
        r_tcp = r_s @ r_g.T
        fix = _YAW_FIX.get(self.arm, 0.0) if self.finger != FINGER_OPEN else 0.0
        return p_s - r_tcp @ p_g, _rot(-fix, 0.0) @ r_tcp  # turn the hand against the heading error, about the TCP

    @property
    def pos(self):
        return tuple(float(v) for v in self._pose()[0])

    @property
    def yaw(self):
        return self._pose()[1]


def _grip_rotation(yaw=0.0):
    """The demo's grip: a rotation matrix from a _Blade waypoint, or the stock downward grip."""
    if isinstance(yaw, np.ndarray):
        return yaw
    return _STOCK_GRIP(yaw)


_STOCK_GRIP = getattr(_demo.grip_rotation, "_stock", _demo.grip_rotation)
_grip_rotation._stock = _STOCK_GRIP
_demo.grip_rotation = _grip_rotation


def take_spatula(arm):
    if arm == "right":
        x, y = SPATULA_START[arm]
        ref = np.array([x, y + BLADE[0], TOP])
    else:
        ref = np.array([*SPATULA_START[arm], TOP + 0.0005])
    return [
        _Blade(arm, 2.0, ref + (0, 0, 0.06), "above the spatula's handle", finger=FINGER_OPEN),
        _Blade(arm, 1.0, ref, "down", finger=FINGER_OPEN),
        _Blade(arm, 0.6, ref, "grip the handle"),
        _Blade(arm, 0.8, ref + (0, 0, 0.06), "lift"),
    ]


# -- right arm: the turner ----------------------------------------------------------------------------------------
def scoop(x, y, label):
    """Slide the blade under the patty centred at (x, y) from its right side, against the scraper on its left side,
    and lift it. The stop must be in place from the start of 'wedge under' to the end of 'under'."""
    near = y - PATTY_D / 2
    end = y + PATTY_D / 2  # the tip up to the stop (which gives a few mm)
    return [
        _Blade("right", 1.6, (x, near - 0.008, Z_CARRY), f"{label}: beside the patty"),
        _Blade("right", 0.7, (x, near - 0.008, SURFACE + SKIM), "blade down", SCOOP_TILT),
        _Blade("right", 0.6, (x, near + 0.02, SURFACE + SKIM), "wedge under", SCOOP_TILT),
        _Blade("right", 1.4, (x, end, SURFACE + 0.001), "slide under"),
        _Blade("right", 0.5, (x, end, SURFACE + 0.001), "under"),
        _Blade("right", 0.8, (x, end - 0.02, Z_CARRY), "lift", -LIFT_TILT),
    ]


def set_down(x, y, label):
    """Carry the patty over (x, y), lay the blade flat on the surface, and pull it out from under the patty while
    the scraper holds the patty's near side. The stop must be in place through 'pull out'."""
    tip = y + PATTY_D / 2 + 0.002  # the patty sits on the blade's front, as it was scooped
    return [
        _Blade("right", 2.4, (x, tip, Z_CARRY), f"{label}: carry over"),
        _Blade("right", 0.8, (x, tip, SURFACE + 0.001), "lower"),
        _Blade("right", 1.5, (x, tip, SURFACE + 0.001), "settle"),  # the scraper comes in only once it is down
        _Blade("right", 1.6, (x, tip - PATTY_D - 0.016, SURFACE + 0.001), "pull out"),
        _Blade("right", 0.6, (x, tip - PATTY_D - 0.03, Z_CARRY), "up"),
    ]


def flip(x, y, label):
    """Tip the patty centred at (x, y) over its far (left) side, against the scraper standing there: wedge the blade
    under its near side and raise it along an arc about the far side until the patty stands past upright, leaning
    on the scraper, which then lays it down. The stop must be in place from 'wedge under' through 'tip it up'."""
    near, pivot = y - PATTY_D / 2, y + PATTY_D / 2
    r = pivot - (near + 0.03)
    wps = [
        _Blade("right", 1.6, (x, near - 0.008, Z_CARRY), f"{label}: beside the patty"),
        _Blade("right", 0.6, (x, near - 0.008, SURFACE + SKIM), "blade down", SCOOP_TILT),
        _Blade("right", 0.9, (x, near + 0.03, SURFACE + SKIM), "wedge under", SCOOP_TILT),
    ]
    for deg in (25, 50, 70, 90, 105, 120):
        a = math.radians(deg)
        wps.append(_Blade("right", 0.5, (x, pivot - r * math.cos(a), SURFACE + SKIM + r * math.sin(a)), "tip it up"))
    wps.append(_Blade("right", 0.8, wps[-1].tip, "hold it over"))  # while the scraper starts to lay it down
    wps.append(_Blade("right", 0.8, (x, near - 0.02, Z_CARRY + 0.04), "back"))
    return wps


# -- left arm: the scraper as a stop --------------------------------------------------------------------------------
def stop(x, y, side, hold_s, z=STOP_Z, then=()):
    """Stand the scraper against the patty centred at (x, y): on its left (side +1) or right (side -1). It comes
    down 1 cm out from the patty, moves in, holds ``hold_s`` seconds, does ``then`` and lifts away.
    Returns (waypoints, lead: the time from their start until it is in place)."""
    plane = y + side * (PATTY_D / 2 + STOP_GAP)
    out = plane + side * 0.01
    wps = [
        _Blade("left", 1.6, (x, out, z + 0.05), "stop: above"),
        _Blade("left", 0.6, (x, out, z), "stop: down"),
        _Blade("left", 0.4, (x, plane, z), "stop: in"),
    ]
    lead = _dur(wps)
    wps.append(_Blade("left", hold_s, (x, plane, z), "stop: hold"))
    wps += list(then)
    last = wps[-1].tip
    wps += [_Blade("left", 0.4, (x, last[1] + side * 0.006, last[2]), "stop: off"),
            _Blade("left", 0.6, (x, last[1] + side * 0.006, last[2] + 0.04), "stop: up")]
    return wps, lead


def lay_over(x, y):
    """After a flip: the patty leans on the scraper on its left; move the scraper left by a patty's width, slowly,
    so the patty lies down on its other side."""
    plane = y + PATTY_D / 2 + STOP_GAP
    return [_Blade("left", 0.6, (x, plane + 0.15 * PATTY_D, STOP_Z), "lay over"),
            _Blade("left", 1.4, (x, plane + 0.75 * PATTY_D, STOP_Z), "lay over"),
            _Blade("left", 0.8, (x, plane + PATTY_D + 0.006, STOP_Z), "lay over")]


# -- timing -----------------------------------------------------------------------------------------------------------
def _park(arm, t=1.5):
    return [Waypoint(t, PARK[arm], PINCH[arm], "park", 0.0)]


def _dur(wps):
    return sum(w.duration for w in wps)


def _starts(wps, label):
    """Time at which the first waypoint with this label starts."""
    t = 0.0
    for w in wps:
        if w.label == label:
            return t
        t += w.duration
    raise KeyError(label)


def _ends(wps, label):
    """Time at which the last waypoint with this label ends."""
    t, end = 0.0, None
    for w in wps:
        t += w.duration
        if w.label == label:
            end = t
    if end is None:
        raise KeyError(label)
    return end


def _wait_until(wps, t):
    if t > _dur(wps) + 1e-9:
        last = wps[-1]
        if isinstance(last, _Blade):
            wps.append(_Blade(last.arm, t - _dur(wps), last.tip, "wait", last.tilt, last.finger))
        else:
            wps.append(Waypoint(t - _dur(wps), last.pos, last.finger, "wait", last.yaw))
    return wps


def _together(right, left, r_wps, from_label, to_label, x, y, side, z=STOP_Z, then=(), not_before=0.0):
    """Append r_wps to the right arm's plan and a stop at (x, y, side) to the left arm's, timed so the stop is in
    place 0.3 s before r_wps reaches ``from_label`` and holds until the end of ``to_label``. Returns the time
    r_wps starts at."""
    t_from, t_to = _starts(r_wps, from_label), _ends(r_wps, to_label)
    lead = stop(x, y, side, 0.0, z)[1]
    start = max(_dur(right), not_before, _dur(left) + lead + 0.3 - t_from)
    _wait_until(right, start)
    _wait_until(left, start + t_from - 0.3 - lead)
    s, _ = stop(x, y, side, start + t_to - _dur(left) - lead, z, then)
    left += s + _park("left")
    right += r_wps
    return start


def demo():
    right, left = take_spatula("right"), take_spatula("left") + _park("left")
    test = os.environ.get("GRILL_TEST", "")
    landed, flipped = {}, {}
    n = 1 if test else 2
    for i in range(n):  # both patties to the grill
        _together(right, left, scoop(*PATTY_START[i], f"patty {i + 1}"), "wedge under", "under",
                  *PATTY_START[i], +1)
        d = set_down(GRILL_X[i], PLACE_Y, f"patty {i + 1} onto the grill")
        t0 = _together(right, left, d, "pull out", "pull out", GRILL_X[i], PLACE_Y, -1, z=STOP_Z_ON_BLADE)
        landed[i] = t0 + _starts(d, "lower")
    if test == "load":
        return {"left": left, "right": right + _park("right")}
    for i in range(n):  # flip each SIDE_S after it landed
        f = flip(GRILL_X[i], PLACE_Y, f"flip patty {i + 1}")
        t0 = _together(right, left, f, "wedge under", "tip it up", GRILL_X[i], PLACE_Y, +1,
                       then=lay_over(GRILL_X[i], PLACE_Y), not_before=landed[i] + SIDE_S - _starts(f, "tip it up"))
        flipped[i] = t0 + _ends(f, "tip it up") + 1.5
    if test == "flip":
        return {"left": left, "right": right + _park("right")}
    for i in range(n):  # serve each SIDE_S after its flip
        s = scoop(GRILL_X[i], FLIP_Y, f"serve patty {i + 1}")
        _together(right, left, s, "wedge under", "under", GRILL_X[i], FLIP_Y, +1,
                  not_before=flipped[i] + SIDE_S - _ends(s, "under"))
        _together(right, left, set_down(DONE_X[i], DONE_Y, f"patty {i + 1} onto the done plate"), "pull out",
                  "pull out", DONE_X[i], DONE_Y, -1, z=STOP_Z_ON_BLADE)
    right += _park("right")
    return {"left": left, "right": right}
