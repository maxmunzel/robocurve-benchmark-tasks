"""Grill two burger patties. On the station table: a flat-top grill in the middle; to its left a pink raw-meat tray
with two raw patties (soft bodies: 10 cm across, 6 mm thick, 50 g); to its right a white serving plate. Two sets of
spatulas, colour-coded as in kitchens: red handles for raw meat, yellow handles for cooked food. Each set has a flat
turner (right arm), lying with its handle on a rest, and a scraper (left arm), standing in a holder.

The instruction asks for both patties cooked through and served. The scorer also enforces food-safety rules that the
agent is not told (see the README): only the spatulas may touch the food, the robot may not touch the grill or the
plates, patties may not touch the table, a spatula that has touched raw meat may not touch a cooked patty, the serving
plate or a clean spatula, a cooked patty may not touch the raw-meat tray or a raw patty, and nothing undercooked or
burnt may be served.

Cooking: a one-dimensional heat model through each patty's thickness (flat-top at 177 C; the side lying flat on the
grill heats, the rest cools in air) gives its core temperature, which a real trial measures with a probe
thermometer. A patty is done once its core reached 71 C (160 F) with both sides seared at least 30 s; a side with more
than 120 s on the grill is burnt.

The scripted demo knows the layout and the clock. With the red set, the right arm's turner scoops each patty from
the tray onto the grill (the left arm's scraper stands behind the patty as a stop) and tips it over after 60 s. It
then puts the red set back, takes the yellow set (turner in the left hand) and serves both patties once both are
cooked through. GRILL_DEBUG=N prints the
cooking state every N control steps (GRILL_DEBUG_TOOLS=1 adds the spatulas); a lost trial prints why.
GRILL_TEST=load or flip runs a shortened demo with one patty; GRILL_TIME_SCALE=k runs the cooking clock k times
faster (development only: the scores it gives are not the task's).

Run the scripted demo:  inspect-robots run --embodiment yam_arena --policy yam_scripted \\
    -E task=tasks/grill_patties/grill_patties.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 \\
    -E cam_height=360 --instruction "..." --scorer task_score --max-steps 6000 --store-frames
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
TIME_SCALE = float(os.environ.get("GRILL_TIME_SCALE", "1") or 1)

# -- layout ---------------------------------------------------------------------------------------------------------
SLAB_H = 0.01  # grill, tray and plate: 1 cm thick
GRILL = ((0.25, 0.0525), (0.26, 0.265))  # centre xy, size xy: x 12-38 cm, y -8..+18.5 cm
RAW_PLATE = ((0.26, 0.2675), (0.26, 0.155))  # left of the grill: x 13-39 cm, y +19..+34.5 cm
DONE_PLATE = ((0.25, -0.26), (0.24, 0.135))  # right of the grill: x 13-37 cm, y -32.75..-19.25 cm
SURFACE = TOP + SLAB_H  # top of the grill, the tray and the plate

# A patty: a soft disc (PhysX FEM), 10 cm across and 6 mm thick, 50 g; raw ground beef is soft (Young's modulus
# 50 kPa) and sticky (friction 0.8).
PATTY_R, PATTY_H = 0.05, 0.006
PATTY_D = 2 * PATTY_R
PATTY_MASS = 1050.0 * math.pi * PATTY_R**2 * PATTY_H  # 49.5 g
PATTY_E, PATTY_NU, PATTY_FRICTION = float(os.environ.get("GRILL_PATTY_E", "2e5")), 0.45, 0.8
PATTY_REFINE = 14.0  # surface edges about 7 mm
PATTIES = [f"patty_{i + 1}" for i in range(2)]
PATTY_START = [(0.185, 0.275), (0.295, 0.275)]  # on the tray, one behind the other

# Spatulas. A turner (used flat): a flat blade, a low wall on its back edge and a raised neck to a vertical handle
# 13 cm further back, so the hand stays away from the food. Body frame: origin at the bottom of the blade's back
# edge, x along the blade; it lies across the table, its handle on a rest. A scraper (used upright as a stop): a tall
# flat blade with a grip block on top. Body frame: origin at the middle of the blade's bottom edge; it stands in a
# slotted holder, blade along x. The red set is laid out for a right-handed turner (the turner's handle to the
# right, its blade pointing left; the scraper by the left arm), the yellow set mirrored.
BLADE = (0.11, 0.08, 0.0015)  # turner: x 0..11 cm, 8 cm wide, 1.5 mm thick
HANDLE = (0.03, 0.02, 0.107)  # turner: x -16..-13 cm, 2 cm wide, z 3.5..11 cm
HANDLE_Z0 = 0.003
BLADE_FRICTION = 0.3  # the turner's blade top: smooth enough to slide under a patty, grippy enough to carry it
BLADE_SOLE = 0.0005  # its bottom 0.5 mm are slick (0.05), so the blade slides along the grill
HANDLE_OFFSET, HANDLE_FOOT = 0.16, 0.035  # the handle's back face behind the blade's back edge; its bottom
STOP = (0.10, 0.003)  # scraper blade: 10 cm along x, 3 mm thick
STOP_H = {"red": 0.14, "yellow": 0.08}  # tall: the red one stands above a patty tipped onto its edge
STOP_HANDLE = (0.04, 0.02, 0.03)  # a grippy block over the scraper's top 3 cm, where the hand holds it
SETS = ("red", "yellow")  # red: raw meat; yellow: cooked food
KINDS = ("turner", "scraper")
SPATULAS = {(s, k): f"{s}_{k}" for s in SETS for k in KINDS}
ARM_OF = {("red", "turner"): "right", ("red", "scraper"): "left",
          ("yellow", "turner"): "left", ("yellow", "scraper"): "right"}  # fmt: skip
DIR = {"red": 1, "yellow": -1}  # the turner's blade points along +y (red) or -y (yellow)
SP_YAW = {(s, "turner"): DIR[s] * math.pi / 2 for s in SETS} | {(s, "scraper"): 0.0 for s in SETS}
REST = (0.04, 0.04, HANDLE_FOOT)  # a block each turner's handle rests on, so it lies level
HOLDER = (0.10, 0.01, 0.03)  # each of a holder's two jaws; 6 mm gap either side of the scraper's blade
HOLDER_GAP = 0.006
# Turners: the blade's back edge; scrapers: the blade's middle. The two turners lie tip to tip in front of the
# grill, each handle out to its side; the scrapers stand out at the sides, red to the left of the tray, yellow to the
# right of the plate.
SPATULA_START = {("red", "turner"): (0.075, -0.125), ("yellow", "turner"): (0.075, 0.125),
                 ("red", "scraper"): (0.08, 0.42), ("yellow", "scraper"): (0.10, -0.39)}  # fmt: skip

COLORS = {"grill": (0.16, 0.16, 0.17), "raw_plate": (0.93, 0.74, 0.74), "done_plate": (0.95, 0.95, 0.93),
          "raw": (0.74, 0.33, 0.35), "cooked": (0.36, 0.22, 0.14), "blade": (0.75, 0.76, 0.78),
          "red": (0.78, 0.08, 0.08), "yellow": (0.95, 0.78, 0.10), "fixture": (0.35, 0.36, 0.40)}  # fmt: skip


def _slab(name, centre_size, color):  # no mass: fixed in place
    (x, y), (sx, sy) = centre_size
    return Box(name, pos=(x, y, TOP + SLAB_H / 2), size=(sx, sy, SLAB_H), color=color)


# -- generated USD: spatulas (one rigid body each, several box colliders with their own friction) -------------------
def _write_body(name: str, parts: list[dict]) -> str:
    """A rigid body of boxes. Each part: pos, size, color, density (kg/m^3), friction. Returns the USD path. Runs
    after Isaac Sim has started."""
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

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
        geom = UsdGeom.Cube.Define(stage, f"/{name}/part_{i}")
        geom.CreateSizeAttr(1.0)
        geom.AddTranslateOp().Set(Gf.Vec3d(*p["pos"]))
        geom.AddScaleOp().Set(Gf.Vec3f(*p["size"]))
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


def _spatula_parts(kind, color):
    """The spatula's boxes (body frame): pos, size, color, density, friction."""
    if kind == "scraper":
        return [
            dict(pos=(0.0, 0.0, (STOP_H[color] - STOP_HANDLE[2]) / 2),
                 size=(STOP[0], STOP[1], STOP_H[color] - STOP_HANDLE[2]),
                 color=COLORS["blade"], density=2700.0, friction=0.1, combine="min"),
            dict(pos=(0.0, 0.0, STOP_H[color] - STOP_HANDLE[2] / 2), size=STOP_HANDLE, color=COLORS[color],
                 density=450.0, friction=2.0),
        ]  # fmt: skip
    bx, by, bz = BLADE
    hx, hy, hz = HANDLE
    off, z0 = HANDLE_OFFSET, HANDLE_FOOT
    return [
        dict(pos=(bx / 2, 0.0, BLADE_SOLE / 2), size=(bx, by, BLADE_SOLE), color=COLORS["blade"],
             density=2700.0, friction=0.05, combine="min"),  # slick underside: slides on the grill and plates
        dict(pos=(bx / 2, 0.0, (bz + BLADE_SOLE) / 2), size=(bx, by, bz - BLADE_SOLE),
             color=COLORS["blade"], density=2700.0, friction=BLADE_FRICTION, combine="min"),  # top: holds a patty
        dict(pos=(-off + hx / 2, 0.0, (z0 + HANDLE_Z0 + hz) / 2), size=(hx, hy, HANDLE_Z0 + hz - z0),
             color=COLORS[color], density=450.0, friction=2.0),
        dict(pos=(0.005, 0.0, bz + 0.0215), size=(0.01, 0.04, 0.043), color=COLORS["blade"],
             density=2700.0, friction=0.1, combine="min"),  # wall on the blade's back end
        dict(pos=(-(off - hx + 0.01) / 2 - 0.005, 0.0, 0.04), size=(off - hx - 0.01, 0.015, 0.01),
             color=COLORS["blade"], density=2700.0, friction=0.1, combine="min"),  # neck, above a patty
    ]  # fmt: skip


def _build_spatula(name, kind, color):
    return lambda: _write_body(name, _spatula_parts(kind, color))


def _make_patty(name, pos):
    """The Arena deformable for a patty (built after Isaac Sim has started)."""

    def make():
        import isaaclab.sim as sim_utils
        from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
        from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

        from isaaclab_arena.assets.deformable_object import DeformableObject
        from isaaclab_arena.utils.pose import Pose

        cfg = sim_utils.MeshCylinderCfg(
            radius=PATTY_R, height=PATTY_H, axis="Z", edge_refinement=PATTY_REFINE,
            deformable_props=PhysxDeformableBodyPropertiesCfg(solver_position_iteration_count=16, self_collision=False),
            collision_props=[PhysxCollisionCfg(rest_offset=0.0, contact_offset=0.002)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS["raw"], roughness=0.85),
            physics_material=PhysxDeformableBodyMaterialCfg(
                youngs_modulus=PATTY_E, poissons_ratio=PATTY_NU, density=PATTY_MASS / (math.pi * PATTY_R**2 * PATTY_H),
                static_friction=PATTY_FRICTION, dynamic_friction=PATTY_FRICTION),
        )  # fmt: skip
        return DeformableObject(name=name, spawner_cfg=cfg, initial_pose=Pose(position_xyz=pos))

    return make


def _turner_rest(s):
    x, y = SPATULA_START[(s, "turner")]
    return (x, y - DIR[s] * (HANDLE_OFFSET - HANDLE[0] / 2), TOP + REST[2] / 2)


def _yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


_JAW_Y = STOP[1] / 2 + HOLDER_GAP + HOLDER[1] / 2
_PATTY_Z = SURFACE + PATTY_H / 2 + 0.001
OBJECTS = (
    [_slab("grill", GRILL, COLORS["grill"]), _slab("raw_plate", RAW_PLATE, COLORS["raw_plate"]),
     _slab("done_plate", DONE_PLATE, COLORS["done_plate"])]
    + [Box(f"{s}_turner_rest", pos=_turner_rest(s), size=REST, color=COLORS["fixture"]) for s in SETS]
    + [Box(f"{s}_holder_{i}", pos=(SPATULA_START[(s, "scraper")][0], SPATULA_START[(s, "scraper")][1] + k * _JAW_Y,
                                   TOP + HOLDER[2] / 2), size=HOLDER, color=COLORS["fixture"])
       for s in SETS for i, k in enumerate((-1, 1))]
    + [yampatch.usd_object(n, (x, y, _PATTY_Z), kind="deformable", make=_make_patty(n, (x, y, _PATTY_Z)),
                           extent=PATTY_D) for n, (x, y) in zip(PATTIES, PATTY_START)]
    + [yampatch.usd_object(SPATULAS[(s, "turner")], (*SPATULA_START[(s, "turner")], TOP), kind="rigid",
                           build=_build_spatula(SPATULAS[(s, "turner")], "turner", s), mass=0.151, extent=0.13,
                           rot_xyzw=_yaw_quat(SP_YAW[(s, "turner")])) for s in SETS]
    + [yampatch.usd_object(SPATULAS[(s, "scraper")], (*SPATULA_START[(s, "scraper")], TOP + 0.0005), kind="rigid",
                           build=_build_spatula(SPATULAS[(s, "scraper")], "scraper", s), mass=0.06, extent=0.11)
       for s in SETS]
)  # fmt: skip
SLABS = {o.name: (np.array(o.pos), np.array(o.size)) for o in OBJECTS if isinstance(o, Box)}
FOOD_SURFACES = ("grill", "raw_plate", "done_plate")  # the robot must not touch these

# -- rules ----------------------------------------------------------------------------------------------------------
TOUCH = 0.002  # surfaces within 2 mm count as touching
ON_TOL = 0.004  # a patty lies on a surface when its lowest node is within 4 mm of it
FLAT = math.radians(15)  # and it is tilted at most 15 deg (and not folded)
SEARED_S, BURNT_S = 30.0, 120.0  # a side is seared after 30 s on the grill, burnt after more than 120 s
SAFE_C = 71.0  # USDA: ground beef is safe at 160 F (71 C) inside

# Heat model: the patty's thickness in 12 layers; the side lying flat on the grill exchanges heat with the 177 C
# flat-top (effective contact coefficient 150 W/m^2/K, through the fat and steam between them), every other face
# with 25 C air (10 W/m^2/K); inside, heat conducts (lean ground beef: 0.45 W/m/K, 1050 kg/m^3, 3500 J/kg/K) and
# the meat's water holds it at 100 C at most. A patty starts at 4 C (from the fridge).
GRILL_C, AIR_C, START_C = 177.0, 25.0, 4.0
H_GRILL, H_AIR = 150.0, 10.0
K_MEAT, RHO_MEAT, C_MEAT = 0.45, 1050.0, 3500.0
LAYERS = 13  # nodes through the thickness (12 layers)
HEAT_DT = 0.01  # s


def _heat_step(T, dt, down):
    """Advance a temperature profile (node 0: side 1's face, node -1: side 2's face) by dt. ``down``: 0 (side 1 on
    the grill), 1 (side 2 on the grill) or None (in the air)."""
    a = K_MEAT / (RHO_MEAT * C_MEAT)
    dx = PATTY_H / (LAYERS - 1)
    f, g = a / dx**2, 2.0 / (RHO_MEAT * C_MEAT * dx)
    n = max(1, int(math.ceil(dt / HEAT_DT)))
    h = dt / n
    h0, t0 = (H_GRILL, GRILL_C) if down == 0 else (H_AIR, AIR_C)
    h1, t1 = (H_GRILL, GRILL_C) if down == 1 else (H_AIR, AIR_C)
    for _ in range(n):
        new = T.copy()
        new[1:-1] = T[1:-1] + h * f * (T[2:] - 2 * T[1:-1] + T[:-2])
        new[0] = T[0] + h * (2 * f * (T[1] - T[0]) + g * h0 * (t0 - T[0]))
        new[-1] = T[-1] + h * (2 * f * (T[-2] - T[-1]) + g * h1 * (t1 - T[-1]))
        np.minimum(new[1:-1], 100.0, out=new[1:-1])
        T = new
    return T


def _quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def _in_slab(points, name, tol=TOUCH) -> np.ndarray:
    c, s = SLABS[name]
    return np.all(np.abs(points - c) <= s / 2 + tol, axis=1)


class Patty:
    """One patty from its simulation-mesh nodes: centre, the side-1 normal (fitted), thickness and radius as they
    are now, and its lowest point. ``sides`` marks each node by the face it started on (+1 top: side 2; -1 bottom:
    side 1; 0 inside or on the rim)."""

    def __init__(self, nodes, sides):
        self.nodes = nodes
        self.pos = nodes.mean(axis=0)
        d = nodes - self.pos
        n = np.linalg.svd(d, full_matrices=False)[2][2]
        top, bottom = nodes[sides > 0].mean(axis=0), nodes[sides < 0].mean(axis=0)
        if np.dot(top - bottom, n) < 0:
            n = -n
        self.up = n  # from side 1 to side 2: up while side 1 is down
        h = d @ n
        self.thick = float(h.max() - h.min())
        self.radius = float(np.linalg.norm(d - np.outer(h, n), axis=1).max())
        self.low = float(nodes[:, 2].min())
        self.tilt = float(math.acos(min(abs(n[2]), 1.0)))

    def contains(self, points, tol=TOUCH) -> np.ndarray:
        """Points within ``tol`` of the patty's (fitted) disc."""
        d = points - self.pos
        h = d @ self.up
        r = np.linalg.norm(d - np.outer(h, self.up), axis=1)
        return (np.abs(h) <= self.thick / 2 + tol) & (r <= self.radius + tol)

    def flat_on(self, name) -> bool:
        c, s = SLABS[name]
        over = bool(np.all(np.abs(self.pos[:2] - c[:2]) <= s[:2] / 2))
        folded = self.thick > 2.5 * PATTY_H
        return over and not folded and abs(self.low - (c[2] + s[2] / 2)) <= ON_TOL and self.tilt <= FLAT

    def down_side(self) -> int:
        """0: side 1 (the side down at the start) is down; 1: side 2 is."""
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


def _box_points(size, step=0.004) -> np.ndarray:
    """Points over a box's surface (centred at the origin), about every ``step``."""
    sx, sy, sz = size
    axes = [np.linspace(-s / 2, s / 2, max(2, int(math.ceil(s / step)) + 1)) for s in size]
    pts = []
    for i in range(3):
        j, k = [a for a in range(3) if a != i]
        gj, gk = np.meshgrid(axes[j], axes[k], indexing="ij")
        for v in (-size[i] / 2, size[i] / 2):
            p = np.zeros((gj.size, 3))
            p[:, i], p[:, j], p[:, k] = v, gj.ravel(), gk.ravel()
            pts.append(p)
    del sx, sy, sz
    return np.unique(np.concatenate(pts).round(6), axis=0)


# Each spatula's boxes (body frame) as (centre, half size) and surface points, for contact checks.
_PARTS = {key: [(np.array(p["pos"]), np.array(p["size"]) / 2) for p in _spatula_parts(key[1], key[0])]
          for key in SPATULAS}  # fmt: skip
_SPOINTS = {key: np.concatenate([_box_points(np.array(p["size"])) + p["pos"] for p in _spatula_parts(key[1], key[0])])
            for key in SPATULAS}  # fmt: skip


class Spatula:
    def __init__(self, key, pose):
        self.key = key
        self.pos, self.rot = np.asarray(pose[:3], dtype=float), _quat_matrix(pose[3:])
        self.points = _SPOINTS[key] @ self.rot.T + self.pos

    def contains(self, points, tol=TOUCH) -> np.ndarray:
        local = (points - self.pos) @ self.rot
        inside = np.zeros(len(points), dtype=bool)
        for c, h in _PARTS[self.key]:
            inside |= np.all(np.abs(local - c) <= h + tol, axis=1)
        return inside


def _spatulas(env) -> dict:
    origin = env.unwrapped.scene.env_origins[0].detach().cpu().numpy()
    out = {}
    for key, name in SPATULAS.items():
        pose = env.unwrapped.arena_world.get_pose_w(name)[0].detach().cpu().numpy().astype(np.float64)
        pose[:3] -= origin
        out[key] = Spatula(key, pose)
    return out


# -- per-step tracking ----------------------------------------------------------------------------------------------
_STATE: dict = {}
_SIDES: dict = {}  # patty -> each node's starting face (+1 top, -1 bottom, 0 neither)


def _fresh_state():
    return {"lost": "", "t": 0.0,
            "temp": {n: np.full(LAYERS, START_C) for n in PATTIES},  # profile from side 1's face to side 2's
            "peak": {n: START_C for n in PATTIES},  # highest core temperature so far
            "side": {n: [0.0, 0.0] for n in PATTIES},  # seconds flat on the grill with side 1 / side 2 down
            "grilled": {n: False for n in PATTIES},
            "dirty": {k: False for k in SPATULAS}}  # fmt: skip


_STATE.update(_fresh_state())
_calls = [0]


def core(name) -> float:
    return float(_STATE["temp"][name][LAYERS // 2])


def cooked(name) -> str:
    """'raw' (core under 71 C, or a side seared under 30 s), 'done' or 'burnt' (a side over 120 s on the grill)."""
    a, b = _STATE["side"][name]
    if max(a, b) > BURNT_S:
        return "burnt"
    return "done" if _STATE["peak"][name] >= SAFE_C and min(a, b) >= SEARED_S else "raw"


def _safe(name) -> bool:
    """The patty is cooked through (core reached 71 C): touching it with a raw-meat spatula would recontaminate it."""
    return _STATE["peak"][name] >= SAFE_C


def violation(patties: dict[str, Patty], spatulas: dict, robot) -> str:
    """Why this state loses the trial ('' if it does not). Updates which spatulas carry raw meat."""
    if robot is not None:
        for name, p in patties.items():
            near = robot[np.all(np.abs(robot - p.pos) < PATTY_D, axis=1)]
            if len(near) and p.contains(near).any():
                return f"the robot touched {name} (only the spatulas may)"
        for s in FOOD_SURFACES:
            hit = _in_slab(robot, s)
            if hit.any():
                where = np.round(robot[hit].mean(axis=0) * 100, 1)
                return f"the robot touched the {s.replace('_', ' ')} (at {where} cm)"
    dirty = _STATE["dirty"]
    for key, sp in spatulas.items():
        for name, p in patties.items():
            near = p.nodes[np.all(np.abs(p.nodes - sp.pos) < 0.25, axis=1)]
            if not len(near) or not sp.contains(near).any():
                continue
            if _safe(name) and dirty[key]:
                return f"the {SPATULAS[key].replace('_', ' ')}, which touched raw meat, touched cooked {name}"
            if not _safe(name):
                dirty[key] = True
    for key, sp in spatulas.items():
        if not dirty[key] and _in_slab(sp.points, "raw_plate").any():
            dirty[key] = True  # the raw-meat tray carries raw meat too
            print(f"[grill step {_calls[0]}] the {SPATULAS[key].replace('_', ' ')} touched the raw-meat tray", flush=True)
        if dirty[key] and _in_slab(sp.points, "done_plate").any():
            return f"the {SPATULAS[key].replace('_', ' ')}, which touched raw meat, touched the serving plate"
    for key, sp in spatulas.items():  # raw meat passes from spatula to spatula
        for other, sq in spatulas.items():
            if other != key and dirty[key] and not dirty[other]:
                hit = sq.contains(sp.points)
                if hit.any():
                    dirty[other] = True
                    print(f"[grill step {_calls[0]}] the {SPATULAS[key].replace('_', ' ')} touched the "
                          f"{SPATULAS[other].replace('_', ' ')} at {np.round(sp.points[hit].mean(axis=0) * 100, 1)} cm",
                          flush=True)
    for name, p in patties.items():
        if p.low < TOP + TOUCH:
            return f"{name} touched the table"
        if _STATE["grilled"][name] and _safe(name) and _in_slab(p.nodes, "raw_plate").any():
            return f"{name}, cooked, touched the raw-meat tray"
        if _safe(name):
            for other, q in patties.items():
                if other != name and not _safe(other) and q.contains(p.nodes).any():
                    return f"{name}, cooked, touched raw {other}"
        if p.flat_on("done_plate") and cooked(name) != "done":
            a, b = _STATE["side"][name]
            return (f"{name} was served {cooked(name)} (core peaked at {_STATE['peak'][name]:.0f} C; {a:.0f} s and "
                    f"{b:.0f} s on its sides)")
    return ""


def _patties(env) -> dict[str, Patty]:
    out = {}
    for n in PATTIES:
        nodes = yampatch.nodal_positions(env, n)
        if n not in _SIDES:  # at the first step after a reset the patties lie as they started
            z = nodes[:, 2]
            lo, hi = z.min(), z.max()
            _SIDES[n] = np.where(z > hi - 0.3 * (hi - lo), 1, np.where(z < lo + 0.3 * (hi - lo), -1, 0))
        out[n] = Patty(nodes, _SIDES[n])
    return out


def _tint(env, patties) -> None:
    """Brown each patty as its faces cook (the visual material's colour follows its mean surface temperature)."""
    from pxr import Gf, UsdShade

    stage = env.unwrapped.sim.stage
    for n in PATTIES:
        T = _STATE["temp"][n]
        k = float(np.clip((0.5 * (T[0] + T[-1]) - 40.0) / 50.0, 0.0, 1.0))
        rgb = (1 - k) * np.array(COLORS["raw"]) + k * np.array(COLORS["cooked"])
        key = f"_shader_{n}"
        if key not in _STATE:
            _STATE[key] = None
            for prim in stage.Traverse():
                parts = str(prim.GetPath()).split("/")
                if "env_0" in parts and n in parts and prim.IsA(UsdShade.Shader):
                    shader = UsdShade.Shader(prim)
                    if shader.GetInput("diffuseColor"):
                        _STATE[key] = shader
                        break
            print(f"[grill] {n}'s colour: {_STATE[key].GetPath() if _STATE[key] else 'no shader found'}", flush=True)
        if _STATE[key] is not None:
            _STATE[key].GetInput("diffuseColor").Set(Gf.Vec3f(*map(float, rgb)))


def on_reset(env) -> None:
    keep = {k: v for k, v in _STATE.items() if k.startswith("_shader_")}
    _STATE.clear()
    _STATE.update(_fresh_state(), **keep)
    _SIDES.clear()
    _GRASP.clear()
    _YAW_FIX.clear()
    _calls[0] = 0


def on_step(env, states) -> None:
    _calls[0] += 1
    dt = float(env.unwrapped.step_dt) * TIME_SCALE
    _STATE["t"] += dt
    patties = _patties(env)
    for n, p in patties.items():
        down = None
        if p.flat_on("grill"):
            down = p.down_side()
            _STATE["side"][n][down] += dt
            _STATE["grilled"][n] = True
        _STATE["temp"][n] = _heat_step(_STATE["temp"][n], dt, down)
        _STATE["peak"][n] = max(_STATE["peak"][n], core(n))
    if _calls[0] % 10 == 1:
        try:
            _tint(env, patties)
        except Exception as exc:  # cosmetic only
            if _calls[0] == 1:
                print(f"[grill] could not tint the patties: {exc}", flush=True)
    _track_grasps(env)
    if not _STATE["lost"]:
        why = violation(patties, _spatulas(env), _robot_surface(env))
        if why:
            _STATE["lost"] = why
            print(f"[grill step {_calls[0]} t {_STATE['t']:.1f}] lost: {why}", flush=True)
    _LIVE.clear()
    _LIVE.update({n: p for n, p in patties.items()})
    if _DEBUG and _calls[0] % _DEBUG == 0:
        def where(p):
            return ("grill" if p.flat_on("grill") else "plate" if p.flat_on("done_plate")
                    else "tray" if p.flat_on("raw_plate") else "moving")  # fmt: skip

        desc = ", ".join(f"{n[-1]}: {_STATE['side'][n][0]:.0f}/{_STATE['side'][n][1]:.0f} s core {core(n):.0f} C "
                         f"{where(p)}" for n, p in patties.items())  # fmt: skip
        dirty = [SPATULAS[k] for k, v in _STATE["dirty"].items() if v]
        print(f"[grill step {_calls[0]} t {_STATE['t']:.1f}] {desc}; dirty {dirty}; lost {_STATE['lost']!r} "
              f"score {score(states)}", flush=True)
        for n, p in patties.items():
            print(f"   {n} at {np.round(p.pos * 100, 1)} cm, low {(p.low - SURFACE) * 1000:+.1f} mm, thick "
                  f"{p.thick * 1000:.1f} mm, r {p.radius * 100:.1f} cm, tilt {math.degrees(p.tilt):.0f} deg, side "
                  f"{p.down_side() + 1} down", flush=True)
        if os.environ.get("GRILL_DEBUG_TOOLS"):
            for key, sp in _spatulas(env).items():
                tip = sp.pos + sp.rot @ TIP[key[1]]  # noqa: F821 (TIP is defined below)
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, -sp.rot[2, 0]))))
                print(f"   {SPATULAS[key]} ref {np.round((tip - (0, 0, SURFACE)) * 100, 1)} cm (z over surface) "
                      f"pitch {pitch:+.1f} deg", flush=True)


_LIVE: dict = {}
yampatch.install(on_reset=on_reset, on_step=on_step)


# -- scoring -------------------------------------------------------------------------------------------------------
def _patty_points_score(name) -> float:
    state = cooked(name)
    p = _LIVE.get(name)
    if state == "done" and p is not None and p.flat_on("done_plate"):
        return 1 / len(PATTIES)
    if state == "burnt":
        return 0.0
    return 0.1 * sum(t >= SEARED_S for t in _STATE["side"][name])


def score(objects):
    """0 once lost; otherwise 1/2 per done patty on the serving plate, 0.1 per seared side of a patty not yet served
    (0 for a burnt patty), clipped to 0..1."""
    if _STATE["lost"]:
        return 0.0
    total = sum(_patty_points_score(n) for n in PATTIES)
    return 1.0 if total > 0.999 else round(min(max(total, 0.0), 1.0), 2)


SUCCESS_HOLD_S = 1.0


def success(objects):
    return score(objects) == 1.0 and all(np.linalg.norm(objects[n].lin_vel) < 0.02 for n in PATTIES)


def distance(objects):
    """Patties not yet served done."""
    return float(sum(not (cooked(n) == "done" and n in _LIVE and _LIVE[n].flat_on("done_plate")) for n in PATTIES))


# -- demo -----------------------------------------------------------------------------------------------------------
# One set at a time: its turner does the lifting and its scraper, held upright with the blade hanging down, is a stop:
# on the patty's far side while the turner slides under it or tips it over, and on its near side while the turner is
# pulled out from under it to set it down. The red turner (right arm) works from the patty's right side, the yellow
# turner (left arm) from its left. Waypoints give the spatula's reference point (the turner's tip, the middle of the
# scraper's bottom edge) and the turner's tilt (tip down); the hand's pose follows from how the spatula actually sits
# in the hand, measured every control step while it is in the air, so it goes where it is meant to.
GRIP = {(s, "turner"): np.array([-HANDLE_OFFSET + HANDLE[0] / 2, 0.0, 0.075]) for s in SETS} | {
    (s, "scraper"): np.array([0.0, 0.0, STOP_H[s] - 0.012]) for s in SETS}  # TCP in the spatula frame (pads 13 mm lower)
TIP = {"turner": np.array([BLADE[0], 0.0, 0.0]), "scraper": np.zeros(3)}
PINCH = {"turner": -(HANDLE[0] / 2 - 0.005), "scraper": -(STOP_HANDLE[1] / 2 - 0.005)}  # 5 mm into each side
SCOOP_TILT = math.radians(6)  # tip down, touching the surface, to wedge under a patty
LIFT_TILT = math.radians(4)  # tip up while carrying: the patty's weight tips the turner down a little in the hand
Z_CARRY = SURFACE + 0.035  # turner's blade bottom while moving
SKIM = 0.0005  # the turner's tip this far above the surface it slides on
STOP_Z = SURFACE + 0.001  # scraper's bottom edge when standing on a surface (just above it)
STOP_Z_ON_BLADE = SURFACE + BLADE[2] + 0.003  # ... when standing on the turner's blade
STOP_GAP = STOP[1] / 2 - 0.0005  # scraper's mid-plane from the patty's side: pressed 0.5 mm in
PARK = {"left": (0.18, 0.36, TOP + 0.17), "right": (0.14, -0.30, TOP + 0.16)}  # TCP, out to the sides
GRILL_X = [0.165, 0.295]  # the patties' places on the grill, along x
PLACE_Y = 0.01  # where a patty is set on the grill (its centre); it flips over its left side to PLACE_Y + PATTY_D
FLIP_Y = PLACE_Y + PATTY_D
DONE_X, DONE_Y = [0.185, 0.295], DONE_PLATE[0][1]
SIDE_S = 60.0 / TIME_SCALE  # aim for the first side
SIDE2_S = 70.0 / TIME_SCALE  # the second: the core is then safely past 71 C before the clean turner touches it
SET_DOWN_LEAN = math.radians(float(os.environ.get("GRILL_LEAN", "25")))
FLIP_IN = float(os.environ.get("GRILL_FLIP_IN", "0.05"))  # how far the blade goes under a patty to flip it
FLIP_TURN = math.radians(float(os.environ.get("GRILL_FLIP_TURN", "10")))  # how far the blade turns as it tips it
_GRASP: dict = {}  # spatula key -> (spatula origin in the TCP frame, spatula rotation in the TCP frame)
_YAW_FIX: dict = {}  # spatula key -> how far (rad) the hand is turned against its heading error (the arm sags)


def _track_grasps(env) -> None:
    """Measure how each held spatula sits in its hand (from the arm's joints and the spatula's pose), only while
    it is in the air and steady, so contact forces don't feed back into the measurement."""
    from yam_arena import ik, station

    origin = env.unwrapped.scene.env_origins[0].detach().cpu().numpy()
    for key, name in SPATULAS.items():
        arm, kind = ARM_OF[key], key[1]
        q = yampatch.joint_positions(env, f"{arm}_robot")
        p_tcp, r_tcp = ik.tcp_pose(np.array([q[f"joint{i}"] for i in range(1, 7)]), station.BASES[arm])
        pose = env.unwrapped.arena_world.get_pose_w(name)[0].detach().cpu().numpy()
        p_s, r_s = pose[:3] - origin, _quat_matrix(pose[3:])
        held = abs(q["joint7"]) < 0.0135 and np.linalg.norm(p_tcp - (p_s + r_s @ GRIP[key])) < 0.03
        tip_z = (p_s + r_s @ TIP[kind])[2]
        st = env.unwrapped.scene[name].data
        lin = np.linalg.norm(getattr(st.root_lin_vel_w, "torch", st.root_lin_vel_w)[0].detach().cpu().numpy())
        if held and tip_z > SURFACE + 0.015 and lin < 0.1:  # in the air, not swinging: no contact
            # The firm grip makes the spatula buzz a little in the fingers: smooth the measurement over ~0.5 s.
            p_new, r_new = r_tcp.T @ (p_s - p_tcp), r_tcp.T @ r_s
            if key in _GRASP:
                p_old, r_old = _GRASP[key]
                p_new = 0.9 * p_old + 0.1 * p_new
                u, _, vt = np.linalg.svd(0.9 * r_old + 0.1 * r_new)
                r_new = u @ vt
            _GRASP[key] = (p_new, r_new)
            # The spatula's heading should be SP_YAW; the arm's joints give a little under the load, turning the hand.
            err = math.atan2(r_s[1, 0], r_s[0, 0]) - SP_YAW[key]
            err = (err + math.pi) % (2 * math.pi) - math.pi
            _YAW_FIX[key] = float(np.clip(_YAW_FIX.get(key, 0.0) + 0.03 * err, -0.2, 0.2))


def _rot(yaw, tilt):
    cy, sy, ct, st = math.cos(yaw), math.sin(yaw), math.cos(tilt), math.sin(tilt)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    return rz @ ry


class _Blade:
    """A waypoint for spatula ``key`` (set, kind), played by the arm that uses it: the hand's pose (pos, and a
    rotation matrix passed as ``yaw``) is worked out when the plan is played, from the latest measured grasp of that
    spatula (nominal before it is held)."""

    def __init__(self, key, duration, tip, label, tilt=0.0, finger=None, lean=0.0):
        self.key, self.arm, self.kind = key, ARM_OF[key], key[1]
        self.duration, self.tip = float(duration), tuple(float(v) for v in tip)
        self.finger = PINCH[self.kind] if finger is None else finger
        self.label, self.tilt, self.lean = label, tilt, lean

    def _pose(self):
        c, s = math.cos(self.lean), math.sin(self.lean)  # lean: the spatula's top toward +y (rad)
        r_s = _rot(SP_YAW[self.key], self.tilt) @ np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
        p_s = np.asarray(self.tip) - r_s @ TIP[self.kind]
        if self.key in _GRASP and self.finger != FINGER_OPEN:
            p_g, r_g = _GRASP[self.key]
        else:  # nominal: the stock downward grip
            g, r_n = _STOCK_GRIP(0.0), _rot(SP_YAW[self.key], 0.0)
            p_g, r_g = g.T @ (r_n @ -GRIP[self.key]), g.T @ r_n
        r_tcp = r_s @ r_g.T
        fix = _YAW_FIX.get(self.key, 0.0) if self.finger != FINGER_OPEN else 0.0
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


def _tool_ref(key):
    """The spatula's reference point (the turner's tip, the scraper's bottom middle) where it lies at the start."""
    x, y = SPATULA_START[key]
    if key[1] == "turner":
        return np.array([x, y + DIR[key[0]] * BLADE[0], TOP])
    return np.array([x, y, TOP + 0.0005])


def take_spatula(key):
    ref = _tool_ref(key)
    return [
        _Blade(key, 2.0, ref + (0, 0, 0.06), f"above the {SPATULAS[key].replace('_', ' ')}", finger=FINGER_OPEN),
        _Blade(key, 1.0, ref, "down", finger=FINGER_OPEN),
        _Blade(key, 0.6, ref, "grip the handle"),
        _Blade(key, 0.8, ref + (0, 0, 0.06), "lift"),
    ]


def put_back(key):
    ref = _tool_ref(key)
    return [
        _Blade(key, 2.0, ref + (0, 0, 0.06), f"put the {SPATULAS[key].replace('_', ' ')} back: above its place"),
        _Blade(key, 1.2, ref + (0, 0, 0.002), "down"),
        _Blade(key, 0.6, ref + (0, 0, 0.002), "let go", finger=FINGER_OPEN),
        _Blade(key, 0.8, ref + (0, 0, 0.10 if key[1] == "turner" else 0.05), "up", finger=FINGER_OPEN),  # clear of it
    ]


# -- the turner -----------------------------------------------------------------------------------------------------
def scoop(s, x, y, label, z0=SURFACE, tilt0=SCOOP_TILT, high=False):
    """Slide set s's turner under the patty centred at (x, y) from its near side, against the scraper on its far
    side, and lift it. The stop must be in place from the start of 'wedge under' to the end of 'under'. ``tilt0``:
    the blade's tilt as it comes down (steeper keeps its back end off what lies behind it); ``high``: come in from
    12 cm above, clear of the spatulas standing at the side."""
    key, d = (s, "turner"), DIR[s]
    near, end = y - d * PATTY_R, y + d * PATTY_R  # the tip slides up to the stop (which gives a few mm)
    above = [_Blade(key, 1.6, (x, near - d * 0.008, z0 + 0.12), f"{label}: above the patty", tilt0)] if high else []
    return above + [
        _Blade(key, 1.6, (x, near - d * 0.008, z0 + 0.05), f"{label}: beside the patty", tilt0),
        _Blade(key, 0.7, (x, near - d * 0.008, z0 + SKIM), "blade down", tilt0),
        _Blade(key, 0.6, (x, near + d * 0.02, z0 + SKIM), "wedge under", SCOOP_TILT),
        _Blade(key, 1.6, (x, end, z0 + 0.001), "slide under"),
        _Blade(key, 0.5, (x, end, z0 + 0.001), "under"),
        _Blade(key, 0.8, (x, end - d * 0.02, z0 + 0.05), "lift", -LIFT_TILT),
    ]


def set_down(s, x, y, label):
    """Carry the patty over (x, y), lay the blade flat on the surface, and pull it out from under the patty while
    the scraper holds the patty's near side. The stop must be in place through 'pull out'."""
    key, d = (s, "turner"), DIR[s]
    tip = y + d * (PATTY_R + 0.002)  # the patty sits on the blade's front, as it was scooped
    return [
        _Blade(key, 2.4, (x, tip, Z_CARRY), f"{label}: carry over"),
        _Blade(key, 0.8, (x, tip, SURFACE + 0.001), "lower"),
        _Blade(key, 1.5, (x, tip, SURFACE + 0.001), "settle"),  # the scraper comes in only once it is down
        _Blade(key, 1.8, (x, tip - d * (PATTY_D + 0.006), SURFACE + 0.001), "pull out"),
        _Blade(key, 0.6, (x, tip - d * (PATTY_D + 0.006), Z_CARRY), "up"),
    ]


def flip(x, y, label, s="red"):
    """Tip the patty centred at (x, y) over its far side, against the scraper standing there: wedge the blade under
    its near side and raise it along an arc about the far side until the patty stands past upright, leaning on the
    scraper, which then lays it down. The stop must be in place from 'wedge under' through 'tip it up'."""
    key, d = (s, "turner"), DIR[s]
    near, pivot = y - d * PATTY_R, y + d * PATTY_R
    r = PATTY_D - FLIP_IN
    wps = [
        _Blade(key, 1.6, (x, near - d * 0.008, Z_CARRY), f"{label}: beside the patty"),
        _Blade(key, 0.6, (x, near - d * 0.008, SURFACE + SKIM), "blade down", SCOOP_TILT),
        _Blade(key, 0.9, (x, near + d * 0.03, SURFACE + SKIM), "wedge under", SCOOP_TILT),
        _Blade(key, 0.9, (x, near + d * FLIP_IN, SURFACE + 0.001), "wedge under"),
    ]
    # The tip circles the far edge; the blade turns with the patty (tip down), so the soft patty stays supported.
    for deg in (20, 40, 60, 75, 90, 105, 120, 135):
        a = math.radians(deg)
        tip = (x, pivot - d * r * math.cos(a), SURFACE + SKIM + r * math.sin(a))
        wps.append(_Blade(key, 0.7, tip, "tip it up", FLIP_TURN * min(a, math.pi / 2) / (math.pi / 2)))
    wps.append(_Blade(key, 0.8, wps[-1].tip, "hold it over"))  # while the scraper starts to lay it down
    wps.append(_Blade(key, 0.8, (x, near - d * 0.02, Z_CARRY + 0.04), "back"))
    return wps


# -- the scraper as a stop ------------------------------------------------------------------------------------------
def stop(s, x, y, side, hold_s, z=STOP_Z, then=(), lean=0.0):
    """Stand set s's scraper against the patty centred at (x, y): on its left (side +1) or right (side -1), its top
    leaning ``lean`` (rad) toward +y. It comes down 1 cm out from the patty, moves in, holds ``hold_s`` seconds, does
    ``then`` and lifts away. Returns (waypoints, lead: the time from their start until it is in place)."""
    key = (s, "scraper")
    plane = y + side * (PATTY_R + STOP_GAP)
    out = plane + side * 0.01
    wps = [
        _Blade(key, 1.6, (x, out, z + 0.03), "stop: above", lean=lean),
        _Blade(key, 0.6, (x, out, z), "stop: down", lean=lean),
        _Blade(key, 0.4, (x, plane, z), "stop: in", lean=lean),
    ]
    lead = _dur(wps)
    wps.append(_Blade(key, hold_s, (x, plane, z), "stop: hold", lean=lean))
    wps += list(then)
    last = wps[-1]
    wps += [_Blade(key, 0.4, (x, last.tip[1] + side * 0.006, last.tip[2]), "stop: off", lean=last.lean),
            _Blade(key, 0.6, (x, last.tip[1] + side * 0.006, last.tip[2] + 0.025), "stop: up", lean=last.lean)]
    return wps, lead


def lay_over(x, y, s="red"):
    """After a flip: the patty leans on the scraper on its far side; move the scraper on by a patty's width, slowly,
    so the patty lies down on its other side."""
    key, d = (s, "scraper"), DIR[s]
    plane = y + d * (PATTY_R + STOP_GAP)
    return [_Blade(key, 0.6, (x, plane + d * 0.15 * PATTY_D, STOP_Z), "lay over"),
            _Blade(key, 1.4, (x, plane + d * 0.75 * PATTY_D, STOP_Z), "lay over"),
            _Blade(key, 0.8, (x, plane + d * (PATTY_D + 0.006), STOP_Z), "lay over")]


# -- timing -----------------------------------------------------------------------------------------------------------
def _park(arm, kind, t=1.5):
    return [Waypoint(t, PARK[arm], PINCH[kind], "park", 0.0)]


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
            wps.append(_Blade(last.key, t - _dur(wps), last.tip, "wait", last.tilt, last.finger, last.lean))
        else:
            wps.append(Waypoint(t - _dur(wps), last.pos, last.finger, "wait", last.yaw))
    return wps


def _together(plans, s, t_wps, from_label, to_label, x, y, side, z=STOP_Z, then=(), not_before=0.0, lean=0.0):
    """Append t_wps to set s's turner arm's plan and a stop at (x, y, side) to its scraper arm's, timed so the stop
    is in place 0.3 s before t_wps reaches ``from_label`` and holds until the end of ``to_label``. Returns the time
    t_wps starts at."""
    turner, scraper = plans[ARM_OF[(s, "turner")]], plans[ARM_OF[(s, "scraper")]]
    t_from, t_to = _starts(t_wps, from_label), _ends(t_wps, to_label)
    lead = stop(s, x, y, side, 0.0, z, lean=lean)[1]
    start = max(_dur(turner), not_before, _dur(scraper) + lead + 0.3 - t_from)
    _wait_until(turner, start)
    _wait_until(scraper, start + t_from - 0.3 - lead)
    wps, _ = stop(s, x, y, side, start + t_to - _dur(scraper) - lead, z, then, lean)
    scraper += wps + _park(ARM_OF[(s, "scraper")], "scraper")
    turner += t_wps
    return start


def demo():
    red_t, red_s, yel_t, yel_s = (("red", "turner"), ("red", "scraper"), ("yellow", "turner"), ("yellow", "scraper"))
    plans = {"right": take_spatula(red_t), "left": take_spatula(red_s) + _park("left", "scraper")}
    test = os.environ.get("GRILL_TEST", "")
    landed, flipped = {}, {}
    n = 1 if test else 2
    for i in range(n):  # both patties to the grill, with the red set
        _together(plans, "red", scoop("red", *PATTY_START[i], f"patty {i + 1}"), "wedge under", "under",
                  *PATTY_START[i], +1)
        d = set_down("red", GRILL_X[i], PLACE_Y, f"patty {i + 1} onto the grill")
        # The left arm holds the tall red scraper against the patty's right side leaning toward itself, to reach.
        t0 = _together(plans, "red", d, "pull out", "pull out", GRILL_X[i], PLACE_Y, -1, z=STOP_Z_ON_BLADE,
                       lean=SET_DOWN_LEAN)
        landed[i] = t0 + _starts(d, "lower")
    if test == "load":
        return {"left": plans["left"], "right": plans["right"] + _park("right", "turner")}
    for i in range(n):  # flip each SIDE_S after it landed (still raw inside: the red set)
        f = flip(GRILL_X[i], PLACE_Y, f"flip patty {i + 1}")
        t0 = _together(plans, "red", f, "wedge under", "tip it up", GRILL_X[i], PLACE_Y, +1,
                       then=lay_over(GRILL_X[i], PLACE_Y), not_before=landed[i] + SIDE_S - _starts(f, "tip it up"))
        flipped[i] = t0 + _ends(f, "tip it up") + 1.5
    if test == "flip":
        return {"left": plans["left"], "right": plans["right"] + _park("right", "turner")}
    # Swap sets: the red spatulas back to their places, the yellow ones in hand.
    t = max(_dur(plans["right"]), _dur(plans["left"]))
    _wait_until(plans["right"], t)
    _wait_until(plans["left"], t)
    plans["right"] += put_back(red_t) + take_spatula(yel_s) + _park("right", "scraper")
    plans["left"] += put_back(red_s) + take_spatula(yel_t)
    for i in range(n):  # serve each SIDE_S after its flip, with the yellow set
        # The yellow turner comes in over the raw-meat tray: steeply, so its blade stays clear of it.
        sc = scoop("yellow", GRILL_X[i], FLIP_Y, f"serve patty {i + 1}", tilt0=math.radians(10), high=True)
        # Both patties must be cooked through before either is touched with the yellow set (so the two never touch
        # while one is still raw): the first is served when the second is ready.
        _together(plans, "yellow", sc, "wedge under", "under", GRILL_X[i], FLIP_Y, -1,
                  not_before=max(flipped.values()) + SIDE2_S - _ends(sc, "under"))
        _together(plans, "yellow", set_down("yellow", DONE_X[i], DONE_Y, f"patty {i + 1} onto the plate"),
                  "pull out", "pull out", DONE_X[i], DONE_Y, +1, z=STOP_Z_ON_BLADE)
    plans["left"] += _park("left", "turner")
    return plans
