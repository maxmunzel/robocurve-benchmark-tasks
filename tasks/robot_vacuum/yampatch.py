"""Extensions to the ``yam_arena`` task-file format, installed at runtime by patching the adapter (the adapter's files
are not changed). Copy this file next to a task file and load it as ``patched_template.py`` does.

What it adds:
- :class:`UsdObject`: spawn any USD file (rigid body, fixed collider, or articulation with joints) at a full pose.
  The USD may be generated lazily by a ``build`` function, which runs after Isaac Sim has started (``pxr`` is safe
  to import there). For full control (actuators, articulation settings, initial joint state), ``make`` returns a
  ready Arena ``Object`` instead, also after Isaac Sim has started. :func:`build_button_panel` generates a fixed-base articulation of spring-loaded push buttons.
- ``on_reset(env)``: called after every episode reset (after the scene has settled), so a task can clear its state.
- ``on_step(env, states)``: called once per control step, before ``success``/``distance``/``score``; it may read
  the simulator (e.g. :func:`joint_positions`) and add entries to ``states``.

Patched (all in the pinned adapter, ``yam-isaac-lab-arena`` 0059d02):
- ``yam_arena.scene.task_object``: spawns ``UsdObject``s (wrapped once ``yam_arena.scene`` is imported, which happens
  after Isaac Sim starts; an import hook waits for it).
- ``yam_arena.scene.object_states``: calls ``on_step`` (``_task_step`` calls it once per step).
- ``yam_arena.inspect_embodiment.YamArenaEmbodiment.reset``: calls ``on_reset``.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from yam_arena.task_file import Asset

_HOOKS: dict[str, Callable | None] = {"on_reset": None, "on_step": None}


def install(*, on_reset: Callable | None = None, on_step: Callable | None = None) -> None:
    """Register the task's hooks and patch the adapter (once per process)."""
    _HOOKS.update(on_reset=on_reset, on_step=on_step)
    _patch_embodiment()
    if "yam_arena.scene" in sys.modules:
        _patch_scene(sys.modules["yam_arena.scene"])
    elif not any(isinstance(f, _AfterSceneImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterSceneImport())


# -- objects ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UsdObject(Asset):
    """A USD file spawned at ``pos`` / ``rot_xyzw`` (world, m). ``kind``: ``"rigid"`` (free, needs a ``mass`` for the
    task-file checks; the USD's own mass is used), ``"fixed"`` (static collider) or ``"articulation"`` (its joints and
    drives as authored; give it a fixed base in the USD to keep it in place). ``build`` makes the USD and returns its
    path, after Isaac Sim starts; otherwise ``usd_path`` names it. ``asset``/``xy``/``extent`` only satisfy the
    task-file checks: use :func:`usd_object`."""

    usd_path: str = ""
    build: Callable[[], str] | None = field(default=None, compare=False)
    make: Callable[[], object] | None = field(default=None, compare=False)  # returns an isaaclab_arena Object
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: float = 1.0
    kind: str = "rigid"

    @property
    def free(self) -> bool:  # seeded resets jitter free objects only
        return self.kind == "rigid"


def usd_object(name: str, pos, *, kind: str = "rigid", usd_path: str = "", build=None, make=None, rot_xyzw=(0, 0, 0, 1),
               scale: float = 1.0, mass: float | None = None, extent: float = 0.1) -> UsdObject:
    if kind not in ("rigid", "fixed", "articulation"):
        raise ValueError(f"{name}: kind must be rigid, fixed or articulation, got {kind!r}")
    if kind == "rigid" and mass is None:
        raise ValueError(f"{name}: a rigid UsdObject needs a mass")
    return UsdObject(
        name, asset=name, xy=(pos[0], pos[1]), extent=extent, mass=mass if kind == "rigid" else None,
        usd_path=usd_path, build=build, make=make, pos=tuple(pos), rot_xyzw=tuple(rot_xyzw), scale=scale, kind=kind,
    )


def _spawn(scene, spec: UsdObject):
    if spec.make is not None:
        return spec.make()
    path = spec.build() if spec.build is not None else spec.usd_path
    kind = {"rigid": scene.ObjectType.RIGID, "fixed": scene.ObjectType.BASE,
            "articulation": scene.ObjectType.ARTICULATION}[spec.kind]
    return scene.Object(
        name=spec.name, usd_path=str(path), object_type=kind, scale=(spec.scale,) * 3,
        initial_pose=scene.Pose(position_xyz=spec.pos, rotation_xyzw=spec.rot_xyzw),
    )


def joint_positions(env, name: str) -> dict[str, float]:
    """An articulation's joint positions (m or rad) by joint name, env 0."""
    art = env.unwrapped.scene[name]
    q = art.data.joint_pos
    q = getattr(q, "torch", q)
    return dict(zip(art.joint_names, q[0].detach().cpu().numpy().tolist(), strict=True))


# -- generated USD: a panel of spring-loaded push buttons ------------------------------------------------------------


@dataclass(frozen=True)
class Button:
    """A box that slides down ``travel`` (m) on a prismatic joint and springs back. ``pos``: its centre at rest,
    relative to the panel's origin; ``size``: full x, y, z (m)."""

    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float] = (0.85, 0.85, 0.85)
    travel: float = 0.004
    stiffness: float = 150.0  # N/m
    damping: float = 1.0  # N s/m
    mass: float = 0.005  # kg


def build_button_panel(name: str, base_size, base_color, buttons: list[Button], base_mass: float = 1.0) -> str:
    """Write a fixed-base articulation: a box base (``base_size``, its bottom at the origin, fixed to the world) and
    one prismatic joint per button (joint name = button name, 0 at rest, ``-travel`` fully pressed). Returns the path.
    Call it after Isaac Sim has started (it imports ``pxr``)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    path = Path(tempfile.gettempdir()) / "yam_tasks" / f"{name}.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    def body(prim_path: str, at, size, color, mass: float, box_at=(0.0, 0.0, 0.0)):
        xf = UsdGeom.Xform.Define(stage, prim_path)
        xf.AddTranslateOp().Set(Gf.Vec3d(*at))
        UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(float(mass))
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/box")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*box_at))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return xf

    bx, by, bz = base_size
    # The base body's frame is the panel's origin (joint anchors are given in it); its box sits on top of it.
    body(f"/{name}/base", (0.0, 0.0, 0.0), base_size, base_color, base_mass, box_at=(0.0, 0.0, bz / 2))
    fix = UsdPhysics.FixedJoint.Define(stage, f"/{name}/joints/fix_base")
    fix.CreateBody1Rel().SetTargets([Sdf.Path(f"/{name}/base")])
    for b in buttons:
        body(f"/{name}/{b.name}", b.pos, b.size, b.color, b.mass)
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/{name}/joints/{b.name}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/{name}/base")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/{name}/{b.name}")])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*b.pos))  # in the base body's frame: the panel's origin
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLowerLimitAttr(-float(b.travel))
        joint.CreateUpperLimitAttr(0.0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(b.stiffness))
        drive.CreateDampingAttr(float(b.damping))
        drive.CreateTargetPositionAttr(0.0)
    stage.GetRootLayer().Export(str(path))
    return str(path)


# -- patches ---------------------------------------------------------------------------------------------------------


def _patch_scene(scene) -> None:
    if getattr(scene, "_yampatch", False):
        return
    stock_object, stock_states = scene.task_object, scene.object_states

    def task_object(spec):
        return _spawn(scene, spec) if isinstance(spec, UsdObject) else stock_object(spec)

    def object_states(env, names):
        # Arena's geometry queries (centroid, bounding box) do not cover articulations: give those their root pose.
        articulated = set(env.unwrapped.scene.articulations) & set(names)
        states = stock_states(env, [n for n in names if n not in articulated])
        for name in articulated:
            states[name] = _root_state(env, name)
        if _HOOKS["on_step"] is not None:
            _HOOKS["on_step"](env, states)
        return states

    scene.task_object, scene.object_states, scene._yampatch = task_object, object_states, True


def _root_state(env, name: str):
    """An articulation's root pose as an ``ObjectState``: no velocities, a zero-size bounding box at its origin."""
    import numpy as np

    from yam_arena.task_file import ObjectState

    base = env.unwrapped
    pose = base.arena_world.get_pose_w(name)[0].detach().cpu().numpy().astype(np.float64)
    pos = pose[:3] - base.scene.env_origins[0].detach().cpu().numpy()
    return ObjectState(pos, pose[3:], np.zeros(3), np.zeros(3), pos.copy(), pos.copy())


def _patch_embodiment() -> None:
    from yam_arena import inspect_embodiment

    cls = inspect_embodiment.YamArenaEmbodiment
    if getattr(cls.reset, "_yampatch", False):
        return
    stock = cls.reset

    def reset(self, scene, *, seed=None):
        obs = stock(self, scene, seed=seed)
        if _HOOKS["on_reset"] is not None:
            _HOOKS["on_reset"](self._env)
        return obs

    reset._yampatch = True
    cls.reset = reset


class _AfterSceneImport(importlib.abc.MetaPathFinder):
    """Patches ``yam_arena.scene`` right after its first import (it needs Isaac Sim, which starts later)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "yam_arena.scene":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _patch_scene(module)

        spec.loader.exec_module = exec_module
        return spec

