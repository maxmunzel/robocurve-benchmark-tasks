# Stack two Stonehenge-style trilithons from wooden blocks

The robot builds two trilithons (two standing stones with a lintel across them, as at Stonehenge), one on top of the
other, from six identical 1 x 2 x 8 cm blocks. The finished structure is 18 cm tall and free-standing: every upright
stands on a 1 x 2 cm end, so each placement has to be accurate to a few millimetres and gentle enough not to topple
what is already built.

The task is the task file `double_trilithon.py` for the `yam_arena` embodiment of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (YAM arms at the calibrated `station` in
Isaac Lab-Arena 0.3.1 / Isaac Lab 3.0 / Isaac Sim 6.1), plus `yampatch.py` beside it, which lets the task read
the grippers' finger positions (see [Implementation](#implementation)).

## Setup

Coordinates: x forward from the line between the two arm bases, y to the left, from the midpoint between the bases.

| Object | Size | Mass | Material | Start |
|---|---|---|---|---|
| block 1 | 1 x 2 x 8 cm | 10 g | wood (pine), painted grey | upright on a 1 x 2 cm end, 2 cm side along x, at (20, 19) cm |
| block 2 | same | 10 g | same | upright, at (30, 19) cm |
| block 3 | same | 10 g | same | upright, at (20, -19) cm |
| block 4 | same | 10 g | same | upright, at (30, -19) cm |
| block 5 | same | 10 g | same | flat on a 2 x 8 cm face, long side along y, at (31, 9) cm |
| block 6 | same | 10 g | same | flat, long side along y, at (31, -9) cm |

Nothing else is on the table. The structure may be built anywhere on the table.

## Scoring

Heights are measured from the table top with a ruler. An *upright* block's bounding box is at least 7.5 cm tall; a
*flat* block's is at most 1.5 cm tall and at least 7.5 cm long. A *trilithon* is two upright blocks whose bottoms are
within 5 mm of the same surface, with a clear gap of at least 2 cm between them, and a flat block whose bottom is
within 5 mm of both their tops and directly above both.

With both grippers open (each finger at least 2 cm open, so no block is held):

- **0.25:** a trilithon stands on the table.
- **0.5:** also, an upright block stands on its flat block.
- **0.75:** two upright blocks stand on that flat block, at least 2 cm apart.
- **1.0:** a second flat block lies across those two, and the topmost block's top is 18 cm (within 5 mm) above the
  table.
- **0** if either gripper is closed at the end.

The trial ends early with 1.0 once the full structure has stood still (every block under 1 cm/s and 0.5 rad/s) for
1 s with both grippers open. `score`, `success` and `distance` (how far the topmost block's top is from 18 cm, in
metres) are in `double_trilithon.py`; run with `--scorer task_score`. `TRILITHON_DEBUG=N` prints the stage reached,
the topmost height and both grippers' finger opening every N control steps.

## Running

Needs Linux with an NVIDIA RTX GPU (Isaac Sim needs RT cores) and about 200 GB of disk. Install the pinned
adapter, which installs Isaac Lab-Arena and Inspect Robots (from the root of this repository):

```bash
mkdir -p .deps
git clone https://github.com/kairosiann/yam-isaac-lab-arena.git .deps/yam-isaac-lab-arena
git -C .deps/yam-isaac-lab-arena checkout 0059d02a1a91bc6834c1d0846022891df0c78730
ARENA_DIR="$PWD/.deps/IsaacLab-Arena" .deps/yam-isaac-lab-arena/setup.sh
```

In each shell:

```bash
source .deps/IsaacLab-Arena/yam.env
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
```

Scripted full-score demo (from the repository root):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=tasks/double_trilithon/double_trilithon.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("tasks/double_trilithon/task.json"))["instruction"])')" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/trilithon-demo
```

A policy under test runs the same command with its own `--policy` (Dream-1: `--policy dream --epochs 1`).
`inspect-robots video runs/trilithon-demo/<log>.json` renders the stored frames to MP4, one file per camera.

- **Run limit:** 2,400 steps (2 minutes at 20 Hz).
- **Seeds and layout:** one fixed starting layout; use `-E object_jitter=0`. With jitter, seeded resets shift the
  blocks by up to 3 cm.
- **Runtime:** the scripted demo succeeds after about 1,680 steps (84 s simulated) with `task_score` 1.0; the run
  takes about 5 minutes wall-clock on one GPU, including Isaac Sim start-up.
- **Dream-1 result:** one `dream` episode (`--epochs 1`, same command with `--policy dream`) reached `task_score` 0.0
  at the 2,400-step limit, with no trial error; it moved blocks but built no trilithon. It took about 6 minutes
  wall-clock, including simulator start-up and API inference.

The demo moves one block at a time, straight down grips only: the left arm stands block 1 at the left end of the
lower trilithon, the right arm block 3 at the right end (6 cm opening), the left arm lays block 5 across them, then
the left and right arms stand blocks 2 and 4 on it and the right arm lays block 6 on top. The structure is centred at
(22, 0) cm, where both arms reach the 18 cm top with the gripper pointing straight down.

## Simulation notes

- Blocks are rigid cuboids with the adapter's rubber-like friction for graspable objects (static 1.2, dynamic 1.0);
  no wood grain, edge rounding or surface imperfections.
- Physics runs at 240 Hz, control at 20 Hz.

## Implementation

The stock task-file format cannot read the robot. `yampatch.py` (shared with the other patched tasks) adds an
`on_step` hook, called once per control step, in which the task reads the finger joints `joint7`/`joint8` of
`left_robot` and `right_robot`. It wraps `yam_arena.scene.object_states` and `YamArenaEmbodiment.reset` of the
pinned adapter; the adapter's files are not changed.

## Media

The setup screenshot, the full-score scripted demo and the Dream-1 evaluation video (top camera) are uploaded
separately to the task board and identified by the `media` entry in `task.json`. The demo was recorded with the
command above at `-E cam_width=640 -E cam_height=360`; the Dream-1 run at 320 x 176.
