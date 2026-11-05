# Move diced salmon onto a plate with a knife, without touching the food

In professional kitchens, raw fish is moved with utensils rather than hands. Here the robot moves five diced salmon
pieces from a cutting board on a counter onto a plate at table level, without touching the salmon with any part of
the robot. It has to use the knife as a tool and position the board so the pieces can be swept onto the plate.

The task is the task file `salmon_transfer.py` for the `yam_arena` embodiment of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (YAM arms at the calibrated `station` in
Isaac Lab-Arena 0.3.1 / Isaac Lab 3.0 / Isaac Sim 6.1), plus `yampatch.py` beside it, which lets the task check every
control step whether the robot touches the salmon (see [Implementation](#implementation)).

## Setup

Coordinates: x forward from the line between the two arm bases, y to the left, from the midpoint between the bases.
Heights are from the table top.

| Object | Size | Mass | Material | Start |
|---|---|---|---|---|
| kitchen counter block | 30 x 28 x 10 cm (x by y by z) | 30 kg | grey, rigid | centred at (29, 20) cm: right edge at y 6 cm, near edge at x 14 cm |
| cutting board | 18 x 7 x 1.5 cm, long side along x | 110 g | wood | on the counter, centred at (29, 10.5) cm |
| 5 salmon pieces | 1.5 cm cubes | 3.5 g each | raw salmon | on the board's centre line, at x 27, 29, 31, 33, 35 cm |
| knife | 1 x 1 x 20 cm bar, long side along x | 100 g | steel | on the counter, centred at (29, 30) cm |
| plate | 18 cm across, 1.5 cm tall | 400 g | white ceramic | on the table, centred at (21, -3.5) cm, 5 mm right of the counter |

## Scoring

- **0.2 per salmon piece on the plate** at the end: its centre is within the plate's 9 cm radius (horizontally from the
  plate's centre) and its bottom rests on the plate (at most 2 cm above the plate's top, so a piece still on the
  board does not count).
- **0** if any part of the robot touched any salmon piece at any time. The knife, board and plate may touch it.
- The trial ends early with **1.0** once all five pieces have lain still on the plate (under 1 cm/s and 0.5 rad/s)
  for 6 s without having been touched.

On a real station, a touch is judged from the video. In simulation, the robot's collision meshes are sampled about
every 2 mm, posed every control step from both arms' joint positions with the URDF kinematics, and a point within
3 mm of a piece (in the piece's own, rotated frame) counts as a touch. The touch is reported with the arm and link.
`score`, `success` and `distance` (summed horizontal distance of the pieces from the plate's centre, in metres) are in
`salmon_transfer.py`; run with `--scorer task_score`. `SALMON_DEBUG=N` prints every N control steps how many pieces
are on the plate, the board's pose and the pieces' positions relative to the board.

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
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=tasks/salmon_transfer/salmon_transfer.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("tasks/salmon_transfer/task.json"))["instruction"])')" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/salmon-demo
```

A policy under test runs the same command with its own `--policy` (Dream-1: `--policy dream --epochs 1`).
`inspect-robots video runs/salmon-demo/<log>.json` renders the stored frames to MP4, one file per camera.

- **Run limit:** 2,400 steps (2 minutes at 20 Hz).
- **Seeds and layout:** one fixed starting layout; use `-E object_jitter=0`. With jitter, seeded resets shift every
  loose object by up to 3 cm.
- **Runtime:** the scripted demo succeeds after 672 steps (34 s simulated) with `task_score` 1.0; the run takes about
  5 minutes wall-clock on one GPU, including Isaac Sim start-up.
- **Dream-1 result:** one `dream` episode (`--epochs 1`, same command with `--policy dream`) reached `task_score` 0.0
  at the 2,400-step limit, with no trial error and no touch; the arms moved towards the counter but did not move the
  board, knife or salmon. It took about 12 minutes wall-clock, including simulator start-up and API inference.

The demo uses both arms:

1. The left arm pushes the board 10 cm towards the robots, so it overhangs the counter's near edge by 4 cm.
2. The right arm grips the overhang across the board's width with a gentle squeeze, and drags the board 5.5 cm right
   until it sticks out 4.5 cm over the counter's right edge, above the plate. It holds it there.
3. The left arm picks up the knife by its far end and sets it across the board, behind the salmon. It sweeps the
   pieces off the board's right side, and they drop onto the plate. The left fingers stay beyond the board's far
   end, and the right hand stays in front of its near end.
4. The right arm drags the board back until it rests on the counter and lets go. Both arms park.

## Simulation notes

- The salmon pieces are rigid 1.5 cm cubes, with no deformation or stickiness. The knife is a plain bar with no blade
  or handle. The counter, board and plate are a rigid box, a box and a cylinder.
- All objects use the adapter's grippy friction for graspable objects (static 1.2, dynamic 1.0). The counter is a
  free 30 kg block, so pushing on it barely moves it.
- Physics runs at 240 Hz, control at 20 Hz.

## Implementation

The stock task-file format cannot read the robot. `yampatch.py` (shared with the other patched tasks) adds an
`on_step` hook, called once per control step, in which the task reads the joint positions of `left_robot` and
`right_robot` to pose their collision meshes. It also adds an `on_reset` hook that clears the touch flag at every
reset. It wraps `yam_arena.scene.object_states` and `YamArenaEmbodiment.reset` of the pinned adapter; the adapter's
files are not changed.

## Media

The setup screenshot, the full-score scripted demo and the Dream-1 evaluation video (top camera) are uploaded
separately to the task board and identified by the `media` entry in `task.json`. The demo was recorded with the
command above at `-E cam_width=640 -E cam_height=360`; the Dream-1 run at 320 x 176.
