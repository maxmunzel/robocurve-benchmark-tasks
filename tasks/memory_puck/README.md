# Remember a marked position after covering it, then place a puck in front of it

The robot sees ten indicator pads, one of them red. It has to hide all ten under a board itself, and only then move a
puck onto the target pad in front of the red one. When the puck moves, the cue is gone, so the policy must remember it.

The task is the task file `memory_puck.py` for the `yam_arena` embodiment of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (YAM arms at the calibrated `station` in
Isaac Lab-Arena 0.3.1 / Isaac Lab 3.0 / Isaac Sim 6.1), plus `yampatch.py` beside it, which lets the task read the
robot's joints every step (see [Implementation](#implementation)).

## Setup

Coordinates: x forward from the line between the two arm bases, y to the left, from the midpoint between the bases.
All pads are flat coasters, 5 mm thick, glued to the table.

| Object | Size | Mass | Material | Start |
|---|---|---|---|---|
| 10 indicator pads | 4 x 4 cm | fixed | white; number 7 from the left red | row at x 36 cm, 5.5 cm apart, y +24.75 to -24.75 cm |
| 10 target pads | 4 x 4 cm | fixed | grey | row at x 28 cm, one in front of each indicator |
| start pad | 5 x 5 cm | fixed | dark grey | at (17, 0) cm |
| board | 60 x 6.5 x 2.5 cm, long side along y | 300 g | wood | flat on the table at (45, 0) cm, behind the indicators |
| puck | cylinder 3 cm across, 4 cm tall | 30 g | yellow plastic | standing on the start pad |

The red indicator is number 7 (y -8.25 cm). `MEMORY_MARKED=1..10` marks another one for extra trials; the submitted
layout is number 7.

## Scoring

The trial is **lost (0)** if at any time:

- the robot touches the puck before all ten indicators are covered;
- an indicator is uncovered again after all ten were covered; or
- the puck touches anything other than the robot, the start pad and the correct target pad: the table, the board or
  another pad.

*Covered:* every indicator pad's top lies inside the board's footprint seen from above, with the board's bottom at
most 1.5 cm above it. Otherwise:

- **0.25:** all indicators covered.
- **0.5:** also, the puck stands on the correct target pad (its centre over the pad, its bottom on the pad's top
  within 3 mm).
- **1.0:** also, both grippers open (each finger at least 2 cm open, so the puck is released).

The trial ends early with 1.0 once the puck and board have been still (under 1 cm/s and 0.5 rad/s) at full score for
3 s. In simulation, touches are detected every control step from geometry, with a 2 mm tolerance: the robot's
collision meshes (posed from the joint positions with the URDF kinematics) and the puck's surface are sampled about
every 2 mm. On a real station they are judged from the video. `score`, `success` and `distance` (the puck's
horizontal distance from the correct target's centre, in metres) are in `memory_puck.py`; run with
`--scorer task_score`. `MEMORY_DEBUG=N` prints the state every N control steps; a lost trial prints why.

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
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=tasks/memory_puck/memory_puck.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("tasks/memory_puck/task.json"))["instruction"])')" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/memory-demo
```

A policy under test runs the same command with its own `--policy` (Dream-1: `--policy dream --epochs 1`).
`inspect-robots video runs/memory-demo/<log>.json` renders the stored frames to MP4, one file per camera.

- **Run limit:** 2,400 steps (2 minutes at 20 Hz).
- **Seeds and layout:** one fixed starting layout (red indicator number 7); use `-E object_jitter=0`. With jitter,
  seeded resets shift the board and puck by up to 3 cm.
- **Runtime:** the scripted demo succeeds after about 400 steps (20 s simulated) and ends at 459 steps with
  `task_score` 1.0; the run takes a few minutes wall-clock on one GPU, including Isaac Sim start-up.
- **Dream-1 result:** one `dream` episode (`--epochs 1`, same command with `--policy dream`) reached `task_score` 0.0
  at the 2,400-step limit, with no trial error: it moved the arms over the pads but neither covered the indicators
  nor touched the puck. It took about 8 minutes wall-clock, including simulator start-up and API inference.

The demo:

1. Both arms grip the board near its ends (20 cm either side of its centre, fingers closing across its depth), lift
   it 3 cm, carry it 9 cm towards the robots and set it down on the indicator pads, then let go.
2. The left arm parks. The right arm picks the puck off the start pad, carries it 10 cm above the table to target
   pad 7, sets it down, lets go and parks.

## Simulation notes

- Pads, board and puck are rigid boxes and a cylinder with the adapter's grippy friction for graspable objects
  (static 1.2, dynamic 1.0). The pads are fixed in place, like coasters taped to the table.
- Physics runs at 240 Hz, control at 20 Hz.

## Implementation

The stock task-file format cannot read the robot. `yampatch.py` (shared with the other patched tasks) adds an
`on_step` hook, called once per control step, in which the task reads the joint positions of `left_robot` and
`right_robot`: the finger joints for the open-gripper check, and all joints to pose the collision meshes for the
touch check. It also adds an `on_reset` hook that clears the trial's state. It wraps `yam_arena.scene.object_states`
and `YamArenaEmbodiment.reset` of the pinned adapter; the adapter's files are not changed.

## Media

The setup screenshot, the full-score scripted demo and the Dream-1 evaluation video (top camera) are uploaded
separately to the task board and identified by the `media` entry in `task.json`. The demo was recorded at
`-E cam_width=640 -E cam_height=360`; the Dream-1 run at 320 x 176.
