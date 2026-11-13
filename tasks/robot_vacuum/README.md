# Free a stuck robot vacuum and get it cleaning again

A robot vacuum has stopped on the table, its status ring lit red. It sits visibly crooked: it has driven onto a TV
remote control, which is wedged under its front edge, with one end sticking out. The instruction only says to get the
vacuum cleaning again; the robot has to work out why it stopped. The vacuum (3 kg) pins the remote, so the robot
should lift the vacuum's edge with one hand and pull the remote out with the other, set the vacuum down flat and press
its start button. The vacuum starts only when it stands flat with nothing under it, as robot vacuums refuse to start
when lifted or blocked.

The task is the task file `robot_vacuum.py` for the `yam_arena` embodiment of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (YAM arms at the calibrated `station` in
Isaac Lab-Arena 0.3.1 / Isaac Lab 3.0 / Isaac Sim 6.1), plus `yampatch.py` beside it, which lets the task spawn its
own objects (the vacuum is an articulation with a sprung button) and check the vacuum every control step (see
[Implementation](#implementation)).

## Setup

Coordinates: x forward from the line between the two arm bases, y to the left, from the midpoint between the bases.
Heights are from the table top.

| Object | Size | Mass | Material | Start |
|---|---|---|---|---|
| robot vacuum | round, 34 cm across, 7.4 cm tall (a 7 cm chassis and a 4 mm lid); a darker bumper band around its side | 3 kg | grey plastic chassis (friction 0.5), black lid | centred at (37, 0) cm, resting tilted on the remote (about 4 degrees; its front edge about 2.5 cm off the table), its back edge on the table |
| start button | 3.6 cm across, standing 4 mm proud of the lid's middle; travels 4 mm, sprung (0.6 N at rest, 3 N fully pressed) | 10 g | silver | |
| status ring | 5.2 cm across, around the button | | lit red while stopped, green once started | |
| TV remote control | 16.4 x 3.6 x 2.5 cm (Arena's library asset `remote_control_hot3d_robolab`, a scan) | 0.2 kg | black plastic | lying flat, buttons up, its length along y, at x 19.7-23.3 cm from y +2.2 to -14.2 cm: its inner 9 cm under the vacuum's front edge (up to 3 cm inside the rim), the other 7 cm sticking out to the right |

The remote's collider is the scan's convex decomposition, as shipped with the asset.

## Scoring

- The remote is **clear** when no part of its outline (its bounding box seen from above) is within 5 mm of the
  vacuum's footprint (a 34 cm disc).
- The vacuum stands **flat** when it is tilted at most 2 degrees and its underside is within 3 mm of the table.
- The vacuum **starts** when its button is pressed (at least 3 mm down) and released while it stands flat with the
  remote clear. Its ring then turns green and it backs away 15 cm over 3 s, as if driving off to clean. Pressing the
  button at any other time does nothing.

Score:

- **1.0** once the vacuum has started (and stands flat);
- else **0.5** if it stands flat with the remote clear;
- else **0.25** if the remote is clear;
- else **0**.

The trial ends with success 1 s after the vacuum has finished backing away.

`score`, `success` and `distance` (steps left: free the remote, set the vacuum flat, start it) are in
`robot_vacuum.py`; run with `--scorer task_score`. `VACUUM_DEBUG=N` prints the vacuum's tilt, the remote's clearance,
the button and the score every N control steps (`VACUUM_DEBUG_HANDS=1` adds both hands' poses).

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
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=tasks/robot_vacuum/robot_vacuum.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  --instruction "$(python -c 'import json; print(json.load(open("tasks/robot_vacuum/task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1800 --store-frames --log-dir runs/vacuum-demo
```

A policy under test runs the same command with its own `--policy` (Dream-1: `--policy dream --epochs 1`).
`inspect-robots video runs/vacuum-demo/<log>.json` renders the stored frames to MP4, one file per camera.

- **Run limit:** 1,800 steps (90 s at 20 Hz).
- **Seeds and layout:** one fixed starting layout; use `-E object_jitter=0`.
- **Runtime:** the scripted demo succeeds after 638 steps (32 s simulated) with `task_score` 1.0.
- **Dream-1 result:** one `dream` episode (`--epochs 1`, same command with `--policy dream`) reached `task_score` 0.0
  at the 1,800-step limit: its hands moved around and over the vacuum but never lifted it or pulled the remote, and
  the vacuum and the remote did not move. It took about 6.5 minutes wall-clock, including simulator start-up and API
  inference.

### The demo

The demo is scripted from the known layout; each move aims at where the vacuum and the remote are when it starts.

1. The left hand, tilted 30 degrees below horizontal and pointing at the vacuum's centre, slides its lower finger
   3 cm in under the raised edge where it faces the left arm, its upper finger over the top, and pinches the edge.
   It lifts the edge 3 cm (the vacuum tilts to about 8 degrees), freeing the remote.
2. The right hand, leaning away from the vacuum, grips the end of the remote that sticks out and pulls the remote
   13.5 cm out along its length, then lets go.
3. The left hand lowers the edge to 12 mm above the table, lets go and draws its lower finger out, so the vacuum drops
   flat.
4. The left hand presses the start button with its closed fingertips and lets go; the ring turns green and the vacuum
   backs away.

## Simulation notes

- The vacuum is a rigid cylinder with a flat underside (no wheels, brushes or motor); it slides on the table with
  friction 0.5. Its button is a prismatic joint with a spring-like position drive.
- The vacuum's driving off after it starts is scripted: the task moves it straight back 15 cm over 3 s.
- The remote is Arena's scanned asset at its real size and mass, with its own convex-decomposition collider.
- Physics runs at 240 Hz, control at 20 Hz.

## Implementation

The stock task-file format scores only the final state and supports only simple primitives. `yampatch.py` (shared
with the other patched tasks) adds an `on_step` hook, called once per control step, in which the task reads the
vacuum's and the remote's poses and the button's joint, starts the vacuum when the button is released under the right
conditions, switches the status ring (by making the red or green ring prim visible) and moves the started vacuum. It
also adds an `on_reset` hook that clears the trial's state, and lets the task spawn its own USD bodies: the vacuum (a
floating articulation generated in code) and the remote (the library asset, laid flat at a full pose). It wraps
`yam_arena.scene.object_states`, `yam_arena.scene.task_object` and `YamArenaEmbodiment.reset` of the pinned adapter;
the adapter's files are not changed.

### Assets

The vacuum is generated by `robot_vacuum.py` (part of this task, under the repository's license). The remote is
`remote_control_hot3d_robolab` from the Isaac Lab-Arena 0.3.1 object library (NVIDIA's SRL RoboLab assets, from the
HOT3D scans); the task references it by name and loads it from Arena's asset server under the terms it is distributed
with. No asset files are copied into this folder.

## Media

The setup screenshot, the full-score scripted demo and the Dream-1 evaluation video (top camera) are uploaded
separately to the task board and identified by the `media` entry in `task.json`. The demo was recorded at
`-E cam_width=640 -E cam_height=360`; the Dream-1 run at 320 x 176.
