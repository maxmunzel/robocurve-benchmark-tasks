# Lift a board and balance a roller on it for 10 seconds

A long wooden board rests on a low stand, with a wooden roller lying across its middle. The robot lifts the board with
both hands and keeps it up for 10 s without the roller falling off or touching the robot. The roller rolls along the
board as soon as the board tilts, so holding a pose is not enough: the policy has to watch the roller and keep
correcting the board's tilt, as when two people carry a tray or a plank with something loose on it.

The task is the task file `balance_board.py` for the `yam_arena` embodiment of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (YAM arms at the calibrated `station` in
Isaac Lab-Arena 0.3.1 / Isaac Lab 3.0 / Isaac Sim 6.1), plus `yampatch.py` beside it, which lets the task check the
rules every control step and spawn the lying roller (see [Implementation](#implementation)).

## Setup

Coordinates: x forward from the line between the two arm bases, y to the left, from the midpoint between the bases.
The posts are glued to the table.

| Object | Size | Mass | Material | Start |
|---|---|---|---|---|
| 2 posts (the stand) | 4 x 4 x 4 cm cubes | fixed | dark grey | x 38 cm, y +12 and -12 cm |
| board | 6 cm deep (x), 54 cm long (y), 2 cm thick | 300 g | wood | on the posts, centred at x 38 cm, y 0; its bottom 4 cm above the table |
| roller | solid cylinder, 4 cm across, 8 cm long | 70 g | orange wood | lying on the board's middle, axis along x (it rolls along the board), 1 cm over each side of the board |

The stand lifts the board 4 cm off the table, so the fingers can reach under its ends. The roller is longer than the
board is deep, so it cannot roll between the fingers of a hand that holds the board by its sides: a hand is an end
stop, and hitting it loses the trial. Its rolling is damped, as on a felt-covered board (see
[Simulation notes](#simulation-notes)).

## Scoring

- *Lifted:* the board's lowest point is at least 5 cm above the stand's top.
- *On the board:* the roller's centre is over the board's top face, and its bottom within 5 mm of that face.

The trial is **lost (0)** if at any time:

- the roller is not on the board (it fell off);
- any part of the robot comes within 2 mm of the roller;
- the board is dropped onto the table.

Otherwise: **0.1 per full second** of the longest stretch with the board lifted (0.1 to 0.9), and **1.0** once that
stretch reaches 10 s; the trial then ends with success. Lowering the board below 5 cm restarts the stretch.

`score`, `success` and `distance` (seconds still to go) are in `balance_board.py`; run with `--scorer task_score`.
`BALANCE_DEBUG=N` prints the board's height, the roller's position along the board and the time balanced every N
control steps. A lost trial prints why.

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
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=tasks/balance_board/balance_board.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("tasks/balance_board/task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1200 --store-frames --log-dir runs/balance-demo
```

A policy under test runs the same command with its own `--policy` (Dream-1: `--policy dream --epochs 1`).
`inspect-robots video runs/balance-demo/<log>.json` renders the stored frames to MP4, one file per camera.

- **Run limit:** 1,200 steps (1 minute at 20 Hz).
- **Seeds and layout:** one fixed starting layout; use `-E object_jitter=0`.
- **Runtime:** the scripted demo succeeds and ends after 333 steps (17 s simulated) with `task_score` 1.0: 8 s
  to grip and lift the board, then 10 s balancing it.
- **Dream-1 result:** one `dream` episode (`--epochs 1`, same command with `--policy dream`) reached `task_score` 0.0
  at the 1,200-step limit, with no rule broken: both hands went to the board's ends but never lifted it. It took
  about 4 minutes wall-clock, including simulator start-up and API inference.

### The demo's controller

Both arms grip the board's ends, 21 cm from its middle, with the fingers closing across its depth, and lift it 10 cm
over 3 s. Then a PD controller balances it, every control step (20 Hz):

    tilt = KP * offset + KD * speed        KP = 0.6 rad/m, KD = 0.4 rad/(m/s), at most 4 deg, at most 20 deg/s

where offset and speed are the roller's position and velocity along the board, measured from its middle. The left
hand rises and the right hand sinks by 21 cm * tan(tilt), or the other way round, so the end the roller is heading
for comes up and rolls it back. The pinch acts as a hinge about the fingers' closing axis, so the board tilts in the
grip. The controller reads the roller's pose from the simulator, through the same per-step hook that checks the rules.

Two switches show the controller's effect:

- `BALANCE_PUSH=0.25` gives the roller a 25 cm/s push along the board once the board is up. With the controller,
  the board tilts up to 4 degrees, the roller swings out to 8 cm from the middle and is back within 1 cm about
  1.5 s later; the demo still scores 1.0.
- `BALANCE_CONTROLLER=0` holds the board level instead. The board's residual tilt (about 0.3 degrees,
  from the lift) rolls the roller into the right hand 3 s after the lift ends, and the trial is lost: "the robot
  touched the roller".

## Simulation notes

- The board, posts and roller are rigid primitives with the adapter's grippy friction (static 1.2, dynamic 1.0).
- Rolling resistance is modelled as angular damping on the roller (1/s, PhysX), standing in for a felt-covered board.
  It slows the roller without stopping it: held level by the robot, the board's small residual tilt still rolls
  the roller off within the 10 s.
- Touching is judged geometrically: the roller's surface (sampled every 2 mm) against the robot's collision meshes
  at its current joint positions.
- Physics runs at 240 Hz, control at 20 Hz.

## Implementation

The stock task-file format scores only the final state and supports only upright cylinders. `yampatch.py` (shared
with the other patched tasks) adds an `on_step` hook, called once per control step, in which the task checks the
rules, counts the time balanced and runs the demo's controller. It also adds an `on_reset` hook that clears the
trial's state, and lets a task spawn its own object (here the roller, a cylinder lying along x with angular
damping). It wraps `yam_arena.scene.object_states`, `yam_arena.scene.task_object` and
`YamArenaEmbodiment.reset` of the pinned adapter; the adapter's files are not changed.

## Media

The setup screenshot, the full-score scripted demo and the Dream-1 evaluation video (top camera) are uploaded
separately to the task board and identified by the `media` entry in `task.json`. The demo was recorded at
`-E cam_width=640 -E cam_height=360`; the Dream-1 run at 320 x 176.
