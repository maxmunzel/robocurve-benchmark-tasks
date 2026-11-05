# Play a matching-pairs game with covered coloured discs

Six coloured discs, three pairs, are hidden under identical covers. The robot plays the usual matching-pairs game
(also known as concentration or memory):
- Each turn, it lifts two covers at once.
- **Match:** it sets both covers aside on a mat.
- **No match:** it puts both covers back.

The colours are visible only during a turn. To find the pairs instead of guessing, the policy has to remember what was
under covers it has already put back.

The task is the task file `pairs_game.py` for the `yam_arena` embodiment of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (YAM arms at the calibrated `station` in
Isaac Lab-Arena 0.3.1 / Isaac Lab 3.0 / Isaac Sim 6.1), plus `yampatch.py` beside it, which lets the task check the
rules every control step and read the grippers (see [Implementation](#implementation)).

## Setup

Coordinates: x forward from the line between the two arm bases, y to the left, from the midpoint between the bases.
Coasters, discs and the mat are glued to the table.

| Object | Size | Mass | Material | Start |
|---|---|---|---|---|
| 6 coasters | round, 7 cm across, 3 mm thick | fixed | grey | triangular grid, each 20 cm from its nearest neighbours: pads 1-4 at x 40.3 cm, y +30, +10, -10, -30 cm; pads 5-6 at x 23.0 cm, y +20, -20 cm |
| 6 discs | round, 3.4 cm across, 2 mm thick | fixed | red, green, blue | centred on the coasters: 1 red, 2 green, 3 blue, 4 green, 5 blue, 6 red |
| 6 covers | closed cylinder (an upturned cup), 6 cm across, 5 cm tall | 50 g | opaque, dark grey | standing centred on each disc |
| mat | 9 x 56 cm, 4 mm thick, long side along y | fixed | white | centred at (12, 0) cm, in front of the pads |

Pads are numbered left to right as seen from the robots, far row first. The coaster's rim, 5 mm wider than a cover all
round, shows where a cover belongs.

## Scoring

- *Uncovered:* a disc is uncovered unless a cover stands on it. The cover's bottom must be centred within 1.3 cm of
  the disc's centre (then the whole disc is hidden), at most 5 mm above the disc, and tilted at most 10°. This is
  exact for round covers over round discs. Lifting, sliding or tipping a cover to peek counts as uncovering.
- *On the mat:* a cover stands upright (at most 10°), with its bottom on the mat (within 5 mm) and its bottom centre
  over the mat.
- *Cleared pair:* both discs of a colour are uncovered, and two covers per cleared pair stand on the mat.

The trial is **lost (0)** if at any time:

- more than two discs are uncovered, not counting cleared pairs. This also catches leaving a non-matching pair
  uncovered and lifting a third cover, or lifting another cover while a matched pair is still in the hands;
- a cover is set aside on the mat while its disc's partner is still covered (no match shown);
- a cover is knocked over (tilted more than 45°) or falls off the table.

Otherwise: **0.3 per cleared pair**; **1.0** with all three cleared and both grippers open (each finger at least 4 cm
open, wider than a cover). The trial ends early with 1.0 once all covers have been still (under 1 cm/s and
0.5 rad/s) at full score for 3 s.

`score`, `success` and `distance` (covers not yet on the mat) are in `pairs_game.py`; run with
`--scorer task_score`. `PAIRS_DEBUG=N` prints the uncovered discs, matched pairs, covers on the mat and the grippers
every N control steps. A lost trial prints why.

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
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=tasks/pairs_game/pairs_game.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("tasks/pairs_game/task.json"))["instruction"])')" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/pairs-demo
```

A policy under test runs the same command with its own `--policy` (Dream-1: `--policy dream --epochs 1`).
`inspect-robots video runs/pairs-demo/<log>.json` renders the stored frames to MP4, one file per camera.

- **Run limit:** 2,400 steps (2 minutes at 20 Hz).
- **Seeds and layout:** one fixed starting layout; use `-E object_jitter=0`. With jitter, seeded resets shift the
  covers by up to 3 cm, which can leave a disc partly uncovered at the start.
- **Runtime:** the scripted demo succeeds and ends after 1,157 steps (58 s simulated) with `task_score` 1.0.
- **Dream-1 result:** one `dream` episode (`--epochs 1`, same command with `--policy dream`) reached `task_score` 0.0
  at the 2,400-step limit, with no trial error and no rule broken: the arms moved over the covers but lifted none. It
  took about 7 minutes wall-clock, including simulator start-up and API inference.

The demo plays five turns. In each, the left arm lifts a cover on its side (pads 1, 2, 5) and the right arm one on
its side (pads 3, 4, 6) at the same time, 4 cm up:

1. Pads 2 and 3, the neighbouring inner pads of the far row: green and blue, no match; both put back.
2. Pads 5 and 4: blue and green, no match; both put back.
3. Pads 5 and 3: blue and blue. The left arm sets its cover on the mat and moves aside, then the right arm sets its
   cover down. Close to the mat, the two forearms would collide if both went at once.
4. Pads 2 and 4: green and green, set aside the same way.
5. Pads 1 and 6: red and red, set aside the same way.

The demo knows the layout. It plays like someone who remembers what they saw, with two misses that are shown to the
viewer first.

## Simulation notes

- Covers are solid rigid cylinders standing in for opaque cups. Coasters and discs are fixed thin cylinders, and the
  mat is a fixed thin box. The covers use the adapter's grippy friction for graspable objects (static 1.2, dynamic
  1.0).
- Physics runs at 240 Hz, control at 20 Hz.

## Implementation

The stock task-file format only scores the final state. `yampatch.py` (shared with the other patched tasks) adds an
`on_step` hook, called once per control step. In it, the task checks the rules on the covers' poses, so a rule broken
at any moment loses the trial, and reads the finger joints `joint7`/`joint8` of `left_robot` and `right_robot`. It
also adds an `on_reset` hook that clears the trial's state. It wraps `yam_arena.scene.object_states` and
`YamArenaEmbodiment.reset` of the pinned adapter; the adapter's files are not changed.

## Media

The setup screenshot, the full-score scripted demo and the Dream-1 evaluation video (top camera) are uploaded
separately to the task board and identified by the `media` entry in `task.json`. The demo was recorded at
`-E cam_width=640 -E cam_height=360`; the Dream-1 run at 320 x 176.
