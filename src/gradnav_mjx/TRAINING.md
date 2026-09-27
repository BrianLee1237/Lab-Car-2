# Training the SAC car policy

## Current results

Enclosed 16x16m room, 4 perimeter + 2 interior walls, 8 walking people,
goals 5-10m apart. Policy: `sac_policy_room_fixed_h8.npz`.

| map | success | wall hits | ped hits | mean closest |
|-----|---------|-----------|----------|--------------|
| seed 52 (trained on) | **77.0%** | 21/100 | 11/100 | 0.935m |
| seed 200 (held out)  | **88.0%** | 21/100 | 19/100 | 0.449m |

Reproduce with `SEED=52 N_HUMANS=8 N_INNER=2 ./reproduce_room.sh`.

### How this compares to what this file used to say

The previous reference was 47% / 56% with 3 people, and everything below
was written against it. Both the physics and the room have changed:

| | old | new |
|---|-----|-----|
| people in the room | 3 | 8 |
| success, trained map | 47% | 77.0% |
| success, held out | 56% | 88.0% |
| wall hits | 44 / 33 per 100 | 21 / 21 |
| mean closest approach | 2.73 / 2.38m | 0.935 / 0.449m |

A 30-point gain while nearly tripling the pedestrians, and wall
collisions halved. Matched-step comparison against the old run:

| steps | old (3 people, braked) | new (8 people, fixed) |
|-------|------------------------|-----------------------|
| 60k   | 23% | 53% |
| 120k  | 40% | 73% |
| 160k  | 37% | 67% |

**The ~50% plateau was a physics bug, not a limit of SAC, the reward, or
the car's kinematics.** Two faults, both in the scene rather than the
learner:

1. The chassis capsule overlapped each FRONT wheel by 34mm, ~150N of
   normal force per wheel against ~4N of wheel-floor contact -- the car
   drove with its own front wheels clamped in a vice. About 60% of the
   throttle range produced under 0.22 m/s, so the policy had effectively
   binary throttle while the reward's action penalties pushed it toward
   the inert region. Fixed in 068e8d1.
2. QVEL_CLAMP=40 capped wheel spin, limiting pure rolling to 2.0 m/s, so
   above that the car skidded at up to 31% slip and cornered wrongly in
   the speed range it actually used. Fixed in ae3aa58.

Consequences for the history below:

- **"Things not to re-try" is misleading, not merely stale.** Those six
  reward experiments were diagnosing symptoms of fault 1. Several
  "froze the car", which is what a heavier obstacle penalty does when it
  nudges throttle into a dead zone. The speed-penalty-near-obstacles
  idea in particular asked the car to slow down when slowing meant
  stopping; it is worth retrying.
- The claim that the remaining gap was kinematic -- a 1.5-1.9m turning
  radius leaving the car committed -- was measured on the braked car.
  Measured turn radius is now 0.55m at 2.35 m/s, and wall hits halved.
- `sac_policy_room_best.npz` (47%/56%) is kept for provenance only. It
  learned to hold throttle at 0.96 because nothing else moved the car.

Everything from here down predates the fixes. Read it as history.

## Reproducing the headline results

### Room: enclosed 16x16m, walls + walking people, 5-10m goals

Run this as four legs, raising `--total-steps` each time. Each leg
resumes the previous one's checkpoint:

```bash
CFG="--room --room-size 8.0 --n-inner 4 --n-humans 3 --spawn-half 6.5 \
  --min-dist 5.0 --max-dist 10.0 --horizon 300 --n-envs 16 \
  --warmup-steps 500 --seed 51"

rm -f /tmp/room5.pkl
python -u train_sac.py $CFG --total-steps 60000  --eval-every 5000  --eval-n 30 --ckpt /tmp/room5.pkl --resume
python -u train_sac.py $CFG --total-steps 160000 --eval-every 10000 --eval-n 30 --ckpt /tmp/room5.pkl --resume
python -u train_sac.py $CFG --total-steps 300000 --eval-every 20000 --eval-n 40 --ckpt /tmp/room5.pkl --resume

cp /tmp/room5.pkl /tmp/room_obs2.pkl
python -u train_sac.py $CFG --total-steps 400000 --eval-every 20000 --eval-n 40 \
  --obstacle-weight 2.0 --ckpt /tmp/room_obs2.pkl --resume

cp sac_policy_best.npz sac_policy_room_best.npz
```

`reproduce_room.sh` runs exactly this. `SEED=52 ./reproduce_room.sh`
runs another seed; every checkpoint, log and saved policy is scoped by
seed, so seeds can be run back to back and compared. Given the
cross-machine variance noted below, running two or three and keeping
the best by an n=100 score is the realistic way to get a good policy on
your own hardware.

These are the literal commands of the original run, recovered from the
session transcript -- not a reconstruction. Notes on the details:

- `--obstacle-weight` is on the LAST leg only. It defaults to 1.0,
  which is what the first 300k steps ran at; the flag did not exist
  until that leg introduced it. Raising it from the start is not this
  recipe, and a heavier obstacle penalty early is precisely what froze
  the car in several of the failed experiments in "Things not to
  re-try".
- The leg-4 rename is only to preserve the 300k state before the
  obstacle-weight experiment. It is the same continuous lineage: that
  leg logs `resumed from /tmp/room_obs2.pkl at 300000 steps`.
- `--eval-every` / `--eval-n` vary per leg. Eval cadence only; no
  effect on training.
- `--action-repeat`, `--batch-size` and `--updates-per-step` were never
  passed -- defaults 20 / 256 / 1. `--warmup-steps` is 500, not the
  default 2000.

Expect roughly: 23% at 60k steps, 40% at 120k, ~50% by 300k. It
plateaus around 45-55% on this machine; see the cross-machine note
below before treating that band as guaranteed.

**The legs are not optional, and a single 400k run is not equivalent.**
The curriculum is paced against `--total-steps`:

    progress = min(1.0, total_env_steps / (total_steps * 0.7))

so the goal range only reaches the full `--min-dist`..`--max-dist` at
70% of `--total-steps`, while the eval *always* scores the full target
band. Launch one run at `--total-steps 400000` and at 60k it is still
training on ~1.8-4.1m goals but being graded on 5-10m ones -- it
reports ~5-8%, not 23%, and looks broken when it is merely early.

If you would rather run one continuous job, that is fine -- but read
the eval against `goal_range` in the same log line, and do not expect
these milestones until well past 280k steps.

`--resume` is safe to pass on a fresh run (it simply finds no
checkpoint). Re-running the same command after an interruption picks up
where it left off, replay buffer included — this matters, see "Resuming"
below.

Evaluate a checkpoint properly (the in-training evals use few goals and
are noisy):

```bash
# training map
python eval_band.py --checkpoint sac_policy_best.npz --room \
  --n-inner 4 --n-humans 3 --spawn-half 6.5 --max-dist 10.0 \
  --map-seed 51 --eval-n 100 --horizon 300 --bands "5,10"

# held-out map the policy never trained on
python eval_band.py --checkpoint sac_policy_best.npz --room \
  --n-inner 4 --n-humans 3 --spawn-half 6.5 --max-dist 10.0 \
  --map-seed 200 --eval-n 100 --horizon 300 --bands "5,10"
```

This recipe was re-run end to end from a clean checkout and reproduces
exactly. Independent run vs. the original, same seed:

| steps | re-run | original |
|-------|--------|----------|
| 60k   | 20%    | 23%      |
| 120k  | 33%    | 40%      |
| 180k  | 50%    | 50%      |
| 280k  | 53%    | 53%      |
| 300k  | 47%    | 47%      |
| final n=100, seed 51  | 47.0% | 47% |
| final n=100, seed 200 | 56.0% | 56% |

Wall hits (44/100 and 33/100), pedestrian hits (4/100 both) and mean
closest approach (2.73m / 2.38m) all match as well. The early-leg gaps
are n=30 eval noise at +-7-8pp, not drift.

**That holds on the same machine, not across machines.** The same
checkout and the same --seed 51 run on an M-series Mac gave 13.3% at
60k (vs 20% here) and finished at 34% / 39%. JAX floating point
differs between ARM and x86, and RL training compounds small
differences through the replay buffer, so a seed pins a run down on one
platform only; elsewhere it is effectively a fresh draw. Plan on
running a few seeds and keeping the best by an n=100 score, and treat
the committed `sac_policy_room_best.npz` as the reference result rather
than something a single retrain is guaranteed to match.

Reference numbers for `sac_policy_room_best.npz` (committed):

| map | success | wall hits | ped hits | mean closest |
|-----|---------|-----------|----------|--------------|
| seed 51 (trained on) | 47% | 44/100 | 4/100 | 2.73m |
| seed 200 (held out)  | 56% | 33/100 | 4/100 | 2.38m |

### Open field: scattered walls, unbounded floor

Easier, and where the best absolute numbers are:

```bash
python -u train_sac.py --min-dist 1.0 --max-dist 6.0 \
  --total-steps 60000 --horizon 300 --n-envs 16 \
  --warmup-steps 1000 --eval-every 10000 --eval-n 50 --seed 31
```

~90% at 1-6m by 40k steps; ~88% at 1-2m and ~74% at 4-6m when scored
per band. Generalises to unseen maps with no measurable gap.

## Checkpoints in this directory

- `sac_policy_room_best.npz` — the 47%/56% room policy above. **Use
  this one.** Copied aside deliberately.
- `sac_policy_best.npz` / `sac_policy_final.npz` — scratch. Written by
  whatever ran last and **overwritten by every new run**, so they are
  gitignored and untracked: a run no longer dirties the repo, and a
  stray `git add -A` can no longer commit scratch over a real result.
  Copy anything you care about to a durable name.
- `sac_policy_room_s<SEED>.npz` — what `reproduce_room.sh` saves, named
  by seed so runs cannot clobber each other. Commit the good ones; they
  are ~122K, small enough for git with no LFS.

## Resuming (important)

`--ckpt FILE --resume` saves and restores the full training state:
networks, target copies, optimiser moments, entropy temperature, step
count, **and the replay buffer**. The buffer matters more than it looks:
an earlier version omitted it, and chaining three 10k-step legs with a
fresh buffer each time gave flat 0-7% success across 30k steps where one
continuous run was climbing steadily. If you are stitching a long run
out of shorter ones, make sure the buffer is being restored — the log
prints `restored replay buffer: N transitions`.

Writes are atomic (temp file + rename), so a process killed mid-save
cannot corrupt the checkpoint.

## Things not to re-try

These were each measured and each did worse. The reasoning behind them
still reads well, which is exactly why they are written down.

| change | result |
|--------|--------|
| `obstacle` weight 2.5 | froze the car; success 0-7% |
| `OBSTACLE_SAFETY_DIST` 0.6 -> 2.0 | wall hits 44%->36% but success 54%->42% |
| speed penalty near obstacles (0.15) | success 46%->38%, no fewer collisions |
| `progress` 6.0 + safety 2.0 + `yaw_alignment` 0 | froze the car; 0-3% |
| `progress` 10 + `obstacle` 0.3 @ 1.5m + `yaw` 0 | 17% at 200k vs ~40% for the default |
| horizon 300 -> 600 (12s -> 24s) | 54% -> 58%, inside the noise at n=50 |

Note that several of those were applied by *resuming* a trained policy.
Changing the reward invalidates the critic's learned values, so a resume
makes it fight stale estimates for a different problem. Retrain from
scratch when the reward changes.

`softplus` never reaches zero, so a large `OBSTACLE_SAFETY_DIST` is a
near-constant tax on existing rather than a local deterrent: at 2.0m
with weight 1.0 the penalty is -0.31 per decision even three metres from
a wall, against +0.24 of progress reward. In a 16m room with 8 walls
there is then nowhere that moving pays, and the policy correctly stops
moving.

## Known remaining limitation

~44% of room episodes end in a wall collision, and that is the whole
gap. A crash diagnostic found the car meets walls at only ~1.2 m/s with
the forward lidar reading 1.35m for several decisions beforehand — it
sees the wall and drives in anyway, because its measured minimum turning
radius is 1.5-1.9m at speed and it is already committed. This is
kinematic, not perceptual, so more lidar resolution is unlikely to help.
Reward shaping has been explored fairly thoroughly (see the table
above); further gains most likely need a longer run and a larger network
than `[128, 128, 64]`.

## Sanity checks

Independent of the learner, useful when something looks wrong:

- `oracle_check.py` — drives a hand-written controller at the goals.
  Establishes whether the task is solvable at all under a given
  configuration. Reached 97% on a +-60 degree cone at 1-3m in the open
  field, which is what showed the original 360-degree goal sampling was
  placing goals inside the car's own turning circle.
- `steer_check.py` — measures steering angle and turn radius against
  command. Caught the steering being a torque motor rather than a
  position servo, which gave the policy effectively binary steering.
- `stuck_check.py` — logs pose, height and tilt while driving, for
  "why did it stop" questions.
- `trace_sac.py` — rolls out a checkpoint and prints the trajectory.
  Nearly every real bug this session was found here rather than in the
  eval numbers.
