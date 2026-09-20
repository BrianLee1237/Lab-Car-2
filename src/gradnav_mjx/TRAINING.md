# Training the SAC car policy

Everything below runs from this directory (`src/gradnav_mjx`).

`train_sac.py` is the entry point. It is Soft Actor-Critic (Haarnoja et
al. 2018) driving an Ackermann car in MJX. It shares the car model and
the reward *function* with the DiffRL pipeline (`train_diffmjx_final.py`)
but not its reward weights or observation encoding — those were tuned
for a different learner and are wrong here. See the comments at the top
of `train_sac.py` for why, in detail.

## Reproducing the headline results

### Room: enclosed 16x16m, walls + walking people, 5-10m goals

```bash
python -u train_sac.py --room --room-size 8.0 --n-inner 4 --n-humans 3 \
  --spawn-half 6.5 --min-dist 5.0 --max-dist 10.0 \
  --obstacle-weight 2.0 \
  --total-steps 400000 --horizon 300 --n-envs 16 \
  --warmup-steps 500 --eval-every 20000 --eval-n 40 \
  --seed 51 --ckpt room.pkl --resume
```

Expect roughly: 23% at 60k steps, 40% at 120k, ~50% by 300k. It
plateaus around 45-55%.

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
- `sac_policy_best.npz` / `sac_policy_final.npz` — written by whatever
  ran last, and **overwritten by every new run**. Currently they hold a
  much weaker policy from a failed experiment. Copy anything you care
  about to a new name.

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
