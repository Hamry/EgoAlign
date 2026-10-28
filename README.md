# EgoAlign

Turn casual egocentric (POV) human video into robot-validated fine-tuning data
for Vision-Language-Action models — no robot, no mocap.

```
video ─► 1. extract ─► 3. task-space ─► 4. MuJoCo filter ─► 5. export (.npz)
            │ MediaPipe     6D EE pose +     Panda IK, collision,     actions, joints,
            │ hand → 3D     gripper          singularity, joint       embeddings,
            └► 2. align (CLIP frame embeddings) ───────────────────► instruction
```

| Stage | Module | What it does |
|---|---|---|
| 1 Ingestion & pose | `extract.py`, `api.py` | MediaPipe hand landmarks, lifted to camera-frame metres by pinhole fitting |
| 2 View alignment | `align.py` | open_clip encoder → L2-normalized per-frame embeddings |
| 3 Task-space mapping | `taskspace.py` | Wrist → x, y, z, roll, pitch, yaw + gripper (pinch with hysteresis); undoes camera tilt |
| 4 Kinematic filter | `robot.py`, `validate.py`, `pipeline.py` | Capsule Franka Panda in MuJoCo; rejects unreachable, self/table collision, singular, or too-fast frames; keeps the valid prefix if it is ≥15 frames |
| 5 Export | `export.py` | Compressed `.npz` episode per accepted video |
| Sim replication | `register.py`, `sim.py`, `replay.py`, `augment.py`, `cloud/` | ArUco camera registration, MuJoCo Panda scene, physics replay, demo augmentation, LeRobot/SmolVLA scripts |

## Install

```bash
pip install -e ".[align,dev]"   # drop `align` to skip torch/open_clip
```

## Use

```bash
egoalign process mug.mp4 --instruction "pick up the mug" --camera-rpy 30 0 0  # tilted 30° down
egoalign serve            # POST /episodes (multipart: video, instruction, camera_*_deg, fov_deg)
docker build -t egoalign . && docker run --gpus all -p 8000:8000 \
    -v $PWD/data:/data -v egoalign-cache:/cache egoalign   # CUDA torch image
```

Episode keys: `actions` (T, 7: xyz, rpy, gripper), `joints` (T, 7), `embeddings`
(T, D), or (0,) without an encoder, `instruction`, `fps`.

Calibration knobs: `--fov` and `--camera-rpy` (degrees about camera x right,
y down, z forward). Others are function defaults: `home` and `grip_threshold` in
`taskspace.to_trajectory`, `min_sigma` and `max_joint_speed` in `validate.validate`.

## Same-POV VLA fine-tune (human video → sim robot → SmolVLA)

A VLA trained on images of a *hand* won't work when it sees a *robot*, so
EgoAlign replays your motion on a simulated Franka Panda and renders the robot
from your camera's exact pose. Those robot frames plus joint actions become a
[LeRobot](https://github.com/huggingface/lerobot) dataset for SmolVLA.

```
video + ArUco ─► process ─► pick in robot frame ─► augment (move mug, physics replay,
                                                   keep lifts, render POV) ─► ≤50 episodes
             ─► to_lerobot ─► lerobot-train (smolvla_base) ─► eval_smolvla (KPI 4)
```

**Record (once):**
1. `egoalign marker` → print `marker.png`; measure the black square (e.g. 0.10 m).
2. Tape it flat on the table. The robot base sits at the origin. The marker is
   axis-aligned: left→right on the paper is robot +x (away from the robot), and
   bottom→top is robot +y (the robot's left). Note the centre's (x, y) in metres.
3. Fix the phone (tripod) so the marker and the mug are both in view. Set its
   horizontal FOV with `FOV=` (≈65° for many main cameras). Measure it in the
   video mode you record with, and turn stabilization off.
4. Record one clear pick: reach, pinch-grasp the mug top-down, and lift ~20 cm.
   One hand, camera still, marker unobstructed for the first 2 s. The mug should
   sit 0.4–0.6 m from the "robot" in x.

**Run (GPU box, ~24 GB GPU, Python ≥ 3.12 for lerobot):**
```bash
VIDEO=pick.mp4 MARKER_X=0.35 MARKER_Y=0.25 MARKER_SIZE=0.10 FOV=65 bash cloud/run_all.sh
SYNTHETIC=1 STEPS=2000 bash cloud/run_all.sh     # smoke test without a video
```
Steps inside: `egoalign process --marker-xy …` → `egoalign augment` (up to 50
successful episodes, mug moved ±8 cm; it prints `N/50 successful episodes`) →
`cloud/to_lerobot.py` → `lerobot-train --policy.path=lerobot/smolvla_base` →
`cloud/eval_smolvla.py` (success rate over 20 randomized placements, and
`work/eval_episode0.mp4`). Other env vars: `TASK` ("pick up the mug"), `EPISODES`
(50), `WORK` (`work`). Each run **deletes** `$WORK/{demo,rollouts,dataset,train}`,
so set a new `WORK` to keep a finished run.

Sim notes: the mug is a 6×6×10 cm box stand-in (flat faces grasp reliably), the
gripper servo is stiffened 3× (stock squeezes ~3 N), and success means the mug
rose ≥10 cm. Actions are absolute joint targets + gripper (1 = close).

## KPIs

| KPI | Check | Status |
|---|---|---|
| 1 ≥10 FPS end-to-end | `python benchmarks/throughput.py VIDEO` | Validation runs ~800 FPS on CPU; full run unmeasured (needs a GPU + real video) |
| 2 <2 cm MAE at 30° off-angle | `python benchmarks/view_shift.py` | 0.74–0.97 cm when the camera tilt is supplied; 3.8–6.0 cm without it |
| 3 100% adversarial rejection | `pytest tests/test_validate.py` | 5/5 adversarial rejected (table, reach, base, wrist spin, teleport), feasible ones pass |
| 4 ≥70% sim success after fine-tune | `cloud/run_all.sh` (SmolVLA) | Pipeline built; not yet run (needs a GPU and your recorded video) |

## Limitations

- View alignment of the *trajectory* needs the camera tilt (`--camera-rpy`); the
  CLIP embeddings are not used to estimate it.
- Export targets LeRobot (SmolVLA); OpenVLA would need an RLDS converter.
- The Panda is capsule-approximated, so collision margins are conservative.

## Test

```bash
pytest -q
```
