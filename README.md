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

## KPIs

| KPI | Check | Status |
|---|---|---|
| 1 ≥10 FPS end-to-end | `python benchmarks/throughput.py VIDEO` | Validation runs ~800 FPS on CPU; full run unmeasured (needs a GPU + real video) |
| 2 <2 cm MAE at 30° off-angle | `python benchmarks/view_shift.py` | 0.74–0.97 cm when the camera tilt is supplied; 3.8–6.0 cm without it |
| 3 100% adversarial rejection | `pytest tests/test_validate.py` | 5/5 adversarial rejected (table, reach, base, wrist spin, teleport), feasible ones pass |
| 4 ≥70% sim success after LoRA | — | Not implemented: needs OpenVLA fine-tuning + a sim task harness |

## Limitations

- View alignment of the *trajectory* needs the camera tilt (`--camera-rpy`); the
  CLIP embeddings are not used to estimate it.
- Export is `.npz`, not RLDS; convert before feeding OpenVLA's LoRA script.
- The Panda is capsule-approximated, so collision margins are conservative.

## Test

```bash
pytest -q
```
