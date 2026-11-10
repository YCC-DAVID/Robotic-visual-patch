# Controlled RoboTwin checkpoint study

Run: `robotwinprobe/out/checkpoint_study_20261023`.
User report: `result/28_robotwin_checkpoint_study.md`.
This is independent of the completed G1 comparisons in results 25/27.

## Controls

- Task: `place_container_plate`, `demo_clean`, `aloha-agilex`, native three cameras.
- Fixed data: first 100 successful expert demos from seeds starting 20000; 20 validation demos starting 21000. Failed expert plans are retained in manifests. Evaluation seeds 22000–22099 are **not** filtered by expert success.
- Base: cached official `pi05_base/params`, fingerprinted by the pipeline. All three models share one train-only norm-stat file and dataset manifest.
- Independent model seeds: 202610230/231/232, 20,000 updates each; batch16/micro4; frozen vision; language LoRA rank16; full action expert; original OpenPI augmentation, LR, AdamW and EMA. No learning-rate or budget tuning per seed.
- Temporal checkpoints: only M0 at 5k/10k/20k; separate from the independent-seed comparison. Total five distinct checkpoints.
- Three patch seeds per checkpoint: 202610240/241/242. Fixed 10cm/Q128/full frequency, gray initialization, site05, 500×8 sign ascent, lr .01. Each seed permutes the same 4000 calibrated geometry records and creates its own 16-noise bank; draws match across target checkpoints for that seed.
- Proxy bank: `out/table9_joint_eot_10cm_20261012`, site05. Same physical texture in all views. EOT disk radius3cm, yaw±15°, three recorded light conditions, no flip. This reuses old-policy states as a fixed patch-training distribution, not old-policy weights.
- Nominal-pose primary evaluation: every texture on every target; clean and one fixed Random on each target. 85×100=8500 primary episodes. Separate 30 clean pilot episodes use seeds23000–23009, and are not part of the final denominators.

## Verified preflight and gates

The standalone smoke run has three model updates and two patch updates; it is not an attack result. Native-camera resize, normalized state/actions and tokenization are checked against upstream OpenPI. Recorded expert absolute-qpos actions must replay successfully through `take_action`. Three shared observations/noises check export fidelity: velocity relativeL2<5%, full action relativeL2<10%; diagnostic Euler integration must match native Torch maxabs<1e-4.

The closed-loop smoke run checks one query/32 actions under clean and Random, including the served checkpoint hash and identical initial RGB/state hashes. Its task success is not used as model performance evidence. Formal patch training starts only if each final model passes the separate clean pilot (at least5/10). A failed gate stops the pipeline for inspection; it does not silently tune one seed or discard initial states.

## Execution and resumption

Run from the repository root in the GPU-capable host environment. Existing collection processes can be adopted by their actual PIDs; do not pass stale PIDs or start duplicate collectors.

```bash
/home/user1/miniconda3/bin/python -u robotwinprobe/checkpoint_study_pipeline.py \
  --gpus 3,5,6 --adopt-train TRAIN_COLLECTOR_PID --adopt-validation VALIDATION_COLLECTOR_PID
```

On a later resume, omit adoption arguments if the collectors have exited. The file lock prevents concurrent pipelines. Collection, checkpoints, patch training and per-episode results resume from verified artifacts. Incomplete exports and recorded runtime errors require inspection; preserve failed artifacts rather than deleting them or counting them as task failures.

The pipeline is: fixed data -> three concurrent model fine-tunes -> strict LoRA-folded exports and JAX/Torch parity -> clean pilot -> 15 patches -> actual-render 15×5 cross matrix plus controls -> fixed-state diagnostics -> report. It only terminates policy-server child processes that it started.

## Reading results

- `pipeline_status.json`: current stage; `logs/*.json` and corresponding `.log`: child status/command.
- `expert/*_manifest.json`: accepted and rejected expert seeds; `data_cache/manifest.json`: fixed dataset and normalization hashes.
- `training/M*/status.json`, `history.jsonl`, `monitor.jsonl`: update counts, optimization loss and held-out monitoring. Four monitoring examples are diagnostics, not a policy-success estimate.
- `matched_recipe_audit.json`: exact recipe equality across model seeds; `base_inventory.json`, `execution_manifest.json`: sources and base fingerprints.
- `patches/<source>/P<seed>/`: patch config, updates, final PNG and validation L0. Final texture list is `texture_manifest.json`.
- `closedloop/<target>/<condition>/`: each initial state's status and full observations/actions; `results.json` only marks complete after all100.
- `comparison/metrics.csv`: overall SR, target-clean-conditioned ASR, paired excess over Random, common-clean ASR and initial-state bootstrap intervals.
- `comparison/ASR_cross_matrix.csv`, `transfer_asymmetry.csv`: all source patch→target results; independent model seeds and temporal steps analyzed separately. Asymmetry uses the pair's common clean-success subset, with three paired patch seeds held fixed for the initial-state bootstrap.
- `diagnostic_frames/`: actual renders of all textures at identical physical states along M0_20k clean trajectories, fixed first12 test seeds, queries0/3/6/9 when reached. Includes repeated clean rendering to quantify rendering variation; no physics steps occur between textures.
- `diagnostics/<target>/responses.csv`: same three noises, full32 and active14 L0, complete10-step action MSE14, physical joint/gripper RMSE. `directions.csv` compares Δv directions within a target model.
- `comparison/texture_metrics.csv`, `texture_pairs.csv`: color, grayscale histogram entropy, centered/windowed radial power and edge/pixel correlations. These are descriptive features, not proof of an independently sufficient attack structure.

Refresh the report at any time:

```bash
MPLCONFIGDIR=/tmp/wam_checkpoint_mpl /home/user1/miniconda3/bin/python robotwinprobe/checkpoint_study_report.py
```

ASR = failures under attack among clean-success initial states / clean-success count. Overall SR uses all100 initial states. Therefore SR generally differs from1−ASR. Raw runtime errors remain distinct from task failure; the primary matrix is not declared complete when any group is missing.
