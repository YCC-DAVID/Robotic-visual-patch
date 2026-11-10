# RoboTwin experiment entry

This directory starts the migration from LIBERO to RoboTwin. It uses separate simulator and policy processes; the LIBERO scripts and data remain historical evidence.

Validated: real perspective checker rendering, three-view fixed-noise policy inference, and a five-seed clean pilot with **4/5 task successes**. Seed10000 repeats exactly across images, states and actions. This is environment/policy integration; RoboTwin attack training is not implemented yet. The validation policy server has been stopped.

## Pinned dependencies

- RoboTwin: `../third_party/RoboTwin`, `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`.
- XPolicyLab submodule: `fa431ecd893ee706883e64fe5fe1464ec8cd928d`.
- CuRobo v0.7.8: `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`.
- Simulator: `/home/user1/miniconda3/envs/RoboTwin/bin/python`, SAPIEN3.0.0b1, Torch2.4.1.
- Model: `/home/user1/miniconda3/envs/openpi-server/bin/python`, existing OpenPI PyTorch implementation and patched Transformers.
- Checkpoint: [motus-robotics/pi0.5_robotwin2](https://huggingface.co/motus-robotics/pi0.5_robotwin2), revision `effc6e1a99bd4d0edf9447e8ba313def67649cfe`.
- Assets: [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0), revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`. Initial clean tasks need `embodiments.zip` and `objects.zip`; randomized-background assets have not been requested.

Additional simulator dependencies are installed into `runtime_py/`, and CuRobo kernels compile into `out/torch_extensions/`. The shared conda environments are not modified.

The exact local additions are pinned in `requirements-sim-overlay.txt`. Restore them with the simulator Python and `pip install --no-deps --target robotwinprobe/runtime_py -r robotwinprobe/requirements-sim-overlay.txt`. After extracting the two asset archives into `third_party/RoboTwin/assets`, run the official `scripts/update_embodiment_config_path.py` from the RoboTwin root. The wrapper disables user-site packages so missing simulator dependencies do not silently come from another environment.

## Initial integration checks

From the WAMattack root, on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=7 bash robotwinprobe/run_sim.sh robotwinprobe/scene_smoke.py
CUDA_VISIBLE_DEVICES=7 /home/user1/miniconda3/envs/openpi-server/bin/python robotwinprobe/policy_server.py --port 8180
CUDA_VISIBLE_DEVICES=7 bash robotwinprobe/run_sim.sh robotwinprobe/clean_rollout.py --port 8180
```

Run the policy server in its own terminal, and stop it after evaluation. Scene/rollout scripts refuse to overwrite completed captures; use `--out` for another run.

`scene_smoke.py` captures native head/left-wrist/right-wrist images before and after inserting a visual-only13cm rectangular textured mesh, with explicit UVs. It preserves the official ray-tracing and OIDN configuration. Clean repeated frames are measured rather than assumed pixel-exact.

Completed captures: `out/scene_smoke/` contains the first partly out-of-frame placement; `out/scene_checker_visible/` uses `--center -0.12 0 0.741` and shows the whole rectangle in the head view. All cameras output240×320RGB. Wrist visibility at this initial pose is limited; this is not evidence of wrist attack coverage throughout a trajectory.

`policy_server.py` uses the checkpoint author's32-step horizon,14joint outputs, Aloha adaptation and delta-joint transforms. It requires caller-supplied32×32noise and verifies repeated inference. This is a separate RoboTwin checkpoint, not the LIBERO policy.

The wire format is HWC uint8 RGB. The server permutes it to CHW for `AlohaInputs`, which converts it back internally; channel values and RGB order are preserved. A native240×320 synthetic color round-trip and exact repeated fixed-noise inference passed.

`clean_rollout.py` is a five-seed integration pilot on `place_container_plate`, using native RGB arrays, full32-action chunks and the official task step limit. There is no expert seed prefilter; instructions describe the actual bowl/cup. It is not an official leaderboard result and supplies no attack-success evidence.

`report_clean.py` reads a completed pilot and plots initial/final native head-camera frames. Run it with a Python environment containing NumPy and Matplotlib; it needs neither GPU nor simulator. The pilot records all three native views, joint states, sampled noise seeds, action chunks, and final frames for later inspection.

## Migration boundary

The new render-transfer experiment is in `out/render_transfer/`. It uses eight training scenes, two held-out validation scenes, and ten test scenes, with one20cm world texture shared by all three cameras. The fixed scene placement has limited wrist visibility; this is not a test of the benefit of joint-view training.

`render_geometry.py` inserts explicit-UV visual geometry and reads true segmentation IDs. SAPIEN RT caches bound textures, so material changes use a fresh visual actor. `render_capture.py` replays saved clean actions and asserts exact clean image/state recovery after removing each texture. `render_proxy.py` provides forward perspective and a RoboTwin-specific candidate imaging proxy; no MuJoCo UV/light/LOD calibration is reused. `render_audit.py` independently checks paired images, model states, split membership and UV direction.

`render_train.py` optimizes a shared full or band32 texture through all three observations, using frozen pi0.5 and a common-noise first-flow-step distance objective. `render_probe.py` separates per-layer prefix KV distances, first velocity differences, and integrated normalized actions. The candidate proxy improves held-out fidelity but retains errors; true-render closed loops remain the primary outcome.

`render_rollout.py` supports digital perspective and actual SAPIEN rendering with matched noise. `render_eval_pipeline.py` owns and cleans up its policy server, runs paired closed loops, recaptures validation images, verifies camera/mask equality, then runs the model probe. `render_report.py` audits initial conditions and reports attack successes only against paired clean successes. See [the current protocol and results](../docs/story/31_robotwin_render_transfer.md). This first pass holds position fixed; position EOT is not enabled.

Current measurements and logs: `out/setup/`, `out/scene_smoke/`, `out/clean_pilot/` when completed. Research narrative: [RoboTwin migration](../docs/story/29_robotwin_migration.md).

## Larger independent cohort for digital rotation

`rotation_expanded_pipeline.py` freezes the original digital Full/Random textures and tests new initial states10600–10699 with Clean and both textures at0°/−10°/+10°. If fewer than50 clean successes occur, the predeclared entire cohort10700–10799 is added. Every selected state receives all seven groups; old pilot results are excluded. No training is performed.

The pipeline owns one policy service and four simulator workers, records immutable protocol/source hashes, resumes completed episodes, and stops only its own children. Default output: `out/rotation_expanded_20261009/`. `status.json` and `summary.json` show progress; [result15](../result/15_rotation_expanded.md) reports only completed-group ASR, Wilson intervals, and differences bootstrapped by independent initial seed. Final audit checks saved raw initial RGB, joint state, prompts, noise rules, texture hashes and paired Full/Random masks.

```bash
/home/user1/miniconda3/bin/python robotwinprobe/rotation_expanded_pipeline.py --model-gpu 2 --sim-gpus 4,5,6,7 --port 8295
```
