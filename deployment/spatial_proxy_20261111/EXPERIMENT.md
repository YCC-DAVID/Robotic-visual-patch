# Full spatial render-proxy estimation

User objective: estimate the parameters of the existing render proxy from observations available in robot datasets. This experiment is not patch optimization or closed-loop attack evaluation.

## Fixed interface

Inputs: current native RGB; available full EEF pose and gripper with validity flags; camera identity and image size; explicitly known patch placement, orientation and physical size. Optional history must be causal. Scene, task, episode and seed identifiers are grouping metadata, not model inputs. Simulator intrinsics, extrinsics, depth, probe images and target H/B/G/S are labels only.

Input audit correction: the exported `query_canonical_to_world_xy` gauge was constructed from simulator base-camera H/height. Despite the capture's `query_known_valid` flag, it is not an allowed deployment observation. Both model/data adapters force its nine reserved input values to zero and validity false. Raw files keep it for provenance only. Similarly, `lowdim_valid[:8]` concerns an older canonical square, so geometry label validity must be computed directly from the actual probe H; out-of-frame but finite convex corners remain valid.

Outputs: image-pixel-to-texture-pixel H, native-resolution black response B, nonnegative spatial gain G=max(white-black,0), and dx/dy/LOD sampling parameters shared by camera/imaging configuration. B/G already encode visibility and edge mixture; do not multiply another hard mask into rendering. The model must support a single image and missing views/state.

First candidate: actual TinyViT-5M multiscale encoder, shallow RGB skip and lightweight spatial decoder, state/query MLP, four-corner geometry head, and shared sampling head. All heads and the encoder train. Record the actual parameter count, dependencies and pretrained weights. A larger architecture is not an assumed solution.

## Data and split

Use real native probe captures. Keep the existing scene-level train/val/test assignment from the 499-scene data; all views, times and texture anchors from one scene stay together. For the first pilot, select three query timestamps per scene without using labels or scores. Keep original resolution and exact pixels. A 187-frame aligned single-task historical set is a separate diagnostic, not a replacement for multi-task training or an independent unknown-rig test.

The exported pilot selects the last timestamp from each scene's first/middle/last bundle (1179 train / 156 val / 162 test). The first bundle is inside the previous shared-calibration model's 12-frame prefix. It is legitimate training/evaluation data for the new single-frame model, but comparisons against the old scene-calibrated predictor must use only the common post-prefix timestamps. Do not label all 1497 targets as post-calibration queries.

Training random-texture anchors may supervise rendering. Check textures are held out from training; validation scenes may use them for model selection, while test scenes stay untouched until the model and configuration are frozen. All normalization, mean baselines and sampling calibration use training scenes only. Existing low-dimensional-C sampling fits are not equivalent to the old spatial-proxy S labels; retain provenance and validity, and do not supervise RoboTwin from invalid S labels. Fit shared S against training native random anchors using oracle H/B/G if needed; record fit residuals and identifiability.

## Renderer contract

Preserve pi05probe/dual_view_proxy.py: float64 projective geometry, texture pixel-center conventions, dx/dy before projection, log2(rho).clamp_min(0) before LOD bias, border/bilinear/trilinear sampling, STE uint8 rounding of Q and every mipmap, B+G*sample, native rounding, then antialiased resize to 224 and final rounding. Do not reuse cached grids when H changes. STE is a surrogate derivative; continuous finite-difference tests use quantize=False.

## Execution and evaluation

1. CPU: exact old-renderer parity on real fixtures; differentiable H/DLT; every model output receives a finite gradient; missing-view and single-frame interfaces.
2. On dl after GPU preflight: tiny training-only overfit check, verify actual updates of encoder and every head, then full pilot with validation-selected early stopping and reproducible checkpoints. Failures must be fixed or explicitly diagnosed before a long run. No model selection on test.
3. Evaluate H corner error; B/G MAE and RMSE in affected region, boundary and background; sampling fit errors/residuals; native and 224 held-out-texture errors. Separately report predicted-parameter vs teacher-parameter rendering and teacher-parameter vs simulator errors.
4. Run a common-reference-H diagnostic to isolate B/G/S from geometry, alongside full predicted H/B/G/S. Compare the existing scalar-C/mask model on the same inputs/split and retain the teacher oracle. No policy/patch optimization is required for this comparison.
5. Record actual parameter count, memory and synchronized single-frame inference latency, GPU UUID, hardware, batch size and resolution. Existing rigs remain fixed: do not claim unknown-rig or sim-to-real generalization from these splits.

Only dl may launch this experiment. nnmc62 is forbidden for all new project compute/controllers. Before every GPU stage, follow the current gpu-preflight skill, audit the specified original fleet and dl, recheck the selected healthy idle card, and set CUDA_VISIBLE_DEVICES explicitly. Do not terminate unrelated tasks. Existing checkpoints and results remain intact in separate experiment directories.
