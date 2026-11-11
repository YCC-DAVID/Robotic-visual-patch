# Spatial proxy model and renderer contract

This implementation predicts the old CameraProxy representation: image-to-texture
H, native spatial B and G, and shared sampling S. It does not convert the old
7-value C fit into a spatial B/G target. It does not add a hard visibility mask to
the final renderer.

## Architecture and dependencies

`SpatialProxyEstimator` uses the real official timm architecture
`tiny_vit_5m_224.in1k`, with `features_only=True` and four stage outputs of
64/128/160/320 channels at reductions 4/8/16/32. A shared 48-channel skip decoder,
native RGB skip, EEF MLP, known-query MLP and camera embedding feed spatial B/G
and the corner geometry head. Only valid cameras are packed into the encoder:
missing-camera padding never contaminates TinyViT BatchNorm.

CPU-verified dependencies are Python 3.10, torch 2.2.0, torchvision from the
existing openvla-oft environment, and timm 0.9.10. No environment was modified.
Use `PYTHONNOUSERSITE=1`; without it this host's unrelated user-site ONNX package
breaks the existing torchvision import. The official model and weight source is
https://huggingface.co/timm/tiny_vit_5m_224.in1k and timm source is
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/tiny_vit.py.

For the default 3 camera IDs and 6 imaging configurations:

- Encoder: **5,071,124** parameters.
- Complete model: **5,330,085** parameters, all trainable by default.
- CPU tests instantiate `pretrained=False`; they validate the real architecture,
  not trained prediction quality. A production trainer should explicitly choose
  `pretrained=True` if using official ImageNet initialization and preserve its
  weight hash. No pretrained weights were downloaded or silently substituted.
- Each encoder input is resized to 224 square and ImageNet-normalized. Native H/W
  are also explicit conditioning inputs. The native decoder keeps RGB detail.
- This is a single-timestamp model. No temporal or scene-calibration guarantee is
  claimed. Camera/config vocabularies are fixed device properties, not task IDs,
  episode IDs, timestamps, sample IDs, or target-derived clusters.

## Forward input schema

All tensors live on one device; observations are float32. Group different native
resolutions into separate batches. Let B be batch size, V padded camera slots,
and Hn/Wn the native image size.

| Argument | Shape | Meaning |
|---|---|---|
| `native_clean` | B,V,3,Hn,Wn | Current unpatched RGB in [0,1], in the same pixel convention as the targets |
| `camera_valid` | B,V bool | Whether this current camera image exists |
| `T_world_eef` | B,2,4,4 | Full measured transforms of two EEF slots |
| `eef_valid` | B,2 bool | Valid measured EEF slots |
| `gripper` | B,2 | Measured gripper state |
| `gripper_valid` | B,2 bool | Gripper validity; also requires valid EEF |
| `query` | B,V,21 | Explicit known physical patch-placement query described below |
| `query_valid` | B,V,21 bool | Per-scalar validity, not label availability |
| `camera_id` | B,V int | Fixed declared camera identity; invalid slots can be -1 |
| `imaging_config_id` | B,V int | Fixed camera/imaging configuration for shared S |
| `imaging_config_valid` | B,V bool | Whether that configuration is in the declared vocabulary |
| `image_orientation_xy` | B,V,2 optional | Known image-axis orientation signs, default [1,1] |

EEF features retain xyz and the first two rotation columns (six values, full
rotation information for a valid SO(3) matrix), plus gripper and validity flags.
Invalid EEF/gripper values and inactive RGB may contain NaNs; they are masked
before participating in neural computation. Known inputs containing NaNs raise
an explicit error. An all-missing-camera batch produces flagged finite empty
outputs without invoking the encoder; the trainer must skip its loss.

The 21 query values are ordered as follows:

1. World probe centre xyz: 3 values.
2. Physical size wh in metres: 2 values.
3. Plane z: 1 value.
4. **Reserved disabled gauge slot: nine zeros with all validity flags false.**
   The exported canonical-to-world matrix was derived by `reference_plane_gauge`
   from true camera H and height. It is label-only, not deployment-known input.
5. Texture shape hw: 2 values (currently 128,128).
6. Known physical orientation quaternion wxyz: 4 values.

`pack_exported_query(sidecar)` packs the allowed physical fields from the six
`query_*` groups in `dl_spatial_proxy_targets_20261111`. The sidecar's
`query_known_valid` has six group flags; the helper expands them to 21 flags,
overriding the gauge group to false regardless of the old sidecar flag. It never reads
`query_canonical_to_world_xy`, and forward rejects declaring dimensions 6:15 valid.
Even arbitrary/NaN values in those disabled slots cannot affect predictions.
The helper also never accesses
`image_to_texture_H`, B/G labels, masks, fitted S, task, or episode metadata.
Per-camera probe centres must remain per-camera; RoboTwin probe positions can
differ across views. The supplied quaternion is the known table-aligned physical
placement; texture UV y still points along world -Y, as in `texture_to_plane`.

The broad sidecar uses top-left RGB without the legacy policy's extra horizontal
reflection. If mixing legacy policy-oriented images, explicitly provide that
known image-axis convention and apply identical conventions to native images,
H targets and query geometry. Never derive an orientation flag from target H.

## Outputs and losses

- `H`: B,V,3,3 float64, **native image pixel centre -> 128x128 texture pixel centre**.
- `corners_native`: B,V,4,2, native positions of texture TL/TR/BR/BL centres
  (texture coordinates 0 or 127).
- `H_valid`: B,V bool. A convexity/finite check flags invalid quads; their DLT uses
  a finite fallback rectangle only to prevent NaNs. Invalid predictions must not
  be scored as good geometry. Keep direct corner supervision so they can recover.
- `B`: B,V,3,Hn,Wn, `clamp(native_clean + tanh(residual), 0, 1)`.
- `G`: B,V,3,Hn,Wn, `sigmoid(spatial_logits)`, hence nonnegative and <=1.
- `S`: B,V,3, dx_native_px / dy_native_px / LOD bias, shared via configuration
  embedding + small MLP. Bounds are [-2,2], [-2,2], [-3,3]. It cannot vary with
  image, EEF, query or per-sample identity. Unknown configurations emit neutral S
  with `S_valid=False`, rather than borrowing a hidden per-frame target.
- `camera_valid`, `S_valid`: explicit masks for downstream losses/metrics.

Train B/G against exact black/native-white differences and use render supervision
from declared training probes. H corners should come from the inverse target H
projecting texture corner centres. Normalize geometric losses by native size.
The sidecar's `sampling_reference` is only a low-dimensional-fit reference, not
exact native B/G sampling supervision; do not directly label S accurate from it.
Test check textures and heldout frames must not enter training/model selection.

## Renderer

`render_camera_proxy(q,H,B,G,S,quantize=True,native=False)` takes packed valid
cameras: q is 1 or N,3,128,128; H is N,3,3; B/G are N,3,Hn,Wn; S is N,3.
The implementation follows `pi05probe/dual_view_proxy.py::CameraProxy`:

1. Float64 H/grid/LOD, including pixel centres and safe homogeneous denominator.
2. `log2(rho).clamp_min(0)`, then shared LOD bias, then clamp [0,7].
3. Q STE uint8; eight mip levels; avg-pool and STE uint8 after each pooled level.
4. Float32 `grid_sample`, bilinear/border/align_corners=False, trilinear LOD blend.
5. `clamp(B + G*sample,0,1)`, native STE uint8.
6. If requested, bilinear antialiased resize to224 and another STE uint8.

It never caches a trainable H grid and never multiplies a second hard footprint
or visibility mask. Spatial G already contains those effects. Quantization STE
is a surrogate derivative, not the derivative of discrete native image rounding.
Directional finite-difference checks therefore use `quantize=False`.

## CPU validation and current limits

Run from the repository root:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/user1/miniconda3/envs/openvla-oft/bin/python \
  deployment/spatial_proxy_20261111/source/test_core.py \
  --out deployment/spatial_proxy_20261111/cpu_model_validation.json
```

Tests use actual exported base/wrist CameraProxy fixtures. Across both cameras,
two sampling settings, float/STE and native/224 outputs (16 comparisons), output
and Q-gradient maximum errors were exactly zero on CPU. Geometry corner
roundtrip error was 2.84e-14 px; float64 DLT gradcheck passed. Float Q directional
difference relative error was 7.95e-4. H/B/G/S all have nonzero finite renderer
gradients; changing H changes the grid. Real TinyViT forward/backward and one
optimizer step changed encoder, state/query, geometry, B, G, S and decoder groups.
Missing-camera/EEF isolation and S invariance to RGB were checked.
The forbidden gauge can be removed entirely from the input archive passed to the
helper; its reserved input values stay zero and invalid. Changing those disabled
values leaves every output unchanged, and declaring the gauge valid is rejected.
An outer CPU bfloat16 autocast was also checked: the exported H remains float64
and B/G/S remain float32, with finite outputs.

The renderer supports 240x320 tensors and that path has finite-gradient tests;
this is **not** a claim of exact equivalence to RoboTwin's native renderer.
CPU bitwise equivalence does not establish GPU bitwise reproducibility; the dl
trainer must run its own numerical and resource preflight before GPU use.
The 1497-target data set includes each scene's first/middle/last selected query;
first targets are inside the old shared-calibration prefix. Only the common
post-prefix subset is eligible for a fair comparison against that old model.
