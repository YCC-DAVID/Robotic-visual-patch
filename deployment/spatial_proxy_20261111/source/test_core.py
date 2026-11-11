"""CPU validation of the real legacy renderer and actual TinyViT model.

Run with CUDA_VISIBLE_DEVICES='' and PYTHONNOUSERSITE=1. Uses no CUDA context.
Tests are numerical checks of the required contract, not a training-quality test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
import numpy as np
import torch

from model import SpatialProxyEstimator, pack_exported_query
from renderer import native_grid, project, render_camera_proxy, texture_corners_to_image_H


def legacy_equivalence(root: Path, teacher: Path) -> dict:
    sys.path.insert(0, str(root / "pi05probe"))
    from dual_view_proxy import CameraProxy
    report = {}
    for camera in ("base", "wrist"):
        source = teacher / f"{camera}.npz"
        with np.load(source, allow_pickle=False) as z:
            blacks, whites = z["black256"], z["white256"]
            coverage = np.maximum(whites.astype(np.int16)-blacks.astype(np.int16), 0).sum((1, 2, 3))
            # Numerical nonzero-gradient fixture selection, not an evaluation sample selection.
            k = int(np.argmax(coverage))
            H = z["image256_to_texture_pixel_H"][[k]]
            black, white = blacks[[k]], whites[[k]]
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fixture.npz"
            np.savez(fixture, image256_to_texture_pixel_H=H, black256=black, white256=white)
            old = CameraProxy(fixture)
            b = torch.from_numpy(black.transpose(0, 3, 1, 2).astype(np.float32)/255)
            g = (torch.from_numpy(white.transpose(0, 3, 1, 2).astype(np.float32)/255)-b).clamp_min(0)
            h = torch.from_numpy(H)
            rows = []
            for settings in ([0., 0., 0.], [.21, -.17, .37]):
                s = torch.tensor([settings], dtype=torch.float64)
                for quantize in (False, True):
                    for native in (False, True):
                        torch.manual_seed(812)
                        q1 = (.2+.6*torch.rand(1, 3, 128, 128)).requires_grad_()
                        q2 = q1.detach().clone().requires_grad_()
                        a = old(q1, 0, parameters=settings, quantize=quantize, native=native)
                        c = render_camera_proxy(q2, h, b, g, s, quantize=quantize, native=native)
                        probe = torch.randn_like(a)
                        ga, = torch.autograd.grad((a*probe).sum(), q1)
                        gc, = torch.autograd.grad((c*probe).sum(), q2)
                        im = float((a-c).abs().max())
                        grad = float((ga-gc).abs().max())
                        assert im == 0 and grad == 0, (camera, settings, quantize, native, im, grad)
                        assert float(ga.norm()) > 0
                        rows.append(dict(S=settings, quantize=quantize, native=native,
                                         max_image_diff=im, max_Q_gradient_diff=grad,
                                         Q_gradient_norm=float(ga.norm())))
            report[camera] = dict(source=str(source), selected_frame=k, checks=rows)
    return report


def geometry_checks() -> dict:
    c = torch.tensor([[[31.2, 22.5], [215.8, 18.3], [224.2, 209.7], [24.5, 219.1]]],
                     dtype=torch.float64, requires_grad=True)
    h, valid = texture_corners_to_image_H(c, (240, 320))
    want = c.new_tensor([[[0, 0], [127, 0], [127, 127], [0, 127]]])
    error = float((project(h, c)-want).abs().max())
    assert valid.all() and error < 1e-9
    assert torch.autograd.gradcheck(lambda x: texture_corners_to_image_H(x, (240, 320))[0],
                                    (c,), eps=1e-5, atol=1e-5, rtol=1e-3)
    bad = torch.zeros_like(c)
    safe, flag = texture_corners_to_image_H(bad, (240, 320))
    assert not flag.any() and torch.isfinite(safe).all()
    return dict(native_hw=[240, 320], corner_roundtrip_max_px=error,
                DLT_float64_gradcheck=True, degenerate_corner_flagged=True)


def renderer_gradients() -> dict:
    torch.manual_seed(814)
    # Anisotropic map away from equal-norm/max ties and LOD knots.
    H = torch.tensor([[[.71, .03, -17.3], [.01, .57, -12.1], [.00012, -.00008, 1.]]],
                     dtype=torch.float64, requires_grad=True)
    B = torch.full((1, 3, 240, 320), .1, requires_grad=True)
    G = torch.full((1, 3, 240, 320), .6, requires_grad=True)
    S = torch.tensor([[.13, -.21, .37]], dtype=torch.float64, requires_grad=True)
    Q = (.2+.6*torch.rand(1, 3, 128, 128)).requires_grad_()
    y = render_camera_proxy(Q, H, B, G, S, native=True, quantize=False)
    probe = torch.randn_like(y)
    gradients = torch.autograd.grad((y*probe).sum(), (H, B, G, S, Q))
    norms = {}
    for name, gradient in zip(("H", "B", "G", "S", "Q"), gradients):
        assert torch.isfinite(gradient).all() and float(gradient.norm()) > 0, name
        norms[name] = float(gradient.norm())
    old_grid, _ = native_grid(H, S, (240, 320))
    H2 = H.detach().clone(); H2[:, 0, 2] += 1
    new_grid, _ = native_grid(H2, S, (240, 320))
    assert not torch.equal(old_grid, new_grid), "dynamic H must not use a stale grid cache"
    # Full per-pixel gradient checks are prohibitively large; directional finite
    # differences verify float sampling. STE quantization is deliberately off.
    d = torch.randn_like(Q); d = d/d.norm()
    analytic = float((gradients[-1]*d).sum())
    eps = .01
    with torch.no_grad():
        a = render_camera_proxy(Q+eps*d, H, B, G, S, native=True, quantize=False)
        b = render_camera_proxy(Q-eps*d, H, B, G, S, native=True, quantize=False)
        numeric = float(((a-b)*probe).sum()/(2*eps))
    rel = abs(analytic-numeric)/max(abs(analytic), abs(numeric), 1e-6)
    assert rel < .015, (analytic, numeric, rel)
    return dict(gradient_norms=norms, dynamic_H_grid_changes=True,
                float_Q_directional_difference=dict(analytic=analytic, numeric=numeric, relative_error=rel),
                native_hw_supported=[240, 320], RoboTwin_native_equivalence_claim=False)


def model_checks(target: Path) -> dict:
    torch.manual_seed(816)
    net = SpatialProxyEstimator(pretrained=False)
    with np.load(target, allow_pickle=False) as z:
        clean = torch.from_numpy(z["native_clean"].transpose(0, 3, 1, 2).copy()).float()/255
        query, qvalid = pack_exported_query(z)
        # Reconstruct inputs without even exposing the target-derived gauge key.
        no_gauge = {k:z[k] for k in z.files if k.startswith("query_") and k != "query_canonical_to_world_xy"}
        query_without_gauge, valid_without_gauge = pack_exported_query(no_gauge)
        assert torch.equal(query, query_without_gauge) and torch.equal(qvalid, valid_without_gauge)
        assert not qvalid[:, 6:15].any() and torch.count_nonzero(query[:, 6:15]) == 0
    # One valid and one deliberately unusable/missing camera. Native decoder
    # path is checked at actual fixture resolution, not a toy replacement model.
    h, w = clean.shape[-2:]
    inp = dict(native_clean=torch.cat([clean[:1], torch.full_like(clean[:1], float("nan"))])[None],
               camera_valid=torch.tensor([[True, False]]),
               T_world_eef=torch.eye(4)[None, None].repeat(1, 2, 1, 1),
               eef_valid=torch.tensor([[True, False]]), gripper=torch.tensor([[.4, float("nan")]]),
               gripper_valid=torch.tensor([[True, False]]),
               query=query[:1, None].expand(1, 2, -1).clone(),
               query_valid=qvalid[:1, None].expand(1, 2, -1).clone(),
               camera_id=torch.tensor([[0, -1]]), imaging_config_id=torch.tensor([[0, -1]]),
               imaging_config_valid=torch.tensor([[True, False]]))
    inp["T_world_eef"][:, 1] = float("nan")
    net.eval()
    with torch.no_grad():
        out = net(**inp)
        alter = {k:v.clone() for k, v in inp.items()}
        alter["native_clean"][:, 1] = 1e20
        other = net(**alter)
        for key in ("H", "B", "G", "S"):
            assert torch.equal(out[key][:, 0], other[key][:, 0]), key
        alter["native_clean"][:, 0] = 1-alter["native_clean"][:, 0]
        changed = net(**alter)
        assert torch.equal(out["S"], changed["S"]), "S must not depend on RGB"
        alter = {k:v.clone() for k,v in inp.items()}
        alter["query"][..., 6:15] = float("nan")
        gauge_changed = net(**alter)
        for key in ("H", "B", "G", "S"):
            assert torch.equal(out[key], gauge_changed[key]), key
        alter["query_valid"][:, 0, 6:15] = True
        try:
            net(**alter)
        except ValueError as e:
            assert "Canonical gauge" in str(e)
        else:
            raise AssertionError("Target-derived gauge was accepted as known input")
    net.train()
    groups = {"encoder": net.encoder, "state": net.state_mlp, "query": net.query_mlp,
              "geometry": net.geometry_head, "B": net.background_head, "G": net.gain_head,
              "S_embedding": net.s_embedding, "S_mlp": net.s_mlp, "decoder": net.pixel_fusion}
    before = {k:[p.detach().clone() for p in m.parameters()] for k,m in groups.items()}
    pred = net(**inp)
    assert pred["H"].dtype == torch.float64 and pred["H_valid"][0, 0]
    q = .2+.6*torch.rand(1, 3, 128, 128)
    rendered = render_camera_proxy(q, pred["H"][:, 0], pred["B"][:, 0], pred["G"][:, 0],
                                   pred["S"][:, 0], quantize=False, native=True)
    # Render residual backpropagates through H,B,G,S together; corner/B/G losses
    # provide a realistic supervised signal rather than a sum-of-parameters test.
    target_image = .2+.5*inp["native_clean"][:, 0]
    corner_target = pred["corners_native"][:, 0].detach()+torch.tensor([2., -1.])
    loss = (rendered-target_image).square().mean()
    loss = loss+.01*((pred["corners_native"][:, 0]-corner_target)/max(h,w)).square().mean()
    loss = loss+.1*(pred["B"][:, 0]-.7*inp["native_clean"][:, 0]).square().mean()
    loss = loss+.1*(pred["G"][:, 0]-.15).square().mean()
    loss.backward()
    gradient_norms = {}
    for name,module in groups.items():
        values = [p.grad for p in module.parameters() if p.grad is not None]
        assert values and all(torch.isfinite(g).all() for g in values), name
        norm = sum(float(g.abs().sum()) for g in values)
        assert norm > 0, name
        gradient_norms[name] = norm
    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    optimizer.step()
    updates = {name:max(float((a-p.detach()).abs().max()) for a,p in zip(before[name],module.parameters()))
               for name,module in groups.items()}
    assert all(x > 0 for x in updates.values()), updates
    net.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = net(**inp)
    assert all(torch.isfinite(value).all() for value in mixed.values())
    assert mixed["H"].dtype == torch.float64
    assert all(mixed[key].dtype == torch.float32 for key in ("B", "G", "S"))
    meta = net.metadata()
    assert meta["total_parameters"] < 10_000_000
    return dict(**meta, input_native_hw=[h,w], missing_camera_isolated=True,
                missing_EEF_nan_isolated=True, S_invariant_to_RGB=True,
                forbidden_gauge_key_never_read=True, forbidden_gauge_values_do_not_affect_outputs=True,
                forbidden_gauge_validity_rejected=True,
                CPU_bfloat16_autocast_finite=True, output_geometry_dtype="float64",
                gradient_l1=gradient_norms, max_parameter_updates=updates, loss=float(loss.detach()),
                initialized_with_pretrained_weights=False,
                target_fixture=str(target), target_sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run CPU-only with CUDA_VISIBLE_DEVICES='' explicitly")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    report = dict(CPU_only=True, torch_version=torch.__version__)
    report["legacy_equivalence"] = legacy_equivalence(args.root, args.root/"deployment/dl_full_proxy_teacher_20261111/payload")
    print("legacy CameraProxy image/Q-gradient equivalence passed", flush=True)
    report["geometry"] = geometry_checks()
    report["renderer"] = renderer_gradients()
    print("geometry and differentiable renderer checks passed", flush=True)
    report["model"] = model_checks(args.root/"deployment/dl_spatial_proxy_targets_20261111/payload/targets/target_00003.npz")
    report["passed"] = True
    report["source_sha256"] = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                               for name in ("model.py", "renderer.py", "test_core.py", "MODEL_NOTES.md")}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
