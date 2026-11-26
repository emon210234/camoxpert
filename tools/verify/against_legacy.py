"""Prove that the new CamoXpert code computes exactly what the legacy v11 / v12 / v12-Progressive
classes computed (same architecture, same loss), so the paper describes the trained model.

    uv run python tools/verify/against_legacy.py --legacy-root ../_archive/camoxpert_pre_P2

For each variant: build the legacy class with random weights, copy the weights into the new
class, feed both the same input, and compare (a) the eval-mode logits and (b) the train-mode
loss incl. the MoE auxiliary term. CPU, float32, small input (224) to stay fast.
Differences must be at float32 round-off level (< 1e-5 relative).

The only weights the new code drops are the legacy dead ones: hmu_*.conv1x1_2 (built, never
used) and, in progressive mode, proj (built, never used).
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from camoxpert.losses import SegLoss  # noqa: E402
from camoxpert.models import build_model  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shim import import_root  # noqa: E402

DEAD = ("conv1x1_2.",)


def random_mask(b, s, seed):
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(b, 1, s, s)
    for i in range(b):
        y, x = torch.randint(s // 4, 3 * s // 4, (2,), generator=g)
        r = int(torch.randint(s // 8, s // 4, (1,), generator=g))
        yy, xx = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
        m[i, 0] = (((yy - y) ** 2 + (xx - x) ** 2) < r * r).float()
    return m


def compare(name, legacy, new_cfg, loss_cfg, size=224, batch=2):
    new = build_model(new_cfg, pretrained=False)
    sd = legacy.state_dict()
    dropped = [k for k in sd if any(d in k for d in DEAD) or (new_cfg.get("progressive_iters") and k.startswith("proj."))]
    sd_new = {k: v for k, v in sd.items() if k not in dropped}
    missing, unexpected = new.load_state_dict(sd_new, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    n_legacy = sum(p.numel() for p in legacy.parameters())
    n_new = sum(p.numel() for p in new.parameters())
    x = torch.rand(batch, 3, size, size, generator=torch.Generator().manual_seed(1))
    legacy_in = dict(
        image_l=x,
        image_m=F.interpolate(x, scale_factor=0.75, mode="bilinear", align_corners=False),
        image_s=F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False),
    )

    # (a) eval-mode logits
    legacy.eval()
    new.eval()
    with torch.no_grad():
        lo = legacy.body(legacy_in)
        lo = lo[-1] if isinstance(lo, list) else lo
        no = new(x)["logits"]
    d_eval = (lo - no).abs().max().item()

    # (b) train-mode loss (BatchNorm uses batch statistics; dropout draws the same random numbers
    #     because both models call their dropout layers in the same order after the same seed)
    legacy.train()
    new.train()
    mask = random_mask(batch, size, seed=2)
    torch.manual_seed(3)
    l_loss = legacy(dict(legacy_in, mask=mask))["loss"]
    torch.manual_seed(3)
    out = new(x)
    n_loss, parts = SegLoss(**loss_cfg)(out, mask)
    d_loss = abs(l_loss.item() - n_loss.item()) / max(abs(l_loss.item()), 1e-12)

    ok = d_eval < 1e-4 and d_loss < 1e-5
    print(f"{name:22s} params legacy {n_legacy:,} new {n_new:,} (dropped dead: {n_legacy - n_new:,} in {len(dropped)} tensors) | "
          f"max|logit diff| {d_eval:.2e} | loss legacy {l_loss.item():.6f} new {n_loss.item():.6f} rel diff {d_loss:.2e} "
          f"| {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-root", required=True, help="folder that contains the legacy `methods/` package")
    args = ap.parse_args()
    sys.path.insert(0, import_root(args.legacy_root))
    from methods.zoomnext.camoxpert_v12 import CamoXpertV12_Base, CamoXpertV12_Progressive
    from methods.zoomnext.zoomnext import CamoXpertV11

    torch.set_grad_enabled(True)
    base = dict(name="camoxpert", scales=[1.0, 0.75, 0.5])
    v11_loss = dict(bce=1.0, aux=0.1)
    v12_loss = dict(bce=1.0, iou=0.5, ssim=0.3, edge=0.2, aux=0.1)
    results = []
    torch.manual_seed(0)
    results.append(compare("v11 -> CamoXpert-Base", CamoXpertV11(num_frames=1, pretrained=False),
                           dict(base, asw=False, brm=False), v11_loss))
    torch.manual_seed(0)
    results.append(compare("v12 -> CamoXpert", CamoXpertV12_Base(num_frames=1, pretrained=False),
                           dict(base, asw=True, brm=True), v12_loss))
    torch.manual_seed(0)
    results.append(compare("v12-Prog -> CamoXpert-PR", CamoXpertV12_Progressive(num_frames=1, pretrained=False),
                           dict(base, asw=True, brm=True, progressive_iters=2), v12_loss))
    print("ALL OK" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
