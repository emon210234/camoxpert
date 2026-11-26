"""Prove that models/zoomnext.py + losses.SegLoss(bce=1, ual=1) equal upstream ZoomNeXt-B5.

    git clone https://github.com/lartpang/ZoomNeXt ../ZoomNeXt_upstream   (commit 614af43 checked)
    uv run python tools/verify/zoomnext_port.py --upstream-root ../ZoomNeXt_upstream

Builds upstream PvtV2B5_ZoomNeXt with random weights, copies them into ZoomNeXtB5, and compares
eval logits and the training loss (BCE + UAL at several training-progress values).
Upstream gets s/m/l = 0.5x/1x/1.5x of a base; ours gets the large image and builds the rest.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-root", required=True)
    args = ap.parse_args()
    sys.path.insert(0, import_root(args.upstream_root))
    from methods.zoomnext.zoomnext import PvtV2B5_ZoomNeXt

    torch.manual_seed(0)
    up = PvtV2B5_ZoomNeXt(pretrained=False, num_frames=1)
    ours = build_model(dict(name="zoomnext_b5", scales=[1.0, 2.0 / 3.0, 1.0 / 3.0]), pretrained=False)
    ours.load_state_dict(up.state_dict(), strict=True)  # identical parameter names
    n_up = sum(p.numel() for p in up.parameters())
    n_ours = sum(p.numel() for p in ours.parameters())

    base = 96  # small base for CPU: s/m/l = 48 / 96 / 144
    x_l = torch.rand(2, 3, int(base * 1.5), int(base * 1.5), generator=torch.Generator().manual_seed(1))
    data = dict(
        image_l=x_l,
        image_m=F.interpolate(x_l, size=(base, base), mode="bilinear", align_corners=False),
        image_s=F.interpolate(x_l, size=(base // 2, base // 2), mode="bilinear", align_corners=False),
    )
    up.eval()
    ours.eval()
    with torch.no_grad():
        d_eval = (up(data) - ours(x_l)["logits"]).abs().max().item()

    mask = (torch.rand(2, 1, base, base, generator=torch.Generator().manual_seed(2)) > 0.5).float()
    up.train()
    ours.train()
    crit = SegLoss(bce=1.0, ual=1.0)
    worst = 0.0
    for prog in (0.0, 0.3, 0.7, 1.0):
        l_up = up(dict(data, mask=mask), iter_percentage=prog)["loss"].item()
        out = ours(x_l)
        l_ours = crit(out, mask, progress=prog)[0].item()  # logits already at mask size (base)
        worst = max(worst, abs(l_up - l_ours) / abs(l_up))
        print(f"progress {prog:.1f}: upstream loss {l_up:.6f}  ours {l_ours:.6f}")
    ok = d_eval < 1e-4 and worst < 1e-5
    print(f"params upstream {n_up:,} ours {n_ours:,} | max|logit diff| {d_eval:.2e} | worst rel loss diff {worst:.2e} | "
          f"{'OK' if ok else 'FAIL'}")
    print("NOTE: inputs here are built exactly like upstream (bilinear resize). In training, our pipeline builds the "
          "pyramid from the 576 image with F.interpolate, upstream used cv2.resize on the uint8 image.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
