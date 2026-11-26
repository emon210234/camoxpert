"""Evaluate the FINAL checkpoint of a run on the four COD test sets.

    uv run python tools/eval.py --run C:/runs/R5_camoxpert_s0 --save-maps
    uv run python tools/eval.py --run C:/runs/R5_camoxpert_s0 --tta        # reported separately

Single pass (default) = what goes in the main comparison table: one forward pass per image at the
training input size, sigmoid, bilinear upsampling to the GT size, metrics from PySODMetrics.

--tta = the legacy multi-scale + flip test-time augmentation: input sizes 0.75 S, S, 1.25 S, each
with and without horizontal flip, the 6 probability maps averaged. Other papers' numbers are
single-pass, so TTA results may only appear in a separate, clearly labelled table (C3).

Writes <run>/eval_single.json or <run>/eval_tta.json and, with --save-maps,
<run>/maps_single/<DATASET>/<name>.png (uint8 probability maps, used for the qualitative figure
and for re-computing metrics without the GPU).
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from camoxpert import data as D  # noqa: E402
from camoxpert import engine as E  # noqa: E402
from camoxpert.metrics import CODMetrics  # noqa: E402
from camoxpert.models import build_model  # noqa: E402

TTA_SCALES = (0.75, 1.0, 1.25)


def _collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "gt_path": [b["gt_path"] for b in batch],
        "name": [b["name"] for b in batch],
    }


@torch.no_grad()
def predict_probs(model, image, tta):
    """image [B,3,S,S] in [0,1] -> list of probability maps [B,1,h,w] at the model's output
    resolution: 1 map (single pass) or 6 maps (TTA). Each map is later resized ONCE, straight to
    the GT size, and the maps are averaged there (as the legacy TTA code did)."""
    if not tta:
        return [model(image)["logits"].float().sigmoid()]
    maps = []
    for s in TTA_SCALES:
        x = image if s == 1.0 else F.interpolate(image, scale_factor=s, mode="bilinear", align_corners=False)
        for flip in (False, True):
            xi = torch.flip(x, dims=[3]) if flip else x
            p = model(xi)["logits"].float().sigmoid()
            maps.append(torch.flip(p, dims=[3]) if flip else p)
    return maps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory with config.json and model_final.pth")
    ap.add_argument("--ckpt", default=None, help="default: <run>/model_final.pth")
    ap.add_argument("--data-root", default=None, help="override config data.root")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--save-maps", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(os.path.join(args.run, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    ckpt = args.ckpt or os.path.join(args.run, "model_final.pth")
    root = args.data_root or cfg["data"]["root"]
    names = args.datasets or cfg["data"]["test_sets"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tag = "tta" if args.tta else "single"

    model = build_model(cfg["model"], pretrained=False)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    report = dict(
        run=os.path.abspath(args.run),
        checkpoint=os.path.abspath(ckpt),
        checkpoint_sha256=E.sha256_file(ckpt),
        mode=tag,
        tta_scales=list(TTA_SCALES) if args.tta else None,
        input_size=cfg["data"]["size"],
        precision="float32",
        protocol="sigmoid -> bilinear resize to GT size -> uint8 -> PySODMetrics (GT>128, per-image min-max)",
        env=E.env_info(),
        git=E.git_info(ROOT),
        results={},
    )
    size = cfg["data"]["size"]
    for name in names:
        pairs = D.test_pairs(root, name)
        loader = DataLoader(D.TestDataset(pairs, size), batch_size=cfg["eval"]["batch_size"], shuffle=False,
                            num_workers=cfg["eval"]["num_workers"], collate_fn=_collate)
        metrics = CODMetrics()
        map_dir = os.path.join(args.run, f"maps_{tag}", name)
        if args.save_maps:
            os.makedirs(map_dir, exist_ok=True)
        t0 = time.time()
        for batch in tqdm(loader, desc=name, ncols=80):
            maps = [m.cpu() for m in predict_probs(model, batch["image"].to(device), args.tta)]
            for i, (gt_path, n) in enumerate(zip(batch["gt_path"], batch["name"])):
                gt = D.read_gray(gt_path)
                p = torch.stack([F.interpolate(m[i : i + 1], size=gt.shape, mode="bilinear", align_corners=False)
                                 for m in maps]).mean(0)[0, 0]
                pred = (p.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                metrics.step(pred, gt)
                if args.save_maps:
                    cv2.imwrite(os.path.join(map_dir, n + ".png"), pred)
        res = metrics.results()
        res["fingerprint"] = D.pairs_fingerprint(pairs)
        res["seconds"] = time.time() - t0
        report["results"][name] = res
        print(f"{name:10s} n={res['num_images']:5d}  Sm {res['Sm']:.4f}  wFm {res['wFm']:.4f}  "
              f"MAE {res['MAE']:.4f}  meanEm {res['meanEm']:.4f}  maxEm {res['maxEm']:.4f}  adpEm {res['adpEm']:.4f}")

    out = os.path.join(args.run, f"eval_{tag}.json")
    E.write_json(report, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
