"""Parameters, FLOPs and speed of a model (no trained weights needed: random init gives the same
counts and practically the same speed).

    uv run python tools/profile_model.py --config configs/r5_camoxpert.py --out C:/runs/profile
    uv run python tools/profile_model.py --config configs/r5_camoxpert.py --out profile --device cpu --no-speed   # laptop

What is counted
  * Params: all parameters, split into backbone (PVTv2-B5) and decoder (everything else).
  * FLOPs: torch.utils.flop_counter.FlopCounterMode on one image at the config's input size.
    It counts the multiply-adds of convolutions and matrix products (incl. the GRU and attention
    matmuls) as 2 FLOPs each; element-wise ops are not counted. MACs = FLOPs / 2. Many COD papers
    label MACs as "FLOPs" (thop/ptflops); the paper must say which one a column is.
  * Speed: forward pass only, batch 1, after 20 warm-up passes, mean over 100 timed passes with
    CUDA synchronisation; float32 and bf16 autocast. Input pyramid construction is included
    (it is inside forward); image loading and resizing are not.
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from camoxpert import engine as E  # noqa: E402
from camoxpert.config import load_config  # noqa: E402
from camoxpert.models import build_model  # noqa: E402


@torch.no_grad()
def time_forward(model, x, amp, warmup=20, iters=100):
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.autocast("cuda", enabled=False)
    with ctx:
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    return {"ms_per_image": ms, "fps": 1000.0 / ms}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", required=True, help="directory for profile_<run_name>.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-speed", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    model = build_model(cfg["model"], pretrained=False).eval()
    prefix = model.backbone_prefix
    n_total = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))
    size = cfg["data"]["size"]

    x = torch.rand(1, 3, size, size)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    flops = fc.get_total_flops()

    report = dict(
        run_name=cfg.get("run_name"),
        model=cfg["model"],
        input_size=size,
        params_total=n_total,
        params_backbone=n_backbone,
        params_decoder=n_total - n_backbone,
        gflops=flops / 1e9,
        gmacs=flops / 2e9,
        flop_counter="torch.utils.flop_counter.FlopCounterMode (conv + matmul, 2 FLOPs per MAC)",
        env=E.env_info(),
    )
    if not args.no_speed:
        if not torch.cuda.is_available() or args.device != "cuda":
            raise SystemExit("speed needs a CUDA GPU (use --no-speed on CPU)")
        model = model.cuda()
        xc = x.cuda()
        report["speed_fp32"] = time_forward(model, xc, amp=False)
        report["speed_bf16"] = time_forward(model, xc, amp=True)
        report["speed_protocol"] = "batch 1, forward only, 20 warm-up + 100 timed passes, cuda.synchronize"

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"profile_{cfg.get('run_name', 'model')}.json")
    E.write_json(report, path)
    print(f"params {n_total / 1e6:.2f}M (backbone {n_backbone / 1e6:.2f}M, decoder {(n_total - n_backbone) / 1e6:.2f}M)"
          f"  {flops / 1e9:.1f} GFLOPs = {flops / 2e9:.1f} GMACs at {size}x{size}")
    for k in ("speed_fp32", "speed_bf16"):
        if k in report:
            print(f"{k}: {report[k]['ms_per_image']:.1f} ms/image, {report[k]['fps']:.1f} FPS")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
