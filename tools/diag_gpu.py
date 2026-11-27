"""Find out which GPU setting crashes on this machine. Each test runs in its OWN Python process,
because after a CUDA error the GPU context of that process is unusable.

    uv run python tools/diag_gpu.py --out C:/runs/diag_gpu.json

Tests, cheapest first (random data and random weights; no dataset needed):
    basic            small matmul + conv on the GPU
    pinned           page-locked host memory + non_blocking copy (DataLoader pin_memory)
    dwconv_*         one depthwise 3x3 conv shaped like PVTv2's first stage, fp32/bf16, cudnn.benchmark off/on
    fwd_fp32         CamoXpert forward pass, batch 1, fp32, benchmark off
    train_*          3 CamoXpert training steps, batch 8, 448 px: {fp32, bf16} x {benchmark off, on}
After a failure the script waits 15 s and re-runs `basic`, to see whether the GPU recovered (a
Windows TDR driver reset recovers on its own) or stays broken (later results then mean nothing).
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

CASES = [
    "basic",
    "pinned",
    "dwconv_fp32_nobench",
    "dwconv_bf16_nobench",
    "dwconv_fp32_bench",
    "dwconv_bf16_bench",
    "fwd_fp32",
    "train_fp32_nobench",
    "train_bf16_nobench",
    "train_fp32_bench",
    "train_bf16_bench",
]


def run_case(name):
    import torch

    dev = torch.device("cuda")
    sync = torch.cuda.synchronize
    info = {}
    if name == "basic":
        a = torch.randn(2048, 2048, device=dev)
        (a @ a).sum().item()
        x = torch.randn(4, 64, 112, 112, device=dev)
        torch.nn.Conv2d(64, 64, 3, padding=1).to(dev)(x).sum().item()
        free, total = torch.cuda.mem_get_info()
        info.update(free_gb=free / 2**30, total_gb=total / 2**30, capability=torch.cuda.get_device_capability())
    elif name == "pinned":
        h = torch.randn(8, 3, 448, 448).pin_memory()
        d = h.to(dev, non_blocking=True)
        sync()
        assert torch.equal(d.cpu(), h)
    elif name.startswith("dwconv"):
        torch.backends.cudnn.benchmark = name.endswith("_bench")
        # PVTv2-B5 stage-1 MLP: 64 channels x 4 = 256 hidden, stride 4 -> 112 x 112 at 448 px
        conv = torch.nn.Conv2d(256, 256, 3, padding=1, groups=256).to(dev)
        x = torch.randn(8, 256, 112, 112, device=dev, requires_grad=True)
        bf16 = "bf16" in name
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                y = conv(x)
            y.float().sum().backward()
        sync()
    else:
        from camoxpert.losses import SegLoss
        from camoxpert.models import build_model

        torch.backends.cudnn.benchmark = name.endswith("_bench")
        model = build_model(dict(name="camoxpert"), pretrained=False).to(dev)
        if name == "fwd_fp32":
            model.eval()
            with torch.no_grad():
                model(torch.rand(1, 3, 448, 448, device=dev))
            sync()
        else:
            model.train()
            crit = SegLoss(bce=1.0, iou=0.5, ssim=0.3, edge=0.2, aux=0.1)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            bf16 = "bf16" in name
            times = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                t0 = time.time()
                x = torch.rand(8, 3, 448, 448, device=dev)
                m = (torch.rand(8, 1, 448, 448, device=dev) > 0.5).float()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    out = model(x)
                loss, _ = crit(out, m)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sync()
                times.append(time.time() - t0)
            info.update(s_per_step=times, peak_gb=torch.cuda.max_memory_allocated() / 2**30)
    return info


def child(name):
    try:
        info = run_case(name)
        print("DIAG_RESULT " + json.dumps({"ok": True, **info}))
    except Exception as e:  # report the first line of the CUDA error
        print("DIAG_RESULT " + json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0]}"}))


def spawn(name, timeout):
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")  # report the failing kernel at the right place
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--case", name], capture_output=True, text=True,
                           timeout=timeout, env=env)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("DIAG_RESULT ")]
        res = json.loads(lines[-1][12:]) if lines else {"ok": False, "error": f"process died (exit {r.returncode}): "
                                                         + (r.stderr.strip().splitlines() or ["no output"])[-1][:300]}
    except subprocess.TimeoutExpired:
        res = {"ok": False, "error": f"timeout after {timeout} s"}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def gpu_driver_info():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
            text=True, timeout=30).strip()
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="run only these tests")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    if args.case:
        child(args.case)
        return

    import torch

    report = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "nvidia_smi": gpu_driver_info(), "results": {}}
    print(f"torch {report['torch']}  CUDA runtime {report['cuda_runtime']}  cuDNN {report['cudnn']}")
    print(f"nvidia-smi: {report['nvidia_smi']}\n")
    for name in args.only or CASES:
        res = spawn(name, args.timeout)
        report["results"][name] = res
        extra = ""
        if res.get("s_per_step"):
            extra = f"  steps {', '.join(f'{t:.2f}' for t in res['s_per_step'])} s  peak {res['peak_gb']:.1f} GB"
        print(f"{name:22s} {'PASS' if res['ok'] else 'FAIL'}  ({res['seconds']} s){extra}"
              + ("" if res["ok"] else f"\n{'':25s}{res['error']}"), flush=True)
        if not res["ok"]:
            time.sleep(15)
            again = spawn("basic", 120)
            report["results"][name + "__basic_after"] = again
            print(f"{'  basic re-check':22s} {'PASS (GPU recovered)' if again['ok'] else 'FAIL (GPU still broken)'}",
                  flush=True)
            if not again["ok"]:
                print("Stopping: the GPU did not recover. Reboot the VM, then run the remaining tests with --only.")
                break
    print(f"\nnvidia-smi after: {gpu_driver_info()}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
