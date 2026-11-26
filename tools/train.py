"""Train one model. No test data is touched here: the final checkpoint is evaluated afterwards with
tools/eval.py (fixed epoch budget, last checkpoint, no test-set model selection).

Examples (PowerShell on the lab VM, from C:\\work\\camoxpert):
    uv run python tools/train.py --config configs/r5_camoxpert.py --out C:/runs/R5_camoxpert_s0
    uv run python tools/train.py --config configs/r5_camoxpert.py --out C:/runs/R5_camoxpert_s1 --set seed=1
    # timing run: 300 iterations, then print the projected time per epoch / per run
    uv run python tools/train.py --config configs/r5_camoxpert.py --out C:/runs/timing_R5 --max-iters 300
    # continue after a crash or reboot (restores model, optimizer, schedule position, RNG states)
    uv run python tools/train.py --config configs/r5_camoxpert.py --out C:/runs/R5_camoxpert_s0 --resume

Files written to --out:
    config.json        resolved config (incl. --set overrides)
    run_meta.json      git commit, library versions, GPU, data fingerprint
    train_log.jsonl    one line every `log_every` iterations (losses, lr, speed, memory)
    train.log          human-readable console log
    last.pth           full training state, overwritten at the end of every epoch (for --resume)
    model_final.pth    weights after the last epoch -> the ONLY checkpoint that gets evaluated
    train_summary.json total time, iterations, peak GPU memory
"""

import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from camoxpert import data as D  # noqa: E402
from camoxpert import engine as E  # noqa: E402
from camoxpert.config import load_config  # noqa: E402
from camoxpert.losses import SegLoss  # noqa: E402
from camoxpert.models import build_model  # noqa: E402

LOGGER = logging.getLogger("camoxpert")


def setup_logging(out_dir):
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(out_dir, "train.log"), encoding="utf-8")):
        h.setFormatter(fmt)
        LOGGER.addHandler(h)


def autocast_ctx(mode, device):
    if mode == "none" or device.type != "cuda":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[mode]
    return torch.autocast(device_type="cuda", dtype=dtype)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="run directory (created)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pth")
    ap.add_argument("--max-iters", type=int, default=None, help="stop after N iterations (timing run)")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    os.makedirs(args.out, exist_ok=True)
    setup_logging(args.out)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    cfg_path = os.path.join(args.out, "config.json")
    if args.resume:
        import json

        with open(cfg_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved != cfg:
            raise SystemExit("--resume: config differs from the saved config.json; refusing to mix settings")
    else:
        if os.path.exists(os.path.join(args.out, "model_final.pth")) or os.path.exists(os.path.join(args.out, "last.pth")):
            raise SystemExit(f"{args.out} already holds a run; use --resume or a new --out")
        E.write_json(cfg, cfg_path)

    E.seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------------------- data
    pairs = D.train_pairs(cfg["data"]["root"])
    if cfg["data"]["expected_train"] and len(pairs) != cfg["data"]["expected_train"]:
        raise SystemExit(f"Found {len(pairs)} training pairs, expected {cfg['data']['expected_train']} (C1)")
    dataset = D.TrainDataset(pairs, cfg["data"]["size"])
    tcfg = cfg["train"]
    gen = torch.Generator().manual_seed(cfg["seed"])  # fixes the shuffling order
    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=tcfg["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=tcfg["num_workers"] > 0,
        worker_init_fn=E.worker_init_fn,
        generator=gen,
    )
    iters_per_epoch = len(loader)
    total_iters = tcfg["epochs"] * iters_per_epoch

    # --------------------------------------------------------------- model
    model = build_model(cfg["model"], pretrained=tcfg["pretrained"], pretrained_path=tcfg["pretrained_path"]).to(device)
    criterion = SegLoss(**cfg["loss"])
    optimizer = E.build_optimizer(model, cfg["optim"])
    use_fp16_scaler = tcfg["amp"] == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    n_params = sum(p.numel() for p in model.parameters())
    meta = dict(
        env=E.env_info(),
        git=E.git_info(ROOT),
        argv=sys.argv,
        num_train=len(pairs),
        train_fingerprint=D.pairs_fingerprint(pairs),
        iters_per_epoch=iters_per_epoch,
        total_iters=total_iters,
        num_params=n_params,
    )
    if not args.resume:
        E.write_json(meta, os.path.join(args.out, "run_meta.json"))
    LOGGER.info(f"model={cfg['model']['name']} params={n_params / 1e6:.2f}M train={len(pairs)} "
                f"iters/epoch={iters_per_epoch} total_iters={total_iters} amp={tcfg['amp']}")
    if meta["git"]["dirty"]:
        LOGGER.warning("WARNING: the git working tree has uncommitted changes; commit before paper runs")

    start_epoch, it = 0, 0
    if args.resume:
        state = torch.load(os.path.join(args.out, "last.pth"), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        gen.set_state(state["loader_generator"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch, it = state["epoch"], state["iter"]
        LOGGER.info(f"Resumed after epoch {start_epoch} (iteration {it})")

    jlog = E.JsonlLogger(os.path.join(args.out, "train_log.jsonl"))
    t_run = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stop = False
    warm_it, t_warm = None, None  # timing runs measure speed after the first 50 iterations

    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        t_epoch = t_last = time.time()
        for batch in loader:
            # lr for THIS iteration is set before the update (legacy set it after, so iteration 0
            # ran at the full lr instead of the warm-up start)
            factor = E.lr_factor(it, total_iters, **cfg["sched"])
            E.set_lr(optimizer, factor)

            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with autocast_ctx(tcfg["amp"], device):
                out = model(image)
            loss, parts = criterion(out, mask, progress=it / total_iters)  # float32, outside autocast

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at iteration {it}: {parts}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tcfg["grad_clip"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            it += 1
            if args.max_iters and warm_it is None and it - start_epoch * iters_per_epoch == min(50, max(1, args.max_iters // 3)):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                warm_it, t_warm = it, time.time()

            if it % tcfg["log_every"] == 0 or it == 1:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.time()
                n = 1 if it == 1 else tcfg["log_every"]
                s_per_it = (now - t_last) / n
                t_last = now
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                jlog.log(iter=it, epoch=epoch + 1, lr=optimizer.param_groups[1]["lr"], s_per_it=s_per_it, gpu_gb=mem, **parts)
                eta_h = s_per_it * (total_iters - it) / 3600
                LOGGER.info(f"ep {epoch + 1}/{tcfg['epochs']} it {it}/{total_iters} lr {optimizer.param_groups[1]['lr']:.2e} "
                            f"loss {parts['total']:.4f} {s_per_it:.3f}s/it mem {mem:.1f}G eta {eta_h:.1f}h")
            if args.max_iters and it >= args.max_iters:
                stop = True
                break

        if stop:
            break
        LOGGER.info(f"epoch {epoch + 1} done in {(time.time() - t_epoch) / 60:.1f} min")
        E.atomic_save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                loader_generator=gen.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                epoch=epoch + 1,
                iter=it,
            ),
            os.path.join(args.out, "last.pth"),
        )

    jlog.close()
    elapsed = time.time() - t_run
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0

    if stop:  # timing run: report the projection, save nothing else
        if device.type == "cuda":
            torch.cuda.synchronize()
        done = it - start_epoch * iters_per_epoch
        s_it = (time.time() - t_warm) / max(it - warm_it, 1)
        summary = dict(timing_iters=done, measured_iters=it - warm_it, s_per_iter=s_it, peak_gpu_gb=peak,
                       startup_and_total_s=elapsed, batch_size=tcfg["batch_size"], amp=tcfg["amp"],
                       projected_epoch_min=s_it * iters_per_epoch / 60,
                       projected_run_h=s_it * total_iters / 3600, epochs=tcfg["epochs"])
        E.write_json(summary, os.path.join(args.out, "timing.json"))
        LOGGER.info(f"TIMING: {s_it:.3f} s/it -> {summary['projected_epoch_min']:.1f} min/epoch, "
                    f"{summary['projected_run_h']:.1f} h for {tcfg['epochs']} epochs, peak {peak:.1f} GB")
        return

    final = os.path.join(args.out, "model_final.pth")
    E.atomic_save(model.state_dict(), final)
    E.write_json(dict(epochs=tcfg["epochs"], iters=it, hours=elapsed / 3600, peak_gpu_gb=peak,
                      model_final_sha256=E.sha256_file(final)), os.path.join(args.out, "train_summary.json"))
    LOGGER.info(f"Finished: {it} iterations, {elapsed / 3600:.2f} h. Weights: {final}")
    LOGGER.info("BACK UP model_final.pth today (external drive or cloud).")


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this file
    main()
