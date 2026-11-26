"""End-to-end smoke test on a tiny synthetic dataset (CPU, ~2 minutes). Not part of `pytest`:

    uv run python tests/e2e_smoke.py

Checks: train.py runs; same seed -> identical loss curve; --resume restores the state; eval.py
(single + TTA) writes metrics; profile_model.py runs; make_tables.py builds a table from the JSONs.
The numbers produced here are meaningless (random weights, fake data) and must never be reported.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def make_fake_cod(root, n_train=8, n_test=4, seed=0):
    rng = np.random.default_rng(seed)

    def write(split_dir, n):
        os.makedirs(os.path.join(split_dir, "Imgs"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "GT"), exist_ok=True)
        for i in range(n):
            h, w = rng.integers(80, 140, size=2)
            img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
            gt = np.zeros((h, w), np.uint8)
            cv2.circle(gt, (int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))), int(rng.integers(8, 18)), 255, -1)
            cv2.imwrite(os.path.join(split_dir, "Imgs", f"img_{i:03d}.jpg"), img)
            cv2.imwrite(os.path.join(split_dir, "GT", f"img_{i:03d}.png"), gt)

    write(os.path.join(root, "TrainDataset"), n_train)
    for name in ("CAMO", "CHAMELEON", "COD10K", "NC4K"):
        write(os.path.join(root, "TestDataset", name), n_test)
    os.makedirs(os.path.join(root, "TrainDataset", "__MACOSX"), exist_ok=True)  # junk must be ignored
    open(os.path.join(root, "TrainDataset", "Imgs", "._img_000.jpg"), "w").close()


def run(*args):
    print("$", " ".join(args), flush=True)
    r = subprocess.run([PY, "-W", "ignore", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {args}")
    return r.stdout


def losses(run_dir):
    with open(os.path.join(run_dir, "train_log.jsonl")) as f:
        return [json.loads(line)["total"] for line in f]


def main():
    tmp = tempfile.mkdtemp(prefix="cxp_e2e_")
    data = os.path.join(tmp, "COD")
    make_fake_cod(data)
    common = ["--device", "cpu", "--set", f"data.root={data}", "data.size=128", "data.expected_train=8",
              "train.epochs=2", "train.batch_size=2", "train.num_workers=0", "train.amp=none",
              "train.pretrained=false", "train.log_every=1", "eval.batch_size=2", "eval.num_workers=0"]

    for cfg in ("configs/r5_camoxpert.py", "configs/r6_camoxpert_pr.py", "configs/r0_zoomnext_b5.py", "configs/r1_camoxpert_base.py"):
        tag = os.path.basename(cfg)[:-3]
        a, b = os.path.join(tmp, f"{tag}_a"), os.path.join(tmp, f"{tag}_b")
        run("tools/train.py", "--config", cfg, "--out", a, *common)
        if tag in ("r5_camoxpert", "r0_zoomnext_b5"):
            run("tools/train.py", "--config", cfg, "--out", b, *common)
            assert losses(a) == losses(b), f"{tag}: same seed gave different losses"
        with open(os.path.join(a, "train_summary.json")) as f:
            sha_before = json.load(f)["model_final_sha256"]
        run("tools/train.py", "--config", cfg, "--out", a, "--resume", *common)  # nothing left to train
        with open(os.path.join(a, "train_summary.json")) as f:
            assert json.load(f)["model_final_sha256"] == sha_before, "resume did not restore the weights"
        run("tools/eval.py", "--run", a, "--device", "cpu", "--save-maps")
        if tag == "r5_camoxpert":
            run("tools/eval.py", "--run", a, "--device", "cpu", "--tta")
        with open(os.path.join(a, "eval_single.json")) as f:
            rep = json.load(f)
        assert set(rep["results"]) == {"CAMO", "CHAMELEON", "COD10K", "NC4K"}
        assert all(r["num_images"] == 4 for r in rep["results"].values())
        assert len(os.listdir(os.path.join(a, "maps_single", "CAMO"))) == 4
        print(f"  {tag}: OK  (loss curve reproducible, resume OK, eval OK)")

    # timing mode
    t = os.path.join(tmp, "timing")
    run("tools/train.py", "--config", "configs/r5_camoxpert.py", "--out", t, "--max-iters", "3", *common)
    assert os.path.exists(os.path.join(t, "timing.json")) and not os.path.exists(os.path.join(t, "model_final.pth"))

    # profile (CPU, no speed)
    run("tools/profile_model.py", "--config", "configs/r5_camoxpert.py", "--out", os.path.join(tmp, "prof"),
        "--device", "cpu", "--no-speed", "--set", "data.size=128")

    # tables from JSON (incl. an "external" row and mean over 2 "seeds")
    ext = os.path.join(tmp, "external.json")
    with open(ext, "w") as f:
        json.dump({"Other": {"cite": "x", "where": "y", "results": {"CAMO": {"Sm": 0.5, "MAE": 0.1}}}}, f)
    spec = dict(mode="single", datasets=["CAMO", "NC4K"], metrics=["Sm", "meanEm", "wFm", "MAE"], show_std=True,
                rows=[dict(label="Other", external=ext, key="Other"),
                      dict(label="CamoXpert", runs=[os.path.join(tmp, "r5_camoxpert_a"), os.path.join(tmp, "r5_camoxpert_a")]),
                      dict(label="PR", runs=[os.path.join(tmp, "r6_camoxpert_pr_a")])])
    sp = os.path.join(tmp, "spec.json")
    with open(sp, "w") as f:
        json.dump(spec, f)
    run("tools/make_tables.py", "--spec", sp, "--out", os.path.join(tmp, "table.tex"))
    print(open(os.path.join(tmp, "table.tex")).read())

    shutil.rmtree(tmp)
    print("E2E SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
