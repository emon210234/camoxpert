"""Fast CPU unit tests:  uv run pytest -q"""

import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from camoxpert import engine as E  # noqa: E402
from camoxpert.config import apply_overrides, load_config  # noqa: E402
from camoxpert.losses import SegLoss, edge_targets, soft_iou_loss, ssim, ual_coef  # noqa: E402
from camoxpert.metrics import CODMetrics  # noqa: E402
from camoxpert.models import build_model  # noqa: E402

CONFIGS = sorted(f for f in os.listdir(os.path.join(ROOT, "configs")) if f.startswith("r") and f.endswith(".py"))


# ------------------------------------------------------------------------------------- losses
def test_ssim_identical_is_one():
    x = torch.rand(2, 1, 32, 32)
    assert ssim(x, x).item() == pytest.approx(1.0, abs=1e-5)


def test_iou_perfect_and_disjoint():
    m = torch.zeros(1, 1, 8, 8)
    m[..., :4, :] = 1
    assert soft_iou_loss(m, m).item() == pytest.approx(0.0, abs=1e-6)
    assert soft_iou_loss(1 - m, m).item() == pytest.approx(1.0, abs=1e-6)


def test_edge_targets_on_square():
    m = torch.zeros(1, 1, 16, 16)
    m[..., 4:12, 4:12] = 1
    e = edge_targets(m)
    assert e[0, 0, 8, 8] == 0 and e[0, 0, 0, 0] == 0  # inside and far outside: no edge
    assert e[0, 0, 4, 8] == 1 and e[0, 0, 3, 8] == 1  # both sides of the boundary row


def test_ual_ramp():
    assert ual_coef(0.0) == 0.0 and ual_coef(1.0) == 1.0
    assert ual_coef(0.5) == pytest.approx(0.5)


def test_edge_loss_requires_edge_head():
    out = dict(logits=torch.zeros(1, 1, 8, 8), side_logits=[], edge_logits=None, aux_loss=torch.zeros(()))
    with pytest.raises(ValueError):
        SegLoss(bce=1, edge=0.2)(out, torch.zeros(1, 1, 8, 8))


def test_progressive_weights():
    """L = 1.0 * BCE(p0) + 1.2 * BCE(p1) for two side outputs (BCE only)."""
    z0, z1 = torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8)
    mask = (torch.rand(1, 1, 8, 8) > 0.5).float()
    out = dict(logits=z1, side_logits=[z0, z1], edge_logits=None, aux_loss=torch.zeros(()))
    total, _ = SegLoss(bce=1.0)(out, mask)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    assert total.item() == pytest.approx((bce(z0, mask) + 1.2 * bce(z1, mask)).item(), rel=1e-6)


# --------------------------------------------------------------------------------- schedule
def test_lr_schedule_shape():
    kw = dict(total=10_000, warmup_iters=1000, warmup_start=0.01, min_factor=0.01)
    assert E.lr_factor(0, **kw) == pytest.approx(0.01)
    assert E.lr_factor(500, **kw) == pytest.approx(0.505)
    assert E.lr_factor(1000, **kw) == pytest.approx(1.0)
    assert E.lr_factor(5500, **kw) == pytest.approx(0.01 + 0.99 * 0.5)
    assert E.lr_factor(10_000, **kw) == pytest.approx(0.01)
    vals = [E.lr_factor(i, **kw) for i in range(1000, 10_001, 100)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))  # monotone after warm-up


# ----------------------------------------------------------------------------------- config
@pytest.mark.parametrize("name", CONFIGS)
def test_configs_load(name):
    cfg = load_config(os.path.join(ROOT, "configs", name))
    assert cfg["data"]["expected_train"] == 4040
    if cfg["loss"]["edge"]:
        assert cfg["model"].get("brm"), "edge loss without an edge head"


def test_override_rejects_typos():
    cfg = {"train": {"epochs": 1}}
    assert apply_overrides(cfg, ["train.epochs=5"])["train"]["epochs"] == 5
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["train.epoch=5"])


def test_protocol_identical_across_runs():
    """Only model, loss, run_name (and R0's input size) may differ between paper configs."""
    cfgs = {n: load_config(os.path.join(ROOT, "configs", n)) for n in CONFIGS}
    ref = cfgs["r5_camoxpert.py"]
    for n, c in cfgs.items():
        for key in ("seed", "train", "optim", "sched", "eval"):
            assert c[key] == ref[key], (n, key)
        d = dict(c["data"])
        if n.startswith("r0_"):
            d["size"] = ref["data"]["size"]
        assert d == ref["data"], n


# ----------------------------------------------------------------------------------- models
@pytest.mark.parametrize("name", CONFIGS)
def test_forward_backward_tiny(name):
    cfg = load_config(os.path.join(ROOT, "configs", name))
    model = build_model(cfg["model"], pretrained=False)
    x = torch.rand(2, 3, 96, 96)
    out = model(x)
    assert out["logits"].shape[0] == 2 and out["logits"].shape[1] == 1
    mask = (torch.rand(2, 1, 96, 96) > 0.5).float()
    loss, parts = SegLoss(**cfg["loss"])(out, mask, progress=0.5)
    assert math.isfinite(loss.item())
    loss.backward()
    model.eval()
    with torch.no_grad():
        assert model(x)["aux_loss"].item() == 0.0  # no aux loss at test time


def test_param_groups_split_backbone():
    cfg = load_config(os.path.join(ROOT, "configs", "r5_camoxpert.py"))
    model = build_model(cfg["model"], pretrained=False)
    opt = E.build_optimizer(model, cfg["optim"])
    bb, dec = opt.param_groups
    assert bb["lr"] == pytest.approx(5e-6) and dec["lr"] == pytest.approx(5e-5)
    assert sum(p.numel() for p in bb["params"]) + sum(p.numel() for p in dec["params"]) == sum(
        p.numel() for p in model.parameters()
    )


# ---------------------------------------------------------------------------------- metrics
def test_metrics_perfect_prediction():
    gt = np.zeros((64, 64), np.uint8)
    gt[16:48, 16:48] = 255
    m = CODMetrics()
    m.step(gt.copy(), gt)
    r = m.results()
    assert r["MAE"] == pytest.approx(0.0, abs=1e-9)
    assert r["Sm"] == pytest.approx(1.0, abs=1e-6)
    assert r["wFm"] == pytest.approx(1.0, abs=1e-4)
