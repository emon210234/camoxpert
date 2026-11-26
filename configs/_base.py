"""Shared protocol for EVERY paper run. Change a value here and it changes for all runs.

Only `model` and `loss` differ between the experiment configs (r*.py). Values marked (legacy)
are the ones the pre-P2 v11/v12 configs used; see CHANGES.md for the ones that changed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from camoxpert.config import merge  # noqa: E402,F401  (re-exported for the r*.py configs)


def base_config():
    return dict(
        seed=0,
        data=dict(
            root="C:/data/COD",  # TrainDataset/ and TestDataset/ live here (lab VM)
            size=448,  # S: the LARGE input is S x S; masks are supervised at S x S (legacy)
            expected_train=4040,  # 3,040 COD10K + 1,000 CAMO; training refuses to start otherwise
            test_sets=["CAMO", "CHAMELEON", "COD10K", "NC4K"],
        ),
        model=dict(
            name="camoxpert",
            asw=True,
            brm=True,
            progressive_iters=0,
            use_scan=True,
            use_moe=True,
            num_experts=2,
            scales=[1.0, 0.75, 0.5],  # large, medium, small (legacy: 448 / 336 / 224)
            use_checkpoint=False,  # gradient checkpointing in the backbone (saves GPU memory)
        ),
        loss=dict(bce=1.0, iou=0.5, ssim=0.3, edge=0.2, ual=0.0, aux=0.1),
        train=dict(
            epochs=120,  # (legacy) PROVISIONAL: fixed after the P3 timing run, then same for all runs
            batch_size=8,  # (legacy)
            num_workers=2,  # the VM has 8 GB RAM; each Windows worker is a full Python process
            amp="bf16",  # "bf16", "fp16" or "none". Legacy flag use_amp=True never enabled autocast
            grad_clip=None,  # max grad-norm, None = off (legacy: off)
            log_every=20,
            pretrained=True,  # ImageNet PVTv2-B5 weights (False only for software tests)
            pretrained_path=None,  # local pvt_v2_b5.pth, else downloaded from the PVT GitHub release
        ),
        optim=dict(
            name="adamw",
            lr=5e-5,  # (legacy) decoder lr
            backbone_lr_factor=0.1,  # (legacy) backbone lr = 5e-6
            weight_decay=1e-4,  # (legacy)
            betas=[0.9, 0.999],
        ),
        sched=dict(
            warmup_iters=1000,  # (legacy) linear warm-up
            warmup_start=0.01,  # (legacy)
            min_factor=0.01,  # cosine ends at 1% of the initial lr (legacy code ended at 10%, see C11)
        ),
        eval=dict(batch_size=8, num_workers=2),
    )
