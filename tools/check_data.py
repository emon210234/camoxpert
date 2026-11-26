"""Check the dataset before any training: counts, image/mask pairing, sizes, mask values.

    uv run python tools/check_data.py --root C:/data/COD --out C:/runs/data_manifest.json

Expected counts (standard protocol): train 4,040; CAMO 250, CHAMELEON 76, COD10K 2,026, NC4K 4,121.
The manifest stores a fingerprint (SHA-256 of the file-name list) per split; every training and
evaluation run stores the same fingerprint, so you can prove which files a number came from.
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from camoxpert import data as D  # noqa: E402
from camoxpert import engine as E  # noqa: E402

EXPECTED = {"train": 4040, "CAMO": 250, "CHAMELEON": 76, "COD10K": 2026, "NC4K": 4121}


def inspect(pairs, name, full):
    size_mismatch, nonbinary_masks, empty_masks = [], 0, 0
    for img_path, gt_path in tqdm(pairs, desc=name, ncols=80, disable=not full):
        gt = D.read_gray(gt_path)
        if full:
            img = D.read_rgb(img_path)
            if img.shape[:2] != gt.shape[:2]:
                size_mismatch.append(os.path.basename(img_path))
        vals = np.unique(gt)
        if not np.all(np.isin(vals, (0, 255))):
            nonbinary_masks += 1
        if gt.max() <= 128:
            empty_masks += 1
    return dict(
        count=len(pairs),
        expected=EXPECTED.get(name),
        ok=len(pairs) == EXPECTED.get(name),
        fingerprint=D.pairs_fingerprint(pairs),
        image_mask_size_mismatch=size_mismatch if full else "not checked (use --full)",
        masks_with_values_other_than_0_255=nonbinary_masks,
        masks_without_foreground_above_128=empty_masks,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="C:/data/COD")
    ap.add_argument("--out", default=None)
    ap.add_argument("--full", action="store_true", help="also read every image to compare sizes (slower)")
    args = ap.parse_args()

    report = {"root": os.path.abspath(args.root)}
    report["train"] = inspect(D.train_pairs(args.root), "train", args.full)
    for name in D.TEST_SETS:
        report[name] = inspect(D.test_pairs(args.root, name), name, args.full)

    all_ok = True
    for k, v in report.items():
        if k == "root":
            continue
        all_ok &= v["ok"]
        print(f"{k:10s} {v['count']:5d} (expected {v['expected']})  {'OK' if v['ok'] else 'MISMATCH'}  "
              f"non-binary masks: {v['masks_with_values_other_than_0_255']}  "
              f"no-foreground masks: {v['masks_without_foreground_above_128']}")
    if args.out:
        E.write_json(report, args.out)
        print(f"Wrote {args.out}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
