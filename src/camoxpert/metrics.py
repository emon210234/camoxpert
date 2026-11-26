"""COD metrics via PySODMetrics (https://github.com/lartpang/PySODMetrics, MIT).

What PySODMetrics does with each (prediction, GT) pair, before any metric:
    gt   -> gt > 128                        (binarised)
    pred -> pred / 255, then per-image min-max to [0, 1] (unless the map is constant)
This mirrors the MATLAB evaluation toolbox used by SINet/SINet-V2 and most COD papers, so the
per-image min-max in the legacy test_v12.py was redundant but harmless (CHANGES.md, C4).

Reported (all averaged over the images of a dataset):
    Sm      S-measure (structure similarity, alpha = 0.5)                       higher is better
    wFm     weighted F-measure F_beta^w (beta^2 = 1)                            higher is better
    MAE     mean absolute error between pred in [0, 1] and binary GT            lower is better
    meanEm / maxEm / adpEm   E-measure averaged over 256 thresholds / its max / at the adaptive
                             threshold (2 x mean(pred))                         higher is better
    meanFm / maxFm / adpFm   F-measure (beta^2 = 0.3), same three variants       higher is better
The paper must say which E/F variant it reports (M3 in REVIEW_ROUND0.md); all are stored.
"""

import numpy as np
import py_sod_metrics as M


class CODMetrics:
    names = ("Sm", "wFm", "MAE", "meanEm", "maxEm", "adpEm", "meanFm", "maxFm", "adpFm")

    def __init__(self):
        self.sm = M.Smeasure()
        self.wfm = M.WeightedFmeasure()
        self.mae = M.MAE()
        self.em = M.Emeasure()
        self.fm = M.Fmeasure()
        self.count = 0

    def step(self, pred_uint8: np.ndarray, gt_uint8: np.ndarray):
        assert pred_uint8.dtype == np.uint8 and gt_uint8.dtype == np.uint8
        assert pred_uint8.shape == gt_uint8.shape, (pred_uint8.shape, gt_uint8.shape)
        for m in (self.sm, self.wfm, self.mae, self.em, self.fm):
            m.step(pred=pred_uint8, gt=gt_uint8)
        self.count += 1

    def results(self) -> dict:
        em = self.em.get_results()["em"]
        fm = self.fm.get_results()["fm"]
        return {
            "Sm": float(self.sm.get_results()["sm"]),
            "wFm": float(self.wfm.get_results()["wfm"]),
            "MAE": float(self.mae.get_results()["mae"]),
            "meanEm": float(np.mean(em["curve"])),
            "maxEm": float(np.max(em["curve"])),
            "adpEm": float(em["adp"]),
            "meanFm": float(np.mean(fm["curve"])),
            "maxFm": float(np.max(fm["curve"])),
            "adpFm": float(fm["adp"]),
            "num_images": self.count,
        }
