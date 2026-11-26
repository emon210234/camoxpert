"""Make a legacy/upstream `methods/` package importable on a machine without a CUDA GPU.

The upstream PVT file queries the GPU while the model is being built, so it cannot be built on
CPU. Instead of changing PyTorch at runtime, we copy the package to a temporary folder and put
our backbone file in it. That is safe because we first check that the original backbone file is
byte-identical (line endings aside) to upstream ZoomNeXt's, and ours differs from upstream only
in the two '[CamoXpert patch]' blocks (GPU query guarded; same attention kernels).
On a machine WITH a CUDA GPU nothing is copied: the original package is imported as is.
"""

import hashlib
import os
import shutil
import sys
import tempfile

import torch

UPSTREAM_PVT_MD5 = "d63cc0bb27e61f424ce17f3148a6bd5c"  # ZoomNeXt 614af43 methods/backbone/pvt_v2_eff.py (LF)
OUR_PVT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "camoxpert", "models", "backbone", "pvt_v2_eff.py")


def _md5_lf(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read().replace(b"\r\n", b"\n")).hexdigest()


def import_root(root):
    """Return a folder to put on sys.path that contains an importable `methods/` package."""
    root = os.path.abspath(root)
    if torch.cuda.is_available():
        return root
    orig = os.path.join(root, "methods", "backbone", "pvt_v2_eff.py")
    md5 = _md5_lf(orig)
    if md5 != UPSTREAM_PVT_MD5:
        raise SystemExit(f"{orig} is not the upstream PVT file (md5 {md5}); refusing to swap it")
    tmp = tempfile.mkdtemp(prefix="cxp_verify_")
    shutil.copytree(os.path.join(root, "methods"), os.path.join(tmp, "methods"))
    shutil.copyfile(OUR_PVT, os.path.join(tmp, "methods", "backbone", "pvt_v2_eff.py"))
    print(f"[shim] no CUDA GPU: imported {root}/methods from a copy with our CPU-safe PVT file (original md5 OK)",
          file=sys.stderr)
    return tmp
