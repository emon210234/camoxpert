"""Config files are plain Python: each defines a dict `CFG`. Usually it starts from base_config()
in configs/_base.py and overrides a few keys with merge().

Command-line overrides use dotted keys, e.g.  --set train.epochs=1 data.root=D:/COD
The value is parsed as a Python literal when possible (1 -> int, 1e-4 -> float, true/false ->
bool, [1,2] -> list); otherwise it stays a string.
"""

import ast
import copy
import json
import runpy


def merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; returns a new dict, inputs untouched."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(text):
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_overrides(cfg: dict, items) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"Unknown config key {key!r} (no section {p!r})")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key {key!r}")  # typos must not pass silently
        node[parts[-1]] = _parse_value(value)
    return cfg


def load_config(path: str, overrides=None) -> dict:
    import os
    import sys

    cfg_dir = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, cfg_dir)  # lets a config do `from _base import base_config, merge`
    try:
        ns = runpy.run_path(path)
    finally:
        sys.path.remove(cfg_dir)
    if "CFG" not in ns:
        raise KeyError(f"{path} must define a dict named CFG")
    cfg = apply_overrides(ns["CFG"], overrides)
    json.dumps(cfg)  # configs must be JSON-serialisable (they are saved next to every run)
    return cfg
