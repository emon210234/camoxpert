"""Model registry: build_model(cfg["model"]) -> nn.Module."""

from .camoxpert import CamoXpert
from .zoomnext import ZoomNeXtB5, ZoomNeXtB5Plus

MODELS = {"camoxpert": CamoXpert, "zoomnext_b5": ZoomNeXtB5, "zoomnext_b5_plus": ZoomNeXtB5Plus}


def build_model(model_cfg: dict, pretrained: bool = True, pretrained_path: str | None = None):
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](pretrained=pretrained, pretrained_path=pretrained_path, **cfg)
