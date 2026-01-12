import types
from torch.optim import SGD, Adam, AdamW

def get_optimizer(mode, params, initial_lr, optim_cfg):
    if mode == "adamw":
        return AdamW(params, lr=initial_lr, betas=optim_cfg.get("betas", (0.9, 0.999)), weight_decay=optim_cfg.get("weight_decay", 0))
    elif mode == "adam":
        return Adam(params, lr=initial_lr, betas=optim_cfg.get("betas", (0.9, 0.999)), weight_decay=optim_cfg.get("weight_decay", 0))
    return SGD(params, lr=initial_lr, momentum=optim_cfg["momentum"], weight_decay=optim_cfg["weight_decay"])

def group_params(model, group_mode, initial_lr, optim_cfg):
    if group_mode == "finetune":
        backbone_params, other_params = [], []
        for name, param in model.named_parameters():
            if "backbone" in name: backbone_params.append(param)
            else: other_params.append(param)
        return [{"params": backbone_params, "lr": initial_lr * optim_cfg.get("diff_factor", 0.1)}, {"params": other_params, "lr": initial_lr}]
    return model.parameters()

def construct_optimizer(model, initial_lr, mode, group_mode, cfg):
    params = group_params(model, group_mode, initial_lr, cfg)
    return get_optimizer(mode, params, initial_lr, cfg)
