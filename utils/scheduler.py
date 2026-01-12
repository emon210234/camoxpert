import math
from bisect import bisect_right

def linear_increase(low_bound, up_bound, percentage):
    return low_bound + (up_bound - low_bound) * percentage

def cos_anneal(low_bound, up_bound, percentage):
    cos_percentage = (1 + math.cos(math.pi * percentage)) / 2.0
    return linear_increase(low_bound, up_bound, percentage=cos_percentage)

def poly_anneal(low_bound, up_bound, percentage, lr_decay):
    poly_percentage = pow((1 - percentage), lr_decay)
    return linear_increase(low_bound, up_bound, percentage=poly_percentage)

def linear_anneal(low_bound, up_bound, percentage):
    return linear_increase(low_bound, up_bound, percentage=1 - percentage)

class Scheduler:
    def __init__(self, optimizer, num_iters, epoch_length, scheduler_cfg):
        self.optimizer = optimizer
        self.num_iters = num_iters
        self.epoch_length = epoch_length
        self.scheduler_cfg = scheduler_cfg
        self.warmup_cfg = scheduler_cfg["warmup"]
        self.mode = self.scheduler_cfg["mode"]
        self.learning_rates = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, curr_idx):
        if curr_idx < self.warmup_cfg["num_iters"]:
            percentage = curr_idx / self.warmup_cfg["num_iters"]
            self._update_lr(self.warmup_cfg.get("mode", "linear"), self.warmup_cfg.get("initial_coef", 0.01), 1.0, percentage)
        else:
            percentage = (curr_idx - self.warmup_cfg["num_iters"]) / (self.num_iters - self.warmup_cfg["num_iters"])
            self._update_lr(self.mode, 1.0, self.scheduler_cfg["cfg"].get("gamma", 0.1), percentage)

    def _update_lr(self, mode, factor, target_factor, percentage):
        if mode == "linear": factor = linear_anneal(target_factor, factor, percentage)
        elif mode == "cos": factor = cos_anneal(target_factor, factor, percentage)
        elif mode == "constant": factor = self.scheduler_cfg["cfg"]["coef"]
        
        for group, initial_lr in zip(self.optimizer.param_groups, self.learning_rates):
            group["lr"] = initial_lr * factor
