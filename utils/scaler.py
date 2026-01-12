from functools import partial
from itertools import chain
from torch.cuda.amp import GradScaler, autocast
from . import tensor_ops as ops 

class Scaler:
    def __init__(self, optimizer, use_fp16=False, *, set_to_none=False, clip_grad=False, clip_mode=None, clip_cfg=None):
        self.optimizer = optimizer
        self.set_to_none = set_to_none
        self.autocast = autocast(enabled=use_fp16)
        self.scaler = GradScaler(enabled=use_fp16)
        if clip_grad:
            self.grad_clip_ops = partial(ops.clip_grad, mode=clip_mode, clip_cfg=clip_cfg)
        else:
            self.grad_clip_ops = None

    def calculate_grad(self, loss):
        self.scaler.scale(loss).backward()
        if self.grad_clip_ops is not None:
            self.scaler.unscale_(self.optimizer)
            self.grad_clip_ops(chain(*[group["params"] for group in self.optimizer.param_groups]))

    def update_grad(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=self.set_to_none)

    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict):
        self.scaler.load_state_dict(state_dict)
