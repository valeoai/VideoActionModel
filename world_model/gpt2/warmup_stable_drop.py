from typing import Any, Dict, Optional

from torch.optim.optimizer import Optimizer


class WarmupStableDrop:
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iter: int,
        end_iter: int,
        drop_iter: int = 0,
        num_iter: int = 0,
    ) -> None:
        self.warmup_iter = warmup_iter
        self.end_iter = end_iter
        self.drop_iter = drop_iter
        self.optimizer = optimizer
        self.num_iter = num_iter
        self.start_lr = []
        self.resume_step = num_iter
        for group in self.optimizer.param_groups:
            self.start_lr.append(group["lr"])

        self.step(self.num_iter)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_lr_warmup(self, num_iter: int, base_lr: float, warmup_iter: int) -> float:
        return base_lr * num_iter / warmup_iter

    def get_lr_stable(self, num_iter: int, base_lr: float) -> float:
        return base_lr

    def get_lr_drop(self, num_iter: int, base_lr: float) -> float:
        # progress = (self.end_iter - num_iter) / self.drop_iter
        return base_lr * (0.1 + max(0.9 * (self.end_iter - num_iter) / self.drop_iter, 0))

    def get_lr(self, base_lr: float) -> float:

        assert self.num_iter >= 0

        if self.num_iter < self.warmup_iter and self.warmup_iter > 0:
            return self.get_lr_warmup(self.num_iter, base_lr, self.warmup_iter)

        if self.num_iter > self.end_iter - self.drop_iter and self.drop_iter > 0:
            return self.get_lr_drop(self.num_iter, base_lr)

        return self.get_lr_stable(self.num_iter, base_lr)

    def step(self, num_iter: Optional[int] = None) -> None:
        if num_iter is None:
            num_iter = self.num_iter + 1
        self.num_iter = num_iter

        for group, base_lr in zip(self.optimizer.param_groups, self.start_lr):
            group["lr"] = self.get_lr(base_lr)
