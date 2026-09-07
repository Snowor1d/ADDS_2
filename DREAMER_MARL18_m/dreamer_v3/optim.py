from __future__ import annotations

import torch


class LaProp(torch.optim.Optimizer):
    """LaProp optimizer: RMS-normalize gradients before applying momentum."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-20,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        defaults = dict(
            lr=lr,
            base_lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=int(warmup_steps),
            step=0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = int(group.get("step", 0)) + 1
            base_lr = group.get("base_lr", group["lr"])
            warmup_steps = int(group.get("warmup_steps", 0))
            if warmup_steps > 0:
                lr = base_lr * min(1.0, group["step"] / float(warmup_steps))
            else:
                lr = base_lr
            group["lr"] = base_lr
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2) #  grad^2 평균을gradient 스케일로 추정
                bias_correction2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.div(bias_correction2).sqrt().add_(eps) # bias-corrected RMS gradient scale
                normalized_grad = grad / denom # 최근 gradient들의 평균적인 크기

                exp_avg.mul_(beta1).add_(normalized_grad, alpha=1.0 - beta1)
                bias_correction1 = 1.0 - beta1 ** step
                param.add_(exp_avg, alpha=-lr / bias_correction1)

        return loss
