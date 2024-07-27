# sorted in order of increasing complexity
from typing import List
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer as TorchOptimizer

class Optimizer(TorchOptimizer):
    """
    Base class for all optimizers.
    """
    def __init__(self, params: List[Parameter], lr: float):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def zero_grad(self):
        """
        Zeroes the gradients of all the parameters.
        """
        for param in self.param_groups[0]['params']:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self):
        """
        Performs a single optimization step.
        """
        raise NotImplementedError


class OptimizerGroup(Optimizer):
    """
    Combines multiple optimizers into one.
    """
    def __init__(self, *optimizers: Optimizer):
        params = [param for optimizer in optimizers for param in optimizer.param_groups[0]['params']]
        super().__init__(params, optimizers[0].param_groups[0]['lr'])
        self.optimizers = optimizers

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum and weight decay.

    - Described: https://paperswithcode.com/method/sgd
    """
    def __init__(self, params: List[Parameter], lr=0.001, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.state = {param: {'momentum_buffer': torch.zeros_like(param)} for param in params}

    def step(self):
        for param in self.param_groups[0]['params']:
            if param.grad is None:
                continue
            d_p = param.grad
            if self.weight_decay != 0:
                d_p = d_p.add(param, alpha=self.weight_decay)
            if self.momentum != 0:
                buf = self.state[param]['momentum_buffer']
                buf.mul_(self.momentum).add_(d_p)
                if self.nesterov:
                    d_p = d_p.add(buf, alpha=self.momentum)
                else:
                    d_p = buf
            param.add_(d_p, alpha=-self.param_groups[0]['lr'])


class AdamW(Optimizer):
    """
    AdamW optimizer with optional weight decay.

    - Described: https://paperswithcode.com/method/adamw
    - Paper: https://arxiv.org/abs/1711.05101v3
    """
    def __init__(self, params: List[Parameter], lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {param: {'exp_avg': torch.zeros_like(param),
                              'exp_avg_sq': torch.zeros_like(param)} for param in params}

    def step(self):
        for param in self.param_groups[0]['params']:
            if param.grad is None:
                continue
            grad = param.grad
            state = self.state[param]

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = self.betas

            state['step'] = state.get('step', 0) + 1

            if self.weight_decay != 0:
                grad = grad.add(param, alpha=self.weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(self.eps)
            step_size = self.param_groups[0]['lr']

            param.addcdiv_(exp_avg, denom, value=-step_size)


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer with optional momentum and weight decay.

    - Described: https://paperswithcode.com/method/lars
    - Paper: https://arxiv.org/abs/1708.03888v3
    """
    def __init__(self, params: List[Parameter], lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=False, classic=True, tcoef=0.001):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.classic = classic
        self.tcoef = tcoef
        self.state = {param: {'momentum_buffer': torch.zeros_like(param)} for param in params}

    def step(self):
        for param in self.param_groups[0]['params']:
            if param.grad is None:
                continue
            grad = param.grad

            if self.weight_decay != 0:
                grad = grad.add(param, alpha=self.weight_decay)

            if self.tcoef != 0:
                r1 = param.pow(2).sum().sqrt()
                r2 = grad.pow(2).sum().sqrt()
                trust_ratio = (r1 / (r2 + self.weight_decay * r1)).clamp(max=1.0) * self.tcoef
                grad = grad.mul(trust_ratio)

            if self.classic:
                grad = grad.mul(self.param_groups[0]['lr'])

            if self.momentum != 0:
                buf = self.state[param]['momentum_buffer']
                buf.mul_(self.momentum).add_(grad)
                if self.nesterov:
                    grad = grad.add(buf, alpha=self.momentum)
                else:
                    grad = buf

            if not self.classic:
                grad = grad.mul(self.param_groups[0]['lr'])

            param.add_(grad, alpha=-1)


class LAMB(Optimizer):
    """
    LAMB optimizer with optional weight decay.

    - Described: https://paperswithcode.com/method/lamb
    - Paper: https://arxiv.org/abs/1904.00962
    """
    def __init__(self, params: List[Parameter], lr=0.001, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, adam=False):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.adam = adam
        self.state = {param: {'exp_avg': torch.zeros_like(param),
                              'exp_avg_sq': torch.zeros_like(param),
                              'step': 0} for param in params}

    def step(self):
        for param in self.param_groups[0]['params']:
            if param.grad is None:
                continue
            grad = param.grad
            state = self.state[param]

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = self.betas

            state['step'] += 1

            if self.weight_decay != 0:
                grad = grad.add(param, alpha=self.weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(self.eps)
            step_size = self.param_groups[0]['lr']

            if not self.adam:
                r1 = param.pow(2).sum().sqrt()
                r2 = exp_avg.div(denom).pow(2).sum().sqrt()
                trust_ratio = (r1 / r2).clamp(max=1.0)
            else:
                trust_ratio = 1.0

            param.addcdiv_(exp_avg, denom, value=-step_size * trust_ratio)
