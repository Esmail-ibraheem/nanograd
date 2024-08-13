# sorted in order of increasing complexity
from typing import List
from nanograd.nn.helpers import dedup
from nanograd.nn.tensor import Tensor

from torch.nn import Parameter
import torch 

class Optimizer:
  def __init__(self, params: List[Tensor], lr: float):
    # if it's None, but being put into an optimizer, set it to True
    for x in params:
      if x.requires_grad is None: x.requires_grad = True

    self.params: List[Tensor] = dedup([x for x in params if x.requires_grad])
    assert len(self.params) != 0, "optimizer must have at least one param"
    self.device = self.params[0].device
    self.buffers: List[Tensor] = dedup([x for x in params if not x.requires_grad])   # buffers are still realized
    self.lr = Tensor([lr], requires_grad=False, device=self.device).contiguous()

  def zero_grad(self):
    for param in self.params: param.grad = None

  def realize(self, extra=None):
    # NOTE: in extra is too late for most of the params due to issues with assign
    Tensor.corealize(extra + self.params + self.buffers if extra is not None else self.params + self.buffers)

class SGD(Optimizer):
  def __init__(self, params: List[Tensor], lr=0.001, momentum=0, weight_decay=0.0, nesterov=False):
    super().__init__(params, lr)
    self.momentum, self.wd, self.nesterov = momentum, weight_decay, nesterov
    self.b = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in self.params] if self.momentum else []

  # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
  def step(self) -> None:
    for i, t in enumerate(self.params):
      assert t.grad is not None
      g = t.grad.realize() + self.wd * t.detach()
      if self.momentum:
        self.b[i].assign(self.momentum * self.b[i] + g).realize()  # NOTE: self.b[i] is zero on the first run, no if required
        g = (g + self.momentum * self.b[i]) if self.nesterov else self.b[i]
      t.assign(t.detach() - g * self.lr)
    self.realize(self.b)

# LAMB is essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 its just Adam/W.
def AdamW(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01): return LAMB(params, lr, b1, b2, eps, wd, adam=True)
def Adam(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8): return LAMB(params, lr, b1, b2, eps, 0.0, adam=True)


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
  def __init__(self, params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, wd=0.0, adam=False):
    super().__init__(params, lr)
    self.b1, self.b2, self.eps, self.wd, self.adam, self.t = b1, b2, eps, wd, adam, Tensor([0], requires_grad=False).realize()
    self.m = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in self.params]
    self.v = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in self.params]

  def step(self) -> None:
    self.t.assign(self.t + 1).realize()
    for i, t in enumerate(self.params):
      assert t.grad is not None
      g = t.grad.realize()
      self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * g).realize()
      self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)).realize()
      m_hat = self.m[i] / (1.0 - self.b1**self.t)
      v_hat = self.v[i] / (1.0 - self.b2**self.t)
      up = (m_hat / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()
      if not self.adam:
        r1 = t.detach().square().sum().sqrt()
        r2 = up.square().sum().sqrt()
        r = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
      else:
        r = 1.0
      t.assign(t.detach() - self.lr * r * up)
    self.realize([self.t] + self.m + self.v)