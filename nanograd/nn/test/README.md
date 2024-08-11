
### Quick example comparing nanograd.nn(autograd engine) to PyTorch

```python
from nanograd.nn.tensor import Tensor 

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.numpy())  
print(y.grad.numpy())  
```

The same but in pytorch

```python
import torch

x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.numpy())  
print(y.grad.numpy())  
```
