## this is an autograd engine for writing some features like pytorch
## summation, multiplication, division, relu, sigmoid activation functions.

### Usage:
**run the test.py file, to test the scalar engine**
```Bash
python test.py
```

**Example to use the engine in the test file:**
```python
if __name__ == "__main__":
    from nanograd.nn.engine import Value

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a + b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    d += 3 * d + (b - a).sigmoid(5)
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    print(f'{g.data:.4f}') 
    g.backward()
    print(f'{a.grad:.4f}') 
    print(f'{b.grad:.4f}')  
    print(f'{e.grad:.4f}')  

```

> [!NOTE]
> assuming that you are in this directory nanograd/nanograd/nn/test.py



### Quick example comparing nanograd.nn(autograd engine) to PyTorch

```python
from nanograd.nn.tensor import Tensor

# Create tensors with gradient tracking enabled
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

# Perform some tensor operations
c = a + b  # Element-wise addition
d = a * b  # Element-wise multiplication
e = c.sum()  # Sum all elements of the result of addition

# Compute gradients
e.backward()

# Print the results and gradients
print("Tensor a:")
print(a.numpy())
print("Tensor b:")
print(b.numpy())
print("Result of a + b:")
print(c.numpy())
print("Result of a * b:")
print(d.numpy())
print("Gradient of a:")
print(a.grad.numpy())
print("Gradient of b:")
print(b.grad.numpy())

```

The same but in pytorch

```python
import torch

# Create tensors with gradient tracking enabled
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

# Perform some tensor operations
c = a + b  # Element-wise addition
d = a * b  # Element-wise multiplication
e = c.sum()  # Sum all elements of the result of addition

# Compute gradients
e.backward()

# Print the results and gradients
print("Tensor a:")
print(a)
print("Tensor b:")
print(b)
print("Result of a + b:")
print(c)
print("Result of a * b:")
print(d)
print("Gradient of a:")
print(a.grad)
print("Gradient of b:")
print(b.grad)

```

---


**For testing the neural network training:**
```Bash
python train.py
```
**the output should look something like this:**
```Bash
Epoch 9935, Loss:0.001824420457797021&quot;
Epoch 9936, Loss:0.0018241422932506295&quot;
Epoch 9937, Loss:0.0018238642104392092&quot;
Epoch 9938, Loss:0.0018235862093274602&quot;
Epoch 9939, Loss:0.001823308289880096&quot;
Epoch 9940, Loss:0.0018230304520618333&quot;
Epoch 9941, Loss:0.0018227526958374313&quot;
Epoch 9942, Loss:0.001822475021171659&quot;
Epoch 9943, Loss:0.0018221974280292989&quot;
Epoch 9944, Loss:0.001821919916375175&quot;
Epoch 9945, Loss:0.0018216424861740924&quot;
Epoch 9946, Loss:0.0018213651373909334&quot;
Epoch 9947, Loss:0.0018210878699905417&quot;
Epoch 9948, Loss:0.0018208106839378159&quot;
Epoch 9949, Loss:0.001820533579197663&quot;
Epoch 9950, Loss:0.001820256555735023&quot;
Epoch 9951, Loss:0.001819979613514832&quot;
Epoch 9952, Loss:0.0018197027525020653&quot;
Epoch 9953, Loss:0.0018194259726617152&quot;
Epoch 9954, Loss:0.001819149273958778&quot;
Epoch 9955, Loss:0.001818872656358298&quot;
Epoch 9956, Loss:0.001818596119825315&quot;
Epoch 9957, Loss:0.0018183196643248925&quot;
Epoch 9958, Loss:0.001818043289822136&quot;
Epoch 9959, Loss:0.0018177669962821316&quot;
Epoch 9960, Loss:0.001817490783670024&quot;
Epoch 9961, Loss:0.0018172146519509534&quot;
Epoch 9962, Loss:0.001816938601090092&quot;
Epoch 9963, Loss:0.0018166626310526187&quot;
Epoch 9964, Loss:0.001816386741803748&quot;
Epoch 9965, Loss:0.0018161109333087037&quot;
Epoch 9966, Loss:0.0018158352055327178&quot;
Epoch 9967, Loss:0.0018155595584410794&quot;
Epoch 9968, Loss:0.0018152839919990619&quot;
Epoch 9969, Loss:0.0018150085061719684&quot;
Epoch 9970, Loss:0.001814733100925134&quot;
Epoch 9971, Loss:0.0018144577762238912&quot;
Epoch 9972, Loss:0.0018141825320336126&quot;
Epoch 9973, Loss:0.0018139073683196665&quot;
Epoch 9974, Loss:0.0018136322850474769&quot;
Epoch 9975, Loss:0.0018133572821824565&quot;
Epoch 9976, Loss:0.0018130823596900444&quot;
Epoch 9977, Loss:0.0018128075175357023&quot;
Epoch 9978, Loss:0.0018125327556849204&quot;
Epoch 9979, Loss:0.0018122580741031854&quot;
Epoch 9980, Loss:0.0018119834727560364&quot;
Epoch 9981, Loss:0.0018117089516089886&quot;
Epoch 9982, Loss:0.001811434510627626&quot;
Epoch 9983, Loss:0.0018111601497775086&quot;
Epoch 9984, Loss:0.001810885869024237&quot;
Epoch 9985, Loss:0.0018106116683334363&quot;
Epoch 9986, Loss:0.001810337547670745&quot;
Epoch 9987, Loss:0.0018100635070018084&quot;
Epoch 9988, Loss:0.0018097895462923128&quot;
Epoch 9989, Loss:0.0018095156655079459&quot;
Epoch 9990, Loss:0.0018092418646144216&quot;
Epoch 9991, Loss:0.00180896814357748&quot;
Epoch 9992, Loss:0.0018086945023628718&quot;
Epoch 9993, Loss:0.0018084209409363627&quot;
Epoch 9994, Loss:0.0018081474592637485&quot;
Epoch 9995, Loss:0.0018078740573108493&quot;
Epoch 9996, Loss:0.0018076007350434741&quot;
Epoch 9997, Loss:0.001807327492427506&quot;
Epoch 9998, Loss:0.0018070543294287761&quot;
Epoch 9999, Loss:0.0018067812460131913&quot;
Predictions after training:&quot;
[[0.03517901]
 [0.95797234]
 [0.95579776]
 [0.04762676]]
```
