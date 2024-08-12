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
