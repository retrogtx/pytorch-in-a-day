import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Introduction to Tensors:
Creating Tensors

They're created using torch.tensor() -> https://pytorch.org/docs/stable/tensors.html
"""

scalar = torch.tensor(10)  # scalar is just a magnitude with no direction

print(scalar)
print(scalar.ndim)  # 0 as a scalar as no dimensions
print(scalar.item())  # 10

vector = torch.tensor([10, 20])  # Vector is an object with both magnitude and direction

print(vector)
print(vector.ndim)  # 1 as it has 1 dimension
print(vector.shape)


matrix = torch.tensor(
    [[10, 20], [30, 40]]
)  # Matrix is a 2D array of numbers, comprised of vectors in our case
print(matrix.ndim)
print(matrix.shape)
print(matrix[0])

Tensor = torch.tensor(
    [[[1, 2, 3], [4, 5, 6], [8, 4, 5]]]
)  # Tensor is a multi-dimensional array of numbers
print(Tensor.ndim)
print(Tensor.shape)

"""
Random Tensors:
Why random tensors?

Random tensors are important because the way many neural networks learn is that they start with tensors full of random numbers and then
they adjust those numbers until they represent the patterns in the data that they're trying to learn.

start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers
"""

random_tensor = torch.rand(3, 3)
print(random_tensor)

# Random tensor with similar shape to an image vector
image = torch.rand(size=(224, 224, 3))  # height, width, colour channels
print(image.shape, image.ndim)

# Zero and Ones Tensors
zero_tensor = torch.zeros(3, 3)
print(zero_tensor * random_tensor)

ones_tensor = torch.ones(3, 3)
print(ones_tensor * random_tensor)

"""
Creating a range of Tensors using torch.arange() -> https://pytorch.org/docs/stable/generated/torch.arange.html

Also create zeros_like and ones_like tensors using torch.zeros_like() and torch.ones_like()

Range() unlike regular python code does not work here since it is deprecated in PyTorch
"""
one_to_ten = torch.arange(1, 11)
ten_zeros = torch.zeros_like(one_to_ten)
print(one_to_ten, ten_zeros)

"""
Tensor datatypes is one of the 3 big errors you'll run into with PyTorch & Deep Learning:
1. Tensor of not the right shape
2. Tensor of not the right datatype
3. Tensor on not the right device (CPU or GPU)

Precision in computing -> wikipedia.org/wiki/Precision_(computer_science)
"""

# Changing the datatype of a tensor
float_tensor = torch.tensor([1, 2, 3], dtype=None, device=None, requires_grad=False)
print(float_tensor, float_tensor.dtype)

float_16_tensor = float_tensor.type(torch.half)
print(float_16_tensor, float_16_tensor.dtype)
print(float_16_tensor * float_tensor)

"""
Getting information from Tensors (Tensor Attributes):
1. Tensors not right datatype - to get datatype from a tensor, can use tensor.dtype
2. Tensors not right shape - to get shape of a tensor, can use tensor.shape
3. Tensors not right device - to get device of a tensor, can use tensor.device
"""

some_tensor = torch.rand(3, 4)
print(
    f"Datatype of tensor: {some_tensor.dtype}, Shape of tensor: {some_tensor.shape}, Device of tensor: {some_tensor.device}"
)


"""
Tensor operations
In deep learning, data (images, text, video, audio, protein structures, etc) gets represented as tensors.

A model learns by investigating those tensors and performing a series of operations (could be 1,000,000s+) on tensors to create a representation of the patterns in the input data.

These operations are often a wonderful dance between:

Addition
Substraction
Multiplication (element-wise)
Division
Matrix multiplication
And that's it. Sure there are a few more here and there but these are the basic building blocks of neural networks.

Stacking these building blocks in the right way, you can create the most sophisticated of neural networks (just like lego!).
"""

"""
Finding the min, max, mean, sum (tensor aggregation)

torch.float32 requires the float32 datatype to be used for the tensor

use var.argmin() or var.argmax() to find the index of the min or max value in a tensor
"""

x = torch.arange(0, 100, 10)
print(torch.min(x), torch.max(x), torch.sum(x))

print(torch.mean(x.type(torch.float32)))
print(x.argmin(), x.argmax())

"""
Reshaping, Stacking, Squeezing, Unsqueezing Tensors

Reshaping - Reshapes a tensor into a new shape
View - Return a new tensor with the same data as the original tensor but with a different shape, same memory
Stacking - Combines multiple tensors into a single tensor, on top of each other (v stack) or next to each other (h stack)
Squeezing - Removes all single dimensions from a tensor
Unsqueezing - Adds a dimension with a size of 1 to a tensor's shape
Permute - Rearranges the dimensions of a tensor
"""

x = torch.arange(1.0, 10.0)
