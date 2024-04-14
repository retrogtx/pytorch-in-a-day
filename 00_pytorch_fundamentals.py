import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Introduction to Tensors:
Creating Tensors

They're created using torch.tensor() => https://pytorch.org/docs/stable/tensors.html
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
