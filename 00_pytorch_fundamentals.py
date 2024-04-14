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

Tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [8, 4, 5]]])
print(Tensor.ndim)
print(Tensor.shape)
