import torch.nn as nn
import numpy as np

# Sources
# PyTorch classifier: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# PyTorch kernel: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
# Example ROCKET git: https://github.com/angus924/rocket/blob/master/code/rocket_functions.py

IN_CHANNELS = 10
OUT_CHANNELS = 10
RANGE_SIZE = (2, 10)


def generate_random_kernels(n_kernels: int, groups=1, stride=1, ):
    # Initialize variables
    kernels = []

    # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    for kernel_index in range(n_kernels):
        # Size: int
        # Stride: int
        # Padding: int
        # Dilation: int
        # Bias: bool
        kernels.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5))

    return kernels


if __name__ == "__main__":
    # Parameters
    number_of_kernels = 3
    random_kernels = generate_random_kernels(n_kernels=number_of_kernels)
    h = 1
