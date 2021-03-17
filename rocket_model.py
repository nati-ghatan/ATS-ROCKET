import math

import numpy as np
import torch
import torch.nn as nn


class RocketNet(nn.Module):
    def __init__(self, data, n_kernels, kernel_sizes):
        super(RocketNet, self).__init__()
        self.data = data
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes
        self.__generate_random_kernels(self.n_kernels)

    def __generate_random_kernels(self, n_kernels: int, groups=1, stride=1):
        # Initialize variables
        self.kernels = []
        signal_length = self.data.shape[-1]

        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        for kernel_index in range(n_kernels):
            # Stride: int = 1
            # Size: int
            current_kernel_size = np.random.choice(self.kernel_sizes)

            # Dilation: int
            maximum_allowed_dilation = math.log2((signal_length - 1) / (current_kernel_size - 1))
            current_dilation = np.random.randint(low=0, high=maximum_allowed_dilation)
            dilation_factor = math.floor(math.pow(2, current_dilation))

            # Padding: int
            current_padding_size = 0 if np.random.rand() <= 0.5 else \
                int(((current_kernel_size - 1) * dilation_factor) / 2)

            # Bias: bool
            # TODO: To be implemented

            # Create kernel with selected randomized parameters
            current_kernel = nn.Conv1d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=current_kernel_size,
                                       stride=stride,
                                       groups=groups,
                                       dilation=dilation_factor,
                                       padding=current_padding_size)

            # Initialize kernel weights
            # TODO: Show this to Dana and Adi and ask their opinion
            torch.nn.init.normal_(current_kernel.weight, mean=0.0, std=1.0)  # Draw from a Normal distribution
            current_kernel.weight.data = current_kernel.weight.data - current_kernel.weight.data.mean()  # Mean center

            # Accumulate randomized kernel
            self.kernels.append(current_kernel)

    def forward(self, signal):
        conv_output = []
        for kernel in self.kernels:
            reshaped_signal = signal.view(1, 1, signal.shape[-1])
            conv_output.append(kernel(reshaped_signal))
        return conv_output