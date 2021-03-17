import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

# Sources
# PyTorch classifier: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# PyTorch kernel: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
# Example ROCKET git: https://github.com/angus924/rocket/blob/master/code/rocket_functions.py

IN_CHANNELS = 10
OUT_CHANNELS = 10
RANGE_SIZE = (2, 10)


class Net(nn.Module):
    def __init__(self, n_kernels):
        super(Net, self).__init__()
        self.n_kernels = n_kernels
        self.__generate_random_kernels(self.n_kernels)

    def __generate_random_kernels(self, n_kernels: int, groups=1, stride=1, ):
        # Initialize variables
        self.kernels = []

        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        for kernel_index in range(n_kernels):
            # Size: int
            # Stride: int
            # Padding: int
            # Dilation: int
            # Bias: bool
            self.kernels.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5))

    def forward(self, signal):
        conv_output = []
        for kernel in self.kernels:
            reshaped_signal = signal.view(1, 1, signal.shape[-1])
            conv_output.append(kernel(reshaped_signal))
        return conv_output


if __name__ == "__main__":
    # Parameters
    number_of_kernels = 3

    # Create surrogate data
    signal = torch.from_numpy(np.random.randn(10))
    # Define network
    net = Net(n_kernels=number_of_kernels)
    results = net.forward(signal=signal)
    h = 1
