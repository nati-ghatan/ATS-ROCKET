import math

import numpy as np
import torch

from rocket_model import RocketNet


# Sources
# PyTorch classifier: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# PyTorch kernel: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
# Example ROCKET git: https://github.com/angus924/rocket/blob/master/code/rocket_functions.py

def __compute_expected_output_size(signal_length, kernel_size, padding, dilation, stride):
    # Source: https://arxiv.org/pdf/1603.07285.pdf , Page 28
    nominator = (signal_length + (2 * padding) - kernel_size - ((kernel_size - 1) * (dilation - 1)))
    return math.floor(nominator / stride) + 1


# Debug functions
def main_debug():
    # Parameters
    signal_length = 1000
    number_of_kernels = 10
    permitted_kernel_sizes = [7, 9, 11]  # Taken from the ROCKET article

    # Create surrogate data
    surrogate_signal = torch.from_numpy(np.random.randn(signal_length).astype(np.float32))

    # Define network
    net = RocketNet(data=surrogate_signal,
                    n_kernels=number_of_kernels,
                    kernel_sizes=permitted_kernel_sizes)
    results = net.forward(signal=surrogate_signal)

    # Validate output sizes
    # Source: https://arxiv.org/pdf/1603.07285.pdf , Page 28
    for kernel_index in range(number_of_kernels):
        current_kernel = net.kernels[kernel_index]
        kernel_size = current_kernel.kernel_size[0]
        observed_output_size = results[kernel_index].shape[-1]
        padding_size = current_kernel.padding[0]
        stride_size = current_kernel.stride[0]
        dilation_size = current_kernel.dilation[0]
        expected_output_size = __compute_expected_output_size(signal_length=signal_length,
                                                              kernel_size=kernel_size,
                                                              padding=padding_size,
                                                              dilation=dilation_size,
                                                              stride=stride_size)
        assert expected_output_size == observed_output_size

    print("Results tested successfully!")


if __name__ == "__main__":
    main_debug()
