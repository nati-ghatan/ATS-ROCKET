import math

import numpy as np
import pandas as pd
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
    # Convolutional parameters
    debug_data = False
    number_of_kernels = 1000
    permitted_kernel_sizes = [7, 9, 11]  # Taken from the ROCKET article

    # Classification parameters
    alpha = 1.0  # Regularization strength
    tolerance = 1e-3  # Precision of the solution

    # Read UCR dataset or create surrogate data
    if debug_data:
        signal_length = 1000
        data = torch.from_numpy(np.random.randn(signal_length).astype(np.float32))
        class_labels = None
    else:
        # Read data and convert it to a PyTorch tensor
        data = pd.read_csv('data/ElectricDevices_TRAIN.tsv', header=None, sep='\t')
        data = torch.tensor(data.values.astype(np.float32))

        # Separate between class labels (first column) and actual data (rest of the columns)
        class_labels = data[:, 0]
        data = data[:, 1:]

        # Acquire data signal length
        signal_length = data.shape[1]

    # Define network
    net = RocketNet(data=data,
                    class_labels=class_labels,
                    n_kernels=number_of_kernels,
                    kernel_sizes=permitted_kernel_sizes)
    results = net.forward(signal=data)

    # Validate output sizes
    # Source: https://arxiv.org/pdf/1603.07285.pdf , Page 28
    for kernel_index in range(net.n_kernels):
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
    print("Result sizes validated successfully!")

    # Transform data through convolution kernels
    features = net.train(alpha=alpha, tolerance=tolerance)

    # Examine results
    sample_index = 0
    sample_input = features[sample_index, :]
    sample_input = sample_input.reshape(1, sample_input.shape[0])
    expected_output = class_labels[sample_index]
    observed_output = net.classifier.predict(sample_input)
    print(f"Expected: {expected_output}")
    print(f"Observed: {observed_output}")


if __name__ == "__main__":
    main_debug()
