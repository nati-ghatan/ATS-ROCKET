import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from tqdm import tqdm


class RocketNet(nn.Module):
    def __init__(self, data, class_labels, n_kernels, kernel_sizes):
        super(RocketNet, self).__init__()
        # TODO: Show Nati and Dana
        self.raw_data = data
        self.data = torch.max(data, torch.zeros_like(data))  # See section 3.2 in paper
        self.class_labels = class_labels
        self.n_samples = self.data.shape[0]
        self.signal_length = self.data.shape[1]
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes
        self.__generate_random_kernels(self.n_kernels)
        self.classifier = None

    def __generate_random_kernels(self, n_kernels: int, groups=1, stride=1):
        # Initialize variables
        self.kernels = []
        self.bias_terms = []

        for kernel_index in range(n_kernels):
            # Stride: int = 1
            # Size: int
            current_kernel_size = np.random.choice(self.kernel_sizes)

            # Dilation: int
            maximum_allowed_dilation = math.log2((self.signal_length - 1) / (current_kernel_size - 1))
            current_dilation = np.random.randint(low=0, high=maximum_allowed_dilation)
            dilation_factor = math.floor(math.pow(2, current_dilation))

            # Padding: int
            current_padding_size = 0 if np.random.rand() <= 0.5 else \
                int(((current_kernel_size - 1) * dilation_factor) / 2)

            # Bias: bool
            # TODO: Show Nati and Dana
            current_bias = np.random.uniform(low=-1., high=1.)

            # Create kernel with selected randomized parameters
            current_kernel = nn.Conv1d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=current_kernel_size,
                                       stride=stride,
                                       groups=groups,
                                       dilation=dilation_factor,
                                       padding=current_padding_size,
                                       bias=False)

            # Initialize kernel weights
            # TODO: Show this to Dana and Adi and ask their opinion
            torch.nn.init.normal_(current_kernel.weight, mean=0.0, std=1.0)  # Draw from a Normal distribution
            current_kernel.weight.data = current_kernel.weight.data - current_kernel.weight.data.mean()  # Mean center

            # Accumulate randomized kernel
            self.kernels.append(current_kernel)
            self.bias_terms.append(current_bias)

    def compute_features_for_data(self, data):
        sample_features = []
        print('Computing features for data')
        n_samples, signal_length = data.shape
        for sample_index in tqdm(range(n_samples)):
            # Reshape signal to meet model requirements, and feed it through the network
            current_signal = data[sample_index, :].view(1, signal_length)
            signal_features = self.forward(signal=current_signal)

            max_value = [x.max().item() for x in signal_features]
            ppv = [x[x>0].shape[-1] / x.shape[-1] for x in signal_features]
            sample_features.append(max_value + ppv)

        return sample_features

    def train(self, alpha=1.0, tolerance=1e-3):
        # Transform data through convolution kernels
        features = self.compute_features_for_data(self.data)

        # Perform learning (Ridge regression)
        self.classifier = RidgeClassifier(alpha=alpha, tol=tolerance, normalize=False)
        self.classifier.fit(X=features, y=self.class_labels)
        print('Training score', self.classifier.score(X=features, y=self.class_labels))

        return features

    def predict(self, data, labels=None):
        features = self.compute_features_for_data(data)
        # predicted_labels = self.classifier.predict(X=features)
        print('Test score', self.classifier.score(X=features, y=labels))
        # return float(sum(predicted_labels == labels)) / len(labels)

    def forward(self, signal):
        conv_output = []
        n_samples, signal_length = signal.shape
        for kernel, bias in zip(self.kernels, self.bias_terms):
            reshaped_signal = signal.view(n_samples, 1, signal_length)
            conv_output.append(kernel(reshaped_signal) + bias)
        return conv_output
