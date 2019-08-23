#!/usr/bin/env python3

import numpy as np


class Neuron(object):

    """ A simple Neuron
    Args:
        num_inputs (int): input vector size / number of input values
        activation_fn (callable): activation function
    Attributes:
        W (ndarray): The weight values for each input
        b (float): the bias value added to the weighted sum
        activation_fn (callable): the activation function

    """
    def __init__(self, num_inputs, activation_fn):
        super().__init__()
        # Random initializing the weight vector and bias value
        self.W = np.random.rand(num_inputs)
        self.b = np.random.rand(1)
        self.activation_fn = activation_fn

    def forward(self, x):
        """Forward the input signal through the neuron"""
        z = np.dot(x, self.W) + self.b
        return self.activation_fn(z)


# Fixing the Seed
np.random.seed(42)
# Random input shape = '(1, 3)'
x = np.random.rand(3).reshape(1, 3)

