# -*- coding: utf-8 -*-
"""
Training MNIST dataset using Logistic Regression Model with Stochastic Gradient Descent

CS398 Deep Learning, 2019, Spring Assignment 1

Author: Lin, Meng Zheng
Email: mzlin3@illinois.edu
"""

import numpy as np
import h5py
import time
from random import randint

d = 784
k = 10

learning_rate = 3 * 10 ** -3
learning_iteration = 6 * 10 ** 5

def softmax(z):
	# Fsoftmax function
    return np.exp(z) / np.sum(np.exp(z))

def e(y):
	# One hot indicator function
    return (np.where(np.arange(k)==y, 1, 0))[:,np.newaxis]

def forward(x, y, θ):
    z = np.dot(θ, x)
    f = softmax(z)
    return f

def gradient(x, y, f):
	# -log(J(θ))
    return -((e(y) - f[:,np.newaxis]).dot(x[:,np.newaxis].transpose()))

if __name__ == "__main__":
    # Data Processing
    with h5py.File('MNISTdata.hdf5', 'r') as MNIST_data:
        x_train = np.float32(MNIST_data['x_train'][:])
        y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
        x_test = np.float32(MNIST_data['x_test'][:])
        y_test = np.int32(np.array( MNIST_data['y_test'][:,0]))
        
    # Initialize
    θ = np.zeros((10, 784))
    
    time_start = time.time()        # timer start

    for i in range(learning_iteration):
        sample_index = randint(0, len(x_train) - 1)
        x = x_train[sample_index]
        y = y_train[sample_index]
        forward_result = forward(x, y, θ)
        gradient_result = gradient(x, y, forward_result)
        
        θ -= learning_rate * gradient_result
        
    hit = 0
    for i in range(0, len(x_test)):
        x = x_test[i]
        y = y_test[i]
        forward_result = forward(x, y, θ)
        
        # Checkout the largest probability
        if np.argmax(forward_result) == y:
            hit += 1
            
    time_end = time.time()      # timer end
    
    print("Accuracy: {accuracy}".format(accuracy = hit / len(x_test)))
    print("Time spent: {0:.2f}sec".format(time_end - time_start))
