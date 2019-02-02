# -*- coding: utf-8 -*-
"""
Training MNIST dataset using Single Hidden Layer Fully-Connected Neural Network with Stochastic Gradient Descent

CS398 Deep Learning

Author: Lin, Meng Zheng
Email: mzlin3@illinois.edu
"""

import numpy as np
import h5py
import time
from random import randint

d = 784             # number of input dimension
k = 10              # number of output dimension
dH = 64             # number of hidden units

learning_rate = 3 * 10 ** -3
learning_iteration = 6 * 10 ** 5

def ReLU(z):
    return np.maximum(z, 0)

def dReLU(z):
    return np.where(z >= 0, 1, 0)

def softmax(U):
    return np.exp(U) / np.sum(np.exp(U))

def e(y):
    return (np.where(np.arange(k)==y, 1, 0))[:,np.newaxis]

def forward(x, y, W, b1, C, b2):
    Z = np.dot(W, x) + b1
    H = ReLU(Z)
    U = C.dot(H) + b2
    f = softmax(U)
    return [Z, H, U, f]    

def backpropagrate(Z, H, U, f):
    dp_du = -(e(y) - f)
    dp_db2 = dp_du
    dp_dC = dp_du.dot(H.transpose())
    S = C.transpose().dot(dp_du)
    dp_db1 = np.multiply(S, dReLU(Z))
    dp_dW = np.multiply(S, dReLU(Z)).dot(x.transpose())
    return [dp_dW, dp_db1, dp_dC, dp_db2]

'''
def loss(f, y):
    return L(θ) = E[ρ(f(X; θ), Y)]
'''

if __name__ == "__main__":    
    # Data Processing
    with h5py.File('MNISTdata.hdf5', 'r') as MNIST_data:
        x_train = np.float32((MNIST_data['x_train'][:]))
        y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
        x_test = np.float32(MNIST_data['x_test'][:])
        y_test = np.int32(np.array( MNIST_data['y_test'][:,0]))
    
    # Initialize
    W = np.random.randn(dH, d) / np.sqrt(d)             # weight matrix: W ∈ dH * d
    b1 = np.random.randn(dH)[:,np.newaxis]              # bias 1: b1 ∈ dH
    C = np.random.randn(k, dH) / np.sqrt(k)             # weight matrix: W ∈ dH * d
    b2 = np.random.randn(k)[:,np.newaxis]               # bias 2: b2 ∈ k

    time_start = time.time()
    
    for iteration in range(learning_iteration):
        # Stocastic Gradient Descent - for each iteration, choose one data sample
        sample_index = randint(0, len(x_train) - 1)
        x = x_train[sample_index][:,np.newaxis]
        y = y_train[sample_index]
        
        forward_result = forward(x, y, W, b1, C, b2)
        backpropagrate_result = backpropagrate(*forward_result[0:4])
        
        W -= learning_rate * backpropagrate_result[0]
        b1 -= learning_rate * backpropagrate_result[1]
        C -= learning_rate * backpropagrate_result[2]
        b2 -= learning_rate * backpropagrate_result[3]
        
    hit = 0
    for i in range(0, len(x_test)):
        x = x_test[i][:,np.newaxis]
        y = y_test[i]
        forward_result = forward(x, y, W, b1, C, b2)
        
        # Checkout the largest probability
        if np.argmax(forward_result[3]) == y:
            hit += 1
        
    time_end = time.time()
    
    print("Accuracy: {accuracy}".format(accuracy = hit / len(x_test)))
    print("Time spent: {0:.2f}sec".format(time_end - time_start))