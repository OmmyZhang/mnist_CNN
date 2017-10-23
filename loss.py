from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return 0.5 * np.mean(np.sum(np.square(input - target), axis=1))

    def backward(self, input, target):
        return (input - target) / len(input)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        s =  np.sum(np.exp(input),axis=1)
        input = np.exp(input) / (s.repeat(len(input[0])).reshape(input.shape))
        return -1 * np.mean(np.sum(np.log(input + 1e-7) * target, axis=1))

    def backward(self, input, target):
        '''Your codes here'''
        s =  np.sum(np.exp(input),axis=1)
        input = np.exp(input) / (s.repeat(len(input[0])).reshape(input.shape))
        return (input - target) / len(input) 
