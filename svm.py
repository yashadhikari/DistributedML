# -*- coding: utf-8 -*-
"""
SVM class to calculate GD
@author: ayash
"""

import numpy as np

class svm:
    
    def __init__(self):
        pass
    
    def sigmoid_activation(self, x):
        # compute and return the sigmoid activation value for a
        # given input value
        return 1.0 / (1 + np.exp(-x))
    
    def calculate_grad(self, batch_x, batch_y, W):
        
        preds = self.sigmoid_activation(batch_x.dot(W))
        error = preds - batch_y
        
        loss = np.sum(error ** 2)
        
        gradient = batch_x.T.dot(error) / batch_x.shape[0]
        
        return loss, gradient