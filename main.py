# -*- coding: utf-8 -*-
"""
This file runs the simulation
@author: ayash
"""

import numpy as np

from connected_graph import connected_graph
from sklearn.datasets.samples_generator import make_blobs

if __name__ == "__main__":
    
    # create data
    (X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)
    # insert a column of ones in the feature vector to be treated as the bias 
    # term - not worry about it separately
    X = np.c_[np.ones((X.shape[0])), X]
    
    # initialize the weight matrix otherwise called parameters
    W = np.random.uniform(size=(X.shape[1],))
    
    # send the data and parameters to a connected graph
    graph = connected_graph((X, y), W)
    
    # continue to train until convergence criteria is met
    while not graph.convergence():
        graph.next_iteration()
        break
    