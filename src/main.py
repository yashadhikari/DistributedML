# -*- coding: utf-8 -*-
"""
This file runs the simulation
@author: ayash
"""

import numpy as np

from connected_graph import connected_graph
from sklearn.datasets.samples_generator import make_blobs

if __name__ == "__main__":
    
    cost_comm = []
    cost_processing = []
    epochs_to_convergence = []
    final_epoch_loss = []
    
    simulation_count = 100
    
    for i in range(simulation_count):
    
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
            
            if graph.iteration == 40000:
                break
        
        cost_comm.append(graph.time_cost_comm)
        cost_processing.append(graph.time_cost_processing)
        epochs_to_convergence.append(len(graph.epochloss))
        final_epoch_loss.append(graph.epochloss[-1])
        
        if i % 10 == 0:
            print('Repition: ', i)