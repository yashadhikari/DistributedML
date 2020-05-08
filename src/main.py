# -*- coding: utf-8 -*-
"""
This file runs the simulation
@author: ayash
"""

import numpy as np

from connected_graph import connected_graph
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import norm

def get_num_trial(confidence, epsilon, avg, var):
    # Return an etimate for actual number of runs needed
    z = norm.ppf(0.5 + confidence/200)
    return (var * z**2)/((epsilon * avg)**2)
    

if __name__ == "__main__":
    
    # Set measurrint_time to False when trying to analyze the epoch loss 
    # performance criteria
    measuring_time = True
    cost_comm = []
    cost_processing = []
    epochs_to_convergence = []
    final_epoch_loss = []
    
    simulation_count = 1000
    
    select_nodes = [1, 3, 5, 10, 10, 10, 10]
    select_connection = [0, 2, 2, 2, 1, 5, 9]
    
    # create data
    (X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)
    X = np.c_[np.ones((X.shape[0])), X]
    
    for j in range(len(select_nodes)):
        
        cost_comm = []
        cost_processing = []
        epochs_to_convergence = []
        final_epoch_loss = []
        
        
        for i in range(simulation_count):
            
            if measuring_time:
                (X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)
                # insert a column of ones in the feature vector to be treated as the bias 
                # term - not worry about it separately
                X = np.c_[np.ones((X.shape[0])), X]
            
            # initialize the weight matrix otherwise called parameters
            W = np.random.uniform(size=(X.shape[1],))
            
            # send the data and parameters to a connected graph
            graph = connected_graph((X, y), W, nodes = select_nodes[j], connections = select_connection[j])
            
            
            # continue to train until convergence criteria is met
            while not graph.convergence():
                graph.next_iteration()
                
                if graph.iteration == 40000:
                    break
            
            cost_comm.append(graph.time_cost_comm)
            cost_processing.append(graph.time_cost_processing)
            epochs_to_convergence.append(len(graph.epochloss))
            final_epoch_loss.append(graph.epochloss[-1])
            
            if i % 500 == 0:
                print('Repition: ', i)
        
        
        avg_comm_cost = np.average(cost_comm)
        avg_processing_cost = np.average(cost_processing)
        
        total_time_cost = np.sum([cost_comm, cost_processing], axis = 0)
        avg_cost = np.average(total_time_cost)
        var_cost = np.var(total_time_cost)
        epsilon = 0.05
        avg_final_epoch_loss = np.average(final_epoch_loss)
        
        
        n = get_num_trial(95, epsilon, avg_cost, var_cost)
        print('Nodes: ', select_nodes[j], ', Out degree: ', select_connection[j], ', Avg cost: ', avg_cost, ', Avg. final epoch loss: ', avg_final_epoch_loss, ', Avg. comm cost: ', avg_comm_cost, ', Avg. processing cost: ', avg_processing_cost)
        print('Expected number of trials for +/- ', epsilon*100, '% accuracy is ', n)
    
    
    
    
    
    
    
    
    
    