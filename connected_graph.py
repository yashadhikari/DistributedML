# -*- coding: utf-8 -*-
"""
@author: ayash
"""
import numpy as np

class connected_graph:
    
    
    
    def __init__(self, data, parameters, nodes = 10, connections = 2):
        self.nodes = nodes
        self.outdegree = connections
        self.X = data[0]
        self.y = data[1]
        
        self.loss = [0]
        self.time_cost_processing = 0
        self.time_cost_comm = 0
        
        self.index_data = np.zeros(self.nodes)
        self.updates = np.repeat(parameters[np.newaxis,:], self.nodes, axis = 0)
        self.params = np.repeat(parameters[np.newaxis,:], self.nodes, axis = 0)
        
        
        # At instantiation, divide the data for each worker
        self.divide_data()
        
    def next_iteration(self):
        # calculate SGD at each worker, add time to time_cost, pass the updated
        # parameters to the appropriate worke(s)
        
        self.update_params()
        self.calculate_SGD()
        
        pass
    
    def update_params(self):
        self.params += self.updates
    
    def calculate_SGD(self):
        pass
    
    def divide_data(self):
        # We want each worker to work with a set of data. Data is split into
        # data[worker #][data index]
        
        data_size = len(self.X)
        step_size = int(data_size/self.nodes)
        
        self.X = np.vsplit(self.X, np.arange(step_size, data_size, step_size))
        self.y = np.split(self.y, np.arange(step_size, data_size, step_size))
        
        # if data is unevenly split, add the remaining data to the last worker
        if len(self.X) > self.nodes:
            self.X[-2] = np.concatenate((self.X[-2], self.X[-1]))
            self.X.pop()
            self.y[-2] = np.concatenate((self.y[-2], self.y[-1]))
            self.y.pop()
        elif len(self.X) == self.nodes:
            print('Evenly distributed data')
        else:
            print('eror in distributing data to each worker')
    
    def convergence(self):
        # return true if convergence has been achieved
        criterion = 0.001
        
        if len(self.loss) > 1 and abs(self.loss[-1]-self.loss[-2]) < criterion:
            return True
        else:
            return False