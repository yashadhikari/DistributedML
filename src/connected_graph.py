# -*- coding: utf-8 -*-
"""
@author: ayash
"""
import numpy as np

from svm import svm

class connected_graph:
    
    
    
    def __init__(self, data, parameters, nodes = 10, connections = 1):
        
        # define the model we are working with
        self.model = svm()
        self.alpha = 0.01
        
        # define the scheme and data
        self.nodes = nodes
        self.outdegree = connections
        self.X = data[0]
        self.y = data[1]
        self.examples = len(self.X)
        self.data_index = [0 for i in range(self.nodes)]
        
        # define performance metrics
        self.loss = [0]
        self.epochloss = []
        self.time_cost_processing = 0
        self.time_cost_comm = 0
        self.iteration = 0
        
        # define gradient and parameter information
        temp = np.zeros_like(parameters)
        self.updates = np.repeat(temp[np.newaxis,:], self.nodes, axis = 0)
        self.grads = np.repeat(temp[np.newaxis,:], self.nodes, axis = 0)
        self.params = np.repeat(parameters[np.newaxis,:], self.nodes, axis = 0)
        
        # At instantiation, divide the data for each worker and decide which
        # nodes(worker) are sent updates from each worker
        self.divide_data()
        self.indices = self.get_worker_indices()
        
        # Get  distribution of data (does not contain probabilties)
        self.get_distribution()
    
    def get_distribution(self):
        # Load data which contain time cost as range [col 1] and # of 
        # occurences [col 2]
        directory = '../data/TimeCostHistogram.txt'
        data = np.loadtxt(directory)
        
        # Get data in format of avg. time cost [col 1] and # occurences [col 2]
        start = 2                          # skip first 2 rows (no occurences)
        bins = 10
        occurences = [i[1] for i in data]
        self.separation = np.cumsum(occurences[start:start+bins])
        self.avg_time_cost = [np.average([data[i-1][0], data[i][0]]) for i in range(start, start+bins)]
        
    def get_time_cost(self):
        # Get the cost of computation based on the loaded distribution
        
        # Replace a number of if then statements with finding the proper bin
        x = np.random.randint(0, self.separation[-1])
        find_bin = np.where(self.separation <= x)[0]
        
        if len(find_bin) > 0:
            time_cost_idx = max(find_bin)
            return self.avg_time_cost[time_cost_idx]
        else:
            return self.avg_time_cost[0]
    
    def next_iteration(self):
        # Update parameters first, calculate SGD at each worker while adding 
        # time to time_cost_xxx, pass the updates to the appropriate worker(s)
        
        self.update_params()
        self.calculate_SGD()
        self.send_updates()
        
        self.iteration += 1
        
        
        if self.iteration % len(self.X[0]) == 0:
            self.epochloss.append(np.average(self.loss[-len(self.X[0]):]))
            # print epoch loss if necessary
            # print('Epoch: ', self.iteration/len(self.X[0]), ' Epoch Loss: ', self.epochloss[-1])
    
    def update_params(self):
        self.params += self.updates
    
    def calculate_SGD(self):
        # calculates gradients and loss at each worker
        
        # must kep track of time cost for each worker
        worker_computation_cost = np.zeros(self.nodes)
        
        for i in range(self.nodes):
            l, grad = self.model.calculate_grad(self.X[i][self.data_index[i]:self.data_index[i]+1], self.y[i][self.data_index[i]:self.data_index[i]+1], self.params[i])
            self.loss.append(l)
            self.grads[i] = grad
            self.get_next_data_index(i)
            
            worker_computation_cost[i] = self.get_time_cost()
        
        # Since we are in parallel we default to computation that takes the 
        # maximum amount of time
        self.time_cost_processing += max(worker_computation_cost)
        
    def send_updates(self):
        # At each worker get index for which other workers to send updates to
        # Then update parameters at the selected workers
        
        # Add to communciation cost while sending updates
        cost_per_update = 9e-7
        
        self.updates = np.zeros_like(self.updates)
        for i in range(self.nodes):
            self.updates[self.indices[i]] += -self.alpha * self.grads[i]
            self.time_cost_comm += cost_per_update * self.outdegree
    
    def get_worker_indices(self):
        # We say that updates are passed to the next n connections where n is 
        # decided by self.connections. We simply take the next n workers 
        # starting from index i+1 where i is the index of worker which is 
        # sending updates
        
        workers = []
        for i in range(self.nodes):
            # handle special circumsstances
            if self.nodes > 1:
                if i + self.outdegree < self.nodes:
                    workers.append(list(range(i + 1, i + 1 + self.outdegree)))
                else:
                    # temp_idx creates empty list if you are on the last worker
                    temp_idx = list(range(i + 1, self.nodes))
                    temp_covered = len(temp_idx)
                    temp_idx.extend(list(range(0, self.outdegree-temp_covered)))
                    workers.append(temp_idx)
            else:
                workers.append(0)
            
        return workers
    
    def get_next_data_index(self, current_node):
        # Handle unevenly split data as well as evenly split data to find the 
        # next data set for each worker
        
        sub_data_size = len(self.X[current_node])
        
        if self.data_index[current_node] < sub_data_size - 1:
            self.data_index[current_node] += 1
        else:
            self.data_index[current_node] = 0
    
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
            
            # For debugging if necessary
            # print('Evenly distributed data')
            pass
        else:
            print('eror in distributing data to each worker')
    
    def convergence(self):
        # return true if convergence has been achieved
        criterion = 0.00001
        
        if len(self.epochloss) > 1 and abs(self.epochloss[-1]-self.epochloss[-2]) < criterion:
            return True
        else:
            return False