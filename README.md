# DistributedML
Goal: Explore different schemes of distributed machine learning through a simulation

Steps Taken:
1. Get an approximate probability distribution on a sample computer (worker) on cost in time to calculate gradient for 1 training example
2. Set up distributed ML schemes where each worker draws from the probability distribution to get time cost
3. Calculate the approriate number of simulation runs to attain +/- 5% accuracy with 95% probability
4. Compare different scehemes for performance such as averag time to convergence and average final epoch loss
