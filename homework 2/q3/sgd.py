"""
used for parts (a) through (c) with minor edits
"""

from scipy.io import loadmat
import math
import numpy as np

def model_function(w, x):
    return w[0] + (x - w[1])**2 + math.exp(-w[2] * x)

def cost_function(w, x, t):
    return 0.02 * sum([0.5 * (model_function(w, x[i]) - t[i])**2 for i in range(50)])


import math
import numpy as np

def sgd(x, t, step_size=0.1, epochs=100):
    x = np.asarray(x, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)

    weights = np.zeros(3, dtype=float)
    rng = np.random.default_rng(0)

    for epoch in range(epochs):
        if epoch == 0:
            print(f"Cost function before step 1: {cost_function(weights, x, t):.5f}")

        for index in rng.permutation(x.size):
            sample_x = x[index]
            target = t[index]
            
            error = model_function(weights, sample_x) - target
            
            gradient = np.array([
                error, 
                2 * error * (weights[1] - sample_x),
                -error * sample_x * math.exp(-weights[2] * sample_x)])

            gradient /= x.size
            weights -= step_size * gradient

        if epoch == 0:
            print(f"Cost function after step 1: {cost_function(weights, x, t):.5f}")
            print(f"Weights after step 1: {weights}")

    return weights

if __name__ == "__main__":
    hw2data = loadmat("intro-to-machine-learning/homework 2/hw2data.mat")
    x = hw2data["x"].reshape(-1)
    t = hw2data["t"].reshape(-1)
    weights = sgd(x, t)
    print(f"Learned weights: {weights}")
    print(f"Final cost: {cost_function(weights, x, t):.5f}")