import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import scipy
import cvxpy as cp
from tqdm import tqdm



def solve_compressed_sensing_l1(X, y_star, EPSILON=1e-8):
    """
    Solve the l1-minimization problem to recover a :
    Minimize ||a||_1 subject to ||Xa - y*||_2 <= epsilon
    """
    a = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.norm(a, 1))
    #constraints = [X @ a = y]
    constraints = [cp.norm(X @ a - y_star, 2) <= EPSILON]  # tolerance for numerical precision
    problem = cp.Problem(objective, constraints)
    problem.solve()
    # Recovered sparse representation a
    return a.value