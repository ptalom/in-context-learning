import torch
import numpy as np
from numpy.linalg import svd
import cvxpy as cp
from tqdm import tqdm

import matplotlib.pyplot as plt

# # Set the working directory to the parent directory of your top-level package
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sparse_recovery.compressed_sensing import create_signal, create_orthonormal_basis, get_measures
from utils.svd import np_SVD
from utils.products import face_splitting_product_numpy
from utils.data_tensor_completion import get_matrices, data_tensor_completion_general





def solve_matrix_factorization_nuclear_norm(n_1, n_2, y_star, X1X2=None, X2_bullet_X1=None, X1X2_bar=None, X2_bullet_X1_bar=None, reg=0, EPSILON=1e-6):
    """
    Solve the minimization problem (P5) to recover A*
    # Minimize ||A||_* subject to ||(X2 • X1)A - y*||_2 = sum_i (X1_i^T A X2_i - y*_i)^2 <= epsilon (epsilon = 0 in noiseless case
    
    For matrix completion, we want to solve
        Minimize ||A||_* subject to ||(X2 • X1)A - y*||_2 <= epsilon, 
        ie A_{ij} = A_{ij}^* for (i, j) in Omega
        or, if reg != 0, then
        Minimize ||A||_* subject to ||(X2 • X1)A - y*||_2 + reg * ||(X2_bar • X1_bar)A||_2 <= epsilon, ie A_{ij} = A_{ij}^* for (i, j) in Omega
        ie A_{ij} = A_{ij}^* for (i, j) in Omega and A_{ij} = 0 for (i, j) not in Omega
    """
    # Define the variable A (n_1 x n_2 matrix to be optimized)
    A = cp.Variable((n_1, n_2))

    # Define the objective function (nuclear norm of A)
    objective = cp.Minimize(cp.normNuc(A))

    # Define the constraints 
    if X1X2_bar is None and X2_bullet_X1_bar is None :
        reg = 0
    else :
        # If X1X2_bar and X2_bullet_X1_bar are not None, and reg is not 0, then the constraint is added
        # N_prime = X1X2_bar[0].shape[0] if X1X2_bar is not None else X2_bullet_X1_bar.shape[0]
        N_prime = 0
        N_prime = 0 if X1X2_bar[0] is None else X1X2_bar[0].shape[0]
        N_prime = 0 if X2_bullet_X1_bar is None else X2_bullet_X1_bar.shape[0]
        reg = reg if N_prime == 0 else 0
    #constraints = [F_cvxpy(A, X1X2=X1X2, X2_bullet_X1=X2_bullet_X1) = y_star]
    constraints = [
        cp.norm(F_cvxpy(A, X1X2=X1X2, X2_bullet_X1=X2_bullet_X1) - y_star, 'fro')
        + (reg * cp.norm(F_cvxpy(A, X1X2=X1X2_bar, X2_bullet_X1=X2_bullet_X1_bar), 'fro') if reg!=0 else 0) <= EPSILON]  # tolerance for numerical precision

    # Set up and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Solution
    A = A.value
    return A
