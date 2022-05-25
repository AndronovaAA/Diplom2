import numpy as np
import cvxpy as cp
from scipy.linalg import null_space, orth, svd, solve_continuous_are
# from sympy import Matrix
from numpy.random import default_rng
from control.matlab import lqr,place
# from casadi.casadi import  horzcat, vertcat

class System:
    def __init__(self, A, B, C, G, F1, F2, E1, E2, observerCost, tol):
        self.A = A
        self.B = B
        self.C = C
        self.G = G
        self.F1 = F1
        self.F2 = F2
        self.E1 = E1
        self.E2 = E2

        self.observerCost = observerCost
        #
        self.tol = tol

def Random_System(size_x, size_u, size_y, size_l):
    A = 0.3 * np.random.randn(size_x, size_x) - 1 * np.eye(size_x)
    B = 0.3 * np.random.randn(size_x, size_u)
    C = 0.3 * np.random.randn(size_y, size_x)
    G = 0.1 * np.random.randn(size_l, size_x)
    F_0 = 0.3 * np.random.randn(size_x, size_x)
    E_1 = 0.3 * np.random.randn(size_u, size_x)
    E_2 = 0.3 * np.random.randn(size_x, size_u)

    return A, B, C, G, F_0, E_1, E_2

A, B, C, G, F_0, E_1, E_2 = Random_System(6,6,6,4)

mytol = 0.000001
ObserverCost   = {'Q': 100*np.eye(4), 'R': np.array([[C.shape[0]]])}

# system = System(A, B, C, G, F_0, E_1, E_2, ObserverCost, mytol)