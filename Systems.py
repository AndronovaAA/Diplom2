import numpy as np
from numpy.random import default_rng

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

        self.tol = tol

def Random_System(size_x, size_u, size_y, size_l, F2 = True):
    A = 0.3 * np.random.randn(size_x, size_x) - 1 * np.eye(size_x)
    B = 0.3 * np.random.randn(size_x, size_u)
    C = 0.3 * np.random.randn(size_y, size_x)
    G = 0.1 * np.random.randn(size_l, size_x)
    F_1 = 0.3 * np.random.randn(size_x, size_x)
    E_1 = 0.3 * np.random.randn(size_u, size_x)
    E_2 = 0.3 * np.random.randn(size_x, size_u)
    if F2 == True:
        F_2 = 0.3 * np.random.randn(size_x, size_x)
    else:
        F_2 = np.zeros((size_x,size_x))


    return A, B, C, G, F_1, F_2, E_1, E_2

