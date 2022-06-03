import numpy as np
from control.matlab import lqr

def observer_output(system):
    A_N = system.A
    B_N = system.B
    C_N = system.C

    size_x = A_N.shape[1]
    size_y = C_N.shape[0]

    costQ = system.observerCost['Q']
    costR = system.observerCost['R']

    A_ob = np.vstack((np.hstack((A_N, np.eye(size_x))), np.zeros((size_x, size_x + size_x))))
    C_ob = np.hstack((C_N, np.zeros((size_y, size_x))))
    L = lqr(A_ob.T, C_ob.T, costQ, costR)[0]
    L = L.T

    return L


def observer_output_test3(system):
    A_N = system.A
    B_N = system.B
    C_N = system.C

    size_x = A_N.shape[1]
    size_y = C_N.shape[0]

    # costQ = system.observerCost['Q']
    costR = system.observerCost['R']

    A_ob = np.vstack((np.hstack((A_N, np.eye(size_x))), np.zeros((size_x, size_x + size_x))))
    C_ob = np.hstack((C_N, np.zeros((size_y, size_x))))
    costQ = 100 * np.eye(A_ob.shape[0])
    L = lqr(A_ob.T, C_ob.T, costQ, costR)[0]
    L = L.T

    return L