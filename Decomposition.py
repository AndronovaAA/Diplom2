import numpy as np
import cvxpy as cp
from scipy.linalg import null_space, orth, svd, solve_continuous_are
# from sympy import Matrix
from numpy.random import default_rng
from control.matlab import lqr,place
# from casadi.casadi import  horzcat, vertcat
import Systems

def Decomposition(system):
    A = system.A
    B = system.B
    C = system.C
    G = system.G
    F1 = system.F1
    E1 = system.E1
    E2 = system.E2

    J = G.T
    # mytol = system.tol

    size_x = A.shape[1]
    size_u = B.shape[1]
    size_y = C.shape[0]
    size_l = G.shape[0]

    R = orth(G.T)
    N = null_space(G)
    print("N shape",N.shape)

    # print(np.linalg.matrix_rank(G@J))
    P1 = np.eye(size_x) - J @ np.linalg.pinv(G @ J) @ G
    # print(('P1 rank: ', np.linalg.matrix_rank(P1, tol=mytol)))
    # l, _ = np.linalg.eig(P1)
    # print(('P1 eig: ', np.round(l, 2)))

    P2 = N @ N.T
    # print(('P2 rank: ', np.linalg.matrix_rank(P2, tol=mytol)))
    # l, _ = np.linalg.eig(P2)
    # print(('P2 eig: ', np.round(l, 2)))

    A_c = P2 @ A
    B_c = P2 @ B
    F_c = P2 @ F1

    A_N = N.T @ A_c @ N
    A_R = N.T @ A_c @ R
    B_N = N.T @ B_c
    F_N = N.T @ F_c
    E_N1 = N.T @ E1 @ N
    E_R = N.T @ E1 @ R
    E_N2 = N.T @ E2
    C_N = C @ N

    return A_N, B_N, C_N, F_N, E_N1, E_N2, A_R, E_R, N
# making random matrieces
A, B, C, G, F_0, E_1, E_2 = Systems.Random_System(6,6,6,4)

mytol = 0.000001
ObserverCost   = {'Q': 100*np.eye(4), 'R': np.array([[C.shape[0]]])}
#init system
# system = Systems.System(A, B, C, G, F_0, E_1, E_2, ObserverCost, mytol)
#decomposition
# A_N, B_N, C_N, F_N, E_N1, E_N2, A_R, E_R = Decomposition(system)
