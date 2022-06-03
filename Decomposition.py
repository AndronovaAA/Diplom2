import numpy as np
from scipy.linalg import null_space, orth

def decomposition(system):
    A = system.A
    B = system.B
    C = system.C
    G = system.G
    F1 = system.F1
    F2 = system.F2
    E1 = system.E1
    E2 = system.E2

    J = G.T

    size_x = A.shape[1]

    R = orth(G.T)
    N = null_space(G)

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

    E_N1 = E1 @ N
    E_R = E1 @ R
    E_N2 = E2
    C_N = C @ N

    size_n = N.shape[1]
    size_y = C_N.shape[0]
    F2 = np.zeros((size_y, size_x))

    # print(E_N2.shape)

    class System_N:
        def __init__(self, A_N, B_N, C_N, F_N, E_N1, E_N2, A_R, E_R, N):
            self.A = A_N
            self.B = B_N
            self.C = C_N
            self.F1 = F_N
            self.E1 = E_N1
            self.E2 = E_N2

            self.N = N
            self.A_R = A_R
            self.E_R = E_R

            self.F2 = F2

            self.tol = system.tol
            self.observerCost = system.observerCost

    return System_N(A_N, B_N, C_N, F_N, E_N1, E_N2, A_R, E_R, N)

def decomposition_get_c(system):
    A = system.A
    B = system.B
    C = system.C
    G = system.G
    F1 = system.F1
    F2 = system.F2
    E1 = system.E1
    E2 = system.E2

    J = G.T

    size_x = A.shape[1]

    R = orth(G.T)
    N = null_space(G)

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

    return A_c,B_c,F_c

