import numpy as np
from scipy.linalg import null_space, orth

# class Decomposition:
#     def __init__(self, system):
#         A_N, B_N, C_N, F_N, E_N1, E_N2, A_R, E_R, N = decomposition(system)
#         self.A = A_N
#         self.B = B_N
#         self.C = C_N
#         self.F1 = F_N
#         self.E1 = E_N1
#         self.E2 = E_N2
#
#         self.tol = system.tol
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
    E_N1 = N.T @ E1 @ N
    E_R = N.T @ E1 @ R
    E_N2 = N.T @ E2
    C_N = C @ N

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

            self.F2 = system.F2

            self.tol = system.tol

    return System_N(A_N, B_N, C_N, F_N, E_N1, E_N2, A_R, E_R, N)


