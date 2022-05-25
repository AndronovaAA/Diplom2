import numpy as np
import cvxpy as cp
from scipy.linalg import null_space, orth, svd, solve_continuous_are
# from sympy import Matrix
from numpy.random import default_rng
from control.matlab import lqr,place
# from casadi.casadi import  horzcat, vertcat
import Systems
import Decomposition
import Output_feedback_LMI

# making random matrieces
A, B, C, G, F_0, E_1, E_2 = Systems.Random_System(6,6,6,4)

mytol = 0.000001
ObserverCost   = {'Q': 100*np.eye(4), 'R': np.array([[C.shape[0]]])}

F20 = np.zeros((6,6))
#init system
system = Systems.System(A, B, C, G, F_0, F20, E_1, E_2, ObserverCost, mytol)

#decomposition
A_N, B_N, C_N, F_N, E_N1, E_N2, A_R, E_R, N = Decomposition.Decomposition(system)
F2 = np.zeros((F_N.T.shape))

system_LMI = Systems.System(A_N,B_N,C_N,G,F_N, F20, E_N1,E_N2,ObserverCost,mytol)

num_d = 1000
def check_lmi(system,A_K,B_K,C_K,D_K,numDelta, N = None):
    stable_systems = 0

    A = system.A
    B = system.B
    C = system.C

    F1 = system.F1
    F2 = system.F2
    E1 = system.E1
    E2 = system.E2

    Acl = np.vstack((np.hstack((A + B @ D_K @ C, B @ C_K)), np.hstack((B_K @ C, A_K))))
    Fcl = np.vstack((F1 + B @ D_K @ F2, B_K @ F2))
    Ecl = np.hstack((E1 + E2 @ D_K @ C, E2 @ C_K))
    Hcl = np.array((E2 @ D_K @ F2))

    m = Ecl.shape[0]
    print("Ecl", Ecl.shape)
    #add N for the system with decomposition
    for i in range(numDelta):
        Delta = np.diag(np.random.rand(m, ))

        # Checking eigenvalues to prove stability
        if N is None:
            X_cl = Acl + Fcl @ Delta @ np.linalg.pinv(np.eye(m) - Hcl @ Delta) @ Ecl
        else:
            X_cl = Acl + Fcl @ N @ Delta @ np.linalg.pinv(np.eye(m) - Hcl @ N @ Delta) @ Ecl
        e1, _ = np.linalg.eig(X_cl)
        # print(e1)
        eigNeg = len(list(filter(lambda x: x.real < 0, e1)))
        if eigNeg == len(e1):
            stable_systems += 1
    return stable_systems

A_K1, B_K1, C_K1, D_K1 = Output_feedback_LMI.output_feedback_controller(system)
stab1 = check_lmi(system, A_K1, B_K1, C_K1, D_K1, num_d)

A_K, B_K, C_K, D_K = Output_feedback_LMI.output_feedback_controller(system_LMI)
stab = check_lmi(system_LMI, A_K, B_K, C_K, D_K, num_d, N)
print(stab1)
print(stab)
