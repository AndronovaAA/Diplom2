import numpy as np
from numpy.random import default_rng

def check_output_feedback(system,A_K,B_K,C_K,D_K,numDelta, N = False):
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

    #add N for the system with decomposition
    for i in range(numDelta):
        Delta = np.diag(np.random.rand(m, ))

        # Checking eigenvalues to prove stability
        if N is False:
            X_cl = Acl + Fcl @ Delta @ np.linalg.pinv(np.eye(m) - Hcl @ Delta) @ Ecl
        else:
            N = system.N
            X_cl = Acl + Fcl @ N @ Delta @ np.linalg.pinv(np.eye(m) - Hcl @ N @ Delta) @ Ecl
        e1, _ = np.linalg.eig(X_cl)
        # print(e1)
        eigNeg = len(list(filter(lambda x: x.real < 0, e1)))
        if eigNeg == len(e1):
            stable_systems += 1
    return stable_systems/numDelta * 100

def check_state_feedback(system, K, numDelta, N = False):
    stable_systems = 0

    A = system.A
    B = system.B

    F1 = system.F1
    E1 = system.E1
    E2 = system.E2
    m = E1.shape[0]
    # for i in range(numDelta):
    #     Delta = np.diag(np.random.rand(m, ))
    #     # Checking eigenvalues to prove stability
    #     Acl = A + B @ K + F1 @ Delta @ (E1+E2@K)
    #     e1, _ = np.linalg.eig(Acl)
    #     eigNeg = len(list(filter(lambda x: x.real < 0, e1)))
    #     if eigNeg == len(e1):
    #         stable_systems += 1

        # add N for the system with decomposition
    for i in range(numDelta):
        Delta = np.diag(np.random.rand(m, ))

        # Checking eigenvalues to prove stability
        if N is False:
            Acl = A + B @ K + F1 @ Delta @ (E1+E2@K)
        else:
            N = system.N
            Acl = A + B @ K + F1 @ N @ Delta @ (E1+E2@K)
        e1, _ = np.linalg.eig(Acl)
        eigNeg = len(list(filter(lambda x: x.real < 0, e1)))
        if eigNeg == len(e1):
            stable_systems += 1
    return stable_systems / numDelta * 100



