import numpy as np
import cvxpy as cp

def state_feedback_controller(system):
    A = system.A
    B = system.B

    F = system.F1
    E1 = system.E1
    E2 = system.E2

    size_x = A.shape[1]
    size_u = B.shape[1]
    size_f = F.shape[1]

    Q = cp.Variable((size_x, size_x))
    L = cp.Variable((size_u, size_x))

    LMI = cp.bmat([
        [Q @ A.T + L.T @ B.T + A @ Q + B @ L, F, Q @ E1.T + L.T @ E2.T],
        [F.T, -np.eye(size_f), np.zeros((size_f, size_f))],
        [E1 @ Q + E2 @ L, np.zeros((size_f, size_f)), -np.eye(size_f)]
    ])

    constraints = [Q >> 0.00000000001 * np.eye(size_x), LMI << 0]
    prob = cp.Problem(cp.Minimize(0), constraints=constraints)

    result = prob.solve()
    # print(Q.value)
    if result < 1:
        K = L.value @ np.linalg.pinv(Q.value)
        print("Result for state feedback controller obtained")
    else:
        print("no solution for state feedback LMI")

    return K

def state_feedback_controller_gamma(system):
    A = system.A
    B = system.B

    F = system.F1
    E1 = system.E1
    E2 = system.E2

    size_x = A.shape[0]
    size_u = B.shape[1]
    size_f = F.shape[1]
    coeff = np.array([25, 11, 15, 100, 10, 1])
    # print(np.linalg.eig(A))
    Q = cp.Variable((size_x, size_x), PSD=True)
    L = cp.Variable((size_u, size_x))
    gamma = np.diag(coeff) @ np.eye(size_x)
    # print(np.diag(coeff).shape)
    # Gamma = np.linalg.pinv(gamma)

    LMI = cp.bmat([
        [Q @ A.T + L.T @ B.T + A @ Q + B @ L, F, Q @ E1.T + L.T @ E2.T],
        [F.T, -np.eye(size_f), np.zeros((size_f, size_f))],
        [E1 @ Q + E2 @ L, np.zeros((size_f, size_f)), -np.eye(size_f)]
    ])

    constraints = [LMI << 0, Q >> 0.0001 * np.eye(size_x), Q << gamma]
    prob = cp.Problem(cp.Minimize(0), constraints=constraints)

    result = prob.solve()
    # print(Q.value)
    if result < 1:
        K = L.value @ np.linalg.pinv(Q.value)
        print("Result for state feedback controller obtained")
    else:
        print("no solution for state feedback LMI")

    return K, coeff