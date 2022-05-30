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

    Q = cp.Variable((size_x, size_x))
    L = cp.Variable((size_u, size_x))

    LMI = cp.bmat([
        [Q @ A.T + L.T @ B.T + A @ Q + B @ L, F, Q @ E1.T + L.T @ E2.T],
        [F.T, -np.eye(size_u), np.zeros((size_u, size_x))],
        [E1 @ Q + E2 @ L, np.zeros((size_x, size_u)), -np.eye(size_x)]
    ])

    constraints = [Q >> 0.001 * np.eye(size_x), LMI << 0]
    prob = cp.Problem(cp.Minimize(0), constraints=constraints)

    result = prob.solve()
    if result < 1:
        K = L.value @ np.linalg.pinv(Q.value)
        print("Result for state feedback controller obtained")
    else:
        print("no solution for state feedback LMI")

    return K