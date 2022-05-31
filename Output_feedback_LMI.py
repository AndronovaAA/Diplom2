import numpy as np
import cvxpy as cp

def output_feedback_controller(system):
    A = system.A
    B = system.B
    C = system.C

    F1 = system.F1
    F2 = system.F2
    E1 = system.E1
    E2 = system.E2

    size_x = A.shape[1]
    size_u = B.shape[1]
    size_y = C.shape[0]
    size_f = F1.shape[1]

    Y_1 = cp.Variable((size_x, size_x))
    C_K_hat = cp.Variable((size_u, size_x))
    A_K_hat = cp.Variable((size_x, size_x))
    D_K = cp.Variable((size_u, size_y))#y is right
    X_1 = cp.Variable((size_x, size_x))
    B_K_hat = cp.Variable((size_x, size_y))
    Y_2 = cp.Variable((size_x, size_x))

    LMI = cp.bmat([[A@Y_1 + Y_1 @ A.T + B @ C_K_hat + C_K_hat.T @ B.T, A + A_K_hat.T + B @ D_K @ C, F1+B@D_K@F2,
                    Y_1@E1.T+C_K_hat.T@E2.T],
                   [A.T+ A_K_hat + C.T@D_K.T@B.T, X_1 @ A + A.T@X_1 + B_K_hat @ C + C.T@B_K_hat.T,
                    X_1@F1, E1.T + C.T @ D_K.T @ E2.T],
                   [F1.T+F2.T@D_K.T@B.T, F1.T@X_1+F2.T@B_K_hat.T,-np.eye(size_f),F2.T@D_K.T@E2.T],
                   [E1@Y_1+E2@C_K_hat, E1+E2@D_K@C, E2@D_K@F2, -np.eye(size_f)]])

    constr = cp.bmat([[Y_1, np.eye(size_x)],
                      [np.eye(size_x), X_1]
                      ])

    constraints = [X_1 >> 0.001 * np.eye(size_x), Y_1 >> 0.001 * np.eye(size_x), LMI << -0.01 * np.eye(size_x + size_x + size_f + size_f),
                   constr >> 0.001 * np.eye(size_x + size_x), Y_2 >> 0.001 * np.eye(size_x)]
    prob = cp.Problem(cp.Minimize(0), constraints=constraints)

    result = prob.solve()
    if result < 1:
        print("Result for LMI(output feedback) is obtained")
    else:
        print("no solution for LMI")

    X_2 = (np.eye(size_x) - X_1.value @ Y_1.value) @ np.linalg.pinv(Y_2.value)
    C_K = C_K_hat.value @ np.linalg.pinv(Y_2.value) - D_K.value @ C @ Y_1.value @ np.linalg.pinv(Y_2.value)
    B_K = np.linalg.pinv(X_2) @ B_K_hat.value - np.linalg.pinv(X_2) @ X_1.value @ B @ D_K.value
    A_K = np.linalg.pinv(X_2) @ (-X_2 @ B_K @ C @ Y_1.value - X_1.value @ B @ C_K @ Y_2.value -
                                 X_1.value @ A @ Y_1.value - X_1.value @ B @ D_K.value @ C @ Y_1.value + A_K_hat.value) @ np.linalg.pinv(
        Y_2.value)
    D_K = D_K.value

    return A_K, B_K, C_K, D_K
