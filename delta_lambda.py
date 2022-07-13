import scipy.io
import numpy as np
from scipy.linalg import null_space, orth
import cvxpy as cp

from scipy.integrate import odeint
# from matplotlib.pyplot import *

np.random.seed(10)
mytol = 0.000001

size_x = 6
size_u = 8
size_l = 4

A_0 = 0.3 * np.random.randn(size_x, size_x) - 1 * np.eye(size_x)
B_0 = 0.3 * np.random.randn(size_x, size_u)
G = 0.1 * np.random.randn(size_l, size_x)
J_0 = 0.3 * np.random.randn(size_x, size_x)
# J_0 = G.T
size_lambda = J_0.shape[1]
# print(J_0.shape)
# size f = size N (why?)
size_f = 2
F_0 = 0.3 * np.random.randn(size_x, size_f)
E_0 = 0.3 * np.random.randn(size_f, size_lambda)

P_lambda = np.eye(size_x) - J_0 @ np.linalg.pinv(G @ J_0) @ G
# print(('P1 rank: ', np.linalg.matrix_rank(P_lambda, tol=mytol)))
# l, _ = np.linalg.eig(P_lambda)
# print(('P1 eig: ', np.round(l, 2)))

A_c = P_lambda @ A_0
B_c = P_lambda @ B_0
F_c = P_lambda @ F_0

R = orth(G.T)
N = null_space(G)
# print(N.shape)
# P2 = N @ N.T
# A_c = P2 @ A_0
# B_c = P2 @ B_0
# F_c = P2 @ F_0

A_N = N.T @ A_c @ N
B_N = N.T @ B_c
F_N = N.T @ F_c
# print(J_0.shape)
# print(E_0.shape)
# print(np.linalg.pinv(G @ J_0).shape)
P_E = E_0 @ np.linalg.pinv(G @ J_0) @ G

A_E = P_E @ A_0 @ N
B_E = P_E @ B_0
F_E = P_E @ F_0

size_x_n = A_N.shape[1]
size_u_n = B_N.shape[1]
Q = cp.Variable((size_x_n, size_x_n))
L = cp.Variable((size_u_n, size_x_n))
# print(A_E.shape)
# expr = np.zeros((size_x_n, size_f))
# EXPR1 = cp.bmat([A_N @ Q + Q @ A_N.T + B_N @ L + L.T @ B_N.T, F_N + (Q @ A_E.T + L.T @ B_E.T) @ F_E,
#                  Q @ A_E.T + L.T @ B_E.T, np.zeros((size_x_n, size_f))])
# EXPR2 = cp.bmat([F_N.T + F_E.T @ (A_E @ Q + B_E @ L), F_E.T @ F_E - 2 * np.eye(size_f), np.zeros((size_x_n, size_f)), np.eye(size_f)])
# EXPR3 = cp.bmat([A_E @ Q + B_E @ L, np.zeros((size_f, size_x_n)), - np.eye(size_f), np.zeros((size_x_n, size_x_n))])
# EXPR4 = cp.bmat([np.zeros((size_f, size_x_n,)), np.eye(size_f), np.zeros((size_x_n,size_f)), - np.eye(size_f)])

LMI = cp.bmat([[A_N @ Q + Q @ A_N.T + B_N @ L + L.T @ B_N.T, F_N + (Q @ A_E.T + L.T @ B_E.T) @ F_E,
                Q @ A_E.T + L.T @ B_E.T, np.zeros((size_x_n, size_f))],
               [F_N.T + F_E.T @ (A_E @ Q + B_E @ L), F_E.T @ F_E - 2 * np.eye(size_f), np.zeros((size_x_n, size_f)), np.eye(size_f)],
               [A_E @ Q + B_E @ L, np.zeros((size_f, size_x_n)), - np.eye(size_f), np.zeros((size_x_n, size_x_n))],
               [np.zeros((size_f, size_x_n,)), np.eye(size_f), np.zeros((size_x_n,size_f)), - np.eye(size_f)]])
# print(expr.shape)
constraints = [Q >> 0.0001 * np.eye(size_x_n), LMI << -0.001 * np.eye(size_x_n + size_x_n + size_f + size_f)]
prob = cp.Problem(cp.Minimize(0), constraints=constraints)

result = prob.solve()
# print(Q.value)
if result < 1:
    K = L.value @ np.linalg.pinv(Q.value)
    print("Result for state feedback controller obtained")
else:
    print("no solution for state feedback LMI")

# print(K)

Delta = np.diag(np.random.rand(size_f, ))

# Checking eigenvalues to prove stability

expr = np.hstack((A_N + B_N @ K, F_N @ Delta @ E_0))
# print(expr.shape)
A_cl = np.vstack((np.hstack((A_N + B_N @ K, F_N @ Delta @ E_0)), np.zeros((size_x, size_l + size_l))))
# print(A_cl.shape)
e1, _ = np.linalg.eig(A_cl)
print(e1)