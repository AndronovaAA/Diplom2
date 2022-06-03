import scipy.io
import numpy as np
from Systems import Random_System, System
from Decomposition import decomposition, decomposition_get_c
from Output_feedback_LMI import output_feedback_controller
from State_feedback_LMI import state_feedback_controller, state_feedback_controller_gamma

from Check_stability import check_output_feedback, check_state_feedback,check_observer
from Observers import observer_output,observer_output_test3
from scipy.integrate import odeint
from matplotlib.pyplot import *

# np.random.seed(0)
#Get data
mat = scipy.io.loadmat('TestSystem.mat')
data = mat['System']
content = data[0, 0]
A = content['A']
B = content['B']
C = content['C']
G = content['G']
mytol = content['tol']
ObserverSettings = content['ObserverSettings']
x_desired = content['x_desired']
dx_desired = content['dx_desired']

size_x = A.shape[0]
size_u = B.shape[1]
size_y = C.shape[0]
size_l = G.shape[0]

# Adding uncertainties
E_1 = A
E_2 = B
F_1 = np.eye(size_x)
F_2 = np.eye(size_y, size_x)

#  Initializing system
ObserverCost   = {'Q': 100*np.eye(size_l), 'R': 100*np.eye(size_y)}

system = System(A, B, C, G, F_1, F_2, E_1, E_2, ObserverCost, mytol)

# Decomposition
system_N = decomposition(system)

# Finding K with LMI(gamma for coefficients specific to this system)
K_n, coeff = state_feedback_controller_gamma(system_N)
K = K_n @ np.linalg.pinv(system_N.N)

A_c, B_c, F_c = decomposition_get_c(system)

m = E_1.shape[0]
Delta = np.diag(np.random.rand(m, ))

t0 = 0 # Initial time
tf = 100 # Final time
N = 2E3 # Numbers of points in time span
t = np.linspace(t0, tf, int(N)) # Create time span
#
x_0 = np.zeros((size_x,))
dx_0 = np.zeros((size_x,))
x_02 = x_0, dx_0
x_0 = np.concatenate(x_02)

x_des = x_desired.flatten()
dx_des = dx_desired.flatten()
def StateSpace(q, t, A_c, B_c, F_c, E_1, E_2, Delta, K, x_des, dx_des):
    x = q[:size_x]
    dx = q[size_x:size_x+size_x]
    u_ff = np.linalg.pinv(B_c + F_c @ Delta @E_2) @ (dx_des - (A_c+F_c@Delta@E_1) @ x_des)
    u_fb = K @ (x-x_des)
    u = u_fb + u_ff
    # u = K @ (x-x_des)
    ddx = (A_c+F_c@Delta@E_1) @ x + (B_c + F_c@Delta@E_2) @ u
    dq = ddx, dx
    # print(dx)
    return np.concatenate(dq)

#
figure(1)
x_sol = odeint(StateSpace, x_0, t, args=(A_c, B_c,F_c, E_1, E_2, Delta, K, x_des, dx_des))
y1, dy1 = x_sol[:, :size_x], x_sol[:, size_x:size_x+size_x]
plot(t, y1, 'r', linewidth=1.0, label = r'Position $y$ (m)')
ylabel(r'state(x)')
grid(True)
title(coeff)
xlabel(r'Time $t$ (s)')
show()
print("x_des")
print(x_des)
print("x")
print(y1[-1,:])


########################
z_des = system_N.N.T @ x_desired
dz_des = system_N.N.T @ dx_desired


size_z = len(z_des)

# size_z = 6

z_0 = np.zeros((size_z,))
dz_0 = np.zeros((size_z,))

z_des = z_des.flatten()
dz_des = dz_des.flatten()

def StateSpace_z(q, t, A_N, B_N, F_N, E_N1, E_2, Delta, K_n, z_des, dz_des):
    z = q[:size_z]
    dz = q[size_z:size_z+size_z]
    u_ff = np.linalg.pinv(B_N + F_N @ Delta @E_2) @ (dz_des - (A_N+F_N@Delta@E_N1) @ z_des)
    u_fb = K_n @ (z-z_des)
    u = u_fb + u_ff
    # u = K_n @ (z-z_des)
    ddz = (A_N+F_N@Delta@E_N1) @ z + (B_N + F_N@Delta@E_2) @ u
    dq = ddz, dz
    return np.concatenate(dq)

z_02 = z_0, dz_0
z_0 = np.concatenate(z_02)

A_N=system_N.A
B_N=system_N.B
F_N=system_N.F1
E_N1=system_N.E1

sol = odeint(StateSpace_z, z_0, t, args=(A_N, B_N, F_N, E_N1, E_2, Delta, K_n, z_des, dz_des))

y, dy = sol[:, :size_z], sol[:, size_z:size_z+size_z]
figure(2)
plot(t, y, 'r', linewidth=1.0, label = r'Position $y$ (m)')
ylabel(r'state(z)')
grid(True)
title(coeff)
xlabel(r'Time $t$ (s)')
show()
print("z_des")
print(z_des)
print("z")
print(y[-1,:])

# # L = observer_output(system_N)
# def observer_controller(state, t, A_N, B_N, F_N, E_N1, E_2, Delta, K_n, z_des, dz_des, L, C_N):
#
#     x, x_hat = np.split(state, 2)
#
#     u = -np.dot(K_n, x_hat-z_des)
#
#     ddz = (A_N+F_N@Delta@E_N1) @ x + (B_N + F_N@Delta@E_2) @ u
#
#     y = np.dot(C_N, x)
#
#     #
#     y_hat = np.dot(C_N, x_hat)
#     e = y - y_hat
#
#     dx_hat = np.dot(A, x_hat) + np.dot(B, u) + np.dot(L, e)
#     # print(dx_hat)
#
#     #
#     dstate = np.hstack((ddz, dx_hat))
#     # dstate = dx, dx_hat
#     return dstate