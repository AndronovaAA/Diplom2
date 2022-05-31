import scipy.io
import numpy as np
from Systems import Random_System, System
from Decomposition import decomposition
from Output_feedback_LMI import output_feedback_controller
from State_feedback_LMI import state_feedback_controller
from Check_stability import check_output_feedback, check_state_feedback,check_observer
from Observers import observer_output

mat = scipy.io.loadmat('ForSimon.mat')
data = mat['System']
content = data[0,0]
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

A_not, B_not, C_not, G_not, F_1, F_2, E_1, E_2 = Random_System(size_x, size_u, size_y, size_l, F2 = False, F1=False, E1=False, E2=False)

system = System(A, B, C, G, F_1, F_2, E_1, E_2, ObserverSettings, mytol)

system_N = decomposition(system)

A_K, B_K, C_K, D_K = output_feedback_controller(system_N)
