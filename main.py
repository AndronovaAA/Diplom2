import numpy as np
from Systems import Random_System, System
from Decomposition import decomposition
from Output_feedback_LMI import output_feedback_controller
from State_feedback_LMI import state_feedback_controller
from Check_stability import check_output_feedback, check_state_feedback

#x_dot = (A+F1*Delta*E1)x + (B+F1*Delta*E2)u + J*lambda
#G*x_dot = 0
#y = Cx
# Making Random matrices without F2
size_x = 6
size_y = 6
size_u = 6
size_l = 4
A, B, C, G, F_1, F_2, E_1, E_2 = Random_System(size_x,size_y,size_u,size_l, F2 = False)

# Determining other parameters
mytol = 0.000001
ObserverCost   = {'Q': 100*np.eye(size_l), 'R': np.array([[C.shape[0]]])}

# Initializing system
system = System(A, B, C, G, F_1, F_2, E_1, E_2, ObserverCost, mytol)

#decomposition
system_N = decomposition(system)

# Getting control output feedback for system with decomposition
A_K, B_K, C_K, D_K = output_feedback_controller(system_N)

#Number of deltas to check
num_d = 1000

#Checking eigenvalues
# N=True for system with decomposition, N=False for ordinary system
stab = check_output_feedback(system_N, A_K, B_K, C_K, D_K, num_d, N=True)
print("% of stable systems(output feedback + decomposition):", stab)
print("")

##########
#x_dot = (A+F1*Delta*E1)x + (B+F1*Delta*E2)u
# Making Random matrices without F2
size_x = 6
size_y = 6
size_u = 6
size_l = 4
A, B, C, G, F_1, F_2, E_1, E_2 = Random_System(size_x,size_y,size_u,size_l, F2 = False)

# Determining other parameters
mytol = 0.000001
ObserverCost   = {'Q': 100*np.eye(size_l), 'R': np.array([[C.shape[0]]])}

# Initializing system
system2 = System(A, B, C, G, F_1, F_2, E_1, E_2, ObserverCost, mytol)

# Getting control output feedback for system with decomposition
K = state_feedback_controller(system2)

#Number of deltas to check
num_d = 1000

#Checking eigenvalues
# N=True for system with decomposition, N=False for ordinary system
stab2 = check_state_feedback(system2,K,num_d)
print("% of stable systems(state feedback):", stab2)
print("")

##########
#x_dot = (A+F1*Delta*E1)x + (B+F1*Delta*E2)u
#y = (C+ F2*Delta*E2)x
# Making Random matrices
size_x = 6
size_y = 6
size_u = 6
size_l = 4
A, B, C, G, F_1, F_2, E_1, E_2 = Random_System(size_x,size_y,size_u,size_l, F2 = True)

# Determining other parameters
mytol = 0.000001
ObserverCost   = {'Q': 100*np.eye(size_l), 'R': np.array([[C.shape[0]]])}

# Initializing system
system3 = System(A, B, C, G, F_1, F_2, E_1, E_2, ObserverCost, mytol)

# Getting control output feedback for system with decomposition
A_K3, B_K3, C_K3, D_K3 = output_feedback_controller(system3)

#Number of deltas to check
num_d = 1000

#Checking eigenvalues
# N=True for system with decomposition, N=False for ordinary system
stab3 = check_output_feedback(system3, A_K3, B_K3, C_K3,D_K3, num_d, N=False)
print("% of stable systems(output feedback):", stab3)

