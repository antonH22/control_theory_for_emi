import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from ctrl import discrete_optimal_control as doc
from enhanced_eval_ctrl_approach import myutils

### Inverted pendulum study: stabilizing the pendulum in the upright position (5.2, Figure 8)

# System parameters
m = 1  # mass of the pendulum
M = 1  # mass of the cart
L = 2  # length of the pendulum arm
g = -10  # gravitational acceleration
d = 1  # damping factor
b = 1  # 1: pendulum up
# With b = -1 the equilibrium is pendulum down and the system can be simulated with the linear dynamics (A, B)

# Continuous-time system matrices (linearized around equilibrium point)
A_c = np.array([[0, 1, 0, 0], 
                [0, -d/M, b*m*g/M, 0], 
                [0, 0, 0, 1], 
                [0, -b*d/(M*L), -b*(m+M)*g/(M*L), 0]])

B_c = np.array([[0], 
                [1/M], 
                [0], 
                [b/(M*L)]])

# Time step for discretization
delta_t = 0.01

# Discretize the system 
A = expm(A_c * delta_t) # Matrix exponential expm(A) = e^(A*delta_t)

# Numerical integration to compute the B matrix, because the matrix exponential is not directly invertible
num_points = 100 
tau = np.linspace(0, delta_t, num_points)
dt = delta_t / num_points

B = np.zeros_like(B_c)
for t in tau:
    B += expm(A_c * t) @ B_c * dt

#######################################################################################################################################

# Simulate pendulum (make sure to set d accordingly)
x_0 = np.array([1, 0, np.pi-0.1, 0])  # Initial state (position, velocity, angle, angular velocity)
x_ref = np.array([6, 0, np.pi, 0])  # Reference state

states, inputs = myutils.simulate_pendulum(x_0, x_ref, A, B, 0.0, 5000)

plot_states = np.array(states)
plot_inputs = np.array(inputs)

# Save arrays to file
save_path = os.path.join("results_replicated", "pendulum_up.npz")
np.savez(save_path, plot_states=plot_states, plot_inputs=plot_inputs)
print(f'Saved results to {save_path}')

"""
# Check if the system is stable (its unstable for b = 1) 
# For b = -1, the system is stable? (|lambda0|= 1.0)
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues of A: ", eigenvalues)

# Check if the system is controllable
ctrlb = doc.controllable(A, B)                
print(f'Controllable via input? {ctrlb}')

# Check the Average controlability of each node
ac_nodes = doc.average_ctrb(A, T = 10, B = None)
for i in range(len(ac_nodes)):
    print(f'AC of node {i}: {ac_nodes[i]}')
"""