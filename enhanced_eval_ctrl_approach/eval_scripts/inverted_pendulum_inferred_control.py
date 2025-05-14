import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from enhanced_eval_ctrl_approach import myutils
from ctrl import utils

import numpy as np

### Inverted pendulum study: controlling the pendulum in the down position using control law based on inferred dynamics (5.2, Figure 9)
### - Note: Results may vary from Figure 9 due to random inputs during trajectory generation

steps = 1000 # number of time steps per trajectory

# Function to generate a single trajectory
def generate_trajectory(initial_state, input_func, steps, noise_std=0.0, pendulum_up=False):
    trajectory = []
    state = initial_state
    for _ in range(steps):
        u = input_func()  # Get control input
        if pendulum_up:
            if not (np.pi - 0.01 <= state[2] <= np.pi + 0.01):
                break
            """
            # To try if it works to learn the control law by trajectories controlled by the optimal control law. -> didn't work
            reference = np.array([6, 0, np.pi, 0])  # Reference state
            K = np.array([[-0.96166503, -3.32756487, 75.80102465, 21.98471239]])
            u = -K @ (state - reference)
            """
        trajectory.append((state, u))
        state = myutils.simulate_system_step_nonlinear(state, u, noise_std)
    return trajectory

# Control input function
def random_input():
    return np.array([np.random.uniform(-10, 10)])
# Todo: Try more functions

# Generate a dataset of trajectories for the pendulum in the upright position: doesn't work
def generate_dataset(num_trajectories, steps, noise_std=0.0, pendulum_up=False):
    dataset = []
    """
    # Not needed for pendulum down (Generate trajectory from random initial states near the upright fix point)
    for _ in range(num_trajectories):
        angle_near_equilibrium = np.pi + np.random.uniform(-0.1, 0.1)
        initial_state = np.array([1, 0, angle_near_equilibrium, 0])
        trajectory = generate_trajectory(initial_state, zero_input, steps, noise_std, pendulum_up)
        dataset.append(trajectory)
    """
    
    # Apply random inputs
    for _ in range(num_trajectories):
        initial_state = np.array([0, 0, np.pi, 0])
        trajectory = generate_trajectory(initial_state, random_input, steps, noise_std, pendulum_up)
        dataset.append(trajectory)
    return dataset

# Parameters
num_trajectories = 1
noise_std = 0.0

# Generate the dataset
dataset = generate_dataset(num_trajectories, steps, noise_std, pendulum_up=False)

# Convert dataset to arrays for further processing
state_data = []
input_data = []

for trajectory in dataset:
    for state, control in trajectory:
        state_data.append(state)
        input_data.append(control)
    state_data.append(np.full(4, np.nan))
    input_data.append(np.array([np.nan]))

X = np.array(state_data)
U = np.array(input_data)

#####################################################################################################################################

# Infer the system dynamics using the stable ridge regression
A, B, lmbda = utils.stable_ridge_regression(X, U)

# Simulate pendulum down, set pendulum_up = False in generate trajectory function (pendulum_up should provide trajectory to learn the dynamicsvfor the upwoards fixed point, but it doesnt work)
x_0 = np.array([1, 0, -0.1, 0])  # Initial state (position, velocity, angle, angular velocity)
x_ref = np.array([6, 0, 0, 0])  # Reference state

states, inputs = myutils.simulate_pendulum(x_0, x_ref, A, B, 0.0, 5000)

plot_states = np.array(states)
plot_inputs = np.array(inputs)

# Save arrays to file
save_path = os.path.join("results_replicated", "pendulum_down_inferred.npz")
np.savez(save_path, plot_states=plot_states, plot_inputs=plot_inputs)
print(f'Saved results to {save_path}')

"""
# Check if the system is stable
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