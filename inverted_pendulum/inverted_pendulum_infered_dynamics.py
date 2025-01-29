import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ctrl import discrete_optimal_control as doc
from ctrl import utils

import numpy as np
import random

# Define the simulation time step and duration
delta_t = 0.01  # time step
T = 1000  # total simulation time
steps = int(T / delta_t)  # number of time steps per trajectory

# Nonlinear dynamics simulation function
def simulate_system_step_nonlinear(x, u, delta_t, noise_std=0.0):
    m = 1  # mass of the pendulum
    M = 1  # mass of the cart
    L = 2  # length of the pendulum arm
    g = -10  # gravitational acceleration
    d = 1  # damping factor

    Sx = np.sin(x[2])  # sin(theta)
    Cx = np.cos(x[2])  # cos(theta)
    D = m * L**2 * (M + m * (1 - Cx**2))

    dx = np.zeros(4)

    u = u[0]  # Control input

    # State equations
    dx[0] = x[1]  # x_dot
    dx[1] = (1 / D) * (-m**2 * L**2 * g * Cx * Sx + m * L**2 * (m * L * x[3]**2 * Sx - d * x[1])) + m * L**2 * (1 / D) * u
    dx[2] = x[3]  # theta_dot
    dx[3] = (1 / D) * ((M + m) * m * g * L * Sx - m * L * Cx * (m * L * x[3]**2 * Sx - d * x[1])) - m * L * Cx * (1 / D) * u

    # Add noise to specific nodes
    noise = np.random.normal(0, noise_std)
    noise_vec = np.array([0, 0, noise, 0])

    return x + dx * delta_t + noise_vec  # Euler method

# Function to generate a single trajectory
def generate_trajectory(initial_state, input_func, delta_t, steps, noise_std=0.0):
    trajectory = []
    state = initial_state
    for _ in range(steps):
        u = input_func()  # Get control input
        trajectory.append((state, u))
        state = simulate_system_step_nonlinear(state, u, delta_t, noise_std)
    return trajectory

# Control input functions
def zero_input():
    return np.array([0.0])

def random_input():
    return np.array([np.random.uniform(-10, 10)])

def sinusoidal_input(t):
    return np.array([10 * np.sin(0.5 * t)])

# Generate a dataset of trajectories for the pendulum system in the down position
def generate_dataset_pendulum_down(num_trajectories, delta_t, steps, noise_std=0.0):
    dataset = []
    for _ in range(num_trajectories//2):
        # Near-equilibrium trajectory
        angle_near_equilibrium = random.uniform(-0.1, 0.1)
        initial_state = np.array([0, 0, angle_near_equilibrium, 0])
        trajectory = generate_trajectory(initial_state, zero_input, delta_t, steps, noise_std)
        dataset.append(trajectory)
    # Generate random initial states and also inputs
    for _ in range(num_trajectories//2):
        angle_near_equilibrium = random.uniform(-0.1, 0.1)
        initial_state = np.array([0, 0, 0, 0])
        trajectory = generate_trajectory(initial_state, random_input, delta_t, steps, noise_std)
        dataset.append(trajectory)
    return dataset

# Generate a dataset of trajectories for the pendulum in the upright position: doesn't work
def generate_dataset_pendulum_up(num_trajectories, delta_t, steps, noise_std=0.0):
    dataset = []
    for _ in range(num_trajectories//2):
        # Near-equilibrium trajectory
        angle_near_equilibrium = random.uniform(np.pi-0.2, np.pi+0.2)
        initial_state = np.array([0, 0, angle_near_equilibrium, 0])
        trajectory = generate_trajectory(initial_state, zero_input, delta_t, steps, noise_std)
        dataset.append(trajectory)
    # Generate random initial states and also inputs
    
    for _ in range(num_trajectories//2):
        angle_near_equilibrium = random.uniform(np.pi, np.pi)
        initial_state = np.array([0, 0, angle_near_equilibrium, 0])
        trajectory = generate_trajectory(initial_state, random_input, delta_t, steps, noise_std)
        dataset.append(trajectory)
    
    return dataset

# Parameters
num_trajectories = 20
noise_std = 0.0

# Generate the dataset
dataset = generate_dataset_pendulum_down(num_trajectories, delta_t, steps, noise_std)

# Convert dataset to arrays for further processing
state_data = []
input_data = []

for trajectory in dataset:
    for state, control in trajectory:
        state_data.append(state)
        input_data.append(control)

X = np.array(state_data)
U = np.array(input_data)


"""
# Access one trajectory, e.g., trajectory 0 (first 1000 states of X)
plot_states_trajectory = X[:1000]
plot_inputs_trajectory = U[:1000]


# Plotting the results
plt.figure(figsize=(12, 6))
# Plot for angular position (pendulum angle in radians)
plt.subplot(1, 3, 1)
angles_rad = plot_states_trajectory[:,2]
plt.plot(angles_rad, label="Pendulum Angle (radians)")
plt.title("Angle of the Pendulum")
plt.xlabel("Time Step")
plt.ylabel("Angle (radians)")
plt.grid(True)
plt.legend()

# Plot for cart position
plt.subplot(1, 3, 2)
plt.plot(plot_states_trajectory[:,0], label="Cart Position (meters)")
plt.title("Cart Position")
plt.xlabel("Time Step")
plt.ylabel("Position (meters)")
plt.grid(True)
plt.legend()

# Plot for cart input
plt.subplot(1, 3, 3)
plt.plot(plot_inputs_trajectory, label="Inputs")
plt.title("Input")
plt.xlabel("Time Step")
plt.ylabel("Input")
plt.grid(True)
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
"""
############################################################################################

# Infer the system dynamics using the stable ridge regression

A, B, lmbda = utils.stable_ridge_regression(X, U)


def simulate_system_step_linear(x, u, noise_std=0.0, A=A, B=B):
    # Discrete-time update equation
    x_next = doc.step(A, B, x, u)
    # Add noise to specific nodes
    noise = np.random.normal(0, noise_std)
    noise_vec = np.array([0, 0, noise, 0]) # only noise is on the angle
    return x_next + noise_vec

def simulate_pendulum(x_0, x_ref, A, B, noise_std, num_steps):
    x_t = x_0

    Q = np.diag([1, 1, 1, 1])  # State cost matrix
    R = np.array([[1]]) # Control effort cost matrix (large values reduce control effort)
    reference = x_ref
    
    # Compute the optimal gain matrix K
    K = doc.kalman_gain(A, B, Q, R)
    states = [x_0]
    inputs = []
    for i in range(num_steps):
        # Compute the optimal control input using the Kalman gain
        u = -K @ (x_t - reference.T)
        #u = np.array([0])
        # Update the state using the nonlinear or linear dynamics function
        x_t = simulate_system_step_linear(x_t, u, noise_std)  # Get the next state

        # Append the updated state and control input
        states.append(x_t)
        inputs.append(u)
    return states, inputs

# Simulate pendulum
x_0 = np.array([0, 0, 0.1, 0])  # Initial state (position, velocity, angle, angular velocity)
x_ref = np.array([1, 0, 0, 0])  # Reference state

states, inputs = simulate_pendulum(x_0, x_ref, A, B, 0.0, 2000)

# Bemerkung: Das nonlineare System kann basierend auf dem abgeleiteten linearen System (matrix A und B) kontrolliert werden (durch berechnung der K Matrix)
# Das lineare System kann nur in der Nähe des Gleichgewichts Pendel unten simuliert und kontrolliert werden

# Check if the system is stable (its unstable for b = 1) 
# For b = -1, the system is stable? (|lambda0|= 1.0)
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues of A: ", eigenvalues)

# Closed-loop eigenvalues
#K = doc.kalman_gain(A, B, np.eye(4), np.eye(1))
#A_cl = A - B @ K # Closed-loop system matrix is stable (A_cl: dynamics after a state feedback controller u is applied)
#eigenvalues_cl = np.linalg.eigvals(A_cl)
#print(eigenvalues_cl)

# Check if the system is controllable
ctrlb = doc.controllable(A, B)                      # For all outputs, use the complete B matrix.
print(f'Controllable via input? {ctrlb}')

# Can you even compute these metrics because the system is unstable and we simulate it with the nonlinear dynamics?
# When b-1 (equilibrium pendulum down), the system is stable and the metrics dont explode
# Check the Average controlability of each node
ac_nodes = doc.average_ctrb(A, T = 10, B = None) #when T = inf: matrix is singular error, bc A isnt stable
for i in range(len(ac_nodes)):
    print(f'AC of node {i}: {ac_nodes[i]}')
    # why is AC of node 0 = T?
    # AC values (nodes 1-3) are exponentially increasing, because the system is unstable

# Check the average controlability of the input
ac_input = doc.average_ctrb(A, T = 10, B = B)
print(f'AC of input: {ac_input[0]}')

# Cir ist hier nicht so interessant, weil wir das System nicht in eine bestimmte Richtung lenken wollen, sondern zu einem bestimmten Zustand
# Also the data needs to be normalized, also for AC?
# Check the cir of each node in response to the input
cir = doc.cumulative_impulse_response(A, B[:,0], T=10)
for i in range(len(cir)):
  print(f'CIR of node {i}: {cir[i]}')

# Check the cir of each node in response to a simulated intervention on the highest ac node
max_ac_node = np.argmax(ac_nodes)    # Highest average controllability node
cir = doc.cumulative_impulse_response(A, np.eye(4)[max_ac_node], T = 10)
for i in range(len(cir)):
  print(f'CIR of node {i} in response to simulated intervention on max ac node {max_ac_node}): {cir[i]}')


plot_states = np.array(states)
plot_inputs = np.array(inputs)
# Plotting the results
plt.figure(figsize=(12, 6))
# Plot for angular position (pendulum angle in radians)
plt.subplot(1, 3, 1)
angles_rad = plot_states[:, 2]
plt.plot(angles_rad, label="Pendulum Angle (radians)")
plt.title("Angle of the Pendulum")
plt.xlabel("Time Step")
plt.ylabel("Angle (radians)")
plt.grid(True)
plt.legend()

# Plot for cart position
plt.subplot(1, 3, 2)
plt.plot(plot_states[:, 0], label="Cart Position (meters)")
plt.title("Cart Position")
plt.xlabel("Time Step")
plt.ylabel("Position (meters)")
plt.grid(True)
plt.legend()

# Plot for cart input
plt.subplot(1, 3, 3)
plt.plot(plot_inputs, label="Inputs")
plt.title("Input")
plt.xlabel("Time Step")
plt.ylabel("Input")
plt.grid(True)
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()