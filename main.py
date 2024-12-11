from ctrl import discrete_optimal_control as doc
from ctrl import utils
import numpy as np
import matplotlib.pyplot as plt

dataset_list = []

# Load, adjust and convert the first dataset
csv_file = ''# Path to the csv file
emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']
invert_columns = ['EMA_mood', 'EMA_confidence', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_satisfied', 'EMA_relaxed']
dataset1 = utils.csv_to_dataset(csv_file, emas, emis, invert_columns)

# Convert the dataset back to a csv file and save it
state_columns = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
input_columns = ['emi1', 'emi2', 'emi3', 'emi4', 'emi5', 'emi6', 'emi7', 'emi8']
output_file = 'dataset1.csv'
utils.dataset_to_csv(dataset1, state_columns, input_columns, output_file)

# Todo: Convert the other datasets and append them to dataset_list

dataset_list.append(dataset1)

X, U = dataset_list[0]['X'], dataset_list[0]['Inp']
# Shape of the EMA data: (time x EMA variables)
# Shape of the inputs: (time x input)

"""
# Copied from the notebook

# Fit the model
A, B, lmbda = utils.stable_ridge_regression(X, U)     # the lmbda output is the regularization parameter that is used to render A stable
n_variables = B.shape[0]        # for convenience
n_inputs = B.shape[1]           # for convenience

# Plot the circular graph to get a representation of the VAR model as a graph.
A_positive = np.clip(A, 0, None)        # We plot positive and negative connections separately in different colors
A_positive[A_positive < 0.3] = 0        # We omit weak connections to make the graph less cluttered.
A_negative = (-1) * np.clip(A, None, 0)
A_negative[A_negative < 0.3] = 0
utils.plot_circular_graph(A_positive, directed=True, edge_kwargs={'edge_color':'black'})
utils.plot_circular_graph(A_negative, directed=True, edge_kwargs={'edge_color':'red'})
plt.show()

# Plot data and inputs
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10,6), height_ratios=(15,4))
utils.colorplot_trajectory(X, ax=axes[0], aspect='auto', origin='lower')
axes[0].set_ylabel('EMA variables')
utils.colorplot_trajectory(U, ax=axes[1], aspect='auto', origin='lower')
axes[1].set_ylabel('inputs')
plt.show()

# Check controllability
for k in range(n_inputs):
    ctrlb = doc.controllable(A, B[:,k])             # For a single output, take the respective column in B.
    print(f'Controllable via input {k}? {ctrlb}')
ctrlb = doc.controllable(A, B)                      # For all outputs, use the complete B matrix.
print(f'Controllable via all inputs combined? {ctrlb}')

# Compute and plot average controllability
ac = doc.average_ctrb(A)
plt.barh(range(n_variables), ac, color='#01153E')
plt.ylabel('EMA variable')
plt.xlabel('average controllability')
plt.show()

# Compute and plot CIR
fig, axes = plt.subplots(1, n_inputs, figsize=(12,3))
for k in range(n_inputs):
    cir = doc.cumulative_impulse_response(A, B[:,k], T=100)          # T is the number of timesteps over which the impulse response will be accumulated. Choice depends on theoretical question.
    axes[k].barh(range(n_variables), cir, color='#9ecca6')
    axes[k].set(xlabel='impulse response', yticks=range(n_variables), title=f'input {k}')
axes[0].set_ylabel('EMA variable')
plt.tight_layout()
plt.show()

# Plot the CIR for the highest average controllability node
max_ac_node = np.argmax(ac)         # Highest average controllability node
print(max_ac_node)
cir = doc.cumulative_impulse_response(A, np.eye(n_variables)[max_ac_node], 100)        # Set B to a unit vector 
plt.barh(range(n_variables), cir, color='slategrey')
plt.xlabel('cum. impulse response')
plt.ylabel('EMA variable')
plt.show()

# Compute and plot optimal control
reference = -3 * np.ones_like(X)
rho = 1                     # rho=1 means that control input and trajectory deviation are weighted equally in the loss function.
Q = np.eye(n_variables)
R = np.eye(n_inputs) * rho
optimal_control = doc.tracking_optimal_control(A, B, Q, R, X, reference)
ax, colorbar = utils.colorplot_trajectory(optimal_control, origin='lower')
ax.set_ylabel('inputs')
plt.show()


# Compute and plot optimal control for the case of a single input
R = np.eye(n_variables) * rho
optimal_control = doc.tracking_optimal_control(A, np.eye(n_variables), Q, R, X, reference)
ax, colorbar = utils.colorplot_trajectory(optimal_control, origin='lower')
ax.set_ylabel('EMA variable')
plt.show()  

# Compute and plot optimal control energy
control_energy = (optimal_control**2).mean(axis=0)
plt.barh(range(n_variables), control_energy, color='#8C000F')
plt.xlabel('optimal control energy')
plt.ylabel('EMA variable')
plt.show()

# Plot optimal control energy sorted by average controllability
AC_order = np.argsort(ac)[::-1]
energy_sorted_by_AC = control_energy[AC_order]
line1, = plt.plot(range(n_variables), energy_sorted_by_AC, color='#01153E')
plt.xticks(range(n_variables), labels=(['high']+['']*(n_variables-2)+['low']))
plt.ylabel('optimal control energy')
plt.xlabel('AC')
plt.legend([line1], ['variables sorted by AC'])
plt.show()

# Look at notebook
rho = 100
R = np.eye(n_variables) * rho
Q = np.eye(n_variables)
optimal_control = doc.tracking_optimal_control(A, np.eye(n_variables), Q, R, X, reference)
control_energy = (optimal_control**2).mean(axis=0)
energy_sorted_by_AC = control_energy[AC_order]
line1, = plt.plot(range(n_variables), energy_sorted_by_AC, color='#01153E')
plt.xticks(range(n_variables), labels=(['high']+['']*(n_variables-2)+['low']))
plt.ylabel('optimal control energy')
plt.xlabel('AC')
plt.legend([line1], ['variables sorted by AC'])
plt.show()

# Look at notebook
from ctrl import control_strategies as strategies

admissible_inputs = np.eye(n_inputs)
target_state = np.ones(n_variables) * (-3)
u = strategies.optimal_control_strategy(X[:50], U[:50], target_state, admissible_inputs, 80, online=True)
print(u)

# Look at notebook
time_horizon = np.arange(50, 100)
u_sequence = np.zeros((len(time_horizon), n_inputs))
for t, timestep in enumerate(time_horizon):
    u_sequence[t] = strategies.max_ac_strategy(X[:timestep], U[:timestep], admissible_inputs, online=True)
ax, _ = utils.colorplot_trajectory(u_sequence)
ax.set(xticks=np.arange(0, len(time_horizon), 10), xticklabels=time_horizon[::10], yticks=np.arange(n_inputs), ylabel='input')
plt.show()
"""
