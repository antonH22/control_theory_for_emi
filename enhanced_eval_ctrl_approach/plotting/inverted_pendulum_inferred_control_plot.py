import numpy as np
import matplotlib.pyplot as plt
import os

### Plot inferred control vs. true control in the down position (5.2, Figure 9)

data1 = np.load(os.path.join("results_replicated", "pendulum_down_inferred.npz"))
try:
    data2 = np.load(os.path.join("results_replicated", "pendulum_down.npz"))
    data2_available = True
except FileNotFoundError:
    data2_available = False

# Extract data from both files
plot_states1 = data1['plot_states']
plot_inputs1 = data1['plot_inputs']
if data2_available:
    plot_states2 = data2['plot_states']
    plot_inputs2 = data2['plot_inputs']

# Plotting setup
style = 'fivethirtyeight'
figsize = (12, 6)
label_fontsize = 20
title_fontsize = 24
tick_fontsize = 20

# Create figure
fig, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'wspace': 0.3})
plt.style.use(style)

# Plotting function that works even when data2 is not available
def plot_comparison(ax, data1, data2=None, ylabel="", title="", color1="crimson", color2="blue", legend=False):
    label1 = 'inferred control' if legend else None
    ax.plot(data1, c=color1, label=label1, linewidth=2.5)
    if data2 is not None:
        label2 = 'derived control' if legend else None
        ax.plot(data2, c=color2, label=label2, linewidth=3, linestyle="--")
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Time Step", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(True)
    if legend:
        ax.legend(fontsize=label_fontsize-4)

# Plot trajectories
plot_comparison(axs[0], 
               plot_states1[:, 2], 
               plot_states2[:, 2] if data2_available else None,
               "Angle (radians)", 
               "a) Pendulum angle",
               legend=True)

plot_comparison(axs[1], 
               plot_states1[:, 0], 
               plot_states2[:, 0] if data2_available else None,
               "Position (meters)", 
               "b) Cart position")

plot_comparison(axs[2], 
               plot_inputs1, 
               plot_inputs2 if data2_available else None,
               "Input", 
               "c) Input")

plt.show()