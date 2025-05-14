import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt

### Plot inverted pendulum study: stabilizing the pendulum in the upright position (5.2, Figure 8)

# Load saved data
path = "pendulum_up.npz"
data = np.load(os.path.join("results_replicated", path))
plot_states = data['plot_states']
plot_inputs = data['plot_inputs']

# Plotting setup
style = 'fivethirtyeight'
figsize = (12, 6)
label_fontsize = 20
title_fontsize = 24
tick_fontsize = 20

# Create figure
fig, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'wspace': 0.4})
plt.style.use(style)

# Plot pendulum angle
axs[0].plot(plot_states[:, 2], c="blue", linewidth=2.5)
axs[0].set_title("a) Pendulum angle", fontsize=title_fontsize)
axs[0].set_xlabel("Time step", fontsize=label_fontsize)
axs[0].set_ylabel("Angle (radians)", fontsize=label_fontsize)
axs[0].tick_params(axis='both', labelsize=tick_fontsize)
axs[0].grid(True)

# Plot cart position
axs[1].plot(plot_states[:, 0], c="blue", linewidth=2.5)
axs[1].set_title("b) Cart position", fontsize=title_fontsize)
axs[1].set_xlabel("Time step", fontsize=label_fontsize)
axs[1].set_ylabel("Position (meters)", fontsize=label_fontsize)
axs[1].tick_params(axis='both', labelsize=tick_fontsize)
axs[1].grid(True)

# Plot input
axs[2].plot(plot_inputs, c="blue", linewidth=2.5)
axs[2].set_title("c) Input", fontsize=title_fontsize)
axs[2].set_xlabel("Time step", fontsize=label_fontsize)
axs[2].set_ylabel("Input", fontsize=label_fontsize)
axs[2].tick_params(axis='both', labelsize=tick_fontsize)
axs[2].grid(True)

plt.show()