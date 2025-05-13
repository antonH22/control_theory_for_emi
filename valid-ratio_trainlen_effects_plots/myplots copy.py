import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import pandas as pd
import numpy as np

def myplot_bar(x_values, y_values, yerr=None, color='midnightblue', log_scale=False, xlabel='X-Axis', ylabel='Y-Axis', title=None, legend_label=None, save_path=None):
    style='fivethirtyeight'
    figsize=(8, 6)
    marker='o'

    # Set font sizes
    label_fontsize = 30
    legend_fontsize = 12
    title_fontsize = 16 
    tick_fontsize = 25

    plt.figure(figsize=figsize)
    plt.style.use(style)  # Apply the style sheet
    plt.plot(x_values, y_values, color=color, marker=marker, linewidth=5, label=legend_label, alpha=0.5)
    plt.errorbar(x_values, y_values, yerr=yerr, label=None, fmt='none', color=color, capsize=10, elinewidth=3, capthick=2, alpha=1)  # Make error bars less visible
    
    #Customize the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if log_scale:
        plt.yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300)
    # Show the plot
    plt.show()

def myplot_bar_multiple(x_lists, y_lists, yerr_lists, colors, log_scale=False, xlabel='X-Axis', ylabel='Y-Axis', title=None, legend_labels=None, save_path=None):
    style = 'fivethirtyeight'
    figsize = (8, 6)
    edgecolors= 'k'
    
    plt.figure(figsize=figsize)
    plt.style.use(style)  # Apply the style sheet

    # Plot each set of x and y values with corresponding color
    for i in range(len(x_lists)):
        plt.scatter(x_lists[i], y_lists[i], color=colors[i], marker='o', s=100, zorder=3, edgecolors=edgecolors, label=legend_labels[i] if legend_labels else None)
        plt.plot(x_lists[i], y_lists[i], c=colors[i], alpha=0.5, linewidth=5) 
        plt.errorbar(x_lists[i], y_lists[i], yerr_lists[i], fmt='none', color=colors[i], capsize=10, elinewidth=3, capthick=2, alpha=1)  # Customize error bars

    # Customize the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
 
    if log_scale:
        plt.yscale('log')
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    # Show the plot
    plt.show()

def myplot_scatter(x_values, y_values, log_scale=False, color_rm='b', color_sc='deepskyblue', alpha=1, markersize=20, xlabel='X-Axis', ylabel='Y-Axis', title=None, legend_label=None,  window_size=None, save_path=None):
    style='fivethirtyeight'
    figsize=(8, 6)
    if not title: 
        title=f'{xlabel} vs. {ylabel}'
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.style.use(style)  # Apply the style sheet
    
    if window_size:
        plt.scatter(x_values, y_values, color=color_sc, alpha=alpha, label=legend_label, s=markersize)
        # Compute running mean
        x_sorted, y_sorted = zip(*sorted(zip(x_values, y_values)))
        df = pd.DataFrame({'x': x_sorted, 'y': y_sorted})
        df['running_mean'] = df['y'].rolling(window=window_size, min_periods=1).mean()
        # Plot running mean as a smooth line
        plt.plot(df['x'], df['running_mean'], color=color_rm, linestyle='-', linewidth=3, label=f'Running Mean {legend_label}')
    else:
        plt.scatter(x_values, y_values, color=color_rm, alpha=alpha, label=legend_label, s=markersize, edgecolors='k')
    # Customize the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    #plt.ylim(0.8, None)
    if log_scale:
        plt.yscale('log')
    plt.legend(markerscale=2)

    if save_path:
        plt.savefig(save_path, dpi=300)
    # Show the plot
    plt.show()

def myplot_scatter_compare(x_lists, y_lists, colors, colors_rm, markersize=20, alpha=1,xlabel='X-Axis', ylabel='Y-Axis', title=None, legend_labels=None, window_size=None, save_path=None):
    style = 'fivethirtyeight'
    figsize = (8, 6)
    edgecolors= None
    
    plt.figure(figsize=figsize)
    plt.style.use(style)

    # Plot each set of x and y values with corresponding color and running mean
    for i in range(len(x_lists)):
        if window_size:
            plt.scatter(x_lists[i], y_lists[i], color=colors[i], label=legend_labels[i], s=markersize, alpha=alpha,edgecolors=edgecolors)
            # Compute running mean
            x_sorted, y_sorted = zip(*sorted(zip(x_lists[i], y_lists[i])))
            df = pd.DataFrame({'x': x_sorted, 'y': y_sorted})
            df['running_mean'] = df['y'].rolling(window=window_size, min_periods=1).mean()
            # Plot running mean as a smooth line
            plt.plot(df['x'], df['running_mean'], color=colors_rm[i], linestyle='-', linewidth=3, label=f'Running Mean {legend_labels[i]}')
        else:
            plt.scatter(x_lists[i], y_lists[i], color=colors[i], label=legend_labels[i], s=markersize, alpha=alpha,edgecolors=edgecolors)

    # Customize the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title else f'{xlabel} vs. {ylabel}')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 3)

    # Use log scale for y-axis
    plt.legend(markerscale=2)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

# Plot the scatter plot
def myplot_scatter_compare_participants(x_lists, y_lists, colors, mean=None, std_error=None, markersize=80, alpha=1, xlabel='X-Axis', ylabel='Y-Axis', errorlabel=None, title=None, legend_labels=None, log_scale=False, save_path=None):
    style = 'fivethirtyeight'
    figsize = (12, 6)
    edgecolors= 'k'
    
    plt.figure(figsize=figsize)
    plt.style.use(style)

    # Set font sizes
    label_fontsize = 30
    legend_fontsize = 12
    title_fontsize = 16 
    tick_fontsize = 25
    
    # Iterate through each set of points
    for i, (x, y) in enumerate(zip(x_lists, y_lists)):
        plt.scatter(x, y, c=[colors[i]] * len(x), s=markersize, alpha=alpha, label=legend_labels[i] if legend_labels else None, edgecolors=edgecolors)
        plt.plot(x, y, c=colors[i], alpha=0.5, linewidth=4)  # Connecting line for each participant

    ratios = x_lists[0]
    ratios_reversed = ratios[::-1]
    # Plot mean and std deviation
    if mean is not None:
        plt.errorbar(ratios_reversed, mean, yerr=std_error, fmt='o-', color='black',linewidth=8, label=errorlabel, capsize=8, elinewidth=4, capthick=4, alpha=0.7)
    
    # Labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Labels and title with larger font
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    if title:
        plt.title(title, fontsize=title_fontsize)
    
    # Labels and title with larger font
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    if title:
        plt.title(title, fontsize=title_fontsize)
    
    # Legend with larger font
    #if legend_labels:
    #    plt.legend(title='Participants', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legend_fontsize)

    # Increase tick label size
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    plt.grid(True)

    if log_scale:
        plt.yscale('log')
    
    # Save the plot if save_path is provided
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()