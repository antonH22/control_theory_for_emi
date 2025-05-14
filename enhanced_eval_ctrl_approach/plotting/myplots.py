import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import pandas as pd
import numpy as np

def myplot_bar(x_values, y_values, yerr=None, color='midnightblue', log_scale=False, xlabel=None, ylabel=None, title=None, legend_label=None, save_path=None, ylim = None, counts=None):
    style = 'fivethirtyeight'
    figsize = (10, 6)
    
    plt.figure(figsize=figsize)
    plt.style.use(style)

    # Set font sizes
    label_fontsize = 30
    title_fontsize = 16
    tick_fontsize = 25

    plt.plot(x_values, y_values, 'o', color=color, markersize=10)
    plt.errorbar(x_values, y_values, yerr=yerr,markersize=10, color=color,linewidth=8, label=None, capsize=8, elinewidth=4, capthick=4, alpha=0.7)
    if counts is not None:
        # Plot umber of points at each average point
        for i in range(len(x_values)):
            plt.text(x_values[i], y_values[i] + yerr[i], f"n={counts[i]}", 
                ha='center', fontsize=22, color='black')
            
    # Labels and title with larger font
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    if title:
        plt.title(title, fontsize=title_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if log_scale:
        plt.yscale('log')

    if ylim:
        plt.ylim(None, ylim)

    # Add tight_layout to prevent cutoff
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    # Show the plot
    plt.show()

def myplot_bar_multiple(x_lists, y_lists, yerr_lists, colors, log_scale=False, xlabel='X-Axis', ylabel='Y-Axis', title=None, legend_labels=None, save_path=None):
    style = 'fivethirtyeight'
    figsize = (10, 6)
    edgecolors= 'k'

    # Set font sizes
    label_fontsize = 16
    legend_fontsize = 16
    title_fontsize = 12 
    tick_fontsize = 14
    
    plt.figure(figsize=figsize)
    plt.style.use(style)  # Apply the style sheet

    # Plot each set of x and y values with corresponding color
    for i in range(len(x_lists)):
        plt.scatter(x_lists[i], y_lists[i], color=colors[i], marker='o', s=100, zorder=3, edgecolors=edgecolors, label=legend_labels[i] if legend_labels else None)
        plt.plot(x_lists[i], y_lists[i], c=colors[i], alpha=0.5, linewidth=5) 
        plt.errorbar(x_lists[i], y_lists[i], yerr_lists[i], fmt='none', color=colors[i], capsize=10, elinewidth=3, capthick=2, alpha=1)  # Customize error bars

    # Customize the plot
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=legend_fontsize)

    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
 
    if log_scale:
        plt.yscale('log')
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    # Show the plot
    plt.show()

def myplot_scatter(x_values, y_values, rm_only=False, color_rm='b', color_sc='deepskyblue', alpha=1, markersize=20, xlabel='X-Axis', ylabel='Y-Axis', title=None, legend_label=None,  window_size=None, save_path=None):
    style='fivethirtyeight'
    if rm_only:
        figsize = (12, 6)
    else:
        figsize = (10,6)

    # Set font sizes
    label_fontsize = 30
    legend_fontsize = 30
    title_fontsize = 16 
    tick_fontsize = 25
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.style.use(style)  # Apply the style sheet
    
    if window_size:
        if not rm_only:
            plt.scatter(x_values, y_values, color=color_sc, alpha=alpha, label=legend_label, s=markersize)

        # Compute running mean
        x_sorted, y_sorted = zip(*sorted(zip(x_values, y_values)))
        df = pd.DataFrame({'x': x_sorted, 'y': y_sorted})
        df['running_mean'] = df['y'].rolling(window=window_size, min_periods=window_size).mean()
        # Plot running mean as a smooth line
        plt.plot(df['x'], df['running_mean'], color=color_rm, linestyle='-', linewidth=3, label=None)
    else:
        plt.scatter(x_values, y_values, color=color_sc, alpha=alpha, label=legend_label, s=markersize)
    # Customize the plot
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title)
    plt.grid(True)
    if not rm_only:
        plt.legend(markerscale=2, fontsize=legend_fontsize, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    # Show the plot
    plt.show()

def myplot_scatter_compare(x_lists, y_lists, colors, colors_rm, markersize=20, alpha=1,xlabel='X-Axis', ylabel='Y-Axis', title=None, legend_labels=None, window_size=None, save_path=None, rm_only=False):
    style = 'fivethirtyeight'
    edgecolors= None
    figsize = (12,6)
    plt.figure(figsize=figsize)
    plt.style.use(style)

     # Set font sizes
    label_fontsize = 30
    legend_fontsize = 30
    title_fontsize = 16 
    tick_fontsize = 25

    # Plot each set of x and y values with corresponding color and running mean
    for i in range(len(x_lists)):
        if window_size:
            if not rm_only:
                plt.scatter(x_lists[i], y_lists[i], color=colors[i], label=legend_labels[i], s=markersize, alpha=alpha,edgecolors=edgecolors)
            # Compute running mean
            x_sorted, y_sorted = zip(*sorted(zip(x_lists[i], y_lists[i])))
            df = pd.DataFrame({'x': x_sorted, 'y': y_sorted})
            df['running_mean'] = df['y'].rolling(window=window_size, min_periods=window_size, center=True).mean()
            # Plot running mean as a smooth line
            plt.plot(df['x'], df['running_mean'], color=colors_rm[i], linestyle='-', label=None, linewidth=3)
        else:
            plt.scatter(x_lists[i], y_lists[i], color=colors[i], label=legend_labels[i], s=markersize, alpha=alpha,edgecolors=edgecolors)

    # Customize the plot
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title)
    plt.grid(True)
    if not rm_only:
        plt.legend(markerscale=2, fontsize=legend_fontsize, loc='upper right')

    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.tight_layout()
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

# Plot the scatter plot
def myplot_scatter_compare_participants(x_lists, y_lists, colors, mean=None, std_error=None, markersize=80, alpha=1, xlabel='X-Axis', ylabel='Y-Axis', errorlabel=None, title=None, legend_labels=None, log_scale=False, save_path=None):
    save_path = None
    
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
        plt.errorbar(ratios_reversed, mean[::-1], yerr=std_error[::-1], fmt='o-', color='black',linewidth=8, label=errorlabel, capsize=8, elinewidth=4, capthick=4, alpha=0.7)
    
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