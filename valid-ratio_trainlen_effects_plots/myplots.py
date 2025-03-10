import matplotlib.pyplot as plt

def myplot_bar(x_values, y_values, yerr=None, color='b', xlabel='X-Axis', ylabel='Y-Axis', title=None, legend_label=None, save_path=None):
    style='fivethirtyeight'
    figsize=(8, 6)
    marker='o'
    markersize=12
    capsize=5
    title=f'{xlabel} vs. {ylabel}'

    plt.figure(figsize=figsize)
    plt.style.use(style)  # Apply the style sheet
    plt.plot(x_values, y_values, color=color, marker=marker, markersize=markersize,linewidth=2, label=legend_label)
    plt.errorbar(x_values, y_values, yerr=yerr, fmt='none', color=color, capsize=capsize, elinewidth=1.5, capthick=1.5, alpha=0.3)  # Make error bars less visible
    
    #Customize the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)
    # Show the plot
    plt.show()

def myplot_bar_multiple(x_lists, y_lists, yerr_lists, colors, xlabel='X-Axis', ylabel='Y-Axis', title=None, legend_labels=None, save_path=None):
    style = 'fivethirtyeight'
    figsize = (8, 6)
    
    plt.figure(figsize=figsize)
    plt.style.use(style)  # Apply the style sheet

    # Plot each set of x and y values with corresponding color
    for i in range(len(x_lists)):
        plt.plot(x_lists[i], y_lists[i], color=colors[i], marker='o', markersize=12, linewidth=2, label=legend_labels[i] if legend_labels else None)
        if i < len(y_lists) and len(y_lists[i]) > 0:
            plt.errorbar(x_lists[i], y_lists[i], yerr_lists[i], fmt='none', color=colors[i], capsize=10, elinewidth=1, capthick=3, alpha=0.5)  # Customize error bars

    # Customize the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title else f'{xlabel} vs. {ylabel}')
    plt.grid(True)
    plt.legend()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    # Show the plot
    plt.show()

def myplot_scatter(x_values, y_values, color='b', xlabel='X-Axis', ylabel='Y-Axis', title=None, legend_label=None, save_path=None):
    style='fivethirtyeight'
    figsize=(8, 6)
    markersize = 20
    title=f'{xlabel} vs. {ylabel}'
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.style.use(style)  # Apply the style sheet
    plt.scatter(x_values, y_values, color=color, label=legend_label, s=markersize)
    # Customize the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)
    # Show the plot
    plt.show()