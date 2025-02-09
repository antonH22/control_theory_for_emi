import itertools as it
import os
import warnings
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size, make_axes_locatable
from scipy import stats
from scipy.io import loadmat

import pandas as pd
### Data utils

def load_data(ema_range=3, language='english'):
    if ema_range==6:
        data_dir = 'D:/ZI Mannheim/Control Theory/data_EMIcompass/range_0_to_6'
    elif ema_range==3:
        data_dir = 'D:/ZI Mannheim/Control Theory/data_EMIcompass/range_-3_to_3'
    elif ema_range==-3:
        data_dir = 'D:/ZI Mannheim/Control Theory/data_EMIcompass/range_-3_to_3_wrong'
    else:
        raise ValueError('range must be 3 or 6')
    data = []
    for data_path in os.listdir(data_dir):
        if data_path.endswith('.mat'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dataset = loadmat(os.path.join(data_dir, data_path))
            if language=='english':
                labels = np.array(['anxious', 'cheerful*', 'down', 'irritated', 'relaxed*',
                                'uncomfortable', 'calm*', 'energetic*', 'hungry', 'choose alone*',
                                'rather company', 'soc. unpleasant', 'soc. apprec.*',
                                'agreeable*', 'act. unpleasant'])
            elif language=='german':
                labels = np.array(['ängstlich', 'gut gelaunt*', 'traurig', 'irritiert', 'entspannt*',
                                'unbehaglich', 'ruhig*', 'energetisch*', 'hungrig', 'lieber allein*',
                                'lieber in Gesellschaft', 'unangenehme Gesellschaft', 'Wertschätzung*',
                                'angenehm*', 'unangenehme Aktivität'])
            
            dataset['Xlabs'] = labels
            dataset['Ilabs'] = ['EMI-I', 'EMI-II', 'EMI-III', 'with company']
            dataset['Igroups'] = np.array([3,3,3,3,3,3,2,2,2,1,1,1,1,0,0])
            dataset['Filename'] = data_path
            dataset['B'] = dataset.pop('C')
            data.append(dataset)
    return data


def generate_dataset(data: np.ndarray, inputs: np.ndarray, 
                     data_labels: Optional[List]=None, input_labels: Optional[List]=None):
    assert data.shape[0] == inputs.shape[0]
    if data_labels is not None:
        assert len(data_labels) == data.shape[1]
    else:
        data_labels = [f'item_{k}' for k in range(data.shape[1])]
    if input_labels is not None:
        assert len(input_labels) == inputs.shape[1]
    else:
        input_labels = [f'input_{k}' for k in range(inputs.shape[1])]
    dataset = dict(('X', data),
                   ('Inp', inputs),
                   ('Xlabs', data_labels),
                   ('Ilabs', input_labels)
                   )
    return dataset


def generate_random_dataset(N: int, T: int, seed: int=None):
    ''' Generate a dataset of trajectories drawn from a random model. 
        The format of the dataset is compatible with the original EMIcompass datasets. '''
    
    dist = np.load('model_distribution.npy', allow_pickle=True).item()
    A_shape = (dist['n_features'], dist['n_features'])
    B_shape = (dist['n_features'], dist['n_control'])
    A_dist = stats.multivariate_normal(dist['A_mean'], np.diag(dist['A_var']))
    B_dist = stats.multivariate_normal(dist['B_mean'], np.diag(dist['B_var']))
    data = []
    for n in range(N):
        A = A_dist.rvs(random_state=seed).reshape(A_shape)
        maxeig = np.max(np.abs(np.linalg.eig(A)[0]))
        B = B_dist.rvs(random_state=seed).reshape(B_shape)
        if maxeig >= 1:
            A /= maxeig + 0.001
            B /= maxeig + 0.001
        control_idx = np.random.default_rng(seed).integers(0, dist['n_control'], T//2)
        control_vec = np.zeros((T, dist['n_control']))
        control_vec[np.arange(T//2)*2, control_idx] = 1
        traj = np.zeros((T, dist['n_features']))
        traj[0] = np.random.default_rng(seed).integers(-3, 4, dist['n_features'])
        for t in range(1,T):
            traj[t] = A @ traj[t-1] + B @ control_vec[t-1]
        traj = np.clip(np.round(traj), -3, 3)
        A_est, B_est, _ = stable_ridge_regression(traj, control_vec)     
        data.append({'X':traj, 'Inp':control_vec, 'Xnan':traj,
                     'Xlabs':[f'feature_{i}' for i in range(dist['n_features'])],
                     'Ilabs':[f'input_{i}' for i in range(dist['n_control'])],
                     'Aoriginal': A, 'Boriginal': B,
                     'A': A_est, 'B': B_est})
        if seed is not None:
            seed += 1

    return data


### Algebra utils

def stable_ridge_regression(data, inputs, intercept=False, accepted_eigval_threshold=1, max_regularization=10.5):
    ''' Performs ridge regression for model X[1:] = A@X[:-1] + B@Inp[:-1]. 
        Regularization lambda is chosen as small as possible such that A is stable.
        Returns A, B, lambda. If intercept, returns A, B, intercept, lambda. '''
    if inputs is None:
        inputs = np.zeros((data.shape[0], 0))
    combined_predictor = np.hstack((data, inputs))[:-1]
    target = data[1:]
    if intercept:
        combined_predictor = np.hstack((combined_predictor, np.ones((combined_predictor.shape[0], 1))))
        
    # Remove rows in predictor with NaN values in predictor or target and the corresponding target rows (next time step)
    nan_mask = ~np.isnan(combined_predictor).any(axis=1) & ~np.isnan(target).any(axis=1)
    combined_predictor = combined_predictor[nan_mask]
    target = target[nan_mask]

    size = combined_predictor.shape[1]
    for lmbda in np.arange(0,max_regularization,0.001):
        moment_matrix = combined_predictor.T @ combined_predictor + lmbda * np.eye(size)
        regression_weights = np.linalg.pinv(moment_matrix) @ combined_predictor.T @ target
        A = regression_weights[:data.shape[1]]
        B = regression_weights[data.shape[1]:]
        if np.abs(np.linalg.eig(A)[0]).max() < accepted_eigval_threshold:
            break
    if intercept:
        c = B[-1]
        B = B[:-1]
        return A.T, B.T, lmbda, c
    else:
        return A.T, B.T, lmbda
    
def cohens_d(x: np.array, y: np.array, paired=False, correct=False):
    axis=None
    nx = (~np.isnan(x)).sum()
    ny = (~np.isnan(y)).sum()
    if paired:
        d = np.nanmean(x - y, axis) / np.nanstd(x - y, ddof=1)
    else:
        dof = nx + ny - 2
        d = ((np.nanmean(x, axis) - np.nanmean(y, axis)) / 
            np.sqrt(((nx-1)*np.nanstd(x, ddof=1) ** 2 + (ny-1)*np.nanstd(y, ddof=1) ** 2) / dof))
        if correct:
            d *= (1 - 3 / (4*(nx + ny) - 9))
    return d

def trace(M):
    return np.diag(M).sum()


def round_to_vector(M, Z, p_norm=2):
    ''' Rounds the rows of M to the nearest row of round_to, according to ||M-round_to||^p-norm '''
    distance = np.sum(np.abs(M[np.newaxis] - Z[:, np.newaxis])**p_norm, axis=-1)**(1/p_norm)
    argmin_distance = np.argmin(distance, axis=0)
    return Z[argmin_distance]



### Plotting utils
       
def colorplot_trajectory(trajectory, labels=None, title=None, ax=None, cax_settings={}, **kwargs):
    ''' plots trajectory as heatmap (x axis is time, y axis is trajectory dimensions) '''
    if ax is None:
        fig, ax = plt.subplots()
    image = ax.imshow(trajectory.T, **kwargs)
    cax = steal_space_from_axis(ax, **cax_settings)

    cbar = plt.colorbar(image, cax=cax)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('time steps')
    T = trajectory.shape[0]
    ax.set_xticks(np.linspace(0, T-(T%3), 4))
    if labels is not None:
        ax.set_yticks(range(trajectory.shape[1]))
        ax.set_yticklabels(labels)
        ax.tick_params(left=False)

    return ax, cax


def plot_circular_graph(weights, directed=False, labels=None, 
                        max_edge_width=3, max_edge_rad=0.9,
                        node_kwargs={}, edge_kwargs={}, label_kwargs={}, ax=None):
    ''' plots a graph with nodes arranged around circle, with edges that represent weights '''
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = np.arange(weights.shape[0])
    scaled_weights = weights / weights.max()
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
        if not (weights.T == weights).all():
            raise ValueError('weights matrix must be symmetric for undirected graph')
    size = weights.shape[0]
    node_container = list(range(size))
    graph.add_nodes_from(node_container)
    layout = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos=layout, ax=ax, **node_kwargs)

    edge_container = []
    edge_widths = []
    edge_styles = []
    for x in range(size):
        if directed:
            loopthrough = range(size)
        else:
            loopthrough = range(x)
        for y in loopthrough:
            if weights[x,y] > 0:
                edge_container.append((x,y))
                edge_widths.append(scaled_weights[x,y]*max_edge_width)
                edge_rad = -max_edge_rad*np.abs(y-x) / (size/2 + 1) + max_edge_rad
                helper_graph = graph.copy()
                width = scaled_weights[x,y]*max_edge_width
                style = f'arc3,rad={edge_rad}'
                helper_graph.add_edge(x, y)
                nx.draw_networkx_edges(helper_graph, pos=layout, ax=ax, width=width, arrows=True,
                           connectionstyle=style, **edge_kwargs)

    graph.add_edges_from(edge_container)
    # bboxes = []
    for node, pos in layout.items():
        if pos[0]==0:
            pos[0] = 1e-6
        h_align = 'left' if pos[0]>=0 else 'right'
        label = f'   {labels[node]}' if pos[0]>=0 else f'{labels[node]}   '
        angle = np.arctan(pos[1]/pos[0]) / (0.5*np.pi) * 90
        node_labels = nx.draw_networkx_labels(graph, pos={node: pos}, labels={node: label}, 
                                              horizontalalignment=h_align,
                                              ax=ax, **label_kwargs)
        node_labels[node].set_rotation_mode('anchor')
        node_labels[node].set_rotation(angle)
        # renderer = plt.gcf().canvas.get_renderer()
        # bbox = node_labels[node].get_window_extent(renderer)
        # matplotlib.patches.draw_bbox(bbox, renderer)
        # bboxes.append(bbox)
        # bbox = bbox.transformed(ax.transData.inverted())                
        # ax.update_datalim(bbox.corners())
        # ax.autoscale_view()    

    return ax


def get_axis_size(ax: mpl.axes.Axes):
    sp = ax.figure.subplotpars
    fig_width, fig_height = ax.figure.get_size_inches() 
    ax_width = fig_width * (sp.right - sp.left)
    ax_height = fig_height * (sp.top - sp.bottom)
    return ax_width, ax_height


def set_axis_size(ax: mpl.axes.Axes, width: float|None=None, height: float|None=None):
    # orig_width, orig_height = ax.get_figure().get_size_inches()
    # sp = ax.figure.subplotpars
    # if width is None:
    #     fig_width = orig_width
    # else:
    #     fig_width = float(width) / (sp.right - sp.left)
    # if height is None:
    #     fig_height = orig_height
    # else:
    #     fig_height = float(height) / (sp.top - sp.bottom)
    # ax.figure.set_size_inches(fig_width, fig_height)

    horizontal = [Size.Fixed(0), Size.Fixed(width)]
    vertical = [Size.Fixed(0), Size.Fixed(height)]
    divider = Divider(ax.get_figure(), (0, 0, 1, 1), horizontal, vertical, aspect=False)
    ax.set_axes_locator(divider.new_locator(1,1))


def steal_space_from_axis(ax, side='right', size="5%", pad=0.1):
    divider = make_axes_locatable(ax)
    new_axis = divider.append_axes(side, size=size, pad=pad)
    return new_axis    


def fixed_size_plot(width_inches: float, height_inches: float, pad_inches: float=1.):
    fig = plt.figure(figsize=(width_inches+4*pad_inches, height_inches+4*pad_inches))

    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(pad_inches), Size.Fixed(height_inches)]
    v = [Size.Fixed(pad_inches), Size.Fixed(width_inches)]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(),
                    axes_locator=divider.new_locator(nx=1, ny=1))
    return fig, ax

def csv_to_dataset(file_path, state_columns, input_columns, invert_columns):
    ''' Load a CSV file, adjust data and convert it to a datset (dictionary). '''
    csv_df = pd.read_csv(file_path)
    required_columns = state_columns + input_columns
    csv_df = csv_df[required_columns]
    
    # Delete empty rows in the beginning
    first_non_na_index = csv_df.notna().all(axis=1).idxmax()
    csv_df = csv_df.iloc[first_non_na_index:].reset_index(drop=True)
    
    # Split into state and input variables (ndarrays)
    X = csv_df[state_columns].values
    Inp = csv_df[input_columns].values

    # Regularize state variables to [-3, 3]
    X -= 4
    
    # Invert columns if necessary
    for column in invert_columns:
        idx = state_columns.index(column)
        X[:, idx] = -X[:, idx]
    
    # Return the dataset as a dictionary
    return {'X': X, 'Inp': Inp}


def dataset_to_csv(dataset, state_columns, input_columns, output_file):
    '''' Convert a dataset (dictionary) back to a CSV file. '''
    # Extract the state and input matrices from the dataset
    X = dataset['X']
    Inp = dataset['Inp']
    
    # Concatenate the state and input arrays horizontally
    data = np.hstack((X, Inp))

    columns = state_columns + input_columns
    df = pd.DataFrame(data, columns=columns)
    
    df.to_csv(output_file, index=False)