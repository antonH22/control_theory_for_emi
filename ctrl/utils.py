#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:32:57 2022

@author: janik
"""
import numpy as np
from scipy import stats
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import itertools as it
import warnings
import os

class PrincipalComponents:
    ''' Performs PCA and can project vectors into Principal Component Space '''
    
    def __init__(self, X):
        self.data_mean = X.mean(axis=0)
        self.data_std = X.std(axis=0)
        X = (X - self.data_mean) / self.data_std
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        order = np.argsort(eigvals)[::-1]
        self.princomps = eigvecs[:, order]
        self.explained_var = eigvals[order]        
        
    def project(self, data, n_dims=None):
        ''' Projects data into principal component space up to n_dims dimensions '''
        y = (data - self.data_mean) / self.data_std
        projected = y @ self.princomps[:, :n_dims]
        return projected
    
    def plot(self, feature_names, n_dims=None):
        ''' Plots a summary of the PCA results '''
        fig, axes = plt.subplots(1,2)        
        n_features = self.princomps.shape[0]
        if not n_dims:
            n_dims = n_features
        im = axes[0].matshow(self.princomps, cmap='RdYlGn')
        axes[0].set_xticks(range(n_dims))
        axes[0].set_xticklabels(range(n_dims), rotation=45)
        axes[0].set_xlabel('components')
        axes[0].set_yticks(range(n_dims))
        axes[0].set_yticklabels(feature_names)
        axes[0].set_title('Factor loadings')
        plt.colorbar(im, ax=axes[0])
        axes[1].plot(self.explained_var / self.explained_var.sum())
        # axes[1].plot(self.explained_var.cumsum() / self.explained_var.sum())
        # axes[1].legend(['incremental', 'cumulative'])
        axes[1].set_title('Explained % variance')
        axes[1].set_xticks(range(n_dims))
        axes[1].set_xticklabels(range(1, n_dims+1))
        axes[1].set_xlabel('# components')

def stable_ridge_regression(X, Inp, Y=None):
    ''' Performs ridge regression for model X[1:] = A@X[:-1] + B@Inp[:-1]. 
        Regularization lambda is chosen as small as possible such that A is stable.
        Returns A, B, lambda. '''
    if Y is None:
        combined_predictor = np.hstack((X, Inp))[:-1]
        target = X[1:]
    else:
        combined_predictor = np.hstack((X, Inp))
        target = Y
    size = combined_predictor.shape[1]
    for lmbda in np.arange(0,10.5,0.01):
        moment_matrix = combined_predictor.T @ combined_predictor + lmbda * np.eye(size)
        regression_weights = np.linalg.pinv(moment_matrix) @ combined_predictor.T @ target
        A = regression_weights[:X.shape[1]]
        B = regression_weights[X.shape[1]:]
        if np.abs(np.linalg.eig(A)[0]).max() < 1:
            break

    return A.T, B.T, lmbda

def trace(M):
    return np.diag(M).sum()

def partial_corr(M):
    '''
    Calculates all partial correlations between rows of M,
    partialing out all other rows of M
    '''
    cov = np.cov(M)
    prec = np.linalg.inv(cov)
    pcorr = np.zeros_like(prec)
    for i, j in it.product(range(pcorr.shape[0]), range(pcorr.shape[1])):
        pcorr[i,j] = - prec[i,j] / np.sqrt(prec[i,i]*prec[j,j])
    return pcorr

def pearson_ci(data, ci=95):
    ''' Calculates pearson correlation coefficient of data (across axis 0) and
        confidence interval of correlation coefficient. '''
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    N = data.shape[1]
    probit = stats.norm.ppf((ci/100+1)/2)
    correl = np.corrcoef(data)
    fisher = np.arctanh(correl)
    lower_bound = np.tanh(fisher - probit/np.sqrt(N-3))
    upper_bound = np.tanh(fisher + probit/np.sqrt(N-3))
    return correl, lower_bound, upper_bound

def zscore(M, axis=None):
    result = (M - M.mean(axis=axis, keepdims=True)) / M.std(axis=axis, keepdims=True)
    return result

def interpolate(M, steps=100):
    '''  Linear interpolation at evenly spaced grid along rows of M '''
    rel_coord = np.linspace(0, steps-1, len(M))
    return np.interp(range(steps), rel_coord, M)

def finite_differences(data, timestamps, derivative_order=1, grid=[-1,0,1]):
    ''' Calculate finite difference approximation of the derivative of data 
        where <timestamps> defines the times when rows of data have been recorded. '''
    grid = np.array(grid)
    T = data.shape[0]
    timesteps = np.arange(-np.min(np.min(grid),0), T-np.max(np.max(grid),0))
    deriv = np.zeros((len(timesteps), data.shape[1]))
    for j,t in enumerate(timesteps):
        grid_matrix = np.vstack([(timestamps[grid+t] - timestamps[t])**d for d in range(len(grid))])
        coeff = np.linalg.inv(grid_matrix)[:,derivative_order] * np.math.factorial(derivative_order)
        deriv[j] = data[t+grid].T @ coeff
    return deriv, timesteps

def load_data():
    data_dir = 'D:/ZI Mannheim/Control Theory/data_EMIcompass/range_-3_to_3'
    data = []
    for data_path in os.listdir(data_dir):
        if data_path.endswith('.mat'):
            dataset = loadmat(os.path.join(data_dir, data_path))
            labels = np.array(['anxious', 'cheerful*', 'down', 'irritated', 'relaxed*',
                               'uncomfortable', 'calm*', 'energetic*', 'hungry', 'choose alone*',
                               'rather company', 'soc. unpleasant', 'soc. apprec.*',
                               'agreeable*', 'act. unpleasant'])
            dataset['Xlabs'] = labels
            dataset['Ilabs'] = ['EMI-I', 'EMI-II', 'EMI-III', 'with company']
            dataset['Igroups'] = np.array([3,3,3,3,3,3,2,2,2,1,1,1,1,0,0])
            dataset['Filename'] = data_path
            data.append(dataset)
    return data

def bars(data: np.ndarray, *args, horizontal: bool=False, ax=None, **kwargs):
    ''' Plot bar chart. Axis 1 of data defines the x axis, while axis 0 defines individual
    neighbouring bars. Args and kwargs are passed to pyplot.bar.'''
    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(data, tuple) or isinstance(data, list):
        data = np.array(data)
    N = data.shape[0]
    if data.ndim==1:
        data = data[:, np.newaxis]
    if 'width' in kwargs.keys():
        total_width = kwargs['width']
        kwargs.pop('width')
    else:
        total_width = 0.8
    if 'yerr' in kwargs.keys():
        yerr = kwargs['yerr']
    else:
        yerr = None
    if 'color' in kwargs.keys():   
        color = kwargs['color']
    else:
        color = None   
    
    individual_width = total_width/N
    barsize = individual_width
    offsets = -0.5*total_width + 0.5*individual_width + np.linspace(0, (N-1)*individual_width, N)
    
    for j in range(N):
        if yerr is not None:
            kwargs['yerr'] = yerr[j]
        if color is not None:
            kwargs['color'] = color[j]
        x = np.arange(len(data[j])) + offsets[j]
        if horizontal:
            ax.barh(x, data[j], barsize, *args, **kwargs)
        else:
            ax.bar(x, data[j], barsize, *args, **kwargs)
    
    return ax

def plot_trajectory_plane(*trajectories, legend=None, title=None, ax=None):
    ''' plots first 2 dimensions of trajectories into a plane (without time axis)'''
    colors = ['blue','orange','green','red','purple','cyan']
    if ax is None:
        fig, ax = plt.subplots()
    line_handlers = []
    for j, traj in enumerate(trajectories):
        hdlr, = ax.plot(traj[:,0], traj[:,1], color=colors[j])
        ax.plot(*traj[0], marker='s', fillstyle='none', color=colors[j])
        ax.plot(*traj[-1], marker='o', color=colors[j])
        line_handlers.append(hdlr)
    if title is not None:
        ax.set_title(title)
    if legend is not None:
        ax.legend(line_handlers, legend)
        
def plot_trajectories(*trajectories, legend=None, title=None, ax=None):
    ''' plots trajectories into same plots (x axis is time) '''
    if ax is None:
        fig, ax = plt.subplots()
    for j, traj in enumerate(trajectories):
        ax.plot(traj)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('time')
    if legend is not None:
        ax.legend(legend)
        
def colorplot_trajectory(trajectory, labels=None, title=None, ax=None, **kwargs):
    ''' plots trajectory as heatmap (x axis is time, y axis is trajectory dimensions) '''
    if ax is None:
        fig, ax = plt.subplots()
    image = ax.imshow(trajectory.T, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(image, cax=cax)
    # plt.colorbar(image, ax=ax)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('time steps')
    # ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    T = trajectory.shape[0]
    ax.set_xticks(np.linspace(0, T-(T%3), 4))
    if labels is not None:
        ax.set_yticks(range(trajectory.shape[1]))
        ax.set_yticklabels(labels)
        ax.tick_params(left=False)

    return ax, cbar
        
def plot_correlation(X, labels, title=None, ax=None):
    ''' plots heatmap of correlation coefficients '''
    if ax is None:
        fig, ax = plt.subplots()
    M = np.corrcoef(X.T)
    image = ax.matshow(M)
    plt.colorbar(image, ax=ax)
    ax.set_xticks(range(M.shape[0]))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(range(M.shape[1]))
    ax.set_yticklabels(labels)
    if title is not None:
        ax.set_title(title)

def plot_circular_graph(weights, directed=False, labels=None, 
                        max_edge_width=3, max_edge_rad=0.9,
                        node_kwargs={}, edge_kwargs={}, label_kwargs={}, ax=None):
    ''' plots a graph with nodes arranged around circle, with edges that represent weights '''
    if ax is None:
        ax = plt.gca()
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
    

def plot_prediction_against_data(X, U, system, dims, pca=False, axes=None):
    ''' One subplot per dimension of X. dims specifies (number of) dimensios to plot. 
        Predictions ar calculated according to system. If pca, X and predictions
        are projected into principal component space of X. '''
    if isinstance(dims, int):
        dims = np.arange(dims)
    elif isinstance(dims, list):
        dims = np.array(dims)
    if axes is None:
        fig, axes = plt.subplots(len(dims), 1, sharex=True, sharey=True)
    prediction = system.evolve(X.shape[0], X[0], U)
    if pca:
        pca = PrincipalComponents(X)
        X = pca.project(X)
        prediction = pca.project(prediction)
    for i, d in enumerate(dims):
        axes[i].plot(X[:,d])
        axes[i].plot(prediction[:,d])
        plt.legend(['true', 'model'])


def plot_regression(x, y, ax=None, scatter_kwargs={}, line_kwargs={}, 
                    nan_policy='propagate', test_alternative='two-sided'):
    ''' Scatter plots x against y and fits affine-linear curve. 
        Returns axis and stats.PearsonResult '''
    if nan_policy=='omit':
        mask = (~np.isnan(x)) * (~np.isnan(y))
        x = x[mask]
        y = y[mask]
    a, b = np.polyfit(x, y, deg=1)
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x, y, **scatter_kwargs)
    span = np.array(ax.get_xlim())
    ax.plot(span, span*a + b, **line_kwargs)
    corr = stats.pearsonr(x, y, alternative=test_alternative)
    return ax, corr

def get_axis_size(ax: mpl.axes.Axes):
    sp = ax.figure.subplotpars
    fig_width, fig_height = ax.figure.get_size_inches() 
    ax_width = fig_width * (sp.right - sp.left)
    ax_height = fig_height * (sp.top - sp.bottom)
    return ax_width, ax_height

def set_axis_size(ax: mpl.axes.Axes, width: float, height: float):
    sp = ax.figure.subplotpars
    fig_width = float(width) / (sp.right - sp.left)
    fig_height = float(height) / (sp.top - sp.bottom)
    ax.figure.set_size_inches(fig_width, fig_height)