import numpy as np

import ctrl.discrete_optimal_control as doc
from ctrl import utils


def optimal_control_strategy(data, inputs, target_state, admissible_inputs, rho, online=True):

    A, B, _ = utils.stable_ridge_regression(data, inputs)
    n_items = data.shape[1]
    n_inputs = inputs.shape[1]
    Q = np.eye(n_items)
    R = np.eye(n_inputs) * rho

    if online:
        U = doc.tracking_optimal_control(A, B, Q, R, data[-1], target_state)
    else:
        U = doc.tracking_optimal_control(A, B, Q, R, data, target_state)
    U = utils.round_to_vector(U, admissible_inputs)

    return U


def brute_force_strategy(data, inputs, target_state, admissible_inputs, time_horizon, rho, online=True):

    A, B, _ = utils.stable_ridge_regression(data, inputs)
    n_items = data.shape[1]
    n_inputs = inputs.shape[1]
    n_comb = len(admissible_inputs) ** time_horizon

    if online:
        T = [len(data)-1]
    else:
        T = np.arange(len(data))
    U = np.zeros((len(T), n_inputs))
    for i, t in enumerate(T):
        X = np.zeros((time_horizon+1, n_comb, n_items))
        X[0] = np.tile(data[t], (n_comb, 1))
        U_idx = np.array(np.meshgrid(*[np.arange(len(admissible_inputs))]*time_horizon)).reshape(time_horizon, -1)
        for k in range(time_horizon):
            U_suggest = admissible_inputs[U_idx[k]]
            X[k+1] = doc.step(A, B, X[k], U_suggest)    
        Yref = target_state.reshape(1,1,-1)
        loss = (np.einsum('...i,...i', (X[1:]-Yref), (X[1:]-Yref))
                + np.einsum('...i,...i', admissible_inputs[U_idx], admissible_inputs[U_idx]) * rho).sum(axis=0)    
        U[i] = admissible_inputs[U_idx[0,np.argmin(loss)]]

    return U


def max_ac_strategy(data, inputs, admissible_inputs, online=True):

    A, B, _ = utils.stable_ridge_regression(data, inputs)
    n_items = data.shape[1]
    n_inputs = inputs.shape[1]

    AC = np.zeros(n_inputs)
    for j in range(n_inputs):
        AC[j] = np.trace(doc.ctrb_gramian(A, B[:,j:j+1], n_items)).sum()

    if online:
        U = admissible_inputs[np.argmax(AC)][np.newaxis, :]
    else:
        U = np.tile(admissible_inputs[np.argmax(AC)], (data.shape[0], 1))

    return U
