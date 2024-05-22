import numpy as np
from numpy.linalg import eig, inv
from numpy.linalg import matrix_power as mpow
from numpy.linalg import matrix_rank as rk
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

### Discrete linear dynamical system methods

def seminorm(M, x):
    ''' (x.T) @ M @ x '''
    if len(x.shape)==1:
        x = np.expand_dims(x, 1)
    # return np.diag(np.einsum('...i,ij,...j', x, M, x))
    return np.diag(x.T @ M @ x)
    

def step(A, B, x, u=None):
    ''' A@x + B@u '''
    if u is None:
        u = np.zeros(B.shape[1])
    return np.einsum('ij,...j->...i', A, x) + np.einsum('ic,...c->...i', B, u)
        
    
def evolve(A, B, T, x0, U=None):
    ''' calculates trajectory over T time steps '''
    X = np.zeros((T+1, *x0.shape))
    X[0] = x0
    if U is None:
        U = np.zeros((T, B.shape[1]))
    for t in range(T):
        X[t+1] = step(A, B, X[t], U[t])
    return X


### Control theoretic measures

def ctrb(A, B, T=None):
    ''' calculate controllability matrix C = [B, AB, AAB, ..., (A^(T-1)B)] '''
    if T is None:
        T = A.shape[0]
    return np.hstack([mpow(A, k) @ B for k in range(T)])


def ctrb_gramian(A, B, T):
    ''' calculate controllability gramian, either by sum iteration (if T finite),
        or by solving the lyapunov equation W - AWA = BB (if T infinite)'''
    middle = B @ B.T
    if T==np.infty:
        W = solve_discrete_lyapunov(A, middle)
    else:
        summands = np.array([mpow(A, t) @ middle @ mpow(A.T, t) for t in range(T)])
        W = summands.sum(axis=0)
    return W


def controllable(A, B, T=None):
    ''' checks if the system is controllable in time T '''
    return rk(ctrb(A, B, T)) == A.shape[0]


def average_ctrb(A, T=np.inf, B=None, per_column_of_B=True):
    ''' computes AC for inputs specified in B; 
        if B is none, calculates node-wise AC by setting B to the identity. '''
    if B is None:
        B = np.eye(A.shape[0])
    if per_column_of_B:
        ac = np.zeros(B.shape[1])
        for j in range(B.shape[1]):
            ac[j] = np.diag(ctrb_gramian(A, B[:,j:j+1], T)).sum()
    else:
        ac = np.diag(ctrb_gramian(A, B, T)).sum()
    return ac


def modal_ctrb(A):
    ''' computes MC '''
    size = A.shape[0]
    eigvals, eigvecs = eig(A)
    mc = np.zeros(size)
    for i in range(size):
        mc[i] = ((1 - np.abs(eigvals)**2)*(np.abs(eigvecs[i])**2)).sum()
    return mc


def cumulative_impulse_response(A, b, T):
    ''' calculates the CIR_T '''
    return np.array([mpow(A, t) @ b for t in range(T)]).sum(axis=0)


### Optimal control methods


def kalman_gain(A, B, Q, R):
    ''' computes the constant Kalman gain '''
    riccati = solve_discrete_are(A, B, Q, R)
    temp = B.T @ riccati
    kalman = np.linalg.inv(temp @ B + R) @ temp @ A
    return kalman

def closed_loop_optimal_control(A, B, Q, R, X):
    ''' calculate U=-GX where G is Kalman gain matrix, and X is given '''
    if X.ndim==1:
        X = X[np.newaxis, :]
    kalman = kalman_gain(A, B, Q, R)
    U_kalman = -np.einsum('xy,ty->tx', kalman, X)
    return np.squeeze(U_kalman)


def feedback_feedforward_gain(A, B, Q, R):
    riccati = solve_discrete_are(A, B, Q, R)
    feedforward = inv(B.T @ riccati @ B + R) @ B.T
    kalman = feedforward @ riccati @ A
    return kalman, feedforward


def tracking_optimal_control(A, B, Q, R, X, Yref, save_memory=False):
    '''
        X and Yref are assumed to be of time range 0 ... T-1 and 1 ... T resp.
    '''
    def adjoint_forward_step(inv_transp_ABG, C, Q, adjoint, yref):
        return inv_transp_ABG @ adjoint - inv_transp_ABG @ C.T @ Q @ yref

    def feedback_feedforward_sequence(A, B, C, Q, R, Yref, store_kalman=False, store_riccati=False,
                        store_feedforward=False, store_adjoint=True):
        ''' 
            time range of input:
            Yref: 1 ... T
            time range of outputs (if stored): 
            kalman: 0 ... T-1
            riccati: 1 ... T
            feedforward: 0 ... T-1
            adjoint: 1 ... T
        '''
        
        S = Q
        T = Yref.shape[0]
        riccati_step = C.T @ S @ C
        adjoint_step = C.T @ S @ Yref[T-1]
        if store_riccati:
            riccati = np.zeros((T, *S.shape))
        if store_kalman:
            kalman = np.zeros((T, *B.T.shape))
        if store_feedforward:
            feedforward = np.zeros((T, *B.T.shape))
        if store_adjoint:
            adjoint = np.zeros((T, A.shape[0]))
        
        for t in reversed(range(T)):
            if store_adjoint:
                adjoint[t] = adjoint_step
            if store_riccati and t>0:
                riccati[t] = riccati_step
            feedforward_step = np.linalg.inv(B.T @ riccati_step @ B + R) @ B.T
            kalman_step = feedforward_step @ riccati_step @ A
            if t>0:
                adjoint_step = (A - B @ kalman_step).T @ adjoint_step + C.T @ Q @ Yref[t]
                riccati_step = A.T @ riccati_step @ (A - B @ kalman_step) + C.T @ Q @ C
            if store_kalman:
                kalman[t] = kalman_step            
            if store_feedforward:
                feedforward[t] = feedforward_step
        
        if not store_kalman:
            kalman = kalman_step
        if not store_riccati:
            riccati = riccati_step
        if not store_feedforward:
            feedforward = feedforward_step
        if not store_adjoint:
            adjoint = adjoint_step
        
        return kalman, riccati, feedforward, adjoint

    if np.ndim(X)==1:
        X = X[np.newaxis]
    if np.ndim(Yref)==1:
        Yref = np.tile(Yref, (X.shape[0], 1))
    assert (X.shape == Yref.shape)
    C = np.eye(A.shape[0])
    kalman, feedforward = feedback_feedforward_gain(A, B, Q, R)
    if save_memory:
        _, _, _, adjoint = feedback_feedforward_sequence(A, B, C, Q, R, Yref,
                                    store_kalman=False, store_riccati=False, store_feedforward=False,
                                    store_adjoint=False) 
        U = -np.einsum('ux,tx->tu', kalman, X)
        ABG_inv = inv(A - B@kalman).T
        for t in range(X.shape[0]):
            U[t] += np.einsum('ux,x->u', feedforward, adjoint)
            adjoint = adjoint_forward_step(ABG_inv, C, Q, adjoint, Yref[t])
    else:
        _, _, _, adjoint = feedback_feedforward_sequence(A, B, C, Q, R, Yref,
                                    store_kalman=False, store_riccati=False, store_feedforward=False,
                                    store_adjoint=True)  
        U = -np.einsum('ux,tx->tu', kalman, X) + np.einsum('ux,tx->tu', feedforward, adjoint)
    return U        