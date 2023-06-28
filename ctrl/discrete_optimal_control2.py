import numpy as np
from numpy.linalg import inv, eig
from numpy.linalg import matrix_power as mpow, matrix_rank as rk
from scipy.io import loadmat
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are
import os 

class LinearSystem:

    @staticmethod
    def seminorm(M, x):
        ''' (x.T) @ M @ x '''
        if len(x.shape)==1:
            x = np.expand_dims(x, 1)
        return np.diag(x.T @ M @ x)
    
    @staticmethod
    def step(A, B, x, u):
        return np.einsum('ij,...j->...i', A, x) + np.einsum('ic,...c->...i', B, u)
        
    @classmethod
    def evolve(cls, A, B, T, x0, U=None):
        ''' calculates trajectory over T time steps '''
        X = np.zeros((T+1, *x0.shape))
        X[0] = x0
        if U is None:
            U = np.zeros((T, B.shape[1]))
        for t in range(T):
            X[t+1] = cls.step(A, B, X[t], U[t])
        return X
    
    @staticmethod
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
    
    @staticmethod
    def ctrb(A, B, T=None):
        ''' calculate controllability matrix '''
        if T is None:
            T = A.shape[0]
        return np.hstack([mpow(A, k) @ B for k in range(T)])
    
    @classmethod
    def controllable(cls, A, B, T=None):
        return rk(cls.ctrb(A, B, T)) == A.shape[0]
    
    @classmethod
    def average_ctrb(cls, A, T=np.inf):
        ''' according to Henry (2022) '''
        size = A.shape[0]
        B = np.eye(size)
        ac = np.zeros(size)
        for j in range(size):
            ac[j] = np.diag(cls.ctrb_gramian(A, B[:,j:j+1], T)).sum()
        return ac
    
    @classmethod
    def modal_ctrb(cls, A):
        ''' according to Henry (2022) '''
        size = A.shape[0]
        eigvals, eigvecs = eig(A)
        mc = np.zeros(size)
        for i in range(size):
            mc[i] = ((1 - np.abs(eigvals)**2)*(np.abs(eigvecs[i])**2)).sum()
        return mc
    
    @staticmethod
    def impulse_response(A, b, T):
        ''' according to Henry (2022) '''
        return np.array([mpow(A, t) @ b for t in range(T)]).sum(axis=0)
    

class LQR(LinearSystem):
    ''' Linear Quadratic Regulator, i.e. discrete linear system x(t+1)=Ax(t)+Bu(t)
        with loss function L = x(T)Sx(T) + sum(x(t)Qx(t) + u(t)Ru(t)) '''
    
    @staticmethod
    def kalman_gain(A, B, Q, R):
        ''' inifinite-time limit of kalman gain matrix '''
        riccati = solve_discrete_are(A, B, Q, R)
        temp = B.T @ riccati
        kalman = np.linalg.inv(temp @ B + R) @ temp @ A
        return kalman
    
    @staticmethod
    def kalman_riccati_sequence(A, B, Q, R, T, S=None, store_kalman=True, store_riccati=False):
        ''' calculate solution to the riccati sequence and corresponding time varying 
            kalman gain matrix G (one matrix per time step). If store_riccati, 
            keeps every step S of the riccati sequence. Returns (G, S) ''' 
        if S is None:
            S = Q
        if store_riccati:
            riccati = np.zeros((T+1, *S.shape))
            riccati[T] = S
        if store_kalman:
            kalman = np.zeros((T, *B.T.shape))
        riccati_step = S
        for t in reversed(range(T)):
            temp = B.T @ riccati_step
            kalman_step = np.linalg.inv(temp @ B + R) @ temp @ A
            temp = A - B @ kalman_step
            riccati_step = temp.T @ riccati_step @ temp + kalman_step.T @ R @ kalman_step + Q
            if store_kalman:
                kalman[t] = kalman_step
            if store_riccati:
                riccati[t] = riccati_step
        if not store_kalman:
            kalman = kalman_step
        if not store_riccati:
            riccati = riccati_step
        
        return kalman, riccati
    
    @staticmethod
    def feedback_feedforward_sequence(A, B, C, Q, R, Yref, S=None, store_kalman=False, store_riccati=False,
                           store_feedforward=False, store_adjoint=True):
        ''' 
            calculate solution to the riccati sequence, the correspoding time varying kalman gain matrix,
            the feedforward gains, and the adjoint states.
            time range of input:
            Yref: 1 ... T
            time range of outputs (if stored): 
            kalman: 0 ... T-1
            riccati: 1 ... T
            feedforward: 0 ... T-1
            adjoint: 1 ... T
        '''        
        if S is None:
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
    
    @staticmethod
    def adjoint_forward_step( inv_transp_ABG, C, Q, adjoint, yref):
        return inv_transp_ABG @ adjoint - inv_transp_ABG @ C.T @ Q @ yref

    @staticmethod
    def feedback_feedforward_gain(A, B, C, Q, R):
        riccati = solve_discrete_are(A, B, C.T @ Q @ C, R)
        feedforward = inv(B.T @ riccati @ B + R) @ B.T
        kalman = feedforward @ riccati @ A
        return kalman, feedforward
    
    # @classmethod
    # def linear_quadratic_gaussian(cls, A, B, C, Q, R, 
    #                               cov_process_noise=None, cov_obs_noise=None):
    #     if not cov_process_noise:
    #         cov_process_noise = np.zeros(A.shape[0])
    #     if not cov_obs_noise:
    #         cov_obs_noise = np.zeros(C.shape[0])
    #     k_gain = cls.kalman_gain(A, B, Q, R)
    #     k_filter = cls.kalman_gain(A.T, C.T, cov_process_noise, cov_obs_noise).T
        
    #     return k_gain, k_filter
        
    @classmethod
    def closed_loop_optimal_control(cls, A, B, Q, R, X, S=None):
        ''' calculate U=-GX where G is Kalman gain matrix, and X is given '''
        if X.ndim==1:
            X = X[np.newaxis, :]
        kalman = cls.kalman_gain(A, B, Q, R)
        U_kalman = -np.einsum('xy,ty->tx', kalman, X)
        return np.squeeze(U_kalman)
    
    @classmethod
    def time_variant_closed_loop_optimal_control(cls, A, B, Q, R, X, S=None):
        ''' calculate U=-GX where G is time varying Kalman gain matrix, and X is given '''
        T = X.shape[0] - 1
        kalman, _ = cls.kalman_riccati_sequence(A, B, Q, R, T, S)
        U_kalman = -np.einsum('txy,ty->tx', kalman, X)
        return U_kalman
    
    @classmethod
    def free_final_state_optimal_control(cls, A, B, Q, R, x0, T, S=None, constant_gain=True):
        ''' calculate U=-GX where G is Kalman gain matrix, and X evolves according to Ax+Bu
            Also returns the estimated trajectory X. '''
        if constant_gain:
            kalman = cls.kalman_gain(A, B, Q, R)[np.newaxis, :].repeat(T, axis=0)            
        else:
            kalman, _ = cls.kalman_riccati_sequence(A, B, Q, R, T, S)
        X = np.zeros((T+1, *x0.shape))
        X[0] = x0
        U = np.zeros((T, B.shape[1]))
        for t in range(T):
            U[t] = - kalman[t] @ X[t]
            X[t+1] = cls.step(A, B, X[t], U[t])
        
        return X, U
    
    @classmethod
    def tracking_optimal_control(cls, A, B, C, Q, R, X, Yref, S=None, save_memory=False):
        '''
        Calculate U = -GX + FY, where G is a Kalman gain matrix, F the feedforward gain,
        and Y the feedforward state.
        X and Yref are assumed to be of time range 0 ... T-1 and 1 ... T resp.
        '''
        kalman, feedforward = cls.feedback_feedforward_gain(A, B, C, Q, R)
        if save_memory:
            _, _, _, adjoint = cls.feedback_feedforward_sequence(A, B, C, Q, R, Yref, S,
                                        store_kalman=False, store_riccati=False, store_feedforward=False,
                                        store_adjoint=False) 
            U = -np.einsum('ux,tx->tu', kalman, X)
            ABG_inv = inv(A - B@kalman).T
            for t in range(X.shape[0]):
                U[t] += np.einsum('ux,x->u', feedforward, adjoint)
                adjoint = cls.adjoint_forward_step(ABG_inv, C, Q, adjoint, Yref[t])
        else:
            _, _, _, adjoint = cls.feedback_feedforward_sequence(A, B, C, Q, R, Yref, S,
                                        store_kalman=False, store_riccati=False, store_feedforward=False,
                                        store_adjoint=True)  
            U = -np.einsum('ux,tx->tu', kalman, X) + np.einsum('ux,tx->tu', feedforward, adjoint)
        return U

    @classmethod
    def free_final_state_tracker(cls, A, B, C, Q, R, x0, Yref, S=None, save_memory=False):
        '''
        Calculate U = -GX + FY, where G is a Kalman gain matrix, F the feedforward gain,
        and Y the feedforward state; X evolves according to AX + BU. Also returns X.
        '''
        T = Yref.shape[0]
        kalman, feedforward = cls.feedback_feedforward_gain(A, B, C, Q, R)
        X = np.zeros((T+1, *x0.shape))
        X[0] = x0
        U = np.zeros((T, B.shape[1]))
        if save_memory:
            _,_,_,adjoint = cls.feedback_feedforward_sequence(A, B, C, Q, R, Yref, S, store_adjoint=False)
            ABG_inv = inv(A - B@kalman).T
        else:
            _,_,_,adjoint_full = cls.feedback_feedforward_sequence(A, B, C, Q, R, Yref, S, store_adjoint=True)
        for t in range(T):
            if not save_memory:
                adjoint = adjoint_full[t]
            U[t] = -kalman @ X[t] + feedforward @ adjoint 
            X[t+1] = cls.step(A, B, X[t], U[t])
            if save_memory:
                adjoint = cls.adjoint_forward_step(ABG_inv, C, Q, adjoint, Yref[t])

        return X, U
    
    @classmethod
    def fixed_final_state_optimal_control(cls, A, B, R, x0, xT, T):
        '''
        Calculates the open loop optimal control, also returns the intermediate trajectory X.
        '''
        def weighted_ctrb_gramian(T):
            R_inv = np.linalg.inv(R)
            middle = B @ R_inv @ B.T
            summands = np.array([mpow(A, t) @ middle @ mpow(A.T, t) for t in range(T)])
            return summands.sum(axis=0)
        
        G = weighted_ctrb_gramian(T)
        G_inv = np.linalg.inv(G)
        G_inv_final_state = G_inv @ (xT - (mpow(A, T) @ x0))
        R_inv = np.linalg.inv(R)
        X = np.zeros((T+1, *x0.shape))
        X[0] = x0
        X_costate = np.zeros((T+1, *x0.shape))
        X_costate[0] = x0
        U = np.zeros((T, B.shape[1]))
        for t in range(T):
            costate = - mpow(A.T, (T-t-1)) @ G_inv_final_state
            U[t] = - R_inv @ B.T @ costate
            X[t+1] = cls.step(A, B, X[t], U[t])
            X_costate[t+1] = A @ X[t] - B @ R_inv @ B.T @ costate
            
        return X, U
    
    @classmethod
    def objective(cls, Q, R, X, U, S=None, stepwise=False):
        ''' Loss function value over time. Sums over time if stepwise is False '''
        if S is None:
            S = Q
        steps = np.zeros(X.shape[0])
        steps[:-1] = 0.5 * (cls.seminorm(Q, X[:-1].T) + cls.seminorm(R, U.T))
        steps[-1] = 0.5 * cls.seminorm(S, X[-1])
        if stepwise:
            return steps
        else:
            return steps.sum()
    
    @classmethod
    def control_energy(cls, R, U, stepwise=False):
        steps = 0.5 * cls.seminorm(R, U.T)
        if stepwise:
            return steps
        else:
            return steps.sum()


