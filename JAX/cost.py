import numpy as np
from scipy import sparse
from collections import namedtuple

Cost = namedtuple('Cost', 'Q, p')
"""
QR cost
"""
def construct_total_cost(model):
    N = model._N
    Q = sparse.block_diag([sparse.kron(np.eye(N+1), model._state_cost_weights), 
        sparse.kron(np.eye(N),  model._control_cost_weights), np.zeros((N+1, N+1)), np.zeros((N, N))])
    return Cost(Q=Q, p=np.zeros(model._total_nb_optimizers))

"""
tracking cost on com, lin_mom, ang_mom from IK
"""
def construct_state_tracking_cost(model):
    n_total = model._total_nb_optimizers
    gradient = np.zeros(n_total)
    com_x_indices = model._state_optimizers_indices['coms'][0]._optimizer_idx_vector
    tracking_traj = model._init_trajectories['state']
    for time_idx in range(model._N+1):
        com_x_idx = com_x_indices[time_idx]
        gradient[com_x_idx:com_x_idx+9] = -model._state_cost_weights @ tracking_traj[:, time_idx]
    return Cost(Q=np.zeros((n_total, n_total)),p=gradient)    

"""
L1 norm penalty cost on the state trust region
"""
def construct_state_trust_region_cost(model):
    n_all,  N = model._total_nb_optimizers, model._N
    state_slack_idx_0 = model._n_x*(N+1) + model._n_u*N
    p = np.zeros(n_all)
    p[state_slack_idx_0:state_slack_idx_0+N+1] = np.ones(N+1)
    return Cost(Q=np.zeros((n_all, n_all)), p=p)

"""
L1 norm penalty cost on the control trust region
"""
def construct_control_trust_region_cost(model):
    n_all, N = model._total_nb_optimizers, model._N 
    p = np.zeros(n_all)
    p[(n_all-N):] = np.ones(N) 
    return Cost(Q=np.zeros((n_all, n_all)), p=p)


