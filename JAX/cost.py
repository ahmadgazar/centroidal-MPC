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
com regulation cost following a simple zig-zag
"""
def construct_com_cost(self, model):
    optimizer_object = self._cost_related_optimizers
    com_tracking_traj = self.data.init_trajectories['state']
    for time_idx in range(self.N+1):
        com_x_idx = optimizer_object._optimizer_idx_vector[time_idx]
        self._gradient[com_x_idx:com_x_idx+3] = com_tracking_traj[:3, time_idx]

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
    n_all, n_slack = model._total_nb_optimizers, model._nb_slack_optimizers 
    p = np.zeros(n_all)
    p[(n_all-n_slack):] = np.ones(n_slack) 
    return Cost(Q=np.zeros((n_all, n_all)), p=p)


