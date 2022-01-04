from utils import construct_friction_pyramid_constraint_matrix
from collections import namedtuple
from functools import partial 
from scipy.stats import norm
import jax.numpy as jnp
import numpy as np 
import jax 

Constraint = namedtuple('Constraint', 'mat, lb, ub')

@partial(jax.jit, static_argnums=(0,)) 
def construct_initial_constraints(model):
    print('adding initial constraints ..', '\n')
    nx, N = model._n_x, model._N 
    x_init = model._x_init
    A = np.hstack([np.eye(nx), np.zeros((nx, nx*N)), np.zeros((nx, model._total_nb_optimizers-nx*(N+1)))])
    return Constraint(mat=A, lb=x_init, ub=x_init)

@partial(jax.jit, static_argnums=(0,))     
def construct_dynamics_constraints(model, prev_traj_tuple, traj_data):
    print('adding dynamics constraints ..', '\n')
    nx, N = model._n_x, model._N
    state_constraint_related_optimizers = model._state_optimizers_indices['coms'][0]
    if model._robot == 'TALOS':
        control_constraint_related_optimizers = model._control_optimizers_indices[next(iter(model._control_optimizers_indices))]['cops'][0]
    elif model._robot == 'solo12':
        control_constraint_related_optimizers = model._control_optimizers_indices[next(iter(model._control_optimizers_indices))]['forces'][0]

    # pre-allocate memory
    constraint_dict= dict(mat=jnp.zeros((nx*N, model._total_nb_optimizers)), lb =jnp.zeros(nx*N), ub=jnp.zeros(nx*N))
    X, U = prev_traj_tuple['state'], prev_traj_tuple['control']
    x_index_vector = jnp.array(state_constraint_related_optimizers._optimizer_idx_vector)
    u_index_vector = jnp.array(control_constraint_related_optimizers._optimizer_idx_vector)
    A_traj, B_traj = traj_data['gradients']['f_x'], traj_data['gradients']['f_u']
    F_traj = traj_data['dynamics']
    def fill_linearized_dynamics_loop(time_idx, curr):
        x_k, x_k_plus, u_k = X[:, time_idx], F_traj[:, time_idx], U[:, time_idx]
        A_k, B_k = A_traj[time_idx,:,:], B_traj[time_idx, :, :]
        x_index_k = x_index_vector[time_idx]
        u_index_k = u_index_vector[time_idx]
        curr['mat'] = jax.lax.dynamic_update_slice(curr['mat'], A_k, (x_index_k, x_index_k))  # x_{k}
        curr['mat'] = jax.lax.dynamic_update_slice(curr['mat'], B_k, (x_index_k, u_index_k))  # u_{k}
        curr['mat'] = jax.lax.dynamic_update_slice(curr['mat'], -np.eye(nx), (x_index_k, x_index_k+nx)) # x_{k+1}
        linearized_dynamics = A_k@x_k + B_k@u_k - x_k_plus 
        # adding a slack to avoid infeasibility due to numerical issues
        curr['lb'] = jax.lax.dynamic_update_slice_in_dim(curr['lb'], linearized_dynamics - 1e-6, x_index_k, 0)  
        curr['ub'] = jax.lax.dynamic_update_slice_in_dim(curr['ub'], linearized_dynamics + 1e-6, x_index_k, 0) 
        return curr
    updated_dict= jax.lax.fori_loop(0, N, fill_linearized_dynamics_loop, constraint_dict)
    return Constraint(mat=updated_dict['mat'], lb=updated_dict['lb'], ub=updated_dict['ub'])

"""
TODO: properly with step poisiton and timing optimization 
"""
@partial(jax.jit, static_argnums=(0,))     
def construct_com_reachability_constraints(model):
    n_all, N = model._total_nb_optimizers, model._N
    max_l = model._max_leg_length
    contact_trajectory = model._contact_trajectory
    Aineq_total = np.array([]).reshape(0, n_all)
    l_total = np.array([])
    u_total = np.array([])
    for contact in model._contact_trajectory:
        # pre-allocate memory for every contact
        Aineq_x = np.zeros((N, n_all))
        Aineq_y = np.zeros((N, n_all))
        Aineq_z = np.zeros((N, n_all))
        l_x = np.zeros(N)
        l_y = np.zeros(N)
        l_z = np.zeros(N)
        u_x = np.zeros(N)
        u_y = np.zeros(N)
        u_z = np.zeros(N)
        optimizer_objects = model._state_optimizers_indices['coms']
        for time_idx in range(N):
            if contact_trajectory[contact][time_idx].ACTIVE:
                p = contact_trajectory[contact][time_idx].pose.translation
                for com_dir, optimizer_object in enumerate (optimizer_objects):
                    optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
                    # x-direction
                    if com_dir == 0:
                        Aineq_x[time_idx, optimizer_idx] = -1.
                        l_x[time_idx] =  - max_l - p[com_dir]
                        u_x[time_idx] =  max_l - p[com_dir]
                    # y-direction
                    elif com_dir == 1:
                        Aineq_y[time_idx, optimizer_idx] = -1.
                        l_y[time_idx] = - max_l - p[com_dir]
                        u_y[time_idx] = max_l - p[com_dir]
                    # z-direction
                    elif com_dir == 2:
                        Aineq_z[time_idx, optimizer_idx] = -1.
                        u_z[time_idx] = max_l - p[com_dir]            
        Aineq_total = np.vstack([Aineq_total, Aineq_x, Aineq_y, Aineq_z])
        l_total = np.hstack([l_total, l_x, l_y, l_z])
        u_total = np.hstack([u_total, u_x, u_y, u_z])
    return Constraint(mat=Aineq_total, lb=l_total, ub=u_total)

"""
TODO: construct a control invariant set in the future for MPC to guarantee recursive feasibility 
"""
@partial(jax.jit, static_argnums=(0,)) 
def construct_final_constraints(model):
    print('adding final constraints ..', '\n')
    nx, N = model._n_x, model._N
    x_final = model._x_final #+ 5e-5 
    A = np.hstack([np.zeros((nx, nx*N)), np.eye(nx), np.zeros((nx, model._total_nb_optimizers-nx*(N+1)))])
    return Constraint(mat=A, lb=x_final, ub=x_final)

def construct_cop_constraints(model):
    n_all, N = model._total_nb_optimizers, model._N
    foot_range_x, foot_range_y = model._robot_foot_range['x'], model._robot_foot_range['y']
    contact_trajectory = model._contact_trajectory
    Aineq_total = np.array([]).reshape(0, n_all)
    l_total = np.array([])
    u_total = np.array([])
    for contact in model._contact_trajectory:
        # pre-allocate memory for every contact
        Aineq_x = np.zeros((N, n_all))
        Aineq_y = np.zeros((N, n_all))
        l_x = np.zeros(N)
        l_y = np.zeros(N)
        u_x = np.zeros(N)
        u_y = np.zeros(N)
        optimizer_objects = model._control_optimizers_indices[contact]['cops']
        for time_idx in range(N):
            # right foot CoP local constraints
            if contact_trajectory[contact][time_idx].ACTIVE:
                for x_y_idx, optimizer_object in enumerate (optimizer_objects):
                    optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
                    # x-direction
                    if x_y_idx == 0: 
                        Aineq_x[time_idx, optimizer_idx] = 1.
                        l_x[time_idx] = -foot_range_x[1]
                        u_x[time_idx] =  foot_range_x[0] 
                    # y-direction
                    elif x_y_idx == 1:
                        Aineq_y[time_idx, optimizer_idx] = 1.
                        l_y[time_idx] = -foot_range_y[1]
                        u_y[time_idx] =  foot_range_y[0]
        Aineq_total = np.vstack([Aineq_total, Aineq_x, Aineq_y])
        l_total = np.hstack([l_total, l_x, l_y])
        u_total = np.hstack([u_total, u_x, u_y])
    return Constraint(mat=Aineq_total, lb=l_total, ub=u_total)

"""
inner-linear approximation of friction cone constraints 
"""
def construct_friction_pyramid_constraints(model, prev_traj_tuple=None, traj_data=None):
    print('adding friction pyramid constraints ..', '\n')
    nx, nu, n_all, N = model._n_x, model._n_u, model._total_nb_optimizers, model._N 
    friction_pyramid_mat = construct_friction_pyramid_constraint_matrix(model)
    xi = norm.ppf(norm.cdf(1-model._beta_u)/(friction_pyramid_mat.shape[0]*3))
    X, U = prev_traj_tuple['state'], prev_traj_tuple['control']
    dSigma_dx = traj_data['Covs_gradients']['Cov_dx']  
    dSigma_du = traj_data['Covs_gradients']['Cov_du']
    Sigma_traj = traj_data['Covs']
    K_traj = traj_data['LQR_gains']
    contact_trajectory = model._contact_trajectory
    Aineq_total = np.array([]).reshape(0, n_all)
    ub_total = np.array([])
    nb_contacts = len(model._contact_trajectory)
    nb_constraints_per_contact = friction_pyramid_mat.shape[0]*N
    for contact in model._contact_trajectory:
        # pre-allocate memory
        Aineq = np.zeros((nb_constraints_per_contact, n_all))
        ub = np.zeros(nb_constraints_per_contact)
        force_optimizers = model._control_optimizers_indices[contact]['forces']
        for time_idx in range(N):
            u0_idx = force_optimizers[0]._optimizer_idx_vector[time_idx]
            fx_contact_idx = contact_trajectory[contact][time_idx].idx*3        
            if contact_trajectory[contact][time_idx].ACTIVE:
                contact_orientation = contact_trajectory[contact][time_idx].pose.rotation
                rotated_friction_pyramid_mat = friction_pyramid_mat @ contact_orientation.T
                for constraint_idx in range(rotated_friction_pyramid_mat.shape[0]):
                    idx = time_idx*rotated_friction_pyramid_mat.shape[0] + constraint_idx
                    # nominal friction pyramid constraints
                    for x_y_z_idx, optimizer_object in enumerate(force_optimizers):  
                        optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
                        Aineq[idx, optimizer_idx] = rotated_friction_pyramid_mat[constraint_idx, x_y_z_idx]
                    # chance-constraints back-offs 
                    if time_idx>0 and model._STOCHASTIC_OCP:
                        dSigma_dx_k, dSigma_du_k = dSigma_dx[time_idx], dSigma_du[time_idx]
                        K = K_traj[time_idx, fx_contact_idx:fx_contact_idx+3, :]
                        K_Sigma_Kt = K @ Sigma_traj[time_idx, :, :] @ K.T
                        # extract the ith individual constraint
                        Gi = Aineq[idx, u0_idx:u0_idx+3] 
                        K_dSigma_dx_Kt = np.einsum('ij, jgkf->igkf', K, np.einsum('ijkf, gj->igkf', dSigma_dx_k, K))
                        K_dSigma_du_Kt = np.einsum('ij, jgkf->igkf', K, np.einsum('ijkf, gj->igkf', dSigma_du_k, K))
                        # check for every control variable 
                        for u_idx in range(Gi.shape[0]):
                            Gi_u = Gi[u_idx]
                            # extract only the diagonal part of the covariance matrix
                            sqrt_K_Sigma_Kt = np.sqrt(K_Sigma_Kt[u_idx, u_idx])
                            # consider only active  of the individual chance constraint
                            if Gi_u > 1e-6 and sqrt_K_Sigma_Kt > 1e-6:
                                # linearized individual chance-constraint state contribution
                                dSigma_dx_k_total = (0.5*(1/(2*Gi_u*sqrt_K_Sigma_Kt))*(2*Gi_u*K_dSigma_dx_Kt[u_idx, u_idx])).flatten(order='F') 
                                Aineq[idx, :nx*(N+1)] += xi * dSigma_dx_k_total 
                                ub[idx] += xi*((dSigma_dx_k_total @ X.flatten(order='F')))
                                # linearized individual chance-constraint control contribution
                                dSigma_du_k_total = (0.5*(1/(2*Gi_u*sqrt_K_Sigma_Kt))*(2*Gi_u*K_dSigma_du_Kt[u_idx, u_idx]))[:, :N].flatten(order='F') 
                                Aineq[idx, nx*(N+1):nx*(N+1)+nu*N] += xi * dSigma_du_k_total 
                                ub[idx] += xi*(dSigma_du_k_total @ U.flatten(order='F'))
                                # original individual chance-constraint back-off
                                ub[idx] -= xi*(2*Gi_u*sqrt_K_Sigma_Kt)
        Aineq_total = np.vstack([Aineq_total, Aineq])
        ub_total = np.hstack([ub_total, ub])
    return Constraint(mat=Aineq_total, lb=-np.inf*np.ones(nb_contacts*nb_constraints_per_contact), ub=ub_total)

"""
control trust region constraints based on l1-norm exact penalty method
"""
def construct_control_trust_region_constraints(self, prev_traj_tuple, trust_region):
    print('adding control trust region constraints ..', '\n')
    nu, nb_optimizers, N = self._n_u, self._total_nb_optimizers, self._N
    nb_permut = (2**nu)
    nb_l1_norm_constraints = nb_permut*N
    nb_slack_constraints = self._n_t*(N)
    # pre-allocate memory
    slack_constraint_mat = np.zeros((nb_slack_constraints, nb_optimizers))
    slack_upper_bound = np.zeros(nb_slack_constraints)
    slack_lower_bound = -np.inf * np.ones(nb_slack_constraints)
    l1_norm_constraint_mat = np.zeros((nb_l1_norm_constraints, nb_optimizers))
    l1_norm_upper_bound = np.zeros(nb_l1_norm_constraints)
    l1_norm_lower_bound = -np.inf * np.ones(nb_l1_norm_constraints)
    # get relevant data
    optimizer_object = self._control_slack_optimizers_indices 
    U = prev_traj_tuple['control']
    penum_mat = optimizer_object._penum_mat
    for time_idx in range(N):
        u0_idx = optimizer_object._u0_optimizer_idx_vector[time_idx]
        slack0_idx = optimizer_object._slack_optimizers_idx_vector[time_idx]
        # for penum_index in range(2**nu):
        constraint_idx = time_idx*nb_permut
        # |U-U_j| <= delta_j + t/w_j
        l1_norm_constraint_mat[constraint_idx:constraint_idx+nb_permut, u0_idx:u0_idx+nu] = penum_mat
        l1_norm_constraint_mat[constraint_idx:constraint_idx+nb_permut, slack0_idx] = -1./trust_region['weight']
        l1_norm_upper_bound[constraint_idx:constraint_idx+nb_permut] = (trust_region['radius']*np.ones(nb_permut)) + \
                                                                                        penum_mat @ U[:, time_idx]
    # -t <= 0
    slack_constraint_mat[:, nb_optimizers-nb_slack_constraints:] = -np.eye(nb_slack_constraints)
    # stack up all constraints
    print('DONE !')
    return Constraint(mat=np.vstack([l1_norm_constraint_mat, slack_constraint_mat]),
                             ub=np.hstack([l1_norm_upper_bound, slack_upper_bound]),
                             lb=np.hstack([l1_norm_lower_bound, slack_lower_bound]))

"""
state trust region constraints based on l1-norm exact penalty method on angular momentum
"""
def construct_state_trust_region_constraints(model, prev_traj_tuple, trust_region):
    print('adding state trust region constraints ..', '\n')
    nx, n_all, N = model._n_x, model._total_nb_optimizers, model._N
    nb_permut = (2**(nx-6))
    nb_l1_norm_constraints = nb_permut*(N+1)
    nb_slack_constraints = model._n_t*(N+1)
    # pre-allocate memory
    slack_constraint_mat = np.zeros((nb_slack_constraints, n_all))
    slack_upper_bound = np.zeros(nb_slack_constraints)
    slack_lower_bound = -np.inf * np.ones(nb_slack_constraints)
    l1_norm_constraint_mat = np.zeros((nb_l1_norm_constraints, n_all))
    l1_norm_upper_bound = np.zeros(nb_l1_norm_constraints)
    l1_norm_lower_bound = -np.inf * np.ones(nb_l1_norm_constraints)
    # get relevant data
    optimizer_object = model._state_slack_optimizers_indices 
    X = prev_traj_tuple['state']
    penum_mat = optimizer_object._penum_mat
    for time_idx in range(N+1):
        x6_idx = optimizer_object._x0_optimizer_idx_vector[time_idx] + 6
        slack_idx = optimizer_object._slack_optimizers_idx_vector[time_idx]
        # for penum_index in range(2**(nx-6)):
        constraint_idx = time_idx*nb_permut
        # |X-X_j| <= delta_j + t/w_j
        l1_norm_constraint_mat[constraint_idx:constraint_idx+nb_permut, x6_idx:x6_idx+3] = penum_mat
        l1_norm_constraint_mat[constraint_idx:constraint_idx+nb_permut, slack_idx] = -1./trust_region['weight']
        l1_norm_upper_bound[constraint_idx:constraint_idx+nb_permut] = (trust_region['radius']*np.ones(nb_permut)) + \
                                                                                       penum_mat @ X[6:, time_idx]
    # -t <= 0
    slack_idx0 = nx*(N+1) + model._n_u*(N)  
    slack_constraint_mat[:, slack_idx0:slack_idx0+N+1] = -np.eye(nb_slack_constraints)
    # stack up all constraints
    return Constraint(mat=np.vstack([l1_norm_constraint_mat, slack_constraint_mat]),
                             ub=np.hstack([l1_norm_upper_bound, slack_upper_bound]),
                             lb=np.hstack([l1_norm_lower_bound, slack_lower_bound]))

"""
checks if contact forces are inside the linearized friction pyramid
"""
def evaluate_friction_pyramid_constraints(model, forces):
    U_sol = forces.flatten(order='F')
    nx, nu, n_all, N = model._n_x, model._n_u, model._total_nb_optimizers, model._N 
    friction_pyramid_mat = construct_friction_pyramid_constraint_matrix(model)
    contact_trajectory = model._contact_trajectory
    Aineq_total = np.array([]).reshape(0, n_all)
    ub_total = np.array([])
    nb_constraints_per_contact = friction_pyramid_mat.shape[0]*N
    for contact in model._contact_trajectory:
        # pre-allocate memory
        Aineq = np.zeros((nb_constraints_per_contact, n_all))
        ub = np.zeros(nb_constraints_per_contact)
        force_optimizers = model._control_optimizers_indices[contact]['forces']
        for time_idx in range(N):
            if contact_trajectory[contact][time_idx].ACTIVE:
                contact_orientation = contact_trajectory[contact][time_idx].pose.rotation
                rotated_friction_pyramid_mat = friction_pyramid_mat @ contact_orientation.T
                for constraint_idx in range(rotated_friction_pyramid_mat.shape[0]):
                    idx = time_idx*rotated_friction_pyramid_mat.shape[0] + constraint_idx
                    # nominal friction pyramid constraints
                    for x_y_z_idx, optimizer_object in enumerate(force_optimizers):  
                        optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
                        Aineq[idx, optimizer_idx] = rotated_friction_pyramid_mat[constraint_idx, x_y_z_idx]
        Aineq_total = np.vstack([Aineq_total, Aineq])
        ub_total = np.hstack([ub_total, ub])
    for constraint_idx in range(Aineq_total.shape[0]):
        lhs = Aineq_total[constraint_idx, nx*(N+1):nx*(N+1)+nu*N]@ U_sol 
        if lhs <= ub_total[constraint_idx]:
            continue
            # if(abs(lhs-ub[constraint_idx])<=1e-8):
            #     pass 
            #     print('touching constraint number ', constraint_idx)
        else:
            print('violating constraint number ', constraint_idx)     






