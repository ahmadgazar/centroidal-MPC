from cost import construct_total_cost, construct_state_trust_region_cost, Cost
from warnings import warn
from scipy import sparse
import jax.numpy as jnp 
import numpy as np
import constraints
import jax
import osqp 

def sum_up_all_costs(model):
    QR_cost = construct_total_cost(model)
    state_trust_region_cost = construct_state_trust_region_cost(model)
    total_cost = [QR_cost, state_trust_region_cost]
    Q = np.zeros((model._total_nb_optimizers, model._total_nb_optimizers))
    p = np.zeros(model._total_nb_optimizers)
    for cost in total_cost:
        Q += cost.Q 
        p += cost.p 
    return Cost(Q=sparse.csc_matrix(Q), p=p)

def stack_up_all_constraints(model, traj_tuple, traj_data, trust_region_updates):
    initial_constraints = constraints.construct_initial_constraints(model)
    final_constraints = constraints.construct_final_constraints(model)
    dynamics_constraints = constraints.construct_dynamics_constraints(model, traj_tuple, traj_data)   
    cop_constraints = constraints.construct_cop_constraints(model)
    friction_pyramid_constraints = constraints.construct_friction_pyramid_constraints(model) 
    state_trust_region_constraints= constraints.construct_state_trust_region_constraints(model, traj_tuple, trust_region_updates)
    total_constraints = [initial_constraints, dynamics_constraints, final_constraints, cop_constraints, 
                                            friction_pyramid_constraints, state_trust_region_constraints]
    A = np.array([]).reshape(0, model._total_nb_optimizers)
    l = np.array([])
    u = np.array([])
    for constraint in total_constraints:
        A = sparse.vstack([A, constraint.mat], 'csc')
        l = np.hstack([l, constraint.lb])
        u = np.hstack([u, constraint.ub])
    return constraints.Constraint(mat=A, lb=l, ub=u)

# check if solution is converged 
def convergence(traj_tuple_curr, traj_tuple_prev):
    X_prev, U_prev = traj_tuple_prev['state'], traj_tuple_prev['control']
    X_curr, U_curr = traj_tuple_curr['state'], traj_tuple_curr['control']
    l2_norm = np.linalg.norm(U_curr-U_prev, 2)/np.linalg.norm(U_curr, 2) + \
              np.linalg.norm(X_curr-X_prev, 2)/np.linalg.norm(X_curr, 2)
    return l2_norm 

# solve QP iteration using OSQP
def solve_subproblem(cost, constraints):
    QP_FEASIBLITY = True
    prob  = osqp.OSQP()
    prob.setup(cost.Q, cost.p, constraints.mat, constraints.lb, constraints.ub, 
                                                warm_start=True, verbose=False)
    res = prob.solve()
    if res.info.status != 'solved':
        warn("[solve_OSQP]: Problem unfeasible.")
        QP_FEASIBLITY = False
    return QP_FEASIBLITY, res

# ratio between the model error and the linearized model 
def compute_model_accuracy(model, curr_traj, prev_traj, prev_traj_data):
    ratio = dict(num=0., den=0.)
    non_linear_model_traj = model.integrate_dynamics_trajectory(curr_traj)
    f_prev = prev_traj_data['dynamics'] 
    A_prev, B_prev =  prev_traj_data['gradients']['f_x'], prev_traj_data['gradients']['f_u']
    X_curr, U_curr = jnp.array(curr_traj['state']), jnp.array(curr_traj['control'])
    X_prev, U_prev = prev_traj['state'], prev_traj['control']
    def loop_body(time_idx, curr): 
        delta_x = X_curr[:, time_idx] - X_prev[:,time_idx]
        delta_u = U_curr[:,time_idx]  - U_prev[:,time_idx] 
        linear_model = f_prev[:, time_idx] + A_prev[time_idx, :,:] @ delta_x #+ B_prev[time_idx, :,:] @ delta_u 
        model_error = non_linear_model_traj[:, time_idx] - linear_model 
        curr['num'] += model_error.T @ model_error
        curr['den'] += linear_model.T @ linear_model
        return curr
    res = jax.lax.fori_loop(0, model._N, loop_body, ratio) 
    return res['num']/res['den']

def get_QP_solution(model, res):
    n_x, n_u, N = model._n_x, model._n_u, model._N
    X_sol = np.reshape(res.x[:n_x*(N+1)], (n_x, N+1), order='F')
    U_sol = np.reshape(res.x[n_x*(N+1):n_x*(N+1)+n_u*N], (n_u, N), order='F')
    return dict(state=X_sol, control=U_sol)

def solve_scp(model, scp_params):
    all_solution = dict(state=[], control=[])
    rho_0, rho_max = scp_params['rho0'], scp_params['rho1']
    omega_max = scp_params['omega_max']
    beta_succ, beta_fail = scp_params['beta_succ'], scp_params['beta_fail']
    max_iter = scp_params['max_iterations']
    conv_thresh = scp_params['convergence_threshold']
    gamma_fail = scp_params['gamma_fail']
    B_success = False
    trust_region_updates = {'weight':np.copy(scp_params['omega0']),
                            'radius':np.copy(scp_params['trust_region_radius0'])}
    prev_traj_dict = model._init_trajectories  
    traj_tuple = model._init_trajectories 
    scp_iteration = 0
    # TODO convert to JAX while loop to JIT
    while (scp_iteration < max_iter and trust_region_updates['weight'] < omega_max and \
        not(scp_iteration!= 0 and B_success and convergence(traj_tuple, prev_traj_dict) < conv_thresh)):
        print('\n'+'=' * 50)
        print('Iteration ' + str(scp_iteration))
        print('-' * 50)
        B_success = False
        # propagate dynamics given previous solutions
        traj_data = model.compute_trajectory_data(traj_tuple)
        # build cost and constraints matrices 
        total_cost = sum_up_all_costs(model)
        total_constraints = stack_up_all_constraints(model, traj_tuple, traj_data, trust_region_updates)
        QP_SOLVED, res = solve_subproblem(cost=total_cost, constraints=total_constraints)
        # break out when QP fails 
        if QP_SOLVED == False:
            print('QP subproblem Failed at iter #' + str(scp_iteration ))
            return False
        sol_traj_dict =  get_QP_solution(model, res) 
        # check if the solution is inside the trust region 
        if np.linalg.norm((sol_traj_dict['state']-prev_traj_dict['state']), 2) < trust_region_updates['radius']:
            print('solution is inside trust region .. checking model accuracy')
            rho = compute_model_accuracy(model, sol_traj_dict, prev_traj_dict, traj_data)
            print("error ratio between linearized and nonlinear dynamics = ", rho)
            # check model accuracy
            if rho > rho_max:
                print('linearized model is NOT accurate .. rejecting solution and decreasing trust region radius') 
                trust_region_updates['radius'] *= beta_fail 
                B_success = False
            else:
                # accept solution
                print('linearized model is accurate enough .. accepting solution ')
                all_solution['state'].append(sol_traj_dict['state'])
                all_solution['control'].append(sol_traj_dict['control'])
                B_success = True                                                      
                # check if you can decrease trust region radius more than initial guess
                if rho < rho_0:
                    trust_region_updates['radius'] = np.minimum(beta_succ*trust_region_updates['radius'],
                                                                      scp_params['trust_region_radius0'])
        else:
            # reject solution        
            print('solution is outside trust region .. rejecting solution and increasing trust region weight')
            trust_region_updates['weight'] *= gamma_fail  
            B_success = False
        scp_iteration +=1
    print('[solve_ccscp] Success: '+str(B_success)+', Nb of iterations: '+str(scp_iteration))
    return all_solution

