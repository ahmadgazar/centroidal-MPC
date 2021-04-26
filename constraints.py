import numpy as np 
import matplotlib.pyplot as plt 
from scipy import sparse

class Constraints:
    def __init__(self, model, data, contact_trajectory, CONSTRAINT_IDENTIFIER):
        self.nx, self.nu, self.nt = model._n_x, model._n_u, model._n_t
        self.n_all = model._total_nb_optimizers
        self.N = model._N
        self.data = data
        self._CONSTRAINT_IDENTIFIER = CONSTRAINT_IDENTIFIER  
        self.contact_trajectory = contact_trajectory
        self.__construct_constraints(model, data)
     
    def __construct_constraints(self, model, data):
        if self._CONSTRAINT_IDENTIFIER == 'INITIAL_CONDITIONS':
           self.construct_initial_constraints(model)

        elif self._CONSTRAINT_IDENTIFIER == 'DYNAMICS':
            # pre-allocate memory
            self._constraints_matrix = np.zeros((self.nx*(self.N-1), 
                                self.nx*self.N + self.nu*(self.N-1) + self.nt*(self.N) + self.nt*(self.N-1)))
            self._upper_bound = np.zeros(self.nx*(self.N-1))
            self._lower_bound = np.zeros(self.nx*(self.N-1))
            self._constraint_related_optimizers = model._optimizers_objects['dynamics'] 
            self.construct_dynamics_constraints(model, data)   
        
        elif self._CONSTRAINT_IDENTIFIER == 'COM_REACHABILITY':
            self._constraints_matrix = np.zeros((self.N-1, 
                                self.nx*self.N + self.nu*(self.N-1) + self.nt*(self.N-1)))
            self._upper_bound = np.zeros(self.N-1)
            self._lower_bound = np.zeros(self.N-1)
            self._constraint_related_optimizers = model._optimizers_objects['com_z']
                                                   
            self.construct_com_reachability_constraints()

        elif self._CONSTRAINT_IDENTIFIER == 'FINAL_CONDITIONS':
            self.construct_final_constraints(model)
           
        elif self._CONSTRAINT_IDENTIFIER == 'COP':
            # pre-allocate memory
            self._constraints_matrix = np.zeros((4*(self.N-1), self.n_all))
            self._upper_bound = np.zeros(4*(self.N-1))
            self._lower_bound = np.zeros(4*(self.N-1))
            self._constraint_related_optimizers = {'RF':[model._optimizers_objects['cop_x_rf'],
                                                         model._optimizers_objects['cop_y_rf']],
                                                   'LF':[model._optimizers_objects['cop_x_lf'],
                                                         model._optimizers_objects['cop_y_lf']]}
            self.robot_foot_range = model._robot_foot_range
            self.construct_cop_constraints()
        
        elif self._CONSTRAINT_IDENTIFIER == 'UNILATERAL': 
            # pre-allocate memory
            self._constraints_matrix = np.zeros((model._n_x*self.N + model._n_u*(self.N-1), 
                                                        model._N, model._n_x + model._n_u))
            self._constraint_related_optimizers = {'RF':[model._optimizers_objects['fx_rf'], 
                                                         model._optimizers_objects['fy_rf'],
                                                         model._optimizers_objects['fz_rf']],
                                                   'LF':[model._optimizers_objects['fx_lf'], 
                                                         model._optimizers_objects['fy_lf'],
                                                         model._optimizers_objects['fz_lf']]}
            self.construct_unilaterality_constraints()
        
        elif self._CONSTRAINT_IDENTIFIER == 'CONTROL_TRUST_REGION':
            self._constraint_related_optimizers = model._optimizers_objects['control_slack']
            omega, delta = model._scp_params['omega0'], model._scp_params['trust_region_radius0'] 
            self.construct_control_trust_region_constraints(data, omega, delta)

        elif self._CONSTRAINT_IDENTIFIER == 'STATE_TRUST_REGION':
            self._constraint_related_optimizers = model._optimizers_objects['state_slack']
            omega, delta = model._scp_params['omega0'], model._scp_params['trust_region_radius0'] 
            self.construct_state_trust_region_constraints(data, omega, delta)                                                                                                                         
    
    def construct_initial_constraints(self, model):
        self._constraints_matrix = np.hstack([np.eye(self.nx), np.zeros((self.nx, self.nx*(self.N-1))), 
                                   np.zeros((self.nx, self.n_all-self.nx*self.N))])
        self._lower_bound = self._upper_bound = model._x_init
       
    def construct_dynamics_constraints(self, model, data):
        nb_x, nb_u = model._n_x, model._n_u
        x_index_vector = self._constraint_related_optimizers._x_idx_vector
        u_index_vector = self._constraint_related_optimizers._u_idx_vector
        X_traj, X_plus_traj = np.copy(data.previous_trajectories['state']), np.copy(data.dynamics) 
        #print('previous state trajectory = ', X_traj)
        #print('f = ', X_plus_traj)       
        U_traj = np.copy(data.previous_trajectories['control'])
        #print('previous control trajectory = ', U_traj)
        A_traj, B_traj = np.copy(data.gradients['f_x']), np.copy(data.gradients['f_u'])
       
        for time_idx in range(self.N-1):
            x_k, x_k_plus, u_k = X_traj[:, time_idx], X_plus_traj[:, time_idx], U_traj[:, time_idx]
            A_k = np.reshape(A_traj[:, time_idx], (nb_x, nb_x), order='F')
            B_k = np.reshape(B_traj[:, time_idx], (nb_x, nb_u), order='F')
            x_index_k = x_index_vector[time_idx]
            u_index_k = u_index_vector[time_idx]
            self._constraints_matrix[x_index_k:x_index_k+nb_x, x_index_k:x_index_k+nb_x] = A_k # x_{k}
            self._constraints_matrix[x_index_k:x_index_k+nb_x, u_index_k:u_index_k+nb_u] = B_k # u_{k}
            self._constraints_matrix[x_index_k:x_index_k+nb_x, x_index_k+nb_x:x_index_k+(2*nb_x)] = \
                                                                              -np.eye(nb_x) # x_{k+1}
            linearized_dynamics = A_k@x_k + B_k@u_k - x_k_plus 
            #print('linearized_dynamics = ', linearized_dynamics)
            # adding a slack to avoid infeasibility due to numerical issues
            self._lower_bound[x_index_k:x_index_k+nb_x] = linearized_dynamics - 1e-5
            self._upper_bound[x_index_k:x_index_k+nb_x] = linearized_dynamics + 1e-5
        
    """
    TODO: properly for a wallking motion 
    """
    def construct_com_reachability_constraints(self):
        optimizer_object = self._constraint_related_optimizers  
        for time_idx in range(self.N-1):
            optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
            self._constraints_matrix[time_idx, optimizer_idx] = 1.0
            self._lower_bound[time_idx] = 0.88 - 0.001
            self._upper_bound[time_idx] = 0.88 + 0.001                  
    """
    TODO: construct a control invariant set in the future for MPC to guarantee recursive feasibility 
    """
    def construct_final_constraints(self, model):
        self._constraints_matrix = np.hstack([np.zeros((self.nx, self.nx*(self.N-1))), 
                        np.eye(self.nx), np.zeros((self.nx, self.n_all-self.nx*self.N))])
        self._lower_bound = model._x_final - 5e-5
        self._upper_bound = model._x_final + 5e-5
    
    def construct_cop_constraints(self):
        # pre-allocate memory
        Aineq_x_rf = np.zeros((self.N-1, self.n_all))
        Aineq_y_rf = np.zeros((self.N-1, self.n_all))
        Aineq_x_lf = np.zeros((self.N-1, self.n_all))
        Aineq_y_lf = np.zeros((self.N-1, self.n_all))
        u_x_rf = np.zeros(self.N-1)
        u_y_rf = np.zeros(self.N-1)
        l_x_rf = np.zeros(self.N-1)
        l_y_rf = np.zeros(self.N-1)
        u_x_lf = np.zeros(self.N-1)
        u_y_lf = np.zeros(self.N-1)
        l_x_lf = np.zeros(self.N-1)
        l_y_lf = np.zeros(self.N-1)
        optimizer_objects_rf = self._constraint_related_optimizers['RF']
        optimizer_objects_lf = self._constraint_related_optimizers['LF']
        for time_idx in range(self.N-1):
            # right foot CoP local constraints
            if self.contact_trajectory['RF'][time_idx]:
                for x_y_idx, optimizer_object in enumerate (optimizer_objects_rf):
                    optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
                    # x-direction
                    if x_y_idx == 0: 
                        Aineq_x_rf[time_idx, optimizer_idx] = 1.
                        l_x_rf[time_idx] = -self.robot_foot_range['x'][1]
                        u_x_rf[time_idx] = self.robot_foot_range['x'][0] 
                    # y-direction
                    elif x_y_idx == 1:
                        Aineq_y_rf[time_idx, optimizer_idx] = 1.
                        l_y_rf[time_idx] = -self.robot_foot_range['y'][1]
                        u_y_rf[time_idx] = self.robot_foot_range['y'][0]    
            # left foot CoP local constraints
            if self.contact_trajectory['LF'][time_idx]:
                for x_y_idx, optimizer_object in enumerate (optimizer_objects_lf):
                    optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
                    # x-direction
                    if x_y_idx == 0:
                        Aineq_x_lf[time_idx, optimizer_idx] = 1.
                        l_x_lf[time_idx] = -self.robot_foot_range['x'][1]
                        u_x_lf[time_idx] = self.robot_foot_range['x'][0] 
                    # y-direction
                    elif x_y_idx == 1:
                        Aineq_y_lf[time_idx, optimizer_idx] = 1.
                        l_y_lf[time_idx] = -self.robot_foot_range['y'][1]
                        u_y_lf[time_idx] = self.robot_foot_range['y'][0]    
        self._constraints_matrix = np.vstack([Aineq_x_rf, Aineq_y_rf, Aineq_x_lf, Aineq_y_lf])
        self._lower_bound = np.hstack([l_x_rf, l_y_rf, l_x_lf, l_y_lf])
        self._upper_bound = np.hstack([u_x_rf, u_y_rf, u_x_lf, u_y_lf])
    
    def construct_unilaterality_constraints(self):
        # pre-allocate memory
        Aineq_fz_rf = np.zeros((self.N-1, self.n_all))
        Aineq_fz_lf = np.zeros((self.N-1, self.n_all))
        u_fz_rf = np.inf*np.ones(self.N-1)
        l_fz_rf = np.zeros(self.N-1)
        u_fz_lf = np.inf**np.ones(self.N-1)
        l_fz_lf = np.zeros(self.N-1)
        optimizer_objects_rf = self._constraint_related_optimizers['RF'] 
        optimizer_objects_lf = self._constraint_related_optimizers['LF']
        for time_idx in range(self.N-1):       
            # Right foot unilateral constraint
            if self.contact_trajectory['RF'][time_idx]:
                contact_rotation_rf = self.contact_trajectory['RF'][time_idx].pose.rotation
                contact_normal_rf = contact_rotation_rf[:,2]
                for x_y_z_idx, optimizer_object in enumerate(optimizer_objects_rf):  
                    optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
                    Aineq_fz_rf[time_idx, optimizer_idx] = contact_normal_rf[x_y_z_idx]
            # Left foot unilateral constraint
            if self.contact_trajectory['LF'][time_idx]:
                contact_rotation_lf = self.contact_trajectory['LF'][time_idx].pose.rotation
                contact_normal_lf = contact_rotation_lf[:, 2]
                for x_y_z_idx, optimizer_object in enumerate (optimizer_objects_lf): 
                    optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
                    Aineq_fz_lf[time_idx, optimizer_idx] = contact_normal_lf[x_y_z_idx]
        self._constraints_matrix =  np.vstack([Aineq_fz_rf, Aineq_fz_lf])
        self._upper_bound = np.hstack([u_fz_rf,  u_fz_lf])
        self._lower_bound = np.hstack([l_fz_rf,  l_fz_lf])             

    """
    control trust region constraints based on l1-norm exact penalty method
    """
    def construct_control_trust_region_constraints(self, data, omega, delta):
        nb_optimizers = self.n_all
        nb_l1_norm_constraints = (2**self.nu)*(self.N-1)
        nb_slack_constraints = self.nt*(self.N-1)
        # pre-allocate memory
        slack_constraint_mat = np.zeros((nb_slack_constraints, nb_optimizers))
        slack_upper_bound = np.zeros(nb_slack_constraints)
        slack_lower_bound = -np.inf * np.ones(nb_slack_constraints)
        l1_norm_constraint_mat = np.zeros((nb_l1_norm_constraints, nb_optimizers))
        l1_norm_upper_bound = np.zeros(nb_l1_norm_constraints)
        l1_norm_lower_bound = -np.inf * np.ones(nb_l1_norm_constraints)
        # get relevant data
        optimizer_object = self._constraint_related_optimizers 
        U = np.copy(data.previous_trajectories['control'])
        penum_mat_all = optimizer_object._penum_mat_all
        penum_mat_rf = optimizer_object._penum_mat_rf    
        penum_mat_lf = optimizer_object._penum_mat_lf    
        for time_idx in range(self.N-1):
            #constraint_idx = optimizer_object._constraint_idx_vector[time_idx]
            u0_idx = optimizer_object._u0_optimizer_idx_vector[time_idx]
            slack_idx = optimizer_object._slack_optimizers_idx_vector[time_idx]
            # both feet in contact
            if self.contact_trajectory['RF'][time_idx] and self.contact_trajectory['LF'][time_idx]:
                for penum_index in range(2**self.nu):
                    constraint_idx =  penum_index + time_idx*(2**self.nu)
                    # |U-U_j| <= delta_j + t/w_j
                    l1_norm_constraint_mat[constraint_idx, u0_idx:u0_idx + self.nu] = \
                                                          penum_mat_all[penum_index, :]
                    l1_norm_constraint_mat[constraint_idx, slack_idx] = -1./omega
                    l1_norm_upper_bound[constraint_idx] = delta + penum_mat_all[penum_index, :] @ U[:, time_idx]
            # only RF in contact
            elif self.contact_trajectory['RF'][time_idx] and not self.contact_trajectory['LF'][time_idx]:
                for penum_index in range(2**self.nu):
                    constraint_idx =  penum_index + time_idx*(2**self.nu)
                    # |U-U_j| <= delta_j + t/w_j
                    l1_norm_constraint_mat[constraint_idx, u0_idx:u0_idx + self.nu] = \
                                                           penum_mat_rf[penum_index, :]
                    l1_norm_constraint_mat[constraint_idx, slack_idx] = -1./omega                   
                    l1_norm_upper_bound[constraint_idx] = delta + penum_mat_rf[penum_index, :] @ U[:, time_idx]
            # only LF in contact
            elif self.contact_trajectory['LF'][time_idx] and not self.contact_trajectory['RF'][time_idx]:
                for penum_index in range(2**self.nu):
                    constraint_idx =  penum_index + time_idx*(2**self.nu)
                    # |U-U_j| <= delta_j + t/w_j
                    l1_norm_constraint_mat[constraint_idx, u0_idx:u0_idx + self.nu] = \
                                                           penum_mat_lf[penum_index, :]
                    l1_norm_constraint_mat[constraint_idx, slack_idx] = -1./omega                   
                    l1_norm_upper_bound[constraint_idx] = delta + penum_mat_lf[penum_index, :] @ U[:, time_idx]
        # -t <= 0
        slack_constraint_mat[:, nb_optimizers-nb_slack_constraints:] = -np.eye(nb_slack_constraints)
        # stack up all constraints
        self._constraints_matrix = l1_norm_constraint_mat
        self._constraints_matrix =  np.vstack([l1_norm_constraint_mat, slack_constraint_mat])
        self._upper_bound = np.hstack([l1_norm_upper_bound, slack_upper_bound])
        self._lower_bound = np.hstack([l1_norm_lower_bound, slack_lower_bound])

    
    """
    state trust region constraints based on l1-norm exact penalty method
    """
    def construct_state_trust_region_constraints(self, data, omega, delta):
        nb_optimizers = self.n_all
        nb_l1_norm_constraints = (2**self.nu)*(self.N)
        nb_slack_constraints = self.nt*(self.N)
        # pre-allocate memory
        slack_constraint_mat = np.zeros((nb_slack_constraints, nb_optimizers))
        slack_upper_bound = np.zeros(nb_slack_constraints)
        slack_lower_bound = -np.inf * np.ones(nb_slack_constraints)
        l1_norm_constraint_mat = np.zeros((nb_l1_norm_constraints, nb_optimizers))
        l1_norm_upper_bound = np.zeros(nb_l1_norm_constraints)
        l1_norm_lower_bound = -np.inf * np.ones(nb_l1_norm_constraints)
        # get relevant data
        optimizer_object = self._constraint_related_optimizers 
        X = np.copy(data.previous_trajectories['state'])
        penum_mat = optimizer_object._penum_mat
        for time_idx in range(self.N-1):
            x0_idx = optimizer_object._x0_optimizer_idx_vector[time_idx]
            slack_idx = optimizer_object._slack_optimizers_idx_vector[time_idx]
            for penum_index in range(2**self.nx):
                constraint_idx =  penum_index + time_idx*(2**self.nx)
                # |X-X_j| <= delta_j + t/w_j
                l1_norm_constraint_mat[constraint_idx, x0_idx:x0_idx + self.nx] = \
                                                        penum_mat[penum_index, :]
                l1_norm_constraint_mat[constraint_idx, slack_idx] = -1./omega
                l1_norm_upper_bound[constraint_idx] = delta + penum_mat[penum_index, :] @ X[:, time_idx]
        # -t <= 0
        slack_constraint_mat[:, nb_optimizers-nb_slack_constraints:] = -np.eye(nb_slack_constraints)
        # stack up all constraints
        self._constraints_matrix = l1_norm_constraint_mat
        self._constraints_matrix = np.vstack([l1_norm_constraint_mat, slack_constraint_mat])
        self._upper_bound = np.hstack([l1_norm_upper_bound, slack_upper_bound])
        self._lower_bound = np.hstack([l1_norm_lower_bound, slack_lower_bound])
                 

if __name__=='__main__':
    import conf
    from centroidal_model import bipedal_centroidal_model
    from trajectory_data import trajectory_data
    from contact_plan import create_contact_trajectory 
    import numpy as np
    import matplotlib.pyplot as plt
    # create model and data
    contact_trajectory = create_contact_trajectory(conf)         
    model = bipedal_centroidal_model(conf)
    data = trajectory_data(model, contact_trajectory)
    cop_constraints = Constraints(model, data, contact_trajectory, 'COP')
    unilateral_constraints = Constraints(model, data, contact_trajectory, 'UNILATERAL')
    data.evaluate_dynamics_and_gradients(data.previous_trajectories['state'],
                                         data.previous_trajectories['control'] )
    initial_constraints = Constraints(model, data, contact_trajectory, 'INITIAL_CONDITIONS')
    final_constraints = Constraints(model, data, contact_trajectory, 'FINAL_CONDITIONS')
    dynamics_constraints = Constraints(model, data, contact_trajectory, 'DYNAMICS')
    #com_reachability_constraints = Constraints(model, data, contact_trajectory, 'COM_REACHABILITY')
    trust_region_constraints = Constraints(model, data, contact_trajectory, 'CONTROL_TRUST_REGION')
    A_ineq_initial_conditions = initial_constraints._constraints_matrix
    A_ineq_dynamics = dynamics_constraints._constraints_matrix
    A_ineq_final_conditions = final_constraints._constraints_matrix
    A_ineq_all_dynamics = np.vstack([A_ineq_initial_conditions,  A_ineq_dynamics, A_ineq_final_conditions]) 
    #A_ineq_com_reachability = com_reachability_constraints._constraints_matrix
    A_ineq_cop = cop_constraints._constraints_matrix
    A_ineq_unilateral = unilateral_constraints._constraints_matrix
    A_ineq_trust_region = trust_region_constraints._constraints_matrix 
 
    with np.nditer(A_ineq_cop, op_flags=['readwrite']) as it:
         for x in it:
             if x[...] != 0:
                 x[...] = 1
    with np.nditer(A_ineq_initial_conditions, op_flags=['readwrite']) as it:
        for x in it:
            if x[...] != 0:
                x[...] = 1
    with np.nditer(A_ineq_dynamics, op_flags=['readwrite']) as it:
        for x in it:
            if x[...] != 0:
                x[...] = 1
    with np.nditer(A_ineq_final_conditions, op_flags=['readwrite']) as it:
        for x in it:
            if x[...] != 0:
                x[...] = 1            
    # with np.nditer(A_ineq_com_reachability, op_flags=['readwrite']) as it:
    #     for x in it:
    #         if x[...] != 0:
    #             x[...] = 1            
    with np.nditer(A_ineq_unilateral, op_flags=['readwrite']) as it:
        for x in it:
            if x[...] != 0:
                x[...] = 1
    with np.nditer(A_ineq_trust_region, op_flags=['readwrite']) as it:
        for x in it:
            if x[...] != 0:
                x[...] = 1            
    with np.nditer(A_ineq_all_dynamics, op_flags=['readwrite']) as it:
         for x in it:
             if x[...] != 0:
                 x[...] = 1
    plt.figure()                      
    plt.figure()
    plt.grid()
    plt.suptitle('Structure of A_ineq matrix of COP')
    plt.imshow(A_ineq_cop, cmap='Greys', extent =[0,A_ineq_cop.shape[1],
                A_ineq_cop.shape[0],0], interpolation = 'nearest')

    plt.figure()
    plt.grid()
    plt.suptitle('Structure of A_ineq matrix of initial constraints')
    plt.imshow(A_ineq_initial_conditions, cmap='Greys', extent =[0,A_ineq_initial_conditions.shape[1],
                A_ineq_initial_conditions.shape[0],0], interpolation = 'nearest')

    plt.figure()
    plt.grid()
    plt.suptitle('Structure of A_ineq matrix of dynamics')
    plt.imshow(A_ineq_dynamics, cmap='Greys', extent =[0,A_ineq_dynamics.shape[1],
                A_ineq_dynamics.shape[0],0], interpolation = 'nearest')
    
    plt.figure()
    plt.grid()
    plt.suptitle('Structure of A_ineq matrix of final conditions')
    plt.imshow(A_ineq_final_conditions, cmap='Greys', extent =[0,A_ineq_final_conditions.shape[1],
                A_ineq_final_conditions.shape[0],0], interpolation = 'nearest')

    plt.figure()
    plt.grid()
    plt.suptitle('Structure of A_ineq matrix of unilaterality')
    plt.imshow(A_ineq_unilateral, cmap='Greys', extent =[0,A_ineq_unilateral.shape[1],
                A_ineq_unilateral.shape[0],0], interpolation = 'nearest')            
    
    plt.figure()
    plt.grid()
    plt.suptitle('Structure of A_ineq matrix of trust region')
    plt.imshow(A_ineq_trust_region, cmap='Greys', extent =[0,A_ineq_trust_region.shape[1],
                A_ineq_trust_region.shape[0],0], interpolation = 'nearest')  
    # plt.figure()
    # plt.grid()
    # plt.suptitle('Structure of A_ineq matrix of com_z reachability')
    # plt.imshow(A_ineq_com_reachability, cmap='Greys', extent =[0,A_ineq_com_reachability.shape[1],
    #             A_ineq_com_reachability.shape[0],0], interpolation = 'nearest')      
    plt.show()

    # plt.figure()
    # plt.grid()
    # plt.suptitle('Structure of A_ineq matrix of all dynamics constraints')
    # plt.imshow(A_ineq_all_dynamics, cmap='Greys', extent =[0, A_ineq_all_dynamics[1],
    #              A_ineq_all_dynamics.shape[0],0], interpolation = 'nearest')      
    # plt.show()  