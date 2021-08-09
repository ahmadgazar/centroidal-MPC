import numpy as np 
import matplotlib.pyplot as plt 
from scipy import sparse
from utils import construct_friction_pyramid_constraint_matrix

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
            self._constraints_matrix = np.zeros((self.nx*(self.N), self.n_all))
            self._upper_bound = np.zeros(self.nx*(self.N))
            self._lower_bound = np.zeros(self.nx*(self.N))
            self._state_constraint_related_optimizers = model._state_optimizers_indices['coms'][0]
            self._control_constraint_related_optimizers = model._control_optimizers_indices[next(iter(model._control_optimizers_indices))]['cops'][0]
            self.construct_dynamics_constraints(model, data)   
        
        elif self._CONSTRAINT_IDENTIFIER == 'COM_REACHABILITY':
            self._constraints_matrix = np.zeros((self.N+1, self.n_all)) 
            self._upper_bound = np.zeros(self.N+1)
            self._lower_bound = np.zeros(self.N+1)
            self._constraint_related_optimizers = model._optimizers_objects['com_z']         
            self.construct_com_reachability_constraints()

        elif self._CONSTRAINT_IDENTIFIER == 'FINAL_CONDITIONS':
            self.construct_final_constraints(model)
           
        elif self._CONSTRAINT_IDENTIFIER == 'COP':
            # pre-allocate memory
            self._constraints_matrix = np.zeros((4*(self.N), self.n_all))
            self._upper_bound = np.zeros(4*(self.N))
            self._lower_bound = np.zeros(4*(self.N))
            self._constraint_related_optimizers = model._control_optimizers_indices
            self.robot_foot_range = model._robot_foot_range
            self.construct_cop_constraints(model)
        
        elif self._CONSTRAINT_IDENTIFIER == 'UNILATERAL': 
            # pre-allocate memory
            self._constraint_related_optimizers = model._control_optimizers_indices
            self.construct_unilaterality_constraints(model)

        elif self._CONSTRAINT_IDENTIFIER == 'FRICTION_PYRAMID':
            # pre-allocate memory
            self._constraint_related_optimizers = model._control_optimizers_indices
            self.construct_friction_pyramid_constraints(model, data)
                                             
        elif self._CONSTRAINT_IDENTIFIER == 'CONTROL_TRUST_REGION':
            self._constraint_related_optimizers = model._control_slack_optimizers_indices
            omega, delta = model._scp_params['omega0'], model._scp_params['trust_region_radius0'] 
            self.construct_control_trust_region_constraints(data, omega, delta)

        elif self._CONSTRAINT_IDENTIFIER == 'STATE_TRUST_REGION':
            self._constraint_related_optimizers = model._state_slack_optimizers_indices
            omega, delta = model._scp_params['omega0'], model._scp_params['trust_region_radius0'] 
            self.construct_state_trust_region_constraints(model, data, omega, delta)                                                                                                                         
    
    def construct_initial_constraints(self, model):
        self._constraints_matrix = np.hstack([np.eye(self.nx), np.zeros((self.nx, self.nx*(self.N))), 
                                   np.zeros((self.nx, self.n_all-self.nx*(self.N+1)))])
        self._lower_bound = self._upper_bound = model._x_init
       
    def construct_dynamics_constraints(self, model, data):
        nb_x, nb_u = model._n_x, model._n_u
        x_index_vector = self._state_constraint_related_optimizers._optimizer_idx_vector
        u_index_vector = self._control_constraint_related_optimizers._optimizer_idx_vector
        X_traj, X_plus_traj = np.copy(data.previous_trajectories['state']), np.copy(data.dynamics)       
        U_traj = np.copy(data.previous_trajectories['control'])
        A_traj, B_traj = np.copy(data.gradients['f_x']), np.copy(data.gradients['f_u'])
        for time_idx in range(self.N):
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
    TODO: properly with step poisiton and timing optimization 
    """
    def construct_com_reachability_constraints(self):
        return 0.

    """
    TODO: construct a control invariant set in the future for MPC to guarantee recursive feasibility 
    """
    def construct_final_constraints(self, model):
        self._constraints_matrix = np.hstack([np.zeros((self.nx, self.nx*(self.N))), 
                        np.eye(self.nx), np.zeros((self.nx, self.n_all-self.nx*(self.N+1)))])
        self._lower_bound = model._x_final #- 5e-5
        self._upper_bound = model._x_final #+ 5e-5
    
    def construct_cop_constraints(self, model):
        Aineq_total = np.array([]).reshape(0, self.n_all)
        l_total = np.array([])
        u_total = np.array([])
        for contact in model._contacts:
            # pre-allocate memory for every contact
            Aineq_x = np.zeros((self.N, self.n_all))
            Aineq_y = np.zeros((self.N, self.n_all))
            l_x = np.zeros(self.N)
            l_y = np.zeros(self.N)
            u_x = np.zeros(self.N)
            u_y = np.zeros(self.N)
            optimizer_objects = self._constraint_related_optimizers[contact._name]['cops']
            for time_idx in range(self.N):
                # right foot CoP local constraints
                if self.contact_trajectory[contact._name][time_idx].ACTIVE:
                    for x_y_idx, optimizer_object in enumerate (optimizer_objects):
                        optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
                        # x-direction
                        if x_y_idx == 0: 
                            Aineq_x[time_idx, optimizer_idx] = 1.
                            l_x[time_idx] = -self.robot_foot_range['x'][1]
                            u_x[time_idx] = self.robot_foot_range['x'][0] 
                        # y-direction
                        elif x_y_idx == 1:
                            Aineq_y[time_idx, optimizer_idx] = 1.
                            l_y[time_idx] = -self.robot_foot_range['y'][1]
                            u_y[time_idx] = self.robot_foot_range['y'][0]
            Aineq_total = np.vstack([Aineq_total, Aineq_x, Aineq_y])
            l_total = np.hstack([l_total, l_x, l_y])
            u_total = np.hstack([u_total, u_x, u_y])
        self._constraints_matrix = Aineq_total
        self._lower_bound = l_total
        self._upper_bound = u_total
    
    """
    use only if you are walking on a flat terrain otherwise use friction pyramid constraints 
    which includes unilaterality constraints)
    """
    def construct_unilaterality_constraints(self, model):
        # pre-allocate memory
        Aineq_total =  np.array([]).reshape(0, self.n_all)
        for contact in model._contacts:
            Aineq = np.zeros((self.N, self.n_all)) 
            optimizer_objects = self._constraint_related_optimizers[contact._name]['forces'] 
            for time_idx in range(self.N):       
                # Right foot unilateral constraint
                if self.contact_trajectory[contact._name][time_idx].ACTIVE:
                    contact_orientation  = self.contact_trajectory[contact._name][time_idx].pose.rotation.T
                    contact_normal = contact_orientation[:,2]
                    for x_y_z_idx, optimizer_object in enumerate(optimizer_objects):     
                        optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
                        Aineq[time_idx, optimizer_idx] = contact_normal[x_y_z_idx]
            Aineq_total = np.vstack([Aineq_total, Aineq])
        self._constraints_matrix = Aineq_total
        self._upper_bound = np.inf*np.ones(len(model._contacts)*self.N)
        self._lower_bound = np.zeros(len(model._contacts)*self.N)     

    """
    inner-linear approximation of friction cone constraints 
    """
    def construct_friction_pyramid_constraints(self, model, data):
        nx, nu = model._n_x, model._n_u 
        friction_pyramid_mat = construct_friction_pyramid_constraint_matrix(model)
        contact_trajectory = self.contact_trajectory
        Aineq_total = np.array([]).reshape(0, self.n_all)
        u0_indices = next(iter(self._constraint_related_optimizers.items()))[1]['cops'][0]._optimizer_idx_vector
        for contact in model._contacts:
            # pre-allocate memory
            Aineq = np.zeros((friction_pyramid_mat.shape[0]*self.N, self.n_all))
            force_optimizers = self._constraint_related_optimizers[contact._name]['forces']
            for time_idx in range(self.N):        
                if contact_trajectory[contact._name][time_idx].ACTIVE:
                    x_k, u_k = data.previous_trajectories['state'][:, time_idx], data.previous_trajectories['control'][:, time_idx]
                    contact_logic_and_pose = data.contacts_logic_and_pose[time_idx]
                    K = np.reshape(data.LQR_gains[:, time_idx], (nu, nx), order='F')
                    Cov_k = np.reshape(data.Covs[:, time_idx], (nx, nx), order='F')    
                    contact_orientation = contact_trajectory[contact._name][time_idx].pose.rotation
                    rotated_friction_pyramid_mat = friction_pyramid_mat @ contact_orientation.T
                    for constraint_idx in range(rotated_friction_pyramid_mat.shape[0]):
                        idx = time_idx*rotated_friction_pyramid_mat.shape[0] + constraint_idx
                        for x_y_z_idx, optimizer_object in enumerate(force_optimizers):  
                            optimizer_idx = optimizer_object._optimizer_idx_vector[time_idx]
                            Aineq[idx, optimizer_idx] = rotated_friction_pyramid_mat[constraint_idx, x_y_z_idx]
            Aineq_total = np.vstack([Aineq_total, Aineq])
        self._constraints_matrix = Aineq_total
        self._lower_bound = -np.inf*np.ones(len(model._contacts)*rotated_friction_pyramid_mat.shape[0]*self.N) 
        self._upper_bound = np.zeros(len(model._contacts)*friction_pyramid_mat.shape[0]*self.N)

    """
    control trust region constraints based on l1-norm exact penalty method
    """
    def construct_control_trust_region_constraints(self, data, omega, delta):
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
        U = np.copy(data.previous_trajectories['control'])
        penum_mat = optimizer_object._penum_mat
        for time_idx in range(self.N):
            u0_idx = optimizer_object._u0_optimizer_idx_vector[time_idx]
            slack_idx = optimizer_object._slack_optimizers_idx_vector[time_idx]
            for penum_index in range(2**self.nu):
                constraint_idx =  penum_index + time_idx*(2**self.nu)
                # |U-U_j| <= delta_j + t/w_j
                l1_norm_constraint_mat[constraint_idx, u0_idx:u0_idx + self.nu] = \
                                                        penum_mat[penum_index, :]
                l1_norm_constraint_mat[constraint_idx, slack_idx] = -1./omega
                l1_norm_upper_bound[constraint_idx] = delta + penum_mat[penum_index, :] @ U[:, time_idx]
        # -t <= 0
        slack_constraint_mat[:, nb_optimizers-nb_slack_constraints:] = -np.eye(nb_slack_constraints)
        # stack up all constraints
        self._constraints_matrix = l1_norm_constraint_mat
        self._constraints_matrix = np.vstack([l1_norm_constraint_mat, slack_constraint_mat])
        self._upper_bound = np.hstack([l1_norm_upper_bound, slack_upper_bound])
        self._lower_bound = np.hstack([l1_norm_lower_bound, slack_lower_bound])

    """
    state trust region constraints based on l1-norm exact penalty method on angular momentum
    """
    def construct_state_trust_region_constraints(self, model, data, omega, delta):
        nb_optimizers = self.n_all
        nb_l1_norm_constraints = (2**(self.nx-6))*(self.N+1)
        nb_slack_constraints = self.nt*(self.N+1)
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
        for time_idx in range(self.N+1):
            x0_idx = optimizer_object._x0_optimizer_idx_vector[time_idx]
            slack_idx = optimizer_object._slack_optimizers_idx_vector[time_idx]
            for penum_index in range(2**(self.nx-6)):
                constraint_idx =  penum_index + time_idx*(2**(self.nx-6))
                # |X-X_j| <= delta_j + t/w_j
                l1_norm_constraint_mat[constraint_idx, x0_idx+6:x0_idx+9] = \
                                                        penum_mat[penum_index, :]
                l1_norm_constraint_mat[constraint_idx, slack_idx] = -1./omega
                l1_norm_upper_bound[constraint_idx] = delta + penum_mat[penum_index, :] @ X[6:, time_idx]
        # -t <= 0
        slack_idx0 = model._n_x*(model._N+1) + model._n_u*(model._N)  
        slack_constraint_mat[:, slack_idx0:slack_idx0+model._N+1] = -np.eye(nb_slack_constraints)
        # stack up all constraints
        self._constraints_matrix = np.vstack([l1_norm_constraint_mat, slack_constraint_mat])
        self._upper_bound = np.hstack([l1_norm_upper_bound, slack_upper_bound])
        self._lower_bound = np.hstack([l1_norm_lower_bound, slack_lower_bound])     
                 

if __name__=='__main__':
    import conf_talos as conf
    from centroidal_model import Centroidal_model
    from trajectory_data import Data
    from contact_plan import create_contact_trajectory 
    import numpy as np
    import matplotlib.pyplot as plt
    # create model and data
    contact_sequence = conf.contact_sequence
    contact_trajectory = create_contact_trajectory(conf)         
    model = Centroidal_model(conf)
    data = Data(model, contact_sequence, contact_trajectory)
    cop_constraints = Constraints(model, data, contact_trajectory, 'COP')
    unilateral_constraints = Constraints(model, data, contact_trajectory, 'UNILATERAL')
    data.evaluate_dynamics_and_gradients(data.previous_trajectories['state'],
                                         data.previous_trajectories['control'] )
    initial_constraints = Constraints(model, data, contact_trajectory, 'INITIAL_CONDITIONS')
    final_constraints = Constraints(model, data, contact_trajectory, 'FINAL_CONDITIONS')
    dynamics_constraints = Constraints(model, data, contact_trajectory, 'DYNAMICS')
    friction_constraints = Constraints(model, data, contact_trajectory,  'FRICTION_PYRAMID')
    #trust_region_constraints = Constraints(model, data, contact_trajectory, 'CONTROL_TRUST_REGION')
    A_ineq_initial_conditions = initial_constraints._constraints_matrix
    A_ineq_dynamics = dynamics_constraints._constraints_matrix
    A_ineq_final_conditions = final_constraints._constraints_matrix
    A_ineq_all_dynamics = np.vstack([A_ineq_initial_conditions,  A_ineq_dynamics, A_ineq_final_conditions]) 
    A_ineq_cop = cop_constraints._constraints_matrix
    A_ineq_unilateral = unilateral_constraints._constraints_matrix
    A_ineq_friction_pyramid = friction_constraints._constraints_matrix
    #A_ineq_trust_region = trust_region_constraints._constraints_matrix 
 
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
    with np.nditer(A_ineq_unilateral, op_flags=['readwrite']) as it:
        for x in it:
            if x[...] != 0:
                x[...] = 1
    with np.nditer(A_ineq_friction_pyramid, op_flags=['readwrite']) as it:
         for x in it:
             if x[...] != 0:
                 x[...] = 1            
    with np.nditer(A_ineq_all_dynamics, op_flags=['readwrite']) as it:
         for x in it:
             if x[...] != 0:
                 x[...] = 1
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
    plt.suptitle('Structure of A_ineq matrix of friction pyramid')
    plt.imshow(A_ineq_friction_pyramid, cmap='Greys', extent =[0,A_ineq_friction_pyramid.shape[1],
                 A_ineq_friction_pyramid.shape[0],0], interpolation = 'nearest')   
    plt.figure()
    plt.grid()
    plt.suptitle('Structure of A_ineq matrix of all dynamics constraints')
    plt.imshow(A_ineq_all_dynamics, cmap='Greys', extent =[0, A_ineq_all_dynamics.shape[1],
                 A_ineq_all_dynamics.shape[0],0], interpolation = 'nearest')      
    plt.show()  