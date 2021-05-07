import numpy as np
from scipy import sparse
class Cost:
    def __init__(self, model, data, contact_trajectory, COST_IDENTIFIER):
        self.data = data
        self.model = model
        self.N = model._N
        self._nb_slack_optimizers = model._n_t*(self.N) 
        self._total_nb_optimizers = model._total_nb_optimizers
        self._COST_IDENTIFIER = COST_IDENTIFIER  
        self._contact_trajectory = contact_trajectory
        self._hessian = np.zeros((self._total_nb_optimizers, self._total_nb_optimizers))
        self._gradient = np.zeros(self._total_nb_optimizers)
        self.__construct_objective_cost(model)

    def __construct_objective_cost(self, model):
        if self._COST_IDENTIFIER == 'STATE_CONTROL':
            self.__construct_total_cost(model)
            print('constructing QR cost ..')
        elif self._COST_IDENTIFIER == 'COM_REGULATION':
            print('constructing CoM regulation cost ..')
            self._cost_related_optimizers = model._optimizers_objects['dynamics']
            self.__construct_com_cost(self.model)
        elif self._COST_IDENTIFIER == 'TERMINAL':
            print('constructing terminal cost ..')
            self.__construct_terminal_cost(self.model)
        elif self._COST_IDENTIFIER == 'STATE_TRUST_REGION':
            print('constructing state trust region cost ..')
            self._cost_related_optimizers = model._optimizers_objects['state_slack']     
            self.__construct_state_trust_region_cost(self.model)    
        elif self._COST_IDENTIFIER == 'CONTROL_TRUST_REGION':
            print('constructing control trust region cost ..')
            self._cost_related_optimizers = model._optimizers_objects['control_slack']     
            self.__construct_control_trust_region_cost()

    def __construct_total_cost(self, model):
        N = self.N
        running_state_cost, control_cost = model._state_cost_weights, model._control_cost_weights
        self._hessian += sparse.block_diag([sparse.kron(np.eye(N+1), running_state_cost), 
                         sparse.kron(np.eye(N), control_cost), np.zeros((N+1, N+1)),
                                                   np.zeros((N, N))], format='csc')

    def __construct_com_cost(self, model):
        optimizer_object = self._cost_related_optimizers
        com_tracking_traj = self.data.init_trajectories['state']
        for time_idx in range(self.N):
            com_x_cost_idx = optimizer_object._x_idx_vector[time_idx]
            self._gradient[com_x_cost_idx:com_x_cost_idx+model._n_x] = com_tracking_traj[:, time_idx]

    """
    L1 norm penalty cost on the state trust region
    """
    def __construct_state_trust_region_cost(self, model):
        nx, nu, N = model._n_x, model._n_u, model._N
        state_slack_idx_0 = nx*(N+1) + nu*N
        self._gradient[state_slack_idx_0:state_slack_idx_0+N+1] = np.ones(N+1)

    """
    L1 norm penalty cost on the control trust region
    """
    def __construct_control_trust_region_cost(self):
        self._gradient[(self._total_nb_optimizers - self._nb_slack_optimizers):] = \
                                                np.ones(self._nb_slack_optimizers)

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
    QR_cost = Cost(model, data, contact_trajectory, 'STATE_CONTROL')
    trust_region_cost = Cost(model, data, contact_trajectory, 'TRUST_REGION')   
    hessian = QR_cost._hessian
    gradient = QR_cost._gradient
    with np.nditer(hessian, op_flags=['readwrite']) as it:
         for x in it:
             if x[...] != 0:
                 x[...] = 1 
    plt.figure()
    plt.grid()
    plt.suptitle('Structure of Hessian of COP')
    plt.imshow(hessian, cmap='Greys', extent =[0, hessian.shape[1],
                hessian.shape[0],0], interpolation = 'nearest') 
    plt.show()                               