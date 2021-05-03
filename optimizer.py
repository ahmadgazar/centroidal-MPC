import numpy as np
"""
a helper class for book keeping of optimizers and constraints global indices  
"""
class State_optimizer():
    def __init__(self, OPTIMIZER_IDENTIFIER, nb_x_optimizers, horizon_length, optimizer_weight=None):
        self._name = OPTIMIZER_IDENTIFIER 
        self._nx = nb_x_optimizers
        self.horizon_length = horizon_length  
        self._optimizer_idx = None 
        self._optimizer_idx_vector = np.zeros(horizon_length-1, dtype=int)
        self.__fill_optimizer_indices()
    
    def __fill_optimizer_indices(self):
        if self._name == 'com_x':
            self._optimizer_idx = 0
        elif self._name == 'com_y':
            self._optimizer_idx = 1
        elif self._name == 'com_z':
             self._optimizer_idx = 2
        elif self._name == 'linear_mom_x':
            self._optimizer_idx = 3
        elif self._name == 'linear_mom_y':
            self._optimizer_idx = 4    
        elif self._name == 'linear_mom_z':
            self._optimizer_idx = 5
        elif self._name == 'angular_mom_x':
            self._optimizer_idx = 6
        elif self._name == 'angular_mom_y':
            self._optimizer_idx = 7 
        elif self._name == 'angular_mom_z':
            self._optimizer_idx = 8          
        for time_idx in range(self.horizon_length-1):     
            self._optimizer_idx_vector[time_idx] = time_idx*self._nx + self._optimizer_idx

class Control_optimizer():
    def __init__(self, OPTIMIZER_IDENTIFIER, nb_x_optimizers, nb_u_ptimizers, horizon_length, optimizer_weight=None):
        self._name = OPTIMIZER_IDENTIFIER 
        self._nx = nb_x_optimizers
        self._nu = nb_u_ptimizers 
        self.horizon_length = horizon_length  
        self._optimizer_idx = None 
        self._optimizer_idx_vector = np.zeros(horizon_length-1, dtype=int)
        self.__fill_optimizer_params()
    
    def __fill_optimizer_params(self):
        if self._name == 'cop_x_rf':
            self._optimizer_idx = 0
        elif self._name == 'cop_y_rf':
            self._optimizer_idx = 1
        elif self._name == 'fx_rf':
             self._optimizer_idx = 2
        elif self._name == 'fy_rf':
            self._optimizer_idx = 3
        elif self._name == 'fz_rf':
            self._optimizer_idx = 4    
        elif self._name == 'tau_z_rf':
            self._optimizer_idx = 5
        elif self._name == 'cop_x_lf':
            self._optimizer_idx = 6
        elif self._name == 'cop_y_lf':
            self._optimizer_idx = 7
        elif self._name == 'fx_lf':
            self._optimizer_idx = 8
        elif self._name == 'fy_lf':
            self._optimizer_idx =  9
        elif self._name == 'fz_lf':
            self._optimizer_idx = 10    
        elif self._name == 'tau_z_lf':
            self._optimizer_idx = 11
        else: print('this is not a control optimizer name !')
        for time_idx in range(self.horizon_length-1):     
            self._optimizer_idx_vector[time_idx] = self._nx*self.horizon_length + \
                                            time_idx*self._nu + self._optimizer_idx 

class Dynamics_optimizer():
    def __init__(self, OPTIMIZER_IDENTIFIER, nx_optimizers, nu_optimizers, horizon_length):
        self._name = OPTIMIZER_IDENTIFIER 
        self._nb_state_optimizers = nx_optimizers
        self._nb_control_optimizers = nu_optimizers
        self.N = horizon_length
        self._x_idx_vector = np.zeros(horizon_length, dtype=int)
        self._u_idx_vector = np.zeros(horizon_length-1, dtype=int)
        self.__fill_optimizer_params() 
    
    def __fill_optimizer_params(self):
        if self._name == 'dynamics':
            for time_index in range(self.N-1):
                self._x_idx_vector[time_index] = time_index*self._nb_state_optimizers
                self._u_idx_vector[time_index] = self._nb_state_optimizers*self.N + time_index*self._nb_control_optimizers
        else: print('this not a dynamics optimizer name !')    
            
class Slack_optimizer():
    def __init__(self, OPTIMIZER_IDENTIFIER, nx_optimizers, nu_optimizers, nt_optimizers, horizon_length):
        self._name = OPTIMIZER_IDENTIFIER 
        self.N = horizon_length
        self.nx = nx_optimizers
        self.nu = nu_optimizers
        self.nt = nt_optimizers   
        self.__fill_permutation_matrices()
        self.__fill_optimizer_params() 
    
    def __fill_permutation_matrices(self):
        if self._name == 'state':
            # pre-allocate memory
            self._nb_slack_constraints = self.nt*(2**(self.nx-6))
            self._penum_mat = np.zeros((2**(self.nx-6), self.nx-6))     
            self._constraint_idx_vector = np.zeros(self.N, dtype=int)
            self._x0_optimizer_idx_vector = np.zeros(self.N, dtype=int)
            self._slack_optimizers_idx_vector = np.zeros(self.N, dtype=int)
            for x_idx in range(self.nx-6):
                self._penum_mat[:, x_idx] = np.array([(-1)**(j//(2**x_idx)) for j in range(2**(self.nx-6))])
        elif self._name == 'control':
            # pre-allocate memory
            self._nb_slack_constraints = self.nt*(2**self.nu)
            self._penum_mat_all = np.zeros((2**self.nu, self.nu))
            self._penum_mat_rf = np.zeros((2**self.nu, self.nu))
            self._penum_mat_lf = np.zeros((2**self.nu, self.nu))       
            self._constraint_idx_vector = np.zeros(self.N-1, dtype=int)
            self._u0_optimizer_idx_vector = np.zeros(self.N-1, dtype=int)
            self._slack_optimizers_idx_vector = np.zeros(self.N-1, dtype=int)  
            for u_idx in range(self.nu):
                self._penum_mat_all[:, u_idx] = np.array([(-1)**(j//(2**u_idx)) for j in range(2**self.nu)])
                if u_idx < (int(self.nu/2)):
                    self._penum_mat_rf[:2**int(self.nu/2), u_idx] = np.array([(-1)**(j//(2**u_idx)) for j in range(2**int(self.nu/2))])
                    self._penum_mat_lf[(2**self.nu)-(2**int(self.nu/2)):, u_idx+int(self.nu/2)] = np.array([(-1)**(j//(2**u_idx)) for j in range(2**int(self.nu/2))])                     
        
    def __fill_optimizer_params(self):
        if self._name == 'state':
            for time_index in range(self.N):
                self._x0_optimizer_idx_vector[time_index] = time_index*self.nx               
                self._slack_optimizers_idx_vector[time_index] = self.nx*self.N + self.nu*(self.N-1) + time_index
        elif self._name == 'control':
            for time_index in range(self.N-1):
                self._u0_optimizer_idx_vector[time_index] = self.nx*self.N + time_index*self.nu               
                self._slack_optimizers_idx_vector[time_index] = self.nx*self.N + self.nu*(self.N-1) + self.nt*self.N + time_index
        else: print('this not a slack optimizer name !')
       
    