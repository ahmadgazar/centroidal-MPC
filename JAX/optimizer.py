import numpy as np
"""
helper classes for book keeping of optimizers global indices  
"""
class State_optimizer():
    def __init__(self, OPTIMIZER_IDENTIFIER, nb_x_optimizers, horizon_length):
        self._name = OPTIMIZER_IDENTIFIER 
        self.nx = nb_x_optimizers
        self.N = horizon_length  
        self._optimizer_idx = None 
        self._optimizer_idx_vector = np.zeros(self.N+1, dtype=int)
        self.__fill_optimizer_indices()
    
    def __fill_optimizer_indices(self):
        if self._name == 'com_x':
            self._optimizer_idx = 0
        elif self._name == 'com_y':
            self._optimizer_idx = 1
        elif self._name == 'com_z':
             self._optimizer_idx = 2
        elif self._name == 'lin_mom_x':
            self._optimizer_idx = 3
        elif self._name == 'lin_mom_y':
            self._optimizer_idx = 4    
        elif self._name == 'lin_mom_z':
            self._optimizer_idx = 5
        elif self._name == 'ang_mom_x':
            self._optimizer_idx = 6
        elif self._name == 'ang_mom_y':
            self._optimizer_idx = 7 
        elif self._name == 'ang_mom_z':
            self._optimizer_idx = 8
        else: print('this is not a state optimizer name !')             
        for time_idx in range(self.N+1):     
            self._optimizer_idx_vector[time_idx] = time_idx*self.nx + self._optimizer_idx

class Control_optimizer():
    def __init__(self, OPTIMIZER_IDENTIFIER, contact_idx, robot_name, nb_x_optimizers, nb_u_ptimizers, horizon_length):
        self._name = OPTIMIZER_IDENTIFIER 
        self.nx = nb_x_optimizers
        self.nu = nb_u_ptimizers 
        self.N = horizon_length  
        self._optimizer_idx = None 
        self._optimizer_idx_vector = np.zeros(self.N, dtype=int)
        self.__fill_optimizer_params(contact_idx, robot_name)
    
    def __fill_optimizer_params(self, contact_idx, robot_name):
        if robot_name == 'TALOS':
            if self._name == 'cop_x':
                self._optimizer_idx = 0
            elif self._name == 'cop_y':
                self._optimizer_idx = 1
            elif self._name == 'fx':
                self._optimizer_idx = 2
            elif self._name == 'fy':
                self._optimizer_idx = 3
            elif self._name == 'fz':
                self._optimizer_idx = 4    
            elif self._name == 'tau_z':
                self._optimizer_idx = 5
            else: print('this is not a control optimizer name !')
            for time_idx in range(self.N):     
                self._optimizer_idx_vector[time_idx] = self.nx*(self.N+1) + \
                    time_idx*self.nu + (6*contact_idx) + self._optimizer_idx            
        elif robot_name == 'solo12':
            if self._name == 'fx':
                self._optimizer_idx = 0
            elif self._name == 'fy':
                self._optimizer_idx = 1
            elif self._name == 'fz':
                self._optimizer_idx = 2    
            else: print('this is not a control optimizer name !')
            for time_idx in range(self.N):     
                self._optimizer_idx_vector[time_idx] = self.nx*(self.N+1) + time_idx*self.nu + (3*contact_idx) + self._optimizer_idx 
            

class Dynamics_optimizer():
    def __init__(self, OPTIMIZER_IDENTIFIER, nx_optimizers, nu_optimizers, horizon_length):
        self._name = OPTIMIZER_IDENTIFIER 
        self.nx = nx_optimizers
        self.nu = nu_optimizers
        self.N = horizon_length
        self._x_idx_vector = np.zeros(self.N+1, dtype=int)
        self._u_idx_vector = np.zeros(self.N, dtype=int)
        self.__fill_optimizer_params() 
    
    def __fill_optimizer_params(self):
        if self._name == 'dynamics':
            for time_index in range(self.N+1):
                self._x_idx_vector[time_index] = time_index*self.nx
                if time_index < self.N:
                    self._u_idx_vector[time_index] = self.nx*(self.N+1) + time_index*self.nu
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
            self._x0_optimizer_idx_vector = np.zeros(self.N+1, dtype=int)
            self._slack_optimizers_idx_vector = np.zeros(self.N+1, dtype=int)
            for x_idx in range(self.nx-6):
                self._penum_mat[:, x_idx] = np.array([(-1)**(j//(2**x_idx)) for j in range(2**(self.nx-6))])
        elif self._name == 'control':
            # pre-allocate memory
            self._nb_slack_constraints = self.nt*(2**self.nu)
            self._penum_mat = np.zeros((2**self.nu, self.nu))      
            self._u0_optimizer_idx_vector = np.zeros(self.N, dtype=int)
            self._slack_optimizers_idx_vector = np.zeros(self.N, dtype=int)  
            for u_idx in range(self.nu):
                self._penum_mat[:, u_idx] = np.array([(-1)**(j//(2**u_idx)) for j in range(2**self.nu)])
        
    def __fill_optimizer_params(self):
        if self._name == 'state':
            for time_index in range(self.N+1):
                self._x0_optimizer_idx_vector[time_index] = time_index*self.nx               
                self._slack_optimizers_idx_vector[time_index] = self.nx*(self.N+1) + self.nu*(self.N) + time_index
        elif self._name == 'control':
            for time_index in range(self.N):
                self._u0_optimizer_idx_vector[time_index] = self.nx*(self.N+1) + time_index*self.nu               
                self._slack_optimizers_idx_vector[time_index] = self.nx*(self.N+1) + self.nu*(self.N) + self.nt*(self.N+1) + time_index
        else: print('this not a slack optimizer name !')
       
    