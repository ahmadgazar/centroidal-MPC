from utils import compute_centroid
import numpy as np
from sympy import *

class Data():
    # constructor
    def __init__(self, model, contact_sequence, contact_trajectory):
        n_x = model._n_x
        n_u, n_w, N = model._n_u, model._n_w, model._N
        self.model = model 
        self.dynamics = np.zeros((n_x, N))
        self.LQR_gains =  np.zeros((n_u*n_x, N))
        self.gradients = {'f_x':np.zeros((n_x*n_x, N)),
                'f_u':np.zeros((n_x*n_u, N)),
                'f_w':np.zeros((n_x*n_w, N)),
                'f_xx':np.zeros((n_x*n_x*n_x, N)),
                'f_ux':np.zeros((n_x*n_u*n_x, N)),
                'f_wx':np.zeros((n_x*n_w*n_x, N)),
                'f_xu':np.zeros((n_x*n_x*n_u, N)),
                'f_uu':np.zeros((n_x*n_u*n_u, N)),
                'f_wu':np.zeros((n_x*n_w*n_u, N))}
        self.contacts_logic_and_pose = []        
        self.Covs = np.zeros((n_x*n_x, N+1))
        self.contact_sequence = contact_sequence
        self.contact_trajectory = contact_trajectory
        self.previous_trajectories = {'state':np.zeros(shape=[n_x, N+1]),
                                    'control':np.zeros(shape=[n_u, N])}
        # CoM is warm-sarted with a simple zig-zag and rest of the state and controls with zeros                            
        self.__initialize_state_and_control_trajectories()
    
    def __initialize_state_and_control_trajectories(self): 
        contact_trajectory, com_z, N = self.contact_trajectory, self.model._com_z, self.model._N 
        for time_idx in range(N):
            contacts_logic_and_pose = []
            vertices = np.array([]).reshape(0, 3)
            for contact in self.contact_trajectory:
                if contact_trajectory[contact][time_idx].ACTIVE:
                    vertices = np.vstack([vertices, contact_trajectory[contact][time_idx].pose.translation])
                    contact_logic = 1
                    R = contact_trajectory[contact][time_idx].pose.rotation
                    p = contact_trajectory[contact][time_idx].pose.translation.reshape(3,1) 
                else:
                    contact_logic = 0
                    R = np.zeros((3,3))
                    p = np.zeros((3,1))
                contacts_logic_and_pose.append([contact_logic, R, p])     
            contacts_logic_and_pose = list(np.concatenate(contacts_logic_and_pose).flat)
            self.contacts_logic_and_pose.append(contacts_logic_and_pose) 
            centeroid = compute_centroid(vertices)
            self.previous_trajectories['state'][:, time_idx] = np.array([centeroid[0], centeroid[1], com_z+centeroid[2], 
                                                                                                0., 0., 0., 0., 0., 0.]) 
            self.previous_trajectories['control']+= 1e-7    
        self.previous_trajectories['state'][:, -1] = self.previous_trajectories['state'][:, -2]
        self.init_trajectories = self.previous_trajectories.copy() 

    def evaluate_dynamics_and_gradients(self, X, U):
        N = self.model._N
        nx, nu = self.model._n_x, self.model._n_u
        for k in range(N):
            x_k = X[:,k]
            u_k = U[:,k]
            contacts_logic_and_pose = self.contacts_logic_and_pose[k]
            A_k = self.model._A(x_k, u_k, contacts_logic_and_pose)
            B_k = self.model._B(x_k, u_k, contacts_logic_and_pose)
            C_k = self.model._C(x_k, u_k, contacts_logic_and_pose)
            A_dx_k = self.model._A_dx(x_k, u_k, contacts_logic_and_pose)
            A_du_k = self.model._A_du(x_k, u_k, contacts_logic_and_pose)
            B_dx_k = self.model._B_dx(x_k, u_k, contacts_logic_and_pose)
            B_du_k = self.model._B_du(x_k, u_k, contacts_logic_and_pose)
            C_dx_k = self.model._C_dx(x_k, u_k, contacts_logic_and_pose)
            C_du_k = self.model._C_du(x_k, u_k, contacts_logic_and_pose)
            # dynamics
            self.dynamics[:,k] = self.model._f(x_k, u_k, contacts_logic_and_pose).squeeze()
            # first-order derivatives
            self.gradients['f_x'][:,k] = A_k.flatten(order='F')
            self.gradients['f_u'][:,k] = B_k.flatten(order='F')
            self.gradients['f_w'][:,k] = C_k.flatten(order='F')
            # second-order derivatives 
            self.gradients['f_xx'][:,k] = A_dx_k.flatten(order='F')
            self.gradients['f_xu'][:,k] = A_du_k.flatten(order='F')
            self.gradients['f_ux'][:,k] = B_dx_k.flatten(order='F')
            self.gradients['f_uu'][:,k] = B_du_k.flatten(order='F')
            self.gradients['f_wx'][:,k] = C_dx_k.flatten(order='F')
            self.gradients['f_wu'][:,k] = C_du_k.flatten(order='F')  
            # LQR gains
            K = self.model._LQR_gains(A_k, B_k, self.model._Q, self.model._R)
            self.LQR_gains[:,k] = np.reshape(K, (nu*nx), order='F')
            # Covariance propagation
            Cov_curr = np.reshape(self.Covs[:,k], (nx, nx), order='F') 
            self.Covs[:,k+1] = self.model._Cov_next(x_k, u_k, contacts_logic_and_pose, Cov_curr, K, self.model._Cov_w, self.model._Cov_eta).flatten(order='F')
