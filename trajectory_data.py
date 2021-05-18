import numpy as np

# helper functions for computing feedback gains
# Discrete Algebraic Ricatti Equation
def compute_DARE(A, B, Q, R, P):
    AtP           = A.T @ P
    AtPA          = AtP @ A
    AtPB          = AtP @ B
    RplusBtPB_inv = np.linalg.inv(R + B.T @ P @ B)
    P_minus       = (Q + AtPA) - (AtPB @ RplusBtPB_inv @ (AtPB.T))
    return P_minus

def compute_lqr_feedback_gain(A, B, Q, R, niter=5):
    P = Q
    for i in range(niter):
        P = compute_DARE(A, B, Q, R, P)
        K = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K

class trajectory_data():
    # constructor
    def __init__(self, model, contact_sequence, contact_trajectory):
        n_x = model._n_x
        n_u, n_p, N = model._n_u, model._n_p, model._N
        self.model = model 
        self.dynamics = np.zeros((n_x, N))
        self.feedback_gains =  np.zeros((N, n_u, n_x))
        self.gradients = {'f_x':np.zeros((n_x*n_x, N)),
                'f_u':np.zeros((n_x*n_u, N)),
                'f_w':np.zeros((n_x*n_p, N)),
                'f_xx':np.zeros((n_x*n_x*n_x, N)),
                'f_ux':np.zeros((n_x*n_u*n_x, N)),
                'f_wx':np.zeros((n_x*n_p*n_x, N)),
                'f_xu':np.zeros((n_x*n_x*n_u, N)),
                'f_uu':np.zeros((n_x*n_u*n_u, N)),
                'f_wu':np.zeros((n_x*n_p*n_u, N))}
        self.contact_sequence = contact_sequence
        self.contact_trajectory = contact_trajectory
        self.previous_trajectories = {'state':np.zeros(shape=[n_x, N+1]),
                                    'control':np.zeros(shape=[n_u, N])}
        self.__initialize_state_and_control_trajectories()  
    
    def __initialize_state_and_control_trajectories(self): 
        contact_trajectory, com_z, N = self.contact_trajectory, self.model._com_z, self.model._N 
        for time_idx in range(N):
            contact_rf = contact_trajectory['RF'][time_idx]
            contact_lf = contact_trajectory['LF'][time_idx]
            # LF active and RF not active
            if contact_lf and not contact_rf:
                self.previous_trajectories['state'][:, time_idx] = np.array([contact_lf.pose.translation[0],
                                                                             contact_lf.pose.translation[1], 
                                                                       com_z+contact_lf.pose.translation[2], 
                                                                                    0., 0., 0., 0., 0., 0.])
            # RF active and LF not active
            elif contact_rf and not contact_lf:    
                self.previous_trajectories['state'][:, time_idx] = np.array([contact_rf.pose.translation[0], 
                                                                             contact_rf.pose.translation[1],
                                                                       com_z+contact_rf.pose.translation[2],   
                                                                                    0., 0., 0., 0., 0., 0.])                  
            # RF active and LF active                                                                                                  
            elif contact_rf and contact_lf:    
                self.previous_trajectories['state'][:, time_idx] = np.array([0.5*(contact_rf.pose.translation[0]+contact_lf.pose.translation[0]), 
                                                                             0.5*(contact_rf.pose.translation[1]+contact_lf.pose.translation[1]), 
                                                                       0.5*(contact_rf.pose.translation[2]+contact_lf.pose.translation[2])+com_z, 
                                                                                                                         0., 0., 0., 0., 0., 0.])                                      
        self.previous_trajectories['state'][:, -1] = self.previous_trajectories['state'][:, -2]
        
        for time_idx in range(N):
            """
            uncomment if you want to warm start the control trajectory with a normal contact force f_z 
            otherwise let the solver figure it out
            """
            # f_max_weight_local = np.array([0., 0., -self.model._m*self.model._g])
            # f_half_weight_local = np.array([0., 0., -0.5*(self.model._m*self.model._g)])
            # # both feet in contact
            # if self.contact_trajectory['RF'][time_idx] and self.contact_trajectory['LF'][time_idx]:
            #     f_world_rf = self.contact_trajectory['RF'][time_idx].pose.rotation @ f_half_weight_local
            #     print("f_world_rf", f_world_rf)
            #     f_world_lf = self.contact_trajectory['LF'][time_idx].pose.rotation @ f_half_weight_local
            #     print("f_world_lf", f_world_lf)
            #     self.previous_trajectories['control'][4,  time_idx] = f_world_rf[2] #fz_rf
            #     self.previous_trajectories['control'][10, time_idx] = f_world_lf[2] #fz_lf   
            # # only RF in contact, and it was previously in contact
            # elif self.contact_trajectory['RF'][time_idx] and not self.contact_trajectory['LF'][time_idx]:
            #     f_world_rf = self.contact_trajectory['RF'][time_idx].pose.rotation @ f_max_weight_local
            #     self.previous_trajectories['control'][4, time_idx]  = f_world_rf[2]  #fz_rf
            #     self.previous_trajectories['control'][10, time_idx] = 0.             #fz_lf   
            # # only LF in contact, and it was previously in contact
            # elif self.contact_trajectory['LF'][time_idx] and not self.contact_trajectory['RF'][time_idx]:
            #     f_world_lf = self.contact_trajectory['LF'][time_idx].pose.rotation @ f_max_weight_local
            #     self.previous_trajectories['control'][4, time_idx] = 0.            #fz_rf
            #     self.previous_trajectories['control'][10,time_idx] = f_world_lf[2]  #fz_lf  
            self.previous_trajectories['control']+= 1e-7    
        self.init_trajectories = self.previous_trajectories.copy() 

    def evaluate_dynamics_and_gradients(self, X, U):
        N = self.model._N
        contact_trajectory = self.contact_trajectory
        for k in range(N):
            x_k = X[:,k]
            u_k = U[:,k]
            if contact_trajectory['LF'][k]:
                LF_ACTIVE=1
                R_lf = contact_trajectory['LF'][k].pose.rotation
                p_lf = contact_trajectory['LF'][k].pose.translation.reshape(3,1)   
            else:
                LF_ACTIVE=0
                R_lf = np.zeros((3,3))
                p_lf = np.zeros((3,1))
            if contact_trajectory['RF'][k]:
                RF_ACTIVE=1
                R_rf = contact_trajectory['RF'][k].pose.rotation
                p_rf = contact_trajectory['RF'][k].pose.translation.reshape(3,1) 
            else:
                RF_ACTIVE=0
                R_rf = np.zeros((3,3))
                p_rf = np.zeros((3,1))
            A_k = self.model._A(x_k, u_k, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf)
            B_k = self.model._B(x_k, u_k, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf)
            #self.feedback_gains[k,:,:] = compute_lqr_feedback_gain(A_k, B_k, self.model._Q, self.model._R)
            self.dynamics[:,k] = self.model._f(x_k, u_k, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf).squeeze()
            self.gradients['f_x'][:,k] = A_k.flatten(order='F')
            self.gradients['f_u'][:,k] = B_k.flatten(order='F')
            #self.gradients['f_w'][:,k] = self.model._C(x_k, u_k, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf).flatten(order='F')
            #self.gradients['f_xx'][:,k] = self.model._A_dx(x_k, u_k, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf).flatten(order='F')
            #self.gradients['f_ux'][:,k] = self.model._B_dx(x_k, u_k, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf).flatten(order='F')
            #self.gradients['f_wx'][:,k] = self.model._C_dx(x_k, u_k, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf).flatten(order='F')
            #self.gradients['f_xu'][:,k] = self.model._A_du(x_k, u_k, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf).flatten(order='F')
            #self.gradients['f_uu'][:,k] = self.model._B_du(x_k, u_k, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf).flatten(order='F')
            #self.gradients['f_wu'][:,k] = self.model._C_du(x_k, u_k, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf).flatten(order='F')
                