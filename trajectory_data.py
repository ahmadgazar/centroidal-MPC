from utils import compute_centroid
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

class Data():
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
            vertices = np.array([]).reshape(0, 3)
            for contact in self.model._contacts:
                 if contact_trajectory[contact._name][time_idx].ACTIVE:
                    vertices = np.vstack([vertices, contact_trajectory[contact._name][time_idx].pose.translation])
            centeroid = compute_centroid(vertices)
            self.previous_trajectories['state'][:, time_idx] = np.array([centeroid[0], centeroid[1], com_z+centeroid[2], 
                                                                                                0., 0., 0., 0., 0., 0.]) 
            self.previous_trajectories['control']+= 1e-7    
        self.previous_trajectories['state'][:, -1] = self.previous_trajectories['state'][:, -2]
        self.init_trajectories = self.previous_trajectories.copy() 

    def evaluate_dynamics_and_gradients(self, X, U):
        N = self.model._N
        contact_trajectory = self.contact_trajectory
        for k in range(N):
            x_k = X[:,k]
            u_k = U[:,k]
            contacts_logic_and_pose = []
            for contact in self.contact_trajectory:
                if contact_trajectory[contact][k].ACTIVE:
                    contact_logic = 1
                    R = contact_trajectory[contact][k].pose.rotation
                    p = contact_trajectory[contact][k].pose.translation.reshape(3,1) 
                else:
                    contact_logic = 0
                    R = np.zeros((3,3))
                    p = np.zeros((3,1))
                contacts_logic_and_pose.append([contact_logic, R, p])     
            contacts_logic_and_pose = list(np.concatenate(contacts_logic_and_pose).flat)
            A_k = self.model._A(x_k, u_k, contacts_logic_and_pose)
            B_k = self.model._B(x_k, u_k, contacts_logic_and_pose)
            self.dynamics[:,k] = self.model._f(x_k, u_k, contacts_logic_and_pose).squeeze()
            self.gradients['f_x'][:,k] = A_k.flatten(order='F')
            self.gradients['f_u'][:,k] = B_k.flatten(order='F')