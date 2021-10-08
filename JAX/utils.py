import numpy as np
#TODO change to jnp and jit

def construct_friction_pyramid_constraint_matrix(model):
    mu_linear = 0.5*np.sqrt(2)
    pyramid_constraint_matrix = np.array([[1. ,  0., -mu_linear], 
                                    [-1.,  0., -mu_linear],                                     
                                    [0. ,  1., -mu_linear], 
                                    [0. , -1., -mu_linear], 
                                    [0. ,  0., -1.]])
    return pyramid_constraint_matrix

def compute_centroid(vertices):
    centroid = [0., 0., 0.]
    n = len(vertices)
    centroid[0] = np.sum(np.asarray(vertices)[:, 0])/n
    centroid[1] = np.sum(np.asarray(vertices)[:, 1])/n
    centroid[2] = np.sum(np.asarray(vertices)[:, 2])/n
    return centroid

def interpolate_centroidal_traj(conf, data):
    N = conf.N
    N_ctrl = conf.N_ctrl   
    N_inner = int(N_ctrl/N)
    result = {'state':np.zeros((conf.n_x, N_ctrl+N_inner)), 'control':np.zeros((conf.n_u, N_ctrl)),'contact_sequence':np.array([]).reshape(0, 4)}
    for outer_time_idx in range(N+1):
        inner_time_idx = outer_time_idx*N_inner
        result['state'][:, inner_time_idx:inner_time_idx+N_inner] = np.tile(data['state'][:, outer_time_idx], (N_inner,1)).T
        if outer_time_idx < N:
            result['contact_sequence'] = np.vstack([result['contact_sequence'], np.tile(data['contact_sequence'][outer_time_idx], (N_inner, 1))])  
            result['control'][:, inner_time_idx:inner_time_idx+N_inner] = np.tile(data['control'][:, outer_time_idx], (N_inner,1)).T
    return result 


    
