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


    
