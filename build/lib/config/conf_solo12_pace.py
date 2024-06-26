from robot_properties_solo.solo12wrapper import Solo12Config
from src.contact_plan import create_contact_sequence
import example_robot_data
import numpy as np 
import pinocchio

# walking parameters:
# -------------------
DYNAMICS_FIRST = False
dt = 0.01
dt_ctrl = 0.001

gait ={'type': 'PACE',
      'stepLength' : 0.0,
      'stepHeight' : 0.05,
      'stepKnots' : 10,
      'supportKnots' : 3,
      'nbSteps': 4}      

mu = 0.5 # linear friction coefficient

# robot model and parameters
# --------------------------
robot_name = 'solo12'
ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
robot = example_robot_data.load('solo12')
rmodel = robot.model
rdata = rmodel.createData()
robot_mass = pinocchio.computeTotalMass(rmodel)
gravity_constant = -9.81 
max_leg_length = 0.34                          
foot_scaling  = 1.
lxp = 0.01  # foot length in positive x direction
lxn = 0.01  # foot length in negative x direction
lyp = 0.01  # foot length in positive y direction
lyn = 0.01  # foot length in negative y direction

# centroidal state and control dimensions
# ---------------------------------------
n_u_per_contact = 3
nb_contacts = 4
n_u = nb_contacts*n_u_per_contact
n_x = 9
n_t = 1

q0 = np.array(Solo12Config.initial_configuration.copy())
q0[0] = 0.0
gait_templates, contact_sequence = create_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0)
           
# planning and control horizon lengths:
# -------------------------------------
N = int(round(contact_sequence[-1][0].t_end/dt, 2))
N_ctrl = int((N-1)*(dt/dt_ctrl))    

# LQR gains (for stochastic control)      
# ----------------------------------
Q = np.diag([1e4, 1e4, 1e4, 
             1e3, 1e3, 1e3, 
             1e3, 1e3, 1e3])

R = np.diag([1e2,5e2,1e1,
             1e2,5e2,1e1,
             1e2,5e2,1e1,
             1e2,5e2,1e1])

# noise parameters:
# -----------------
n_w = nb_contacts*3  # no. of contact position parameters
# contact position noise
# discretizaton is done inside uncertainty propagation
cov_w = np.diag([0.4**2, 0.4**2, 0.3**2,
                 0.4**2, 0.4**2, 0.3**2,
                 0.4**2, 0.4**2, 0.3**2,
                 0.4**2, 0.4**2, 0.3**2])
# discrete addtive noise
cov_white_noise = dt*np.diag(np.array([0.7**2, 0.5**2, 0.01**2,
                                       0.8**2, 0.6**2, 0.01**2,
                                       0.7**2, 0.5**2, 0.01**2]))
beta_u = 0.01 # probability of constraint violation 

# centroidal cost objective weights:
# ----------------------------------
state_cost_weights = np.diag([1e4, 1e4, 1e4, 1e3, 1e3, 1e3, 1e5, 1e5, 1e5])
control_cost_weights = np.diag([1e2, 1e2, 1e1, 
                                1e2, 1e2, 1e1,
                                1e2, 1e2, 1e1,
                                1e2, 1e2, 1e1])
# whole-body cost objective weights:
# ----------------------------------      
whole_body_task_weights = {'footTrack':{'swing':1e5, 'impact':1e6}, 'impulseVel':20, 'comTrack':1000, 'stateBounds':0e3, 
                            'stateReg':{'stance':0.1, 'impact':1}, 'ctrlReg':{'stance':1, 'impact':10}, 'frictionCone':2,
                            'centroidalTrack': 1e4, 'contactForceTrack':100}                                        

# SCP solver parameters:
# --------------------- 
scp_params  = {"trust_region_radius0":  50, "omega0": 100, "omega_max": 1.0e10, "epsilon": 1.0e-6, "rho0": 0.4, 
               "rho1": 1.5, "beta_succ": 2.,  "beta_fail": 0.5, "gamma_fail": 5, "convergence_threshold": 1e-3, 
               "max_iterations": 20}

# Gepetto viewer:
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
WITHDISPLAY = True
WITHPLOT = True
SAVEDAT = True             