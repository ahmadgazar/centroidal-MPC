from robot_properties_solo.solo12wrapper import Solo12Config
from contact_plan import create_contact_sequence
from utils import compute_centroid
import example_robot_data
import numpy as np 
import pinocchio

# walking parameters:
# -------------------
DYNAMICS_FIRST = False
dt = 0.01
dt_ctrl = 0.001
gait ={'type': 'TROT',
      'stepLength' : 0.10,
      'stepHeight' : 0.05,
      'stepKnots' : 25,
      'supportKnots' : 10,
      'nbSteps': 4}

mu = 0.7 # linear friction coefficient

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
gait_templates, contact_sequence = create_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0)
           
# intiial and final conditions:
# -----------------------------
if DYNAMICS_FIRST:
      com_z = 0.20   
      x_init =  np.array([0., 0., com_z, 0., 0., 0., 0., 0., 0.])
      final_contact_sequence = contact_sequence[-1]
      vertices = np.array([]).reshape(0, 3)
      for contact in final_contact_sequence:
            if contact.ACTIVE:
                  vertices = np.vstack([vertices, contact.pose.translation])
            centeroid = compute_centroid(vertices)
            x_final = np.array([centeroid[0], centeroid[1], com_z+centeroid[2], 
                                                      0., 0., 0., 0., 0., 0.])
# planning and control horizon lengths:
# -------------------------------------
N = int(round(contact_sequence[-1][0].t_end/dt, 2))
N_ctrl = int((N-1)*(dt/dt_ctrl))    

# LQR gains (for stochastic control)      
# ----------------------------------
Q = 100*np.eye(n_x)
R = 10*np.eye(n_u)

# noise parameters:
# -----------------
n_w = nb_contacts*3  # no. of contact position parameters
cov_w = (0.5**2)*np.eye(n_w) 
cov_white_noise = dt*np.diag(np.array([0.3**2, 0.3**2, 0.3**2,
                                       0.2**2, 0.2**2, 0.2**2,
                                       0.1**2, 0.1**2, 0.1**2]))
beta_u = 0.05 # probability of constraint violation 

# centroidal cost objective weights:
# ----------------------------------
state_cost_weights = np.diag([1e3, 1e3, 1e3, 1e4, 1e4, 1e4, 1e8, 1e8, 1e8])
control_cost_weights = np.diag([1e2, 1e2, 1e1, 
                                1e2, 1e2, 1e1,
                                1e2, 1e2, 1e1,
                                1e2, 1e2, 1e1])
# whole-body cost objective weights:
# ----------------------------------
whole_body_task_weights = {'footTrack':{'swing':1e9, 'impact':1e9}, 'impulseVel':1e3, 'comTrack':1e7, 'stateBounds':1e3, 
                            'stateReg':{'stance':1e3, 'impact':1e3}, 'ctrlReg':{'stance':1e2, 'impact':1e2}, 'frictionCone':1e2,
                            'centroidalTrack': 1e7, 'contactForceTrack':1e2}                                           
# SCP solver parameters:
# --------------------- 
scp_params  = {"trust_region_radius0":  100,
              "omega0":                100,
           "omega_max":             1.0e10,
             "epsilon":             1.0e-6,
                "rho0":                0.4,
                "rho1":                1.5, 
           "beta_succ":                 2.,
           "beta_fail":                0.5,
          "gamma_fail":                 5,
"convergence_threshold":              1e-3,
      "max_iterations":                 20,
             "padding":                 0.1}

# Gepetto viewer:
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
WITHDISPLAY = True
WITHPLOT = True
SAVEDAT = True