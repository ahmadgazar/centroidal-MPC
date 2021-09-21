from contact_plan import Debris
from utils import compute_centroid
import numpy as np

robot = 'TALOS'
# walking parameters:
# -------------------
dt = 0.1
contact_duration  = 1.0 # time needed for every step
mu = 0.1                # linear friction coefficient

# state and control dimensions
# ----------------------------
nb_contacts = 2
n_u_per_contact = 6
n_u = nb_contacts*n_u_per_contact
n_x = 9
n_t = 1

# contact_sequence 
# ----------------
contact_sequence = [[Debris(CONTACT='LF', x=0., y=.085, z=0., axis=[1, 0], angle=0., ACTIVE=True), Debris(CONTACT='RF',x=0.,y=-.085,z=0.,axis=[-1, 0],angle=0., ACTIVE=True)],
                    [Debris(CONTACT='LF', x=0., y=.085, z=0., axis=[1, 0], angle=0., ACTIVE=True), Debris(CONTACT='RF', ACTIVE=False)                                      ],
                    [Debris(CONTACT='LF', x=0., y=.085, z=0., axis=[1, 0], angle=0., ACTIVE=True), Debris(CONTACT='RF',x=.2,y=-.2,z=.2,axis=[-1, 0],angle=0.6, ACTIVE=True)],
                    [Debris(CONTACT='LF', ACTIVE=False)                                          , Debris(CONTACT='RF',x=.2,y=-.2,z=.2,axis=[-1, 0],angle=0.6, ACTIVE=True)],
                    [Debris(CONTACT='LF', x=.2, y=.2, z=.2, axis=[1, 0], angle=0.6, ACTIVE=True) , Debris(CONTACT='RF',x=.2,y=-.2,z=.2,axis=[-1, 0],angle=0.6, ACTIVE=True)]] 

nb_steps = len(contact_sequence)
contact_knots = int(round(contact_duration/dt))
N = nb_steps*contact_knots     

# LQR gains (for stochastic control)
# ----------------------------------
Q = 0.1*np.eye(n_x)
R = 1*np.eye(n_u)

# robot parameters:
# -----------------
robot_mass = 95.
gravity_constant = -9.81 
foot_scaling  = 1.
lxp = foot_scaling*0.10  # foot length in positive x direction
lxn = foot_scaling*0.05  # foot length in negative x direction
lyp = foot_scaling*0.05  # foot length in positive y direction
lyn = foot_scaling*0.05  # foot length in negative y direction

# noise parameters:
# -----------------
FRICTION_UNCERTAINTY = 0.90
n_w = nb_contacts*3  # no. of contact position parameters
cov_w = 0.001*np.eye(n_w)
cov_white_noise = 0.001*np.eye(n_x)

# intiial and final conditions:
# -----------------------------
com_z = 0.88  

x_init =  np.array([0., 0., com_z, 0., 0., 0., 0., 0., 0.])
final_contact_sequence = contact_sequence[-1]
vertices = np.array([]).reshape(0, 3)
for contact in final_contact_sequence:
    if contact.ACTIVE:
        vertices = np.vstack([vertices, contact.pose.translation])
centeroid = compute_centroid(vertices)
x_final = np.array([centeroid[0], centeroid[1], com_z+centeroid[2], 0., 0., 0., 0., 0., 0.])
# cost objective weights:
# -----------------------
state_cost_weights = np.diag([1e2, 1e2, 1e2, 1e1, 1e1, 1e7, 1e8, 1e8, 1e8])
control_cost_weights = np.diag([1e6, 1e6, 1e5, 1e5, 1e3, 1e7, 1e6, 1e6, 1e5, 1e5, 1e3, 1e7])

# SCP solver parameters:
# --------------------- 
scp_params  = {"trust_region_radius0":  10,
              "omega0":                100,
           "omega_max":             1.0e10,
             "epsilon":             1.0e-6,
                "rho0":                0.4,
                "rho1":                1.5, 
           "beta_succ":                 2.,
           "beta_fail":                0.5,
          "gamma_fail":                 5.,
"convergence_threshold":              1e-3,
      "max_iterations":                 20,
             "padding":                 0.1}