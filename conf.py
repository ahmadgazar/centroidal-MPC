from contact_plan import Contact, Debris
from utils import compute_centroid
import numpy as np

# walking parameters:
# -------------------
dt = 0.1
contact_duration  = 0.8 # time needed for every step
mu = 0.1                # linear friction coefficient

# robot contacts
# --------------

# talos feet 
rf = Contact("RF", "flat")
lf = Contact("LF", "flat")
feet = [rf, lf]

# solo12 feet
# fr = Contact("FR", "point")
# fl = Contact("FL", "point")
# hr = Contact("HR", "point")
# hl = Contact("HL", "point")
# feet = [fr, fl, hr, hl]

# state and control dimensions
# ----------------------------
n_u_per_contact = 6
n_u = len(feet)*n_u_per_contact
n_x = 9
n_p = 6 
n_t = 1

# contact_sequence 
# ----------------

# -----
# talos
# -----  

contact_sequence = [[Debris(CONTACT='LF', x=0., y=.085, z=0., axis=[1, 0], angle=0., ACTIVE=True), Debris(CONTACT='RF',x=0.,y=-.085,z=0.,axis=[0, 1],angle=0., ACTIVE=True)],
                    [Debris(CONTACT='LF', x=0., y=.085, z=0., axis=[1, 0], angle=0., ACTIVE=True), Debris(CONTACT='RF', ACTIVE=False)                                      ],
                    [Debris(CONTACT='LF', x=0., y=.085, z=0., axis=[1, 0], angle=0., ACTIVE=True), Debris(CONTACT='RF',x=.2,y=-.2,z=.2,axis=[-1, 0],angle=0.3, ACTIVE=True)],
                    [Debris(CONTACT='LF', ACTIVE=False)                                          , Debris(CONTACT='RF',x=.2,y=-.2,z=.2,axis=[-1, 0],angle=0.3, ACTIVE=True)],
                    [Debris(CONTACT='LF', x=.2, y=.2, z=.2, axis=[1, 0], angle=0.3, ACTIVE=True) , Debris(CONTACT='RF',x=.2,y=-.2,z=.2,axis=[-1, 0],angle=0.3, ACTIVE=True)]] 

# ------
# solo12
# ------ 

# trot sequence
# contact_sequence = [[Debris(CONTACT='FR', x=.42,  y=-.0275, z=0., axis=[0, 0], angle=0., ACTIVE=True), Debris(CONTACT='FL', x=0.42, y=.0275, z=0., axis=[0, 0],angle=0., ACTIVE=True),
#                      Debris(CONTACT='HR', x=-.42, y=-.0275, z=0., axis=[0, 0], angle=0., ACTIVE=True), Debris(CONTACT='HL', x=-0.42, y=.0275, z=0., axis=[0, 0],angle=0., ACTIVE=True)],

#                     [Debris(CONTACT='FR'                                               , ACTIVE=False), Debris(CONTACT='FL', x=0.42, y=.0275, z=0., axis=[0, 0],angle=0., ACTIVE=True),
#                      Debris(CONTACT='HR', x=-.42, y=-.0275, z=0., axis=[0, 0], angle=0., ACTIVE=True), Debris(CONTACT='HL'                                              , ACTIVE=False)],
                    
#                     [Debris(CONTACT='FR', x=.52,  y=-.0275, z=0., axis=[0, 0], angle=0., ACTIVE=True), Debris(CONTACT='FL', x=0.42, y=.0275, z=0., axis=[0, 0],angle=0., ACTIVE=True),
#                      Debris(CONTACT='HR', x=-.42, y=-.0275, z=0., axis=[0, 0], angle=0., ACTIVE=True), Debris(CONTACT='HL', x=-0.32, y=.0275, z=0., axis=[0, 0],angle=0., ACTIVE=True)],

#                     [Debris(CONTACT='FR', x=.52,  y=-.0275, z=0., axis=[0, 0], angle=0., ACTIVE=True), Debris(CONTACT='FL'                                              , ACTIVE=False),
#                      Debris(CONTACT='HR'                                                , ACTIVE=False), Debris(CONTACT='HL', x=-0.32, y=.0275, z=0., axis=[0, 0],angle=0., ACTIVE=True)],

#                     [Debris(CONTACT='FR', x=.52,  y=-.0275, z=0., axis=[0, 0], angle=0., ACTIVE=True), Debris(CONTACT='FL', x=0.52, y=.0275, z=0., axis=[0, 0], angle=0., ACTIVE=True),
#                      Debris(CONTACT='HR', x=-.32, y=-.0275, z=0., axis=[0, 0], angle=0., ACTIVE=True), Debris(CONTACT='HL', x=-0.32, y=.0275, z=0., axis=[0, 0],angle=0., ACTIVE=True)],

#                     ] 
nb_steps = len(contact_sequence)
contact_knots = int(round(contact_duration/dt))
N = nb_steps*contact_knots     

# LQR gains (for stochastic control)
# ----------------------------------
Q = np.zeros((n_x, n_x))
R = 1*np.eye(n_u)

# robot parameters:
# -----------------

# -----
# Talos
# -----

robot_mass = 95.
gravity_constant = -9.81 
foot_scaling  = 1.
lxp = foot_scaling*0.10  # foot length in positive x direction
lxn = foot_scaling*0.05  # foot length in negative x direction
lyp = foot_scaling*0.05  # foot length in positive y direction
lyn = foot_scaling*0.05  # foot length in negative y direction

# ------
# Solo12
# ------
# robot_mass = 2.2
# gravity_constant = -9.81 
# foot_scaling  = 1.
# lxp = foot_scaling*0.01  # foot length in positive x direction
# lxn = foot_scaling*0.01  # foot length in negative x direction
# lyp = foot_scaling*0.01  # foot length in positive y direction
# lyn = foot_scaling*0.01  # foot length in negative y direction

# noise parameters:
# -----------------
Cov_noise = 0.01*np.eye(n_x)
Cov_contact_positions = 0.01*np.eye(n_p)

# intiial and final conditions:
# -----------------------------
com_z = 0.88  # talos
#com_z = 0.24   # solo12

x_init =  np.array([0., 0., com_z, 0., 0., 0., 0., 0., 0.])
final_contact_sequence = contact_sequence[-1]
vertices = np.array([]).reshape(0, 3)
for contact in final_contact_sequence:
    if contact.ACTIVE:
        vertices = np.vstack([vertices, contact.pose.translation])
centeroid = compute_centroid(vertices)
x_final = np.array([centeroid[0], centeroid[1], com_z+centeroid[2], 
                                           0., 0., 0., 0., 0., 0.])
# cost objective weights:
# -----------------------

# ------
# talos 
# ------
state_cost_weights = np.diag([1e2, 1e2, 1e2, 1e6, 1e6, 1e7, 1e8, 1e8, 1e8])
control_cost_weights = np.diag([1e6, 1e6, 1e5, 1e5, 1e3, 1e7, 1e6, 1e6, 1e5, 1e5, 1e3, 1e7])

# ------
# solo12 
# ------
# control_cost_weights = np.diag([1e6, 1e6, 1e5, 1e5, 1e3, 1e7, 1e6, 1e6, 1e5, 1e5, 1e3, 1e7,
#                                 1e6, 1e6, 1e5, 1e5, 1e3, 1e7, 1e6, 1e6, 1e5, 1e5, 1e3, 1e7])
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