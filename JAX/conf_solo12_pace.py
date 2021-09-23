from contact_plan import Debris
import numpy as np

# walking parameters:
# -------------------
dt = 0.05
mu = 0.1 # linear friction coefficient

# robot contacts
# --------------
robot = 'solo12'
# state and control dimensions
# ----------------------------
n_u_per_contact = 3
nb_contacts = 4
n_u = nb_contacts*n_u_per_contact
n_x = 9
n_t = 1

# pace sequence
# -------------
contact_sequence = [[Debris(CONTACT='FR', t_start=0.00, t_end=1.00, x=0.20,  y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL',t_start=0.00, t_end=1.00, x=0.20, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                     Debris(CONTACT='HR', t_start=0.00, t_end=1.00, x=-0.20, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='HL',t_start=0.00, t_end=1.00, x=-0.20,y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],

                    [Debris(CONTACT='FR', t_start=1.00, t_end=1.25                                                  , ACTIVE=False), Debris(CONTACT='FL',t_start=1.00, t_end=1.25, x=0.20, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                     Debris(CONTACT='HR', t_start=1.00, t_end=1.25                                                  , ACTIVE=False), Debris(CONTACT='HL', t_start=1.00, t_end=1.25,x=-0.20,y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],                                              
                    
                    [Debris(CONTACT='FR', t_start=1.25, t_end=1.50, x=0.30,  y=-0.10, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL', t_start=1.25, t_end=1.50                                               , ACTIVE=False),
                     Debris(CONTACT='HR', t_start=1.25, t_end=1.50, x=-0.10,  y=-0.10, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='HL',t_start=1.25, t_end=1.50                                               , ACTIVE=False)],

                    [Debris(CONTACT='FR', t_start=1.50, t_end=1.75                                                  , ACTIVE=False), Debris(CONTACT='FL', t_start=1.50, t_end=1.75, x=0.40, y=0.10, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                     Debris(CONTACT='HR', t_start=1.50, t_end=1.75                                                  , ACTIVE=False), Debris(CONTACT='HL', t_start=1.50, t_end=1.75, x=0.00, y=0.10, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],

                    [Debris(CONTACT='FR', t_start=1.75, t_end=2.00, x=0.50, y=-0.10, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL', t_start=1.75, t_end=2.00                                               , ACTIVE=False),
                    Debris(CONTACT='HR', t_start=1.75, t_end=2.00, x=0.10, y=-0.10, z=0.0, axis=[-1, 0], angle=0.0 , ACTIVE=True), Debris(CONTACT='HL', t_start=1.75,t_end=2.00                                               , ACTIVE=False)],

                    [Debris(CONTACT='FR', t_start=2.00, t_end=2.25                                                , ACTIVE=False), Debris(CONTACT='FL',t_start=2.00, t_end=2.25, x=0.60, y=0.10, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                    Debris(CONTACT='HR', t_start=2.00, t_end=2.25                                                 , ACTIVE=False), Debris(CONTACT='HL', t_start=2.00, t_end=2.25,x=0.20, y=0.10, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],

                    [Debris(CONTACT='FR', t_start=2.25, t_end=2.50, x=0.70, y=-0.10, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL',t_start=2.25, t_end=2.50                                               , ACTIVE=False),
                    Debris(CONTACT='HR', t_start=2.25, t_end=2.50, x=0.30, y=-0.10, z=0.0, axis=[-1, 0], angle=0.0 , ACTIVE=True), Debris(CONTACT='HL',t_start=2.25, t_end=2.50                                               , ACTIVE=False)],

                    [Debris(CONTACT='FR', t_start=2.50, t_end=2.75                                                , ACTIVE=False), Debris(CONTACT='FL',t_start=2.50, t_end=2.75, x=0.80, y=0.10, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                    Debris(CONTACT='HR', t_start=2.50, t_end=2.75                                                 , ACTIVE=False), Debris(CONTACT='HL',t_start=2.50, t_end=2.75, x=0.40, y=0.10, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],

                    [Debris(CONTACT='FR',t_start=2.75, t_end=3.00, x=0.90, y=-0.10, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL', t_start=2.75, t_end=3.00                                               , ACTIVE=False),
                    Debris(CONTACT='HR', t_start=2.75, t_end=3.00, x=0.50, y=-0.10, z=0.0, axis=[-1, 0], angle=0.0 , ACTIVE=True), Debris(CONTACT='HL',t_start=2.75, t_end=3.00                                               , ACTIVE=False)],

                    [Debris(CONTACT='FR',t_start=3.00, t_end=3.25,                                                 ACTIVE=False), Debris(CONTACT='FL',t_start=3.00, t_end=3.25, x=1.00, y=0.10, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                    Debris(CONTACT='HR', t_start=3.00, t_end=3.25,                                                 ACTIVE=False), Debris(CONTACT='HL',t_start=3.00, t_end=3.25, x=0.60, y=0.10, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],

                    [Debris(CONTACT='FR', t_start=3.25, t_end=3.50, x=1.10, y=-0.10, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL',t_start=3.25, t_end=3.50                                               , ACTIVE=False),
                    Debris(CONTACT='HR', t_start=3.25, t_end=3.50, x=0.70, y=-0.10, z=0.0, axis=[-1, 0], angle=0.0,  ACTIVE=True), Debris(CONTACT='HL',t_start=3.25, t_end=3.50                                               , ACTIVE=False)],
                    
                    [Debris(CONTACT='FR', t_start=3.50, t_end=3.75                                                 , ACTIVE=False), Debris(CONTACT='FL',t_start=3.50, t_end=3.75, x=1.20, y=0.10, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                    Debris(CONTACT='HR', t_start=3.50, t_end=3.75                                                  , ACTIVE=False), Debris(CONTACT='HL',t_start=3.50, t_end=3.75, x=0.80, y=0.10, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],
                    
                    [Debris(CONTACT='FR', t_start=3.75, t_end=4.00, x=1.30, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL', t_start=3.75, t_end=4.00                                              , ACTIVE=False),
                    Debris(CONTACT='HR' , t_start=3.75, t_end=4.00, x=0.90, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='HL',t_start=3.75, t_end=4.00                                               , ACTIVE=False)],
                    
                    [Debris(CONTACT='FR', t_start=4.00, t_end=4.50, x=1.30, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL', t_start=4.00, t_end=4.50, x=1.30, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                    Debris(CONTACT='HR', t_start=4.00, t_end=4.50, x=0.90, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='HL', t_start=4.00, t_end=4.50, x=0.90, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)]] 


N = int(contact_sequence[-1][0].t_end/dt)    

# LQR gains (for stochastic control)
# ----------------------------------
Q = 0.1*np.eye(n_x)
R = 1*np.eye(n_u)

# robot parameters:
# -----------------
robot_mass = 2.2
gravity_constant = -9.81 

# noise parameters:
# -----------------
n_w = nb_contacts*3  # no. of contact position parameters
cov_w = 0.01*np.eye(n_w)
cov_white_noise = 0.001*np.eye(n_x)

# cost objective weights:
# -----------------------
state_cost_weights = np.diag([1e4, 1e3, 1e3, 1e6, 1e6, 1e6, 1e8, 1e8, 1e8])
control_cost_weights = np.diag([1e2, 1e2, 1e1, 
                                1e2, 1e2, 1e1,
                                1e2, 1e2, 1e1,
                                1e2, 1e2, 1e1])
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
          "gamma_fail":                 5,
"convergence_threshold":              1e-3,
      "max_iterations":                 20,
             "padding":                 0.1}