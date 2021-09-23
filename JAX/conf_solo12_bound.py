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

# bounding sequence
# -----------------
contact_sequence = [[Debris(CONTACT='FR', t_start=0.00, t_end=1.00, x=0.20,  y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL',t_start=0.00, t_end=1.00, x=0.20, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                     Debris(CONTACT='HR', t_start=0.00, t_end=1.00, x=-0.20, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='HL',t_start=0.00, t_end=1.00, x=-0.20,y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],

                    [Debris(CONTACT='FR', t_start=1.00, t_end=1.20                                                  , ACTIVE=False), Debris(CONTACT='FL',t_start=1.00, t_end=1.20                                              , ACTIVE=False),
                     Debris(CONTACT='HR', t_start=1.00, t_end=1.20, x=-0.20, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='HL', t_start=1.00, t_end=1.20, x=-0.20,y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],
                    
                    [Debris(CONTACT='FR', t_start=1.20, t_end=1.50, x=0.20,  y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL', t_start=1.20, t_end=1.50, x=0.20, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                     Debris(CONTACT='HR', t_start=1.20, t_end=1.50                                                  , ACTIVE=False), Debris(CONTACT='HL',t_start=1.20, t_end=1.50                                              , ACTIVE=False)],

                    [Debris(CONTACT='FR', t_start=1.50, t_end=1.70                                                   ,ACTIVE=False), Debris(CONTACT='FL', t_start=1.50, t_end=1.70                                             , ACTIVE=False),
                     Debris(CONTACT='HR', t_start=1.50, t_end=1.70                                                   , ACTIVE=False), Debris(CONTACT='HL', t_start=1.50, t_end=1.70                                             , ACTIVE=False)],

                    [Debris(CONTACT='FR', t_start=1.70, t_end=2.00                                                 , ACTIVE=False), Debris(CONTACT='FL', t_start=1.70, t_end=2.00                                             , ACTIVE=False),
                    Debris(CONTACT='HR', t_start=1.70, t_end=2.00, x=-0.20, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='HL', t_start=1.70, t_end=2.00, x=-0.20,y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],

                    [Debris(CONTACT='FR', t_start=2.00, t_end=2.20                                                  , ACTIVE=False), Debris(CONTACT='FL',t_start=2.00, t_end=2.20                                             ,ACTIVE=False),
                    Debris(CONTACT='HR', t_start=2.00, t_end=2.20                                                   , ACTIVE=False), Debris(CONTACT='HL', t_start=2.00, t_end=2.20                                            ,ACTIVE=False)],

                    [Debris(CONTACT='FR', t_start=2.20, t_end=2.50, x=0.20,  y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL',t_start=2.20, t_end=2.50, x=0.20, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                    Debris(CONTACT='HR', t_start=2.20, t_end=2.50                                                  , ACTIVE=False), Debris(CONTACT='HL',t_start=2.20, t_end=2.50                                              , ACTIVE=False)],

                    [Debris(CONTACT='FR', t_start=2.50, t_end=2.70                                                , ACTIVE=False), Debris(CONTACT='FL',t_start=2.50, t_end=2.70                                              , ACTIVE=False),
                    Debris(CONTACT='HR', t_start=2.50, t_end=2.70                                                 , ACTIVE=False), Debris(CONTACT='HL',t_start=2.50, t_end=2.70                                              , ACTIVE=False)],

                    [Debris(CONTACT='FR',t_start=2.70, t_end=3.00                                                 , ACTIVE=False), Debris(CONTACT='FL', t_start=2.70, t_end=3.00                                             , ACTIVE=False),
                    Debris(CONTACT='HR', t_start=2.70, t_end=3.00, x=-0.20, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='HL',t_start=2.70, t_end=3.00, x=-0.20,y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],

                    [Debris(CONTACT='FR',t_start=3.00, t_end=3.20                                                 , ACTIVE=False), Debris(CONTACT='FL',t_start=3.00, t_end=3.20                                              , ACTIVE=False),
                    Debris(CONTACT='HR', t_start=3.00, t_end=3.20                                                 , ACTIVE=False), Debris(CONTACT='HL',t_start=3.00, t_end=3.20                                              , ACTIVE=False)],

                    [Debris(CONTACT='FR', t_start=3.20, t_end=3.50, x=0.20,  y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL',t_start=3.20, t_end=3.50, x=0.20, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                    Debris(CONTACT='HR', t_start=3.20, t_end=3.50,                                                  ACTIVE=False), Debris(CONTACT='HL',t_start=3.20, t_end=3.50                                               , ACTIVE=False)],
                    
                    [Debris(CONTACT='FR', t_start=3.50, t_end=3.70                                                 , ACTIVE=False), Debris(CONTACT='FL',t_start=3.50, t_end=3.70                                              , ACTIVE=False),
                    Debris(CONTACT='HR', t_start=3.50, t_end=3.70                                                 , ACTIVE=False), Debris(CONTACT='HL',t_start=3.50, t_end=3.70                                               , ACTIVE=False)],
                    
                    [Debris(CONTACT='FR', t_start=3.70, t_end=4.00                                                 , ACTIVE=False), Debris(CONTACT='FL', t_start=3.70, t_end=4.00                                              , ACTIVE=False),
                    Debris(CONTACT='HR' , t_start=3.70, t_end=4.00, x=-0.20, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='HL',t_start=3.70, t_end=4.00, x=-0.20,y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],

                    [Debris(CONTACT='FR', t_start=4.00, t_end=4.20                                                 , ACTIVE=False), Debris(CONTACT='FL', t_start=4.00, t_end=4.20                                              , ACTIVE=False),
                    Debris(CONTACT='HR' , t_start=4.00, t_end=4.20                                                 , ACTIVE=False), Debris(CONTACT='HL',t_start=4.00, t_end=4.20                                               , ACTIVE=False)],

                    [Debris(CONTACT='FR', t_start=4.20, t_end=4.50, x=0.20,  y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL', t_start=4.20, t_end=4.50, x=0.20, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                    Debris(CONTACT='HR' , t_start=4.20, t_end=4.50                                                 , ACTIVE=False), Debris(CONTACT='HL',t_start=4.20, t_end=4.50                                               , ACTIVE=False)],

                    [Debris(CONTACT='FR', t_start=4.50, t_end=4.70                                                 , ACTIVE=False), Debris(CONTACT='FL', t_start=4.50, t_end=4.70                                              , ACTIVE=False),
                    Debris(CONTACT='HR' , t_start=4.50, t_end=4.70                                                 , ACTIVE=False), Debris(CONTACT='HL',t_start=4.50, t_end=4.70                                               , ACTIVE=False)],
                    
                    [Debris(CONTACT='FR', t_start=4.70, t_end=5.00                                                 , ACTIVE=False), Debris(CONTACT='FL', t_start=4.70, t_end=5.00                                              , ACTIVE=False),
                    Debris(CONTACT='HR' , t_start=4.70, t_end=5.00, x=-0.20, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='HL',t_start=4.70, t_end=5.00, x=-0.20, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)],

                    [Debris(CONTACT='FR', t_start=5.00, t_end=5.20                                                 , ACTIVE=False), Debris(CONTACT='FL', t_start=5.00, t_end=5.20                                              , ACTIVE=False),
                    Debris(CONTACT='HR' , t_start=5.00, t_end=5.20                                                 , ACTIVE=False), Debris(CONTACT='HL',t_start=5.00, t_end=5.20                                               , ACTIVE=False)],

                    [Debris(CONTACT='FR', t_start=5.20, t_end=5.70, x=0.20,  y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL', t_start=5.20, t_end=5.70, x=0.20, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                    Debris(CONTACT='HR' , t_start=5.20, t_end=5.70                                                 , ACTIVE=False), Debris(CONTACT='HL',t_start=5.20, t_end=5.70                                               , ACTIVE=False)],
                    
                    [Debris(CONTACT='FR', t_start=5.70, t_end=5.90, x=0.20,  y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='FL', t_start=5.70, t_end=5.90, x=0.20, y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True),
                    Debris(CONTACT='HR', t_start=5.70, t_end=5.90, x=-0.20, y=-0.15, z=0.0, axis=[-1, 0], angle=0.0, ACTIVE=True), Debris(CONTACT='HL', t_start=5.70, t_end=5.90,x=-0.20,y=0.15, z=0.0, axis=[1, 0],angle=0.0, ACTIVE=True)]] 


N = int(contact_sequence[-1][0].t_end/dt)    
# LQR gains (for stochastic control)
# ----------------------------------
Q = 0.1*np.eye(n_x)
R = 1*np.eye(n_u)

# robot parameters:
# -----------------
robot_mass = 2.2
gravity_constant = -9.81 
foot_scaling  = 1.
lxp = foot_scaling*0.01  # foot length in positive x direction
lxn = foot_scaling*0.01  # foot length in negative x direction
lyp = foot_scaling*0.01  # foot length in positive y direction
lyn = foot_scaling*0.01  # foot length in negative y direction

# noise parameters:
# -----------------
n_w = nb_contacts*3  # no. of contact position parameters
cov_w = 0.01*np.eye(n_w)
cov_white_noise = 0.001*np.eye(n_x)

# cost objective weights:
# -----------------------
state_cost_weights = np.diag([1e3, 1e3, 1e3, 1e6, 1e6, 1e6, 1e8, 1e8, 1e8])
control_cost_weights = np.diag([1e2, 1e2, 1e2, 
                                1e2, 1e2, 1e2,
                                1e2, 1e2, 1e2,
                                1e2, 1e2, 1e2])
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