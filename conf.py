from contact_plan import Debris
import numpy as np

# state and control dimensions
# ----------------------------
n_x = 9
n_u = 12
n_p = 6 

# walking parameters:
# -------------------
dt = 0.1
contact_duration  = 0.8 # time needed for every step

# linear friction coefficient
mu = 0.001 

# terrain parameters:
# -------------------
# Terrain is a list of stepstone. They are defined by the SE3 placement.
# terrain = [     
#         #      x --- y --- z---axis---angle
#         Debris(0., .085, 0., [1, 0], 0., ACTIVE_CONTACT='LF'), 
#         Debris(0.,-.085, 0., [0, 1], 0., ACTIVE_CONTACT='RF'),  
#         Debris(.3, -.3, .3, [-1, 0], 0.5, ACTIVE_CONTACT='RF'),
#         Debris(.3,  .3, .3, [1, 0], 0.5, ACTIVE_CONTACT='LF')]

#                         x (m)  y(m) z(m)  axis  angle(rad)                      x (m)  y(m) z(m)  axis  angle(rad)  
contact_sequence = [[Debris(0., .085, 0., [1, 0], 0., ACTIVE_CONTACT='LF') , Debris(0.,  -.085, 0., [0, 1], 0., ACTIVE_CONTACT='RF')],
                    [Debris(0., .085, 0., [1, 0], 0., ACTIVE_CONTACT='LF') ,                                                  None],
                    [Debris(0., .085, 0., [1, 0], 0., ACTIVE_CONTACT='LF') , Debris(.3, -.3, .3, [-1, 0], 0.5, ACTIVE_CONTACT='RF')],
                    [None                                                  , Debris(.3, -.3, .3, [-1, 0], 0.5, ACTIVE_CONTACT='RF')],
                    [Debris(.3, .3, .3, [1, 0], 0.5, ACTIVE_CONTACT='LF')  , Debris(.3, -.3, .3, [-1, 0], 0.5, ACTIVE_CONTACT='RF')]
                    ] 

nb_steps = len(contact_sequence)
contact_knots = int(round(contact_duration/dt))
N = nb_steps*contact_knots     

# LQR gains 
Q = np.zeros((n_x, n_x))
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
Cov_noise = 0.01*np.eye(n_x)
Cov_contact_positions = 0.01*np.eye(n_p)

# intiial and final conditions:
# -----------------------------
com_z = 0.88
x_init =  np.array([0., 0., com_z, 0., 0., 0., 0., 0., 0.])
last_contact_rf = contact_sequence[-1][1]
last_contact_lf = contact_sequence[-1][0]

# LF active and RF not active
if last_contact_lf and not last_contact_rf:
        com_final =  np.array([last_contact_lf.pose.translation[0], 
                               last_contact_lf.pose.translation[1], 
                        com_z+last_contact_lf.pose.translation[2]])
# RF active and LF not active
elif last_contact_rf and not last_contact_lf:    
        com_final = np.array([last_contact_rf.pose.translation[0], 
                              last_contact_rf.pose.translation[1], 
                        com_z+last_contact_rf.pose.translation[2]])                  
# RF active and LF active                                                                                                  
elif last_contact_rf and last_contact_lf:    
        com_final = np.array([0.5*(last_contact_rf.pose.translation[0]+last_contact_lf.pose.translation[0]), 
                              0.5*(last_contact_rf.pose.translation[1]+last_contact_lf.pose.translation[1]), 
                     0.5*(last_contact_rf.pose.translation[2]+last_contact_lf.pose.translation[2]) + com_z])                                      
x_final = np.array([com_final[0], com_final[1], com_final[2], 0., 0., 0., 0., 0., 0.])

# cost objective weights:
# -----------------------
state_cost_weights = np.diag([1e2, 1e2, 1e2, 1e3, 1e4, 1e6, 1e6, 1e6, 1e6])
control_cost_weights = np.diag([1e6, 1e6, 1e4, 1e4, 1e3, 1e3, 1e6, 1e6, 1e4, 1e4, 1e3, 1e3])

# SCP parameters:
# --------------- 
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