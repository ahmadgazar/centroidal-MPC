from contact_plan import debris
import numpy as np

# Model path
# ----------
#filename = str(os.path.dirname(os.path.abspath(__file__)))
#urdf = '/talos_data/robots/talos_reduced.urdf'
#modelPath = getModelPath(urdf)
#urdf = modelPath + urdf
#srdf = modelPath + '/talos_data/srdf/talos.srdf'
#path = os.path.join(modelPath, '../..')

n_x = 9
n_u = 12
n_p = 6 

# contact frames
nb_contacts = 2
rf_frame_name = 'leg_right_sole_fix_joint'  # right foot frame name
lf_frame_name = 'leg_left_sole_fix_joint'   # left foot frame name

# walking parameters:
# -------------------
dt = 0.1
contact_duration  = 0.8 # time needed for every step

# friction coefficients
mu = 0.001 

# terrain parameters:
# -------------------
# Terrain is a list of stepstone. They are defined by the SE3 placement.
terrain = [     
        #      x --- y --- z---axis---angle
        debris(0, -.17, -1.08, [0, 1], 0.0, ACTIVE_CONTACT='RF'), 
        debris(0, .17, -1.08, [1, 0], 0.0, ACTIVE_CONTACT='LF'),  
        #debris(1, -0.7, -1.05, [1, 1], -.1, ACTIVE_CONTACT='RF'),
        #debris(1.5, 0.7, -1.05, [0, 1], 0.3, ACTIVE_CONTACT='LF'),
        #debris(2, -0.7, -1.05, [1, 2], 0.4, ACTIVE_CONTACT='RF'),
        #debris(3.0, 0.7, -1.05, [1, 1], -.2, ACTIVE_CONTACT='LF') 
         ]

nb_steps = len(terrain)-1
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

# intiial trajectories:
# ---------------------
x_init =  np.array([0., 0., 0.88, 0., 0., 0., 0., 0., 0.])
x_final = np.array([0, 0., 0.88, 0., 0., 0., 0., 0., 0.])

# cost objective weights:
# -----------------------
state_cost_weights = 0*np.diag([1e0, 1e0, 0, 1e0, 1e0, 0, 1e0, 1e0, 1e0])
control_cost_weights = np.diag([1e6, 1e6, 1e4, 1e4, 1, 1, 1e6, 1e6, 1e4, 1e4, 1, 1])

# trust region parameters:
# ------------------------ 
scp_params  = {
        "trust_region_radius0":  1,
        "omega0":                100,
        "omega_max":             1.0e10,
        "epsilon":               1.0e-6,
        "rho0":                  0.4,
        "rho1":                  1.5, 
        "beta_succ":             2.,
        "beta_fail":             0.5,
        "gamma_fail":            5.,
        "convergence_threshold": 1e-3,
        "max_iterations":       20,
        "padding":0.1
    }