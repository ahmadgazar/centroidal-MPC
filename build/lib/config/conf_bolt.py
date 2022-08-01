from robot_properties_bolt.config import BoltConfig
from robot_properties_bolt.config import BoltHumanoidConfig
from contact_plan import create_contact_sequence
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np

# create robot :
# --------------
URDF = BoltHumanoidConfig.urdf_path#BoltConfig.urdf_path
MESH = BoltHumanoidConfig.meshes_path#BoltConfig.meshes_path
robot = RobotWrapper.BuildFromURDF(URDF, MESH, pin.JointModelFreeFlyer())
rmodel = robot.model
rdata = rmodel.createData()
ee_frame_names = ['FL_ANKLE', 'FR_ANKLE']
# Define the initial state.
# q0 = np.array(
#     [
#         0.0,
#         0.0,
#         0.35487417,
#         0.0,
#         0.0,
#         0.0,
#         1.0,
#         -0.2,
#         0.78539816,
#         -1.57079633,
#         0.2,
#         0.78539816,
#         -1.57079633,
#     ]
# )
q0 =  np.array(
        [
            0.0,
            0.0,
            0.35487417,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.3,
            0.78539816,
            -1.9,
            0.3,
            0.78539816,
            -1.9,
            0.0,
            -.75,
            -.75
        ]
    )
# walking parameters:
# -------------------
DYNAMICS_FIRST = False
dt = 0.01
dt_ctrl = 0.001
gait ={'type': 'PACE',
      'stepLength' : 0.0,
      'stepHeight' : 0.05,
      'stepKnots' : 10,
      'supportKnots' : 2,
      'nbSteps': 5}
mu = 0.5 # linear friction coefficient
gait_templates, contact_sequence = create_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0)
# planning and control horizon lengths:
# -------------------------------------
N = int(round(contact_sequence[-1][0].t_end/dt, 2))
N_ctrl = int((N-1)*(dt/dt_ctrl))    
# whole-body cost objective weights:
# ----------------------------------      
whole_body_task_weights = {'footTrack':{'swing':1e7, 'impact':1e7}, 'impulseVel':20, 'comTrack':1e7, 'stateBounds':0e3, 
                            'stateReg':{'stance':1.5, 'impact':1}, 'ctrlReg':{'stance':1, 'impact':10}, 'frictionCone':5,
                            'centroidalTrack': 1e4, 'contactForceTrack':100}    

# Gepetto viewer:
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
WITHDISPLAY = True 
