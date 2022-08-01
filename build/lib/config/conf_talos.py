from src.contact_plan import create_contact_sequence
import example_robot_data
import numpy as np 
import pinocchio

robot = example_robot_data.load('talos_legs')
rmodel = robot.model
rmodel.name = 'talos'
# rmodel.armature[6:] = .3
rdata = rmodel.createData()
q0 = rmodel.referenceConfigurations['half_sitting'].copy()
ee_frame_names = ['right_sole_link', 'left_sole_link']
# walking parameters:
# -------------------
DYNAMICS_FIRST = False
dt = 0.03
dt_ctrl = 0.001
gait ={'type': 'PACE',
        'stepLength': 0.,
        'stepHeight': 0.1,
        'stepKnots': 15,
        'supportKnots': 5,
        'nbSteps': 4}
# whole-body cost objective weights:
# ----------------------------------      
whole_body_task_weights = {'footTrack':{'swing':1e8, 'impact':1e8}, 'impulseVel':1e6, 'comTrack':1e6, 'stateBounds':0e3, 
                            'stateReg':{'stance':1e1, 'impact':1e1}, 'ctrlReg':{'stance':1e-3, 'impact':1e-3}, 'frictionCone':10,
                            'centroidalTrack': 1e4, 'contactForceTrack':100}    

mu = 0.5 # linear friction coefficient
gait_templates, contact_sequence = create_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0)
# for contacts in contact_sequence:
#     for contact in contacts:
#         if contact.ACTIVE:    
#             print(contact.pose.translation)
#         else: print(contact.CONTACT, "IS NOT ACTIVE")            
# planning and control horizon lengths:
# -------------------------------------
N = int(round(contact_sequence[-1][0].t_end/dt, 2))
N_ctrl = int((N-1)*(dt/dt_ctrl))    

cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
WITHDISPLAY = True 
