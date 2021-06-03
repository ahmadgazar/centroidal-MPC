from scp_solver import SCP
from centroidal_model import Centroidal_model
from trajectory_data import Data
from contact_plan import create_contact_trajectory 
from cost import Cost
from constraints import Constraints 
import conf 
import numpy as np
import sys
import utils

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=500)


# create a contact sequence and contact trajectory 
contact_sequence = conf.contact_sequence
contact_trajectory = create_contact_trajectory(conf)      

# create model and data
model = Centroidal_model(conf)
data = Data(model, contact_sequence, contact_trajectory)

# build cost
com_cost = Cost(model, data, contact_trajectory, 'COM_REGULATION')
state_trust_region_cost = Cost(model, data, contact_trajectory, 'STATE_TRUST_REGION')
#control_trust_region_cost = Cost(model, data, contact_trajectory, 'CONTROL_TRUST_REGION')
state_control_cost = Cost(model, data, contact_trajectory, 'STATE_CONTROL')
total_cost = [state_control_cost, state_trust_region_cost]

# build constraints
initial_conditions = Constraints(model, data, contact_trajectory, 'INITIAL_CONDITIONS')
dynamics_constraints = Constraints(model, data, contact_trajectory, 'DYNAMICS') 
final_conditions = Constraints(model, data, contact_trajectory, 'FINAL_CONDITIONS')
cop_constraints = Constraints(model, data, contact_trajectory, 'COP')
friction_pyramid = Constraints(model, data, contact_trajectory, 'FRICTION_PYRAMID')
unilaterality_constraints = Constraints(model, data, contact_trajectory, 'UNILATERAL')
state_trust_region_constraints = Constraints(model, data, contact_trajectory, 'STATE_TRUST_REGION')
#control_trust_region_constraints = Constraints(model, data, contact_trajectory, 'CONTROL_TRUST_REGION')

total_constraints = [initial_conditions, dynamics_constraints, final_conditions,
                    cop_constraints, friction_pyramid, state_trust_region_constraints]
# build solver
problem = SCP(model, data, total_cost, total_constraints)
problem.solve_scp()

# plot final solution
utils.plot_state(problem)
utils.plot_controls(problem)

