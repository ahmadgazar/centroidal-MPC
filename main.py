from scp_solver import SCP
from centroidal_model import bipedal_centroidal_model
from trajectory_data import trajectory_data
from contact_plan import create_contact_trajectory 
from cost import Cost
from constraints import Constraints 
import conf 
import numpy as np

np.set_printoptions(linewidth=600)

# create model and data
contact_trajectory = create_contact_trajectory(conf)      
model = bipedal_centroidal_model(conf)
data = trajectory_data(model, contact_trajectory)

# build cost
com_cost = Cost(model, data, contact_trajectory, 'COM_REGULATION')
state_trust_region_cost = Cost(model, data, contact_trajectory, 'STATE_TRUST_REGION')
control_trust_region_cost = Cost(model, data, contact_trajectory, 'CONTROL_TRUST_REGION')
state_control_cost = Cost(model, data, contact_trajectory, 'STATE_CONTROL')
total_cost = [state_control_cost, control_trust_region_cost]

# build constraints
initial_conditions = Constraints(model, data, contact_trajectory, 'INITIAL_CONDITIONS')
dynamics_constraints = Constraints(model, data, contact_trajectory, 'DYNAMICS') 
final_conditions = Constraints(model, data, contact_trajectory, 'FINAL_CONDITIONS')
cop_constraints = Constraints(model, data, contact_trajectory, 'COP')
unilaterality_constraints = Constraints(model, data, contact_trajectory, 'UNILATERAL')
state_trust_region_constraints = Constraints(model, data, contact_trajectory, 'STATE_TRUST_REGION')
control_trust_region_constraints = Constraints(model, data, contact_trajectory, 'CONTROL_TRUST_REGION')
total_constraints = [initial_conditions, dynamics_constraints, final_conditions,
                     cop_constraints, unilaterality_constraints, 
                    control_trust_region_constraints]

# build solver
problem = SCP(model, data, total_cost, total_constraints)
problem.solve_scp()

#utils.plot_state(problem)
#utils.plot_rf_controls(problem)
#utils.plot_lf_controls(problem)
