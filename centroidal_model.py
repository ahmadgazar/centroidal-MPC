# headers
from optimizer import Control_optimizer, Slack_optimizer, State_optimizer
from utils import skew_sym
import numpy as np
import sympy as sp
import conf

class Centroidal_model:
    # constructor
    def __init__(self, conf):
        # protected members
        self._contacts = conf.feet
        for contact_idx, contact in enumerate(self._contacts):
             contact._idx = contact_idx   
        self._n_x = conf.n_x  #x = [com_x, com_y, com_z, lin_mom_x, lin_mom_y, lin_mom_z, ang_mom_x, ang_mom_y, ang_mom_z]
        self._n_u = conf.n_u  #u = [cop_x, cop_y, fx, fy, fz, tau_z]
        self._n_p = conf.n_p  #w = [pr_x, pr_y, pr_z, pl_x, pl_y, pl_z] additive noise on contact locations 
        self._n_t = conf.n_t
        self._N = conf.N 
        self._total_nb_optimizers = self._n_x*(self._N+1) + self._n_u*self._N + \
                                    self._n_t*(self._N+1) + self._n_t*self._N
        self._x_init = conf.x_init 
        self._x_final = conf.x_final 
        self._com_z = conf.com_z 
        self._m = conf.robot_mass        # total robot mass
        self._g = conf.gravity_constant  # gravity constant
        self._dt = conf.dt               # discretization time
        self._contact_time = conf.contact_knots
        self._state_cost_weights = conf.state_cost_weights     # state weight
        self._control_cost_weights = conf.control_cost_weights # control weight    
        self._f = sp.zeros(self._n_x, 1)  # dynamics
        self._linear_friction_coefficient = conf.mu
        self._scp_params = conf.scp_params
        self._Q = conf.Q
        self._R = conf.R
        self._robot_foot_range = {'x':np.array([conf.lxp, conf.lxn]),
                                  'y':np.array([conf.lyp, conf.lyn])}
        # private members
        self.__x = sp.Matrix(sp.symbols('com_x, com_y, com_z, lin_mom_x, lin_mom_y, lin_mom_z,\
                                                   ang_mom_x, ang_mom_y, ang_mom_z',real=True))
        # private methods
        self.__compute_symbolic_dynamics_and_analytical_gradients()
        self.__fill_optimizer_indices()
       
    def __fill_optimizer_indices(self):
        self._control_optimizers_indices = {}
        self._state_optimizers_indices = {}
        state_optimizers = []
        # state optimizer indices
        for state in self.__x:
            state_optimizers.append(State_optimizer(str(state), self._n_x, self._N))
        self._state_optimizers_indices['coms'] = state_optimizers[:2] 
        self._state_optimizers_indices['lin_moms'] = state_optimizers[2:5]
        self._state_optimizers_indices['ang_moms'] = state_optimizers[-1]
        # control optimizers indices for all contacts
        for contact_idx, contact in enumerate(self._contacts):
            control_optimizers = []
            for control in contact._u:
                control_optimizers.append(Control_optimizer(str(control).strip('_'+contact._name), contact_idx, 
                                                                                self._n_x, self._n_u, self._N))
            contact._optimizers_indices['cops'] = control_optimizers[:2] 
            contact._optimizers_indices['forces'] = control_optimizers[2:5]
            contact._optimizers_indices['moment'] = control_optimizers[-1]
            # copy the dictionary of indices of every contact controls
            self._control_optimizers_indices[contact._name] = contact._optimizers_indices.copy() 
        # state slack indices
        self._state_slack_optimizers_indices = Slack_optimizer('state', self._n_x, self._n_u, self._n_t, self._N)
        # control slack indices 
        #self._control_slack_optimizers_indices = Slack_optimizer('control', self._n_x, self._n_u, self._n_t, self._N)
    
    def __compute_symbolic_dynamics_and_analytical_gradients(self):
        n_x, n_u = self._n_x, self._n_u
        x, m, g = self.__x, self._m, self._g
        dt, f = self._dt, self._f
        contacts = self._contacts
        f[0,0] = (1./m) * x[3, 0]
        f[1,0] = (1./m) * x[4, 0]
        f[2,0] = (1./m) * x[5, 0]
        f[5,0] =  m*g
        kappa, ang_mom = sp.zeros(3,1), sp.zeros(3,1)
        u_total = []
        contact_logic_pose_total = []
        for contact in contacts:
            u, ACTIVE, R, p = contact._u, contact._active, contact._orientation, contact._position
            contact_logic_pose_total.append([ACTIVE, R, p])
            p_com = p - x[0:3,0]
            kappa = (ACTIVE*skew_sym(p_com)*sp.Matrix(u[2:5])) +\
                (ACTIVE*skew_sym(R[:,0:2]*sp.Matrix(u[0:2,0]))*sp.Matrix(u[2:5]))
            ang_mom += kappa + ACTIVE*R[:,2]*u[5]
            f[3,0]  += ACTIVE*(u[2])
            f[4,0]  += ACTIVE*(u[3])
            f[5,0]  += ACTIVE*(u[4])
            f[6:, 0] = sp.Matrix(ang_mom)
            u_total.append(u)
        contact_logic_pose_total = list(np.concatenate(contact_logic_pose_total).flat)
        u_total = sp.Matrix(u_total)
        f = sp.simplify(f)
        # discretize dynamics
        f = x + (dt * f)
        # Jacobians
        A = sp.simplify(f.jacobian(x))       # df/dx
        B = sp.simplify(f.jacobian(u_total)) # df/du
        # Hessians
        A_col = A.reshape(n_x*n_x, 1)
        B_col = B.reshape(n_x*n_u, 1)
        A_dx = sp.simplify(A_col.jacobian(x)) # d^2f/dx^2
        A_du = sp.simplify(A_col.jacobian(u)) # d^2f/dxdu
        B_dx = sp.simplify(B_col.jacobian(x)) # d^2f/du^2
        B_du = sp.simplify(B_col.jacobian(u)) # d^2f/dudx
        # lambdify functions
        f = sp.lambdify((x, u_total,contact_logic_pose_total), f,'numpy')
        A = sp.lambdify((x, u_total,contact_logic_pose_total), A,'numpy')
        B = sp.lambdify((x, u_total,contact_logic_pose_total), B,'numpy')
        A_dx = sp.lambdify((x, u_total,contact_logic_pose_total), A_dx,'numpy')
        B_dx = sp.lambdify((x, u_total,contact_logic_pose_total), B_dx,'numpy')
        A_du = sp.lambdify((x, u_total,contact_logic_pose_total), A_du,'numpy')
        B_du = sp.lambdify((x, u_total,contact_logic_pose_total), B_du,'numpy')
        self._f, self._A, self._B = f, A, B
        self._A_dx, self._A_du = A_dx, A_du 
        self._B_dx, self._B_du = B_dx, B_du 

    # public method
    def evaluate_dynamics(self, X, U, contact_trajectory):
        N = self._N
        f = np.zeros((self._n_x, N))
        for k in range(N):
            x_k = X[:,k]
            u_k = U[:,k]
            contacts_logic_and_pose = []
            for contact in contact_trajectory:
                if contact_trajectory[contact][k].ACTIVE:
                    contact_logic = 1
                    R = contact_trajectory[contact][k].pose.rotation
                    p = contact_trajectory[contact][k].pose.translation.reshape(3,1) 
                else:
                    contact_logic = 0
                    R = np.zeros((3,3))
                    p = np.zeros((3,1)) 
                contacts_logic_and_pose.append([contact_logic, R, p])
            contacts_logic_and_pose = list(np.concatenate(contacts_logic_and_pose).flat)
            f[:,k] = self._f(x_k, u_k, contacts_logic_and_pose).squeeze()
        return f







    
