# headers
from optimizer import Control_optimizer, Slack_optimizer, State_optimizer
from utils import skew_sym
import numpy as np
import sympy as sp
import copy 

#helper functions
#Discrete Algebraic Ricatti Equation
def compute_DARE(A, B, Q, R, P):
    AtP           = A.T @ P
    AtPA          = AtP @ A
    AtPB          = AtP @ B
    RplusBtPB_inv = (R + B.T @ P @ B).inv()
    P_minus       = (Q + AtPA) - (AtPB @ RplusBtPB_inv @ (AtPB.T))
    return P_minus

def compute_lqr_feedback_gains(A, B, Q, R, niter=4):
    P = copy.copy(Q)
    for i in range(niter):
        P = compute_DARE(A, B, Q, R, P)
        K = -(R + B.T @ P @ B).inv() @ (B.T @ P @ A)
    return K

class Centroidal_model:
    # constructor
    def __init__(self, conf):
        # protected members
        self._contacts = conf.feet
        for contact_idx, contact in enumerate(self._contacts):
             contact._idx = contact_idx   
        self._n_x = conf.n_x  #x = [com_x, com_y, com_z, lin_mom_x, lin_mom_y, lin_mom_z, ang_mom_x, ang_mom_y, ang_mom_z]
        self._n_u = conf.n_u  #u = [cop_x, cop_y, fx, fy, fz, tau_z]
        self._n_w = conf.n_w   
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
        self._Cov_w = conf.cov_w  
        self._Cov_eta = conf.cov_white_noise                                                           
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
        n_x, n_u, n_w = self._n_x, self._n_u, self._n_w
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
        contact_position_params = []
        for contact in contacts:
            u, ACTIVE, R, p = contact._u, contact._active, contact._orientation, contact._position 
            contact_logic_pose_total.append([ACTIVE, R, p])
            contact_position_params.append([p])
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
        contact_position_params = sp.Matrix(list(np.concatenate(contact_position_params).flat))
        u_total = sp.Matrix(u_total)
        f = sp.simplify(f)
        # discrete dynamics function 
        f = x + (dt * f)
        # Jacobian functions 
        f_x = sp.simplify(f.jacobian(x))       # df/dx
        f_u = sp.simplify(f.jacobian(u_total)) # df/du
        f_w = sp.simplify(f.jacobian(contact_position_params)) # df/dw
        # Hessian functions
        fx_col = f_x.reshape(n_x*n_x, 1)
        fu_col = f_u.reshape(n_x*n_u, 1)
        fw_col = f_w.reshape(n_x*n_w, 1)
        f_xx = sp.simplify(fx_col.jacobian(x))       # d^2f/dx^2
        f_xu = sp.simplify(fx_col.jacobian(u_total)) # d^2f/dxdu
        f_ux = sp.simplify(fu_col.jacobian(x))       # d^2f/du^2
        f_uu = sp.simplify(fu_col.jacobian(u_total)) # d^2f/dudx
        f_wx = sp.simplify(fw_col.jacobian(x))       # d^2f/dwdx
        f_wu = sp.simplify(fw_col.jacobian(u_total)) # d^2f/dwdu
        # LQR function
        Q = sp.MatrixSymbol('Q',n_x,n_x)
        R = sp.MatrixSymbol('R',n_u,n_u)
        A = sp.MatrixSymbol('A',n_x,n_x)
        B = sp.MatrixSymbol('B',n_x,n_u)
        LQR_gains = compute_lqr_feedback_gains(A, B, Q, R)
        # Covariance functions
        K = sp.MatrixSymbol('K', n_u, n_x)
        Cov_curr = sp.MatrixSymbol('Cov_k', n_x, n_x)
        Cov_w = sp.MatrixSymbol('Cov_w', n_w, n_w)
        Cov_eta = sp.MatrixSymbol('Cov_eta', n_x, n_x)
        Cov_Kt_curr = sp.simplify(Cov_curr @ K.T)
        Cov_xu_curr = sp.BlockMatrix([[Cov_curr     , Cov_Kt_curr                ],
                                      [Cov_Kt_curr.T, sp.simplify(K@Cov_Kt_curr)]])
        AB_curr = sp.Matrix([[f_x,f_u]])
        Cov_next = sp.simplify(AB_curr @ Cov_xu_curr @ AB_curr.T + f_w @ Cov_w @ f_w.T + Cov_eta)
        # lambdify functions
        f = sp.lambdify((x, u_total,contact_logic_pose_total), f,'numpy')
        f_x = sp.lambdify((x, u_total,contact_logic_pose_total), f_x,'numpy')
        f_u = sp.lambdify((x, u_total,contact_logic_pose_total), f_u,'numpy')
        f_w = sp.lambdify((x, u_total,contact_logic_pose_total), f_w,'numpy')
        f_xx = sp.lambdify((x, u_total,contact_logic_pose_total), f_xx,'numpy')
        f_ux = sp.lambdify((x, u_total,contact_logic_pose_total), f_ux,'numpy')
        f_xu = sp.lambdify((x, u_total,contact_logic_pose_total), f_xu,'numpy')
        f_uu = sp.lambdify((x, u_total,contact_logic_pose_total), f_uu,'numpy')
        f_wx = sp.lambdify((x, u_total,contact_logic_pose_total), f_wx,'numpy')
        f_wu = sp.lambdify((x, u_total,contact_logic_pose_total), f_wu,'numpy')
        LQR_gains = sp.lambdify((A, B, Q, R), LQR_gains,'numpy')
        Cov_next = sp.lambdify((x, u_total, contact_logic_pose_total, Cov_curr, K, Cov_w, Cov_eta), Cov_next, 'numpy')
        # save functions
        self._f, self._A, self._B, self._C = f, f_x, f_u, f_w
        self._A_dx, self._A_du = f_xx, f_xu 
        self._B_dx, self._B_du = f_ux, f_uu 
        self._C_dx, self._C_du = f_wx, f_wu
        self._LQR_gains, self._Cov_next = LQR_gains, Cov_next  
       
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







    
