# headers
from optimizer import Control_optimizer, Dynamics_optimizer, Slack_optimizer, State_optimizer
import numpy as np
import sympy as sp
import conf

# helper function returning a 3x3 skew-symmetric symbolic matrix 
def skew_sym(v): return sp.Matrix([[0      , -v[2,0], v[1,0]], 
                                   [v[2,0] , 0      , -v[0,0]], 
                                   [-v[1,0], v[0,0] , 0]])
    
class bipedal_centroidal_model:
    # constructor
    def __init__(self, conf):
        # protected members
        self._n_x = 9    #x = [com_x, com_y, com_z, l_x, l_y, l_z, k_x, k_y, k_z]
        self._n_u = 2*6  #u = [cop_x, cop_y, F_x, F_y, F_z, tau_z]
        self._n_p = 2*3  #w = [pr_x, pr_y, pr_z, pl_x, pl_y, pl_z] additive noise on contact locations 
        self._n_t = 1
        self._N = conf.N 
        self._total_nb_optimizers = self._n_x*(self._N+1) + self._n_u*self._N + \
                                    self._n_t*(self._N+1) + self._n_t*self._N
        self._x_init = conf.x_init 
        self._x_final = conf.x_final  
        self._m = conf.robot_mass        # total robot mass
        self._g = conf.gravity_constant  # gravity constant
        self._dt = conf.dt               # discretization time
        self._state_cost_weights = conf.state_cost_weights #state weight
        self._control_cost_weights = conf.control_cost_weights #control weight    
        self._f = sp.zeros(self._n_x, 1)  #dynamics
        self._scp_params = conf.scp_params
        self._Q = conf.Q
        self._R = conf.R
        self._optimizers_objects = {
            'com_z': State_optimizer('com_z', self._n_x, self._N),
            'lin_mom_z': State_optimizer('lin_mom_z', self._n_x, self._N),
             'cop_x_rf':Control_optimizer('cop_x_rf', self._n_x, self._n_u, self._N),
             'cop_y_rf':Control_optimizer('cop_y_rf', self._n_x, self._n_u, self._N),
                'fx_rf':Control_optimizer('fx_rf', self._n_x, self._n_u, self._N),
                'fy_rf':Control_optimizer('fy_rf', self._n_x, self._n_u, self._N),
                'fz_rf':Control_optimizer('fz_rf', self._n_x, self._n_u, self._N),
             'tau_z_rf':Control_optimizer('tau_z_rf', self._n_x, self._n_u, self._N),
             'cop_x_lf':Control_optimizer('cop_x_lf', self._n_x, self._n_u, self._N),
             'cop_y_lf':Control_optimizer('cop_y_lf', self._n_x, self._n_u, self._N),
                'fx_lf':Control_optimizer('fx_lf', self._n_x, self._n_u, self._N),
                'fy_lf':Control_optimizer('fy_lf', self._n_x, self._n_u, self._N),
                'fz_lf':Control_optimizer('fz_lf', self._n_x, self._n_u, self._N),
             'tau_z_lf':Control_optimizer('tau_z_lf', self._n_x, self._n_u, self._N),
             'dynamics':Dynamics_optimizer('dynamics', self._n_x, self._n_u, self._N),
             'state_slack':Slack_optimizer('state', self._n_x, self._n_u, self._n_t, self._N),
                'control_slack':Slack_optimizer('control', self._n_x, self._n_u, self._n_t, self._N)}
        self._robot_foot_range = {'x':np.array([conf.lxp, conf.lxn]),
                                  'y':np.array([conf.lyp, conf.lyn])}
        # private members
        self.__x = sp.Matrix(sp.symbols('rx ry rz lx ly lz kx ky kz',real=True))
        self.__RF_ACTIVE = sp.symbols('RF_ACTIVE')
        self.__cop_rf = sp.Matrix(sp.symbols('cop_x_rf cop_y_rf', real=True))
        self.__F_rf = sp.Matrix(sp.symbols('fx_rf fy_rf fz_rf tau_rf'),real=True)
        self.__u_rf = sp.Matrix([self.__cop_rf, self.__F_rf])
        self.__p_rf = sp.MatrixSymbol('p_rf',3,1)
        self.__R_rf = sp.MatrixSymbol('R_rf',3,3)
        self.__LF_ACTIVE = sp.symbols('LF_ACTIVE')
        self.__cop_lf = sp.Matrix(sp.symbols('cop_x_lf cop_y_lf', real=True))
        self.__F_lf = sp.Matrix(sp.symbols('fx_lf fy_lf fz_lf tau_lf'), real=True)
        self.__u_lf = sp.Matrix([self.__cop_lf, self.__F_lf])  
        self.__p_lf = sp.MatrixSymbol('p_lf',3,1) 
        self.__R_lf = sp.MatrixSymbol('R_lf',3,3) 
        self.__Cov_noise = conf.Cov_noise #additive white noise covariance  
        self.__Cov_contact_positions = conf.Cov_contact_positions 
        self.__compute_symbolic_dynamics_and_analytical_gradients()
       
    # private method
    def __compute_symbolic_dynamics_and_analytical_gradients(self):
        n_x, n_u, n_p = self._n_x, self._n_u, self._n_p
        x, m, g = self.__x, self._m, self._g
        dt, f, u_rf, u_lf = self._dt, self._f, self.__u_rf, self.__u_lf
        u = sp.Matrix([u_rf, u_lf])
        RF_ACTIVE, LF_ACTIVE = self.__RF_ACTIVE, self.__LF_ACTIVE   
        p_rf, R_rf = self.__p_rf, self.__R_rf
        p_lf, R_lf = self.__p_lf, self.__R_lf
        p = sp.Matrix([p_rf, p_lf])
        p_to_com_rf = p_rf - x[0:3,0]
        p_to_com_lf = p_lf - x[0:3,0] 
        l_rf = (RF_ACTIVE*skew_sym(p_to_com_rf)*sp.Matrix(u[2:5])) +\
               (RF_ACTIVE*skew_sym(R_rf[:,0:2]*sp.Matrix(u[0:2,0]))*sp.Matrix(u[2:5]))
        l_lf = (LF_ACTIVE*skew_sym(p_to_com_lf)*sp.Matrix(u[8:11])) +\
               (LF_ACTIVE*skew_sym(R_lf[:,0:2]*sp.Matrix(u[6:8,0]))*sp.Matrix(u[8:11]))
        #centroidal dynamics
        f[0,0] = (1./m) * x[3, 0]
        f[1,0] = (1./m) * x[4, 0]
        f[2,0] = (1./m) * x[5, 0]
        f[3,0] = RF_ACTIVE*(u[2]) + LF_ACTIVE*(u[8])
        f[4,0] = RF_ACTIVE*(u[3]) + LF_ACTIVE*(u[9])
        f[5,0] = (m*g) + RF_ACTIVE*(u[4]) + LF_ACTIVE*(u[10])
        ang_mom = l_rf + RF_ACTIVE*R_rf[:,2]*u[5] + \
                  l_lf + LF_ACTIVE*R_lf[:,2]*u[11]
        f[6:,0] = sp.Matrix(ang_mom)
        f = sp.simplify(f)
        # Jacobians
        A = sp.simplify(f.jacobian(x)) # df/dx
        B = sp.simplify(f.jacobian(u)) # df/du
        C = sp.simplify(f.jacobian(p)) # df/dp     
        # Hessians
        A_col = A.reshape(n_x*n_x, 1)
        B_col = B.reshape(n_x*n_u, 1)
        C_col = C.reshape(n_x*n_p, 1)
        A_dx = sp.simplify(A_col.jacobian(x)) # d^2f/dx^2
        A_du = sp.simplify(A_col.jacobian(u)) # d^2f/dxdu
        B_dx = sp.simplify(B_col.jacobian(x)) # d^2f/du^2
        B_du = sp.simplify(B_col.jacobian(u)) # d^2f/dudx
        C_dx = sp.simplify(C_col.jacobian(x)) # d^2f/dpdx
        C_du = sp.simplify(C_col.jacobian(u)) # d^2f/dpdu

        # discretize dynamics
        f = x + (dt * f)
        A = np.eye(n_x) + (A*dt)
        B = B*dt 

        f = sp.lambdify((x, u, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf), f,'numpy')
        A = sp.lambdify((x, u, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf), A,'numpy')
        B = sp.lambdify((x, u, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf), B,'numpy')
        C = sp.lambdify((x, u, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf), C,'numpy')
        A_dx = sp.lambdify((x, u, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf), A_dx,'numpy')
        B_dx = sp.lambdify((x, u, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf), B_dx,'numpy')
        C_dx = sp.lambdify((x, u, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf), C_dx,'numpy')
        A_du = sp.lambdify((x, u, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf), A_du,'numpy')
        B_du = sp.lambdify((x, u, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf), B_du,'numpy')
        C_du = sp.lambdify((x, u, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf), C_du,'numpy')
        self._f, self._A, self._B, self._C = f, A, B, C
        self._A_dx, self._A_du = A_dx, A_du 
        self._B_dx, self._B_du = B_dx, B_du 
        self._C_dx, self._C_du = C_dx, C_du 
    
    def evaluate_dynamics(self, X, U, contact_trajectory):
        N = self._N
        f = np.zeros((self._n_x, N))
        for k in range(N):
            x_k = X[:,k]
            u_k = U[:,k]
            if contact_trajectory['LF'][k]:
                LF_ACTIVE=1
                R_lf = contact_trajectory['LF'][k].pose.rotation
                p_lf = contact_trajectory['LF'][k].pose.translation.reshape(3,1)   
            else:
                LF_ACTIVE=0
                R_lf = np.zeros((3,3))
                p_lf = np.zeros((3,1))
            if contact_trajectory['RF'][k]:
                RF_ACTIVE=1
                R_rf = contact_trajectory['RF'][k].pose.rotation
                p_rf = contact_trajectory['RF'][k].pose.translation.reshape(3,1)  
            else:
                RF_ACTIVE=0
                R_rf = np.zeros((3,3))
                p_rf = np.zeros((3,1))
            f[:,k] = self._f(x_k, u_k, RF_ACTIVE, R_rf, p_rf, LF_ACTIVE, R_lf, p_lf).squeeze()
        return f

 




   

      
