import osqp 
import numpy as np
from scipy import sparse
from warnings import warn
import matplotlib.pyplot as plt 

class SCP:
    # constructor
    def __init__(self, model, data, list_of_costs, list_of_constraints):   
        self.model = model 
        self.data = data
        self.n_x = model._n_x
        self.n_u = model._n_u
        self.n_t = model._n_t
        self.N = model._N
        self.total_nb_optimizers = model._total_nb_optimizers  
        self._cost_objectives = list_of_costs
        self._constraints = list_of_constraints
        self.scp_params = model._scp_params
        self._constraint_indices_range = {'dynamics':None, 'trust_region':None}
        self.trust_region_updates = {'radius':self.scp_params['trust_region_radius0'],
                                     'weight':self.scp_params['omega0']}
        self.all_solution = {'state':[], 'control':[]}
        # Setup QP suproblem using OSQP solver
        self.__sum_up_all_costs()
        self.__stack_up_all_constraints()
        self.prob  = osqp.OSQP()
        self.prob.setup(self._Q, self._p, self._A, self._l, self._u, 
                         warm_start=True, verbose=False)
        print('setting up initial OSQP subproblem ..')
        print("OSQP Problem size: ",
               "Hessian =", self._Q.shape,"gradient =", self._p.shape,
               "Inequality constraints matrix =", self._A.shape,
               "lower bound =", self._l.shape,"upper bound =", self._u.shape)                
        
    def __stack_up_all_constraints(self):
        A = np.array([]).reshape(0, self.total_nb_optimizers)
        l = np.array([])
        u = np.array([])
        list_of_constraints = self._constraints
        for constraint_object in list_of_constraints:
            A = sparse.vstack([A, sparse.csr_matrix(constraint_object._constraints_matrix)], format='csc')
            l = np.hstack([l, constraint_object._lower_bound])
            u = np.hstack([u, constraint_object._upper_bound])
        self._A, self._l, self._u = A, l, u 
   
    def __update_all_constraints(self):
        for constraint_object in self._constraints:
            if constraint_object._CONSTRAINT_IDENTIFIER == 'INITIAL_CONDITIONS':
                print('adding initial condtions constraints ...')
                constraint_object.construct_initial_constraints(self.model) 
            elif constraint_object._CONSTRAINT_IDENTIFIER == 'DYNAMICS':
                 print('adding dynamics constraints ...')    
                 constraint_object.construct_dynamics_constraints(self.model, self.data)
            elif constraint_object._CONSTRAINT_IDENTIFIER == 'COM_REACHABILITY':
                print('adding com reachability constraints')
                constraint_object.construct_com_reachability_constraints()   
            elif constraint_object._CONSTRAINT_IDENTIFIER == 'FINAL_CONDITIONS':
                print('adding final condtions constraints ...')
                constraint_object.construct_final_constraints(self.model)
            elif constraint_object._CONSTRAINT_IDENTIFIER == 'COP':
                print('adding CoP constraints ...')
                constraint_object.construct_cop_constraints()
            elif constraint_object._CONSTRAINT_IDENTIFIER == 'UNILATERAL':
                print('adding unilateral constraints ...')
                constraint_object.construct_unilaterality_constraints()
            elif constraint_object._CONSTRAINT_IDENTIFIER == 'CONTROL_TRUST_REGION':
                print('adding control trust region constraints ...')
                delta, omega = self.trust_region_updates['radius'], self.trust_region_updates['weight'] 
                constraint_object.construct_control_trust_region_constraints(self.data, omega, delta)
            elif constraint_object._CONSTRAINT_IDENTIFIER == 'STATE_TRUST_REGION':
                print('adding state trust region constraints ...')
                delta, omega = self.trust_region_updates['radius'], self.trust_region_updates['weight'] 
                constraint_object.construct_state_trust_region_constraints(self.model, self.data, omega, delta)
        self.__stack_up_all_constraints()    
        self.prob.update(Ax=self._A.data, l=self._l, u=self._u)    

    def __sum_up_all_costs(self):
        Q = np.zeros((self.total_nb_optimizers, self.total_nb_optimizers))
        p = np.zeros(self.total_nb_optimizers)
        for cost_object in self._cost_objectives: 
            Q+= cost_object._hessian
            p+= cost_object._gradient
        # padding to avoid singularities with hessian        
        #Q[0:self.n_x*self.N-1, 0:self.n_x*self.N-1] = 10e-6*np.eye(self.n_x*self.N-1)
        self._Q, self._p = sparse.csc_matrix(Q), p
    
    # trust region 
    def __convergence(self, X, X_curr, U, U_curr):
        l2_norm = np.linalg.norm(U-U_curr, 2)/np.linalg.norm(U_curr, 2) + \
                  np.linalg.norm(X-X_curr, 2)/np.linalg.norm(X_curr, 2)
        return l2_norm 
    
    # computes l2 norm
    def __l2_norm(self, v):return np.linalg.norm(v,2)

    # solve QP using OSQP
    def __solve_subproblem(self):
        QP_FEASIBLITY = True
        self.res = self.prob.solve()
        if self.res.info.status != 'solved':
            warn("[solve_OSQP]: Problem unfeasible.")
            QP_FEASIBLITY = False
        return QP_FEASIBLITY
    
    # ratio between the model error and the linearized model 
    def __compute_model_accuracy(self, X_curr, U_curr, X_prev, U_prev):
        num = den = 0.0
        model, data = self.model, self.data 
        contact_trajectory = self.data.contact_trajectory
        f_traj_prev = np.copy(data.dynamics)
        A_traj_prev, B_traj_prev = np.copy(data.gradients['f_x']), np.copy(data.gradients['f_u'])
        non_linear_model_traj = model.evaluate_dynamics(X_curr, U_curr, contact_trajectory) 
        for time_idx in range(self.N-1): 
            delta_x = X_curr[:, time_idx] - X_prev[:,time_idx]
            delta_u = U_curr[:,time_idx]  - U_prev[:,time_idx] 
            A_k_prev = np.reshape(A_traj_prev[:, time_idx], (self.n_x, self.n_x), order='F')
            B_k_prev = np.reshape(B_traj_prev[:, time_idx], (self.n_x, self.n_u), order='F')
            linear_model = f_traj_prev[:, time_idx] + A_k_prev @ delta_x + B_k_prev @ delta_u 
            model_error = non_linear_model_traj[:, time_idx] - linear_model            
            num += model_error.T @ model_error
            den += linear_model.T @ linear_model 
        return num/den

    def __get_QP_solution(self):
        n_x, n_u, N = self.n_x, self.n_u, self.N
        X_sol = np.reshape(self.res.x[:n_x*(N+1)], (n_x, N+1), order='F')
        U_sol = np.reshape(self.res.x[n_x*(N+1):n_x*(N+1)+n_u*N], (n_u, N), order='F')
        return X_sol, U_sol
    
    def solve_scp(self):
        rho_0, rho_max = self.scp_params['rho0'], self.scp_params['rho1']
        omega_max = self.scp_params['omega_max']
        beta_succ, beta_fail = self.scp_params['beta_succ'], self.scp_params['beta_fail']
        max_iter = self.scp_params['max_iterations']
        conv_thresh = self.scp_params['convergence_threshold']
        gamma_fail = self.scp_params['gamma_fail']
        model = self.model
        data = self.data
        B_success = False
        X_prev, U_prev = np.copy(data.init_trajectories['state']), np.copy(data.init_trajectories['control'])  
        X, U = np.copy(data.init_trajectories['state']), np.copy(data.init_trajectories['control'])
        scp_iteration = 0
        # do till convergence 
        while (scp_iteration < max_iter and self.trust_region_updates['weight'] < omega_max and \
            not(scp_iteration!= 0 and B_success and self.__convergence(X, X_prev, U, U_prev) < conv_thresh)):
            print('\n'+'=' * 50)
            print('Iteration ' + str(scp_iteration))
            print('-' * 50)
            B_success = False
            # propagate dynamics given previous solutions
            print('previous control trajectory = ', U)
            print('previous state trajectory = ', X)
            data.evaluate_dynamics_and_gradients(X, U)
            #print(X_prev)
            #print(data.dynamics)
            #f_x = data.gradients['f_x'][:, 0].reshape(self.n_x, self.n_x, order='F')
            #print('A_0 = ', f_x, '\n')
            #f_u = data.gradients['f_u'][:, 0].reshape(self.n_x, self.n_u, order='F')
            #print('B_0 = ', f_u)
            # updates dynamics constraints and 
            self.__update_all_constraints()
            QP_SOLVED = self.__solve_subproblem()
            X_sol, U_sol = self.__get_QP_solution()
            print('X_SOL = ', X_sol)
            
            # visualize solution at each iteration
            dt = self.model._dt
            fig, (comx, comy, comz, lx, ly, lz, kx, ky, kz) = plt.subplots(9, 1, sharex=True)
            time = np.arange(0, np.round((X_sol.shape[1])*dt, 2),dt)
            comx.plot(time, X_sol[0,:])
            comx.set_title('CoM$_x$')
            plt.setp(comx, ylabel=r'\textbf(m)')
            comy.plot(time, X_sol[1,:])
            comy.set_title('CoM$_y$')
            plt.setp(comy, ylabel=r'\textbf(m)')
            comz.plot(time, X_sol[2,:])
            comz.set_title('CoM$_z$')
            plt.setp(comz, ylabel=r'\textbf(m)')
            lx.plot(time, X_sol[3,:])
            lx.set_title('lin. mom$_x$')
            plt.setp(lx, ylabel=r'\textbf(kg.m/s)')
            ly.plot(time, X_sol[4,:])
            ly.set_title('lin. mom$_y$')
            plt.setp(ly, ylabel=r'\textbf(kg.m/s)')
            lz.plot(time, X_sol[5,:])
            lz.set_title('lin. mom$_z$')
            plt.setp(lz, ylabel=r'\textbf(kg.m/s)')
            kx.plot(time, X_sol[6,:])
            kx.set_title('ang. mom$_x$')
            plt.setp(kx, ylabel=r'\textbf(kg.m$^2$/s)')
            ky.plot(time, X_sol[7,:])
            ky.set_title('ang. mom$_y$')
            plt.setp(ky, ylabel=r'\textbf(kg.m$^2$/s)')
            kz.plot(time, X_sol[8,:])
            kz.set_title('ang. mom$_z$')
            plt.setp(kz, ylabel=r'\textbf(kg.m/s)')
            plt.xlabel(r'\textbf{time} (s)', fontsize=14)
            fig.suptitle('state trajectories', fontsize=20)

            plt.rc('text', usetex = True)
            plt.rc('font', family ='serif')
            dt = self.model._dt
            fig, (copx, copy, fx, fy, fz, tauz) = plt.subplots(6, 1, sharex=True)
            time = np.arange(0, np.round((U_sol.shape[1])*dt, 2),dt)
            copx.plot(time, U_sol[0,:])
            copx.set_title('CoP$_x$')
            plt.setp(copx, ylabel=r'\textbf(m)')
            copy.plot(time, U_sol[1,:])
            copy.set_title('CoP$_y$')
            plt.setp(copy, ylabel=r'\textbf(m)')
            fx.plot(time, U_sol[2,:], label=r'\textbf{z} (N)')
            fx.set_title('F$_x$')
            plt.setp(fx, ylabel=r'\textbf(N)')
            fy.plot(time, U_sol[3,:])
            fy.set_title('F$_y$')
            plt.setp(fy, ylabel=r'\textbf(N)')
            fz.plot(time, U_sol[4,:])
            fz.set_title('F$_z$')
            plt.setp(fz, ylabel=r'\textbf(N)')
            tauz.plot(time, U_sol[5,:])
            tauz.set_title('M$_x$')
            plt.setp(tauz, ylabel=r'\textbf(N.m)')
            plt.xlabel(r'\textbf{time} (s)', fontsize=14)
            fig.suptitle('control trajectories of the right foot', fontsize=20)

            plt.rc('text', usetex = True)
            plt.rc('font', family ='serif')
            dt = self.model._dt
            fig, (copx, copy, fx, fy, fz, tauz) = plt.subplots(6, 1, sharex=True)
            time = np.arange(0, np.round((U_sol.shape[1])*dt, 2),dt)
            copx.plot(time, U_sol[6,:])
            copx.set_title('CoP$_x$')
            plt.setp(copx, ylabel=r'\textbf(m)')
            copy.plot(time, U_sol[7,:])
            copy.set_title('CoP$_y$')    
            plt.setp(copy, ylabel=r'\textbf(m)')
            fx.plot(time, U_sol[8,:], label=r'\textbf{z} (N)')
            fx.set_title('F$_x$')
            plt.setp(fx, ylabel=r'\textbf(N)')
            fy.plot(time, U_sol[9,:])
            fy.set_title('F$_y$')
            plt.setp(fy, ylabel=r'\textbf(N)')
            fz.plot(time, U_sol[10,:])
            fz.set_title('F$_z$')
            plt.setp(fz, ylabel=r'\textbf(N)')
            tauz.plot(time, U_sol[11,:])
            tauz.set_title('M$_x$')    
            plt.setp(tauz, ylabel=r'\textbf(N.m)')
            plt.xlabel(r'\textbf{time} (s)', fontsize=14)
            fig.suptitle('control trajectories of the left foot', fontsize=20)
            plt.show()
             
            # break out when QP fails 
            if QP_SOLVED == False:
                print('QP subproblem Failed at iter #' + str(scp_iteration ))
                return False
            # check if the solution is inside the trust region 
            if self.__l2_norm(X_sol-X_prev) < self.trust_region_updates['radius']:
                print('solution is inside trust region .. checking model accuracy')
                rho = self.__compute_model_accuracy(X_sol, U_sol, X_prev, U_prev)
                # check model accuracy
                if rho > rho_max:
                    print('linearized model is NOT accurate .. rejecting solution and decreasing trust region radius') 
                    self.trust_region_updates['radius'] *= beta_fail 
                    B_success = False
                else:
                    # accept solution
                    print('linearized model is accurate enough .. accepting solution ')
                    self.all_solution['state'].append(X_sol)
                    self.all_solution['control'].append(U_sol)
                    B_success = True                                                      
                    # check if you can decrease trust region radius more than initial guess
                    if rho < rho_0:
                        self.trust_region_updates['radius'] = \
                             np.minimum(beta_succ*self.trust_region_updates['radius'],
                                               self.scp_params['trust_region_radius0'])
            else:
                # reject solution        
                print('solution is outside trust region .. rejecting solution and increasing trust region weight')
                self.trust_region_updates['weight'] *= gamma_fail  
                B_success = False
            scp_iteration +=1
        print('[solve_ccscp] Success: '+str(B_success)+', Nb of iterations: '+str(scp_iteration))

