# headers
from optimizer import Control_optimizer, Slack_optimizer, State_optimizer 
from jax.tree_util import register_pytree_node_class
from contact_plan import create_contact_trajectory 
from utils import compute_centroid
from functools import partial 
import conf_solo12 as conf 
import jax.numpy as jnp 
import numpy as np
import jax

# TODO get rid of string members to get a purely pytree class to JIT all functions
@register_pytree_node_class 
class Centroidal_model:
    # constructor
    def __init__(self, conf):
        # protected members
        self._contacts = conf.feet 
        self._n_x = conf.n_x  #x = [com_x, com_y, com_z, lin_mom_x, lin_mom_y, lin_mom_z, ang_mom_x, ang_mom_y, ang_mom_z]
        self._n_u_per_contact = conf.n_u_per_contact
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
        self._state_cost_weights = conf.state_cost_weights     # state weight
        self._control_cost_weights = conf.control_cost_weights # control weight    
        self._linear_friction_coefficient = conf.mu
        self._Q = conf.Q
        self._R = conf.R
        self._robot_foot_range = {'x':np.array([conf.lxp, conf.lxn]),
                                  'y':np.array([conf.lyp, conf.lyn])}
        self._Cov_w = conf.cov_w  
        self._Cov_eta = conf.cov_white_noise                                                           
        # private members
        self.__x = ['com_x', 'com_y', 'com_z', 'lin_mom_x', 'lin_mom_y', 'lin_mom_z', 'ang_mom_x', 'ang_mom_y', 'ang_mom_z']
        self.__u = ['cop_x', 'cop_y', 'fx', 'fy', 'fz', 'tau_z']       
        # private methods
        self.__fill_optimizer_indices()
        self.__fill_contact_data(conf)
        self.__fill_initial_trajectory()
   
    """Returns an iterable over container contents, and aux data."""
    def tree_flatten(model):
        flat_contents = (model._contacts,
                        model._n_x,
                        model._n_u,
                        model._n_u_per_contact, 
                        model._n_u, 
                        model._n_w,    
                        model._n_t, 
                        model._N,   
                        model._total_nb_optimizers, 
                        model._x_init,  
                        model._x_final,  
                        model._com_z,  
                        model._m, 
                        model._g, 
                        model._dt,
                        model._state_cost_weights, 
                        model._control_cost_weights,     
                        model._linear_friction_coefficient, 
                        model._Q, 
                        model._R, 
                        model._robot_foot_range, 
                        model._Cov_w,   
                        model._Cov_eta) 
        return (flat_contents, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
                                               
    # private methods
    def __fill_optimizer_indices(self):  
        self._control_optimizers_indices = {}
        self._state_optimizers_indices = {}
        state_optimizers = []
        # state optimizer indices
        for state in self.__x:
            state_optimizers.append(State_optimizer(state, self._n_x, self._N))
        self._state_optimizers_indices['coms'] = state_optimizers[:2] 
        self._state_optimizers_indices['lin_moms'] = state_optimizers[2:5]
        self._state_optimizers_indices['ang_moms'] = state_optimizers[-1]
        # control optimizers indices for all contacts
        for contact_idx, contact in enumerate(self._contacts):
            contact_optimizers_indices = {}
            control_optimizers = [] 
            for control in self.__u:
                control_optimizers.append(Control_optimizer(control, contact_idx, self._n_x, self._n_u, self._N))
            contact_optimizers_indices['cops'] = control_optimizers[:2] 
            contact_optimizers_indices['forces'] = control_optimizers[2:5]
            contact_optimizers_indices['moment'] = control_optimizers[-1]
            # copy the dictionary of indices of every contact controls
            self._control_optimizers_indices[contact] = contact_optimizers_indices.copy() 
        # state slack indices
        self._state_slack_optimizers_indices = Slack_optimizer('state', self._n_x, self._n_u, self._n_t, self._N)
        # control slack indices 
        #self._control_slack_optimizers_indices = Slack_optimizer('control', self._n_x, self._n_u, self._n_t, self._N) 
    
    def __fill_contact_data(self, conf):
        contact_trajectory = create_contact_trajectory(conf) 
        contacts_logic = []
        contacts_orientation = []
        contacts_position = []
        #TODO Try to figure out how to do nested loops in JAX
        for time_idx in range(self._N):
            contacts_logic_k = []
            contacts_position_k = []
            contacts_orientation_k = [] 
            for contact in contact_trajectory:
                if contact_trajectory[contact][time_idx].ACTIVE:
                    contact_logic = 1
                    R = contact_trajectory[contact][time_idx].pose.rotation
                    p = contact_trajectory[contact][time_idx].pose.translation
                else:
                    contact_logic = 0
                    R = jnp.zeros((3,3))
                    p = jnp.zeros(3)
                contacts_logic_k.append(contact_logic)
                contacts_orientation_k.append(R)
                contacts_position_k.append(p)
            contacts_logic.append(contacts_logic_k)
            contacts_orientation.append(contacts_orientation_k)
            contacts_position.append(jnp.array(contacts_position_k).reshape(len(self._contacts)*3))
        contacts_logic = jnp.array(contacts_logic)
        contacts_orientation = jnp.array(contacts_orientation)
        contacts_position = jnp.array(contacts_position)
        self._contact_trajectory = contact_trajectory
        self._contact_data = dict(contacts_logic=contacts_logic, contacts_orient=contacts_orientation, contacts_position=contacts_position)            
    
    def __fill_initial_trajectory(self):
        N = self._N
        contact_trajectory = self._contact_trajectory
        init_trajectories = {'state':jnp.zeros((self._n_x, N+1)), 'control':jnp.zeros((self._n_u, N))}
        for time_idx in range(N):
            vertices = jnp.array([]).reshape(0, 3)
            for contact in contact_trajectory:
                if contact_trajectory[contact][time_idx].ACTIVE:
                    vertices = jnp.vstack([vertices, contact_trajectory[contact][time_idx].pose.translation])
            centeroid = compute_centroid(vertices)
            init_trajectories['state'] = jax.ops.index_update(init_trajectories['state'], jax.ops.index[:, time_idx],\
                            jnp.array([centeroid[0], centeroid[1], self._com_z+centeroid[2], 0., 0., 0., 0., 0., 0.]))
            init_trajectories['control'] = jax.ops.index_update(init_trajectories['control'], jax.ops.index[:, time_idx], 5e-6)
        init_trajectories['state'] = jax.ops.index_update(init_trajectories['state'], jax.ops.index[:, -1], init_trajectories['state'][:, -2])
        self._init_trajectories = init_trajectories

    @partial(jax.jit, static_argnums=(0,)) 
    def integrate_model_one_step(self, x, u, contacts_position_all, contacts_logic_all, contacts_orientation_all):
        m = self._m
        f_init = jnp.array([(1./m)*x[3], (1./m)*x[4], (1./m)*x[5], 0., 0., m*self._g, 0., 0., 0.])
        u_split = jnp.array(jnp.array_split(u, len(self._contacts)))
        p_split = jnp.array(jnp.array_split(contacts_position_all, len(self._contacts)))
        @jax.jit   
        def dynamics_loop_body(contact_idx, curr):
            u_per_contact  = u_split[contact_idx]
            R_per_contact = contacts_orientation_all[contact_idx, :,:]
            CONTACT_ACTIVATION = contacts_logic_all[contact_idx] 
            p_com = p_split[contact_idx] - x[0:3]
            lin_mom_per_contact = CONTACT_ACTIVATION*jnp.array([u_per_contact[2], u_per_contact[3], u_per_contact[4]])
            ang_mom_per_contact = CONTACT_ACTIVATION*(jnp.cross(p_com, u_per_contact[2:5]) + \
                  jnp.cross((R_per_contact[:,0:2]@u_per_contact[0:2]), u_per_contact[2:5]) + \
                                                         R_per_contact[:,2]*u_per_contact[5])
            curr = jax.ops.index_update(curr, jax.ops.index[3:6], curr[3:6]+lin_mom_per_contact) 
            curr = jax.ops.index_update(curr, jax.ops.index[6:],  curr[6:]+ang_mom_per_contact) 
            return curr 
        return x + jax.lax.fori_loop(0, len(self._contacts), dynamics_loop_body, f_init)*self._dt     
    
    @partial(jax.jit, static_argnums=(0,)) 
    def compute_everything(self, x, u, contacts_position_all, contacts_logic_all, contacts_orientation_all, Sigma_curr):
        @jax.jit 
        def compute_lqr_feedback_gains(A, B, Q, R, niter=3):
            def compute_DARE(A, B, Q, R, P):
                AtP           = A.T @ P
                AtPA          = AtP @ A
                AtPB          = AtP @ B
                RplusBtPB     = R + B.T @ P @ B
                P_minus       = (Q + AtPA) - (AtPB @ jnp.linalg.solve(RplusBtPB, AtPB.T))
                return P_minus
            def loop_body(iter, P):
                return compute_DARE(A, B, Q, R, P)
            P_final = jax.lax.fori_loop(0, niter, loop_body, Q)
            return -jnp.linalg.solve((R + B.T @ P_final @ B), (B.T @ P_final @ A)) #K_final     
        f = self.integrate_model_one_step(x, u, contacts_position_all, contacts_logic_all, contacts_orientation_all)
        A = jax.jit(jax.jacfwd(self.integrate_model_one_step, argnums=0))(x, u, contacts_position_all, contacts_logic_all, contacts_orientation_all)
        B = jax.jit(jax.jacfwd(self.integrate_model_one_step, argnums=1))(x, u, contacts_position_all, contacts_logic_all, contacts_orientation_all)
        C = jax.jit(jax.jacfwd(self.integrate_model_one_step, argnums=2))(x, u, contacts_position_all, contacts_logic_all, contacts_orientation_all)
        K = compute_lqr_feedback_gains(A, B, self._Q, self._R)           
        Sigma_Kt = Sigma_curr @ K.T 
        AB = jnp.hstack([A, B])
        Sigma_xu = jnp.vstack([jnp.hstack([Sigma_curr  , Sigma_Kt]),
                                jnp.hstack([Sigma_Kt.T , K@Sigma_Kt])])      
        Sigma_next = AB @ Sigma_xu @ AB.T + C @ self._Cov_w @ C.T + self._Cov_eta
        Sigma_next_fun = lambda x, u: Sigma_next
        Sigma_x, Sigma_u = jax.jit(jax.jacrev(Sigma_next_fun, argnums=0))(x, u), jax.jit(jax.jacrev(Sigma_next_fun, argnums=1))(x, u)                                     
        return f, A, B, C, K, Sigma_next, Sigma_x, Sigma_u

    @partial(jax.jit, static_argnums=(0,)) 
    def integrate_dynamics_trajectory(self, traj_tuple):
        X, U = traj_tuple['state'], traj_tuple['control']
        N = X.shape[1]
        dynamics_traj = jnp.zeros((X.shape[0], N))
        def loop_body(time_idx, f_traj):
            f = self.integrate_model_one_step(X[:, time_idx], U[:,time_idx], self._contact_data['contacts_position'][time_idx,:], 
                           self._contact_data['contacts_logic'][time_idx, :], self._contact_data['contacts_orient'][time_idx,:,:,:])
            f_traj = jax.ops.index_update(f_traj, jax.ops.index[:, time_idx], f) 
            return f_traj
        return jax.lax.fori_loop(0, N, loop_body, dynamics_traj)    

    @partial(jax.jit, static_argnums=(0,)) 
    def compute_trajectory_data(self, traj_tuple):
        X, U = traj_tuple['state'], traj_tuple['control']
        nx, nu, N = X.shape[0], U.shape[0], U.shape[1]
        traj_data = dict(dynamics=jnp.zeros((nx, N)),
                        LQR_gains=jnp.zeros((N, nu, nx)),
                        gradients={'f_x':jnp.zeros((N, nx, nx)), 
                                   'f_u':jnp.zeros((N, nx, nu)), 
                                   'f_w':jnp.zeros((N, nx, int(nu/2)))},
                             Covs=jnp.zeros((N+1, nx, nx)),
                   Covs_gradients={'Cov_dx':jnp.zeros((N+1, nx, nx, nx)),
                                   'Cov_du':jnp.zeros((N+1, nx, nx, nu))})
        def loop_body(time_idx, traj_data):
            f, A, B, C, K, Sigma_next, dSigma_dx, dSigma_du = self.compute_everything(X[:, time_idx], U[:,time_idx], self._contact_data['contacts_position'][time_idx,:], 
                               self._contact_data['contacts_logic'][time_idx, :], self._contact_data['contacts_orient'][time_idx,:,:,:], traj_data['Covs'][time_idx,:,:]) 
            traj_data['dynamics'] = jax.ops.index_update(traj_data['dynamics'], jax.ops.index[:, time_idx], f)
            traj_data['gradients']['f_x'] = jax.ops.index_update(traj_data['gradients']['f_x'], jax.ops.index[time_idx, :,:], A)
            traj_data['gradients']['f_u'] = jax.ops.index_update(traj_data['gradients']['f_u'], jax.ops.index[time_idx, :,:], B)
            traj_data['gradients']['f_w'] = jax.ops.index_update(traj_data['gradients']['f_w'], jax.ops.index[time_idx, :,:], C)
            traj_data['LQR_gains'] = jax.ops.index_update(traj_data['LQR_gains'], jax.ops.index[time_idx, :,:], K)
            traj_data['Covs'] = jax.ops.index_update(traj_data['Covs'], jax.ops.index[time_idx+1, :,:], Sigma_next)
            traj_data['Covs_gradients']['Cov_dx'] = jax.ops.index_update(traj_data['Covs_gradients']['Cov_dx'], jax.ops.index[time_idx+1,:,:,:], dSigma_dx)
            traj_data['Covs_gradients']['Cov_du'] = jax.ops.index_update(traj_data['Covs_gradients']['Cov_du'], jax.ops.index[time_idx+1,:,:,:], dSigma_du)
            return traj_data 
        return jax.lax.fori_loop(0, conf.N, loop_body, traj_data)      

    def sample_pseudorandom_uncertainties(self):
        key = jax.random.PRNGKey(42) #initial seed 
        contacts_positions = self._contact_data['contacts_position']
        init = dict(key=key, random_contacts_positions=jnp.zeros((self._N, len(self._contact_data['contacts_position'][0]))))
        def contact_loop(time_idx, curr):
            new_key, subkey = jax.random.split(curr['key']) 
            sample_k = jax.random.multivariate_normal(subkey, contacts_positions[time_idx], self._Cov_w)
            curr['random_contacts_positions'] = jax.ops.index_update(curr['random_contacts_positions'], jax.ops.index[time_idx, :], sample_k) 
            curr['key'] = new_key #update seed at every time step (i.i.d)
            return curr
        return jax.lax.fori_loop(0, self._N, contact_loop, init)     

        
