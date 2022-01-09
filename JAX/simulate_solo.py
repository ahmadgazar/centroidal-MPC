from robot_properties_solo.solo12wrapper import Solo12Robot
from bullet_utils.env import BulletEnvWithGround
import jax.numpy as jnp 
import pinocchio as pin
import numpy as np
import pybullet
import jax 
import time

class Simulator:
  def __init__(self, sim_env, robot_wrapper, conf):
    self.robot = sim_env.add_robot(robot_wrapper)
    self.q0 = conf.q0 
    self.nu0 = np.zeros(self.robot.pin_robot.model.nv)
    self.ee_frame_names = conf.ee_frame_names 
    self.dt_ctrl = conf.dt_ctrl
    self.dt_plan = conf.dt
    self.N = conf.N_ctrl
    self.m = conf.robot_mass
    self.cov = conf.cov_white_noise
    mu = conf.mu/np.sqrt(2)
    self.pyramid_constraint_matrix = np.array([[1. ,  0., -mu], 
                                    [-1.,  0., -mu],                                     
                                    [0. ,  1., -mu], 
                                    [0. , -1., -mu],
                                    [0. ,  0., -1.]])

  def sample_pseudorandom_centroidal_uncertainties(self, key):
    curr = dict(key=key, centroidal_uncertainties=jnp.zeros((self.N, 6)))
    def contact_loop(time_idx, curr):
        new_key, subkey = jax.random.split(curr['key'])
        com_sample_k = jax.random.multivariate_normal(subkey, np.zeros(3), self.cov[:3, :3]/10)
        lin_mom_sample_k = (self.m*com_sample_k)/10 + jax.random.multivariate_normal(subkey, np.zeros(3), 
                                                                                  self.cov[3:6, 3:6]/10)
        ang_mom_sample_k = jax.random.multivariate_normal(subkey, np.zeros(3), self.cov[6:, 6:]/10)                                   
        hg_tilde = jnp.hstack([lin_mom_sample_k, ang_mom_sample_k])
        curr['centroidal_uncertainties'] = jax.ops.index_update(curr['centroidal_uncertainties'], 
                                                            jax.ops.index[time_idx, :], hg_tilde) 
        curr['key'] = new_key  
        return curr
    return jax.lax.fori_loop(0, self.N, contact_loop, curr)     
  
  def sample_pseudorandom_centroidal_uncertainties_total(self, key, nb_sims):
    curr = dict(key=key, centroidal_uncertainties_total=jnp.zeros((nb_sims, self.N, 6)))
    def sim_loop(sim, curr):
        new_key, subkey = jax.random.split(curr['key'])
        hg_tilde = self.sample_pseudorandom_centroidal_uncertainties(subkey)['centroidal_uncertainties']
        curr['centroidal_uncertainties_total'] = jax.ops.index_update(curr['centroidal_uncertainties_total'], 
                                                 jax.ops.index[sim, :, :], hg_tilde)
        curr['key'] = new_key
        return curr 
    return jax.lax.fori_loop(0, nb_sims, sim_loop, curr)    
      
  def get_contact_positions_and_forces(self):
    foot_link_ids = tuple(self.robot.bullet_endeff_ids)
    contact_forces = [np.zeros(3) for _ in range(len(foot_link_ids))]
    contact_positions = [np.zeros(3) for _ in range(len(foot_link_ids))]
    friction_cone_violations = [0 for _ in range(len(foot_link_ids))]
    all_contacts = pybullet.getContactPoints(bodyA=self.robot.robot_id)
    for contact in all_contacts:
      (unused_flag, body_a_id, body_b_id, link_a_id, unused_link_b_id,
       unused_pos_on_a, unused_pos_on_b, contact_normal_b_to_a, unused_distance,
       normal_force, friction_1, friction_direction_1, friction_2,
       friction_direction_2) = contact
      # Ignore self contacts
      if body_b_id == body_a_id:
        continue
      if link_a_id in foot_link_ids:
        normal_force = np.array(contact_normal_b_to_a) * normal_force
        friction_force = np.array(friction_direction_1) * friction_1 + np.array(
            friction_direction_2) * friction_2
        force = normal_force + friction_force
        force_norm = np.linalg.norm(force)
        toe_link_order = foot_link_ids.index(link_a_id)
        if force_norm >= 1e-8:
          for idx in range(self.pyramid_constraint_matrix.shape[0]):
            if self.pyramid_constraint_matrix[idx, :] @ force <= 0.0:
              continue
              # pybullet.addUserDebugLine(lineFromXYZ=unused_pos_on_a, lineToXYZ=force, 
                # parentLinkIndex=link_a_id, lifeTime=0, lineColorRGB=[0, 1, 0], lineWidth=5.0)
            else:
              friction_cone_violations[toe_link_order]+= 1
                # pybullet.addUserDebugLine(lineFromXYZ=unused_pos_on_a, lineToXYZ=force, 
                # parentLinkIndex=link_a_id, lifeTime=0, lineColorRGB=[1, 0, 0], lineWidth=5.0)
          contact_forces[toe_link_order] += force
          contact_positions[toe_link_order] += unused_pos_on_a
      else:
        continue
    return contact_positions, contact_forces, friction_cone_violations
  
  def project_centoridal_to_WBD_uncertainties(self, contacts_logic, q, delta_hg):
    rmodel, rdata = self.robot.pin_robot.model, self.robot.pin_robot.data 
    ee_frame_names = self.ee_frame_names
    nv, N = rmodel.nv, delta_hg.shape[1]
    nb_sims = delta_hg.shape[0]
    tau_w = np.empty((nb_sims, N, nv-6))
    for sim in range(nb_sims):
      for time_idx in range(N):
        qk = q[time_idx]
        Jc_stacked = np.array([]).reshape(0, nv)
        for contact_idx, logic in enumerate(contacts_logic[time_idx]):
            if logic:
              foot_idx = rmodel.getFrameId(ee_frame_names[contact_idx])
              foot_jacobian_local = pin.getFrameJacobian(rmodel, rdata, foot_idx, pin.ReferenceFrame.LOCAL)
              world_R_foot = pin.SE3(rdata.oMf[foot_idx].rotation, np.zeros(3))
              Jc_stacked = np.vstack([Jc_stacked, world_R_foot.action.dot(foot_jacobian_local)[:3]]) 
        M = pin.crba(rmodel, rdata, qk)
        Ag = pin.computeCentroidalMap(rmodel, rdata, qk)
        null_space_projector = np.eye(nv) - (np.linalg.pinv(np.asarray(Jc_stacked)) @ np.asarray(Jc_stacked))
        tau_w[sim, time_idx, :] = M[6:] @ null_space_projector @ np.linalg.pinv(Ag) @ delta_hg[sim, time_idx, :] 
    return tau_w

  def run(self, tau_ff, q_des, qdot_des, nb_sims=1, tau_tilde=None): 
      nq, nv = self.robot.pin_robot.model.nq, self.robot.pin_robot.model.nv  
      Nu, Nx, N_inner = tau_ff.shape[0], q_des.shape[0], int(self.dt_plan/self.dt_ctrl)
      q0, dq0 = self.q0, self.nu0
      q0[0] = 0.0
      self.robot.reset_state(q0, dq0) 
      self.robot.pin_robot.framesForwardKinematics(q0)
      # pre-allocate memory for data logging
      q_sim = np.empty((nb_sims, Nx, nq))
      qdot_sim = np.empty((nb_sims, Nx, nv))
      centroidal_dynamics_sim = np.empty((nb_sims, Nx, 9))
      contact_forces_N = []
      contact_positions_N = []
      constraint_violations_N = []
      contact_forces_sim = []
      contact_positions_sim = []
      constraint_violations_sim = []
      # fill initial states
      centroidal_dynamics_sim[:, :N_inner, :3] = pin.centerOfMass(self.robot.pin_robot.model, 
                                                  self.robot.pin_robot.data, q0, dq0)
      self.robot.pin_robot.centroidalMomentum(q0, dq0)    
      q_sim[:, :N_inner, :], qdot_sim[:, :N_inner, :] = q0, dq0 
      centroidal_dynamics_sim[:, :N_inner, 3:9] = np.array(self.robot.pin_robot.data.hg)
      # PD gains
      Kp = 30*np.eye(12)
      Kd = 0.4*np.eye(12)
      for sim in range(nb_sims):
        for time_idx in range(Nu):
            if tau_tilde is None:
              tau_tilde_k = np.zeros(nv-6)
            else:
              tau_tilde_k = tau_tilde[sim, time_idx]  
            q, dq = self.robot.get_state()
            # tau = tau_ff + Kp(q_des - q) + Kd(qdot_des - qdot) + tau_tilde
            tau = tau_ff[time_idx, :] + Kp @ (q_des[time_idx][7:] - q[7:]) +\
                        Kd @ (qdot_des[time_idx][6:]- dq[6:]) + tau_tilde_k    
            # apply joint torques plus disturbance
            self.robot.send_joint_command(tau)
            # get robot state after applying disturbance
            q_tilde, dq_tilde = self.robot.get_state()
            self.robot.pin_robot.centroidalMomentum(q_tilde, dq_tilde)
            p_k, f_k, violations = self.get_contact_positions_and_forces()
            # save data 
            hg_tilde = np.array(self.robot.pin_robot.data.hg)
            centroidal_dynamics_sim[sim, time_idx+N_inner, :3] = pin.centerOfMass(self.robot.pin_robot.model, 
                                                          self.robot.pin_robot.data, q_tilde, dq_tilde)
            centroidal_dynamics_sim[sim, time_idx+N_inner, 3:9] = hg_tilde 
            q_sim[sim, time_idx+N_inner,:] = q_tilde
            qdot_sim[sim, time_idx+N_inner, :] = dq_tilde
            contact_positions_N += [p_k]
            contact_forces_N += [f_k]
            constraint_violations_N +=[violations]
            # step simulation 
            env.step(sleep=True) 
        # reset robot to original state for the new simulation
        self.robot.reset_state(q0, dq0) 
        self.robot.pin_robot.framesForwardKinematics(q0)
        contact_positions_sim += [contact_positions_N]
        contact_forces_sim += [contact_forces_N]
        constraint_violations_sim += [constraint_violations_N]
        contact_forces_N = []
        contact_positions_N = []
        constraint_violations_N = []
      return dict(centroidal=centroidal_dynamics_sim, q=q_sim, qdot=qdot_sim, 
             contact_positions=contact_positions_sim, contact_forces=contact_forces_sim,
             constraint_violations=constraint_violations_sim) 

if __name__ == "__main__":
    import matplotlib.pylab as plt
    import conf_solo12_trot as conf 
    # import conf_solo12_pace as conf
    # import conf_solo12_bound as conf
    from centroidal_model import Centroidal_model
    env = BulletEnvWithGround()
    # env.set_floor_frictions(lateral=0.7)
    # load optimized nominal trajectories
    tau_ff_nom = np.load('wholeBody_interpolated_traj.npz')['U']
    q_des_nom = np.load('wholeBody_interpolated_traj.npz')['q']
    qdot_des_nom = np.load('wholeBody_interpolated_traj.npz')['qdot']
    centroidal_des = np.load('wholeBody_interpolated_traj.npz')['X']
    # load optimized stochastic trajectories
    tau_ff_stoch = np.load('wholeBody_interpolated_traj_stoch.npz')['U']
    q_des_stoch = np.load('wholeBody_interpolated_traj_stoch.npz')['q']
    qdot_des_stoch = np.load('wholeBody_interpolated_traj_stoch.npz')['qdot']
    centroidal_des_stoch = np.load('wholeBody_interpolated_traj_stoch.npz')['X']
    # get contact sequence
    model = Centroidal_model(conf)
    contact_sequence = model._contact_data['contacts_logic']
    # create a pybullet simulation environment and
    # sample pseudorandom centroidal disturbances 
    nb_sims = 20
    initial_seed = jax.random.PRNGKey(42)
    simulator = Simulator(env, Solo12Robot(), conf)
    w_hg = simulator.sample_pseudorandom_centroidal_uncertainties_total(initial_seed, nb_sims)
    tau_tilde = simulator.project_centoridal_to_WBD_uncertainties(contact_sequence, q_des_nom, 
                                                       w_hg['centroidal_uncertainties_total'])
    # run monte-carlo simulations 
    data_nom = simulator.run(tau_ff_nom, q_des_nom, qdot_des_nom, nb_sims, tau_tilde)
    # data_stoch = simulator.run(tau_ff_stoch, q_des_stoch, qdot_des_stoch, nb_sims, tau_tilde)
    # print('number of friction pyramid constraint violations nominal  = ', np.sum(data_nom['constraint_violations']))
    # print('number of friction pyramid constraint violations stochastic  = ', np.sum(data_nom['constraint_violations']))

    # --------------------------------------------------
    # plot desired and simulated centroidal trajectories  
    # --------------------------------------------------
    X_des = centroidal_des
    centroidal_sim_nom = data_nom['centroidal']
    # centroidal_sim_stoch = data_stoch['centroidal']
    fig, (comx, comy, comz, lx, ly, lz, kx, ky, kz) = plt.subplots(9, 1, sharex=True)
    time = np.arange(0, np.round((X_des.shape[0])*conf.dt, 2),conf.dt)
    comx.plot(time, X_des[:, 0], label='des', color='red')
    comx.set_title('CoM$_x$')
    plt.setp(comx, ylabel=r'\textbf(m)')
    comy.plot(time, X_des[:,1], label='des', color='red')
    comy.set_title('CoM$_y$')
    plt.setp(comy, ylabel=r'\textbf(m)')
    comz.plot(time, X_des[:,2], label='des', color='red')
    comz.set_title('CoM$_z$')
    plt.setp(comz, ylabel=r'\textbf(m)')
    lx.plot(time, X_des[:,3], label='des', color='red')
    lx.set_title('lin. mom$_x$')
    plt.setp(lx, ylabel=r'\textbf(kg.m/s)')
    ly.plot(time, X_des[:,4], label='des', color='red')
    ly.set_title('lin. mom$_y$')
    plt.setp(ly, ylabel=r'\textbf(kg.m/s)')
    lz.plot(time, X_des[:,5], label='des', color='red')
    lz.set_title('lin. mom$_z$')
    plt.setp(lz, ylabel=r'\textbf(kg.m/s)')
    kx.plot(time, X_des[:,6], label='des', color='red')
    kx.set_title('ang. mom$_x$')
    plt.setp(kx, ylabel=r'\textbf(kg.m$^2$/s)')
    ky.plot(time, X_des[:,7], label='des', color='red')
    ky.set_title('ang. mom$_y$')
    plt.setp(ky, ylabel=r'\textbf(kg.m$^2$/s)')
    kz.plot(time, X_des[:,8], label='des', color='red')
    kz.set_title('ang. mom$_z$')
    plt.setp(kz, ylabel=r'\textbf(kg.m/s)')
    plt.xlabel(r'\textbf{time} (s)', fontsize=14)
    fig.suptitle('centroidal trajectories', color='red')
    for sim in range(nb_sims):
      # nominal 
      comx.plot(time, centroidal_sim_nom[sim, :, 0], color='blue')
      comy.plot(time, centroidal_sim_nom[sim, :, 1], color='blue')
      comz.plot(time, centroidal_sim_nom[sim, :, 2],color='blue')
      lx.plot(time, centroidal_sim_nom[sim, :, 3], color='blue')
      ly.plot(time, centroidal_sim_nom[sim, :, 4], color='blue')
      lz.plot(time, centroidal_sim_nom[sim, :, 5], color='blue')
      kx.plot(time, centroidal_sim_nom[sim, :, 6], color='blue')
      ky.plot(time, centroidal_sim_nom[sim, :, 7], color='blue')
      kz.plot(time, centroidal_sim_nom[sim, :, 8], color='blue')
      # stochastic
      # comx.plot(time, centroidal_sim_stoch[sim, :, 0], color='green')
      # comy.plot(time, centroidal_sim_stoch[sim, :, 1],color='green')
      # comz.plot(time, centroidal_sim_stoch[sim, :, 2],color='green')
      # lx.plot(time, centroidal_sim_stoch[sim, :, 3], color='green')
      # ly.plot(time, centroidal_sim_stoch[sim, :, 4], color='green')
      # lz.plot(time, centroidal_sim_stoch[sim, :, 5], color='green')
      # kx.plot(time, centroidal_sim_stoch[sim, :, 6], color='green')
      # ky.plot(time, centroidal_sim_stoch[sim, :, 7], color='green')
      # kz.plot(time, centroidal_sim_stoch[sim, :, 8], color='green')
      if sim == nb_sims-1:
        # nominal 
        comx.plot(time, centroidal_sim_nom[sim, :, 0],color='blue', label='nominal')
        comy.plot(time, centroidal_sim_nom[sim, :, 1],color='blue', label='nominal')
        comz.plot(time, centroidal_sim_nom[sim, :, 2],color='blue', label='nominal')
        lx.plot(time, centroidal_sim_nom[sim, :, 3], color='blue', label='nominal')
        ly.plot(time, centroidal_sim_nom[sim, :, 4], color='blue', label='nominal')
        lz.plot(time, centroidal_sim_nom[sim, :, 5], color='blue', label='nominal')
        kx.plot(time, centroidal_sim_nom[sim, :, 6], color='blue', label='nominal')
        ky.plot(time, centroidal_sim_nom[sim, :, 7], color='blue', label='nominal')
        kz.plot(time, centroidal_sim_nom[sim, :, 8], color='blue', label='nominal')
        # stochastic
        # comx.plot(time, centroidal_sim_stoch[sim, :, 0], color='green', label='stochastic')
        # comy.plot(time, centroidal_sim_stoch[sim, :, 1],color='green', label='stochastic')
        # comz.plot(time, centroidal_sim_stoch[sim, :, 2],color='green', label='stochastic')
        # lx.plot(time, centroidal_sim_stoch[sim, :, 3], color='green', label='stochastic')
        # ly.plot(time, centroidal_sim_stoch[sim, :, 4], color='green', label='stochastic')
        # lz.plot(time, centroidal_sim_stoch[sim, :, 5], color='green', label='stochastic')
        # kx.plot(time, centroidal_sim_stoch[sim, :, 6], color='green', label='stochastic')
        # ky.plot(time, centroidal_sim_stoch[sim, :, 7], color='green', label='stochastic')
        # kz.plot(time, centroidal_sim_stoch[sim, :, 8], color='green', label='stochastic')
        comx.legend()

    # --------------------------------------------------
    # plot desired and simulated centroidal trajectories  
    # --------------------------------------------------
    # contact_forces = data['contact_forces']
    # for sim in range(nb_sims):
    #   f_N = np.asarray(contact_forces[sim])
    #   for contact_idx, contact in enumerate (conf.ee_frame_names):
    #     if sim == 0:
    #       fig, (fx, fy, fz) = plt.subplots(3, 1, sharex=True) 
    #     plt.rc('font', family ='serif')
    #     contact_name = contact[0:2]
    #     time = np.arange(0, np.round((f_N.shape[0])*conf.dt_ctrl, 2),conf.dt_ctrl)
    #     f_nom = f_N[:, contact_idx]
    #     fx.plot(time, f_nom[:, 0], label='contact forces simulation (N)')
    #     fx.legend()
    #     fx.set_title('force x')
    #     fy.plot(time, f_nom[:, 1])
    #     fy.set_title('force y')
    #     fz.plot(time, f_nom[:, 2])
    #     fz.set_title('force z')
    #     plt.xlabel('time (s)', fontsize=14)
    #     fig.suptitle('swing foot trajectories of '+ str(contact[0:2]))   

    plt.show()    