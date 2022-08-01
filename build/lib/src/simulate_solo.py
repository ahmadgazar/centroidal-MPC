from time import time
from robot_properties_solo.solo12wrapper import Solo12Robot
from bullet_utils.env import BulletEnvWithGround
from typing import Sequence, Tuple
import jax.numpy as jnp 
import pinocchio as pin
import pybullet_data
import numpy as np
import pybullet
import jax 

GRAY = (0.3, 0.3, 0.3, 1)
GREEN = [60/255, 186/255, 84/255, 1]
YELLOW = [244/255, 194/255, 13/255, 1]
RED = [219/255, 50/255, 54/255, 1]
BLUE = [72/255, 133/255, 237/255, 1]

class Simulator:
  def __init__(self, sim_env, robot_wrapper, conf, contact_sequence, K):
    self.env = sim_env
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
    self.centroidal_gains = K
    self.contact_sequence = contact_sequence
    self.pyramid_constraint_matrix = np.array([[1. ,  0., -mu], 
                                    [-1.,  0., -mu],                                     
                                    [0. ,  1., -mu], 
                                    [0. , -1., -mu],
                                    [0. ,  0., -1.]])
  def load_box(self,
               half_extents: Sequence[float] = (0.05, 0.05, 0.02),
               position: Sequence[float] = (0, 0, 0),
               orientation: Sequence[float] = (0, 0, 0, 1),
               rgba_color: Sequence[float] = (0.3, 0.3, 0.3, 1),
               mass: float = 0) -> int:
    col_box_id = pybullet.createCollisionShape(
        pybullet.GEOM_BOX, halfExtents=half_extents)
    visual_box_id = pybullet.createVisualShape(
        pybullet.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba_color)
    return pybullet.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_box_id,
        baseVisualShapeIndex=visual_box_id,
        basePosition=position,
        baseOrientation=orientation)

  def build_one_stepstone(self,
    start_pos = (0.0, 0.14695, 0.0),
    stone_length = 0.1,
    stone_height = 0.01,
    stone_width = 0.1,
    orientation  = (0, 0, 0, 1),
    gap_length = 0.0,
    height_offset = 0,
    rgba_color = RED) -> Tuple[np.ndarray, int]:
    half_length = stone_length / 2.0
    half_width = stone_width / 2.0
    half_height = stone_height / 2.0
    start_pos = np.asarray(start_pos) + np.array([gap_length, 0, height_offset])
    step_stone_id = self.load_box(half_extents=[half_length, half_width, half_height],
        position=start_pos + np.array([half_length, 0, -half_height]),
        orientation=orientation,
        rgba_color=rgba_color,
        mass=0)
    # pybullet.changeDynamics(step_stone_id, -1, lateralFriction=0.4)    
    end_pos = start_pos + np.array([stone_length, 0, 0])
    return end_pos, step_stone_id   
  
  def build_one_sphere(self, start_pos=(0.,0.,0.), COLOR=BLUE): 
    r = 0.02
    if COLOR=='GREEN':
        visualSphereId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=r, rgbaColor=GREEN)
    elif COLOR=='RED':
        visualSphereId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=r, rgbaColor=RED)
    elif COLOR=='ORANGE':
        visualSphereId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=r, rgbaColor=[1,1,0,1])
    else:
        visualSphereId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=r, rgbaColor=BLUE)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    return pybullet.createMultiBody(0, -1, visualSphereId, start_pos, baseOrientation=[0,0,0,1])
  
  def sample_pseudorandom_force_uncertainties(self, key):
    N = self.N
    curr = dict(key=key, force_uncertainties=jnp.empty((N, 3)))
    def contact_loop(time_idx, curr):
        new_key, subkey = jax.random.split(curr['key'])
        force_sample_k = jax.random.multivariate_normal(subkey, np.zeros(3), 15*np.eye(3))
        curr['force_uncertainties'] = curr['force_uncertainties'].at[time_idx, :].set(force_sample_k) 
        curr['key'] = new_key  
        return curr
    return jax.lax.fori_loop(0, self.N, contact_loop, curr)     
  
  def sample_pseudorandom_force_uncertainties_total(self, time_key, force_key, nb_sims):
    N = self.N
    curr = dict(time_key=time_key, force_key=force_key, t=jnp.empty(nb_sims),
           force_uncertainties_total=jnp.empty((nb_sims, N, 3)))
    def sim_loop(sim, curr):
      new_force_key, force_subkey = jax.random.split(curr['force_key'])
      new_time_key, time_subkey = jax.random.split(curr['time_key'])
      x = self.sample_pseudorandom_force_uncertainties(force_subkey)
      t = jax.random.randint(time_subkey, shape=(1,), minval=900, maxval=N-199)
      curr['t'] = curr['t'].at[sim].set(t[0])
      curr['force_uncertainties_total'] = curr['force_uncertainties_total'].at[sim, :, :].set(x['force_uncertainties'])
      curr['force_key'] = new_force_key
      curr['time_key'] = new_time_key
      return curr 
    return jax.lax.fori_loop(0, nb_sims, sim_loop, curr)         
      
  def get_contact_positions_and_forces(self):
    foot_link_ids = tuple(self.robot.bullet_endeff_ids)
    # print(foot_link_ids)
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
        if force_norm >= 0.5:
          contact_forces[toe_link_order] += force
          contact_positions[toe_link_order] += unused_pos_on_a
      else:
        continue
    return contact_positions, contact_forces, friction_cone_violations
  
  def get_contact_jacobians(self, q, contacts_logic):
    rmodel, rdata = self.robot.pin_robot.model, self.robot.pin_robot.data 
    ee_frame_names = self.ee_frame_names
    nv = rmodel.nv
    Jc_stacked = np.array([]).reshape(0, nv)
    self.robot.pin_robot.framesForwardKinematics(q)
    for contact_idx, logic in enumerate(contacts_logic):
        if logic:
          foot_idx = rmodel.getFrameId(ee_frame_names[contact_idx])
          foot_jacobian_local = pin.getFrameJacobian(rmodel, rdata, foot_idx, pin.ReferenceFrame.LOCAL)
          world_R_foot = pin.SE3(rdata.oMf[foot_idx].rotation, np.zeros(3))
          Jc_stacked = np.vstack([Jc_stacked, world_R_foot.action.dot(foot_jacobian_local)[:3]])
        else:
          Jc_stacked = np.vstack([Jc_stacked, np.zeros((3, nv))])
    return Jc_stacked

  # def project_centoridal_to_WBD_uncertainties(self, contacts_logic, q, delta_hg):
  #   rmodel, rdata = self.robot.pin_robot.model, self.robot.pin_robot.data 
  #   ee_frame_names = self.ee_frame_names
  #   nv, N = rmodel.nv, delta_hg.shape[1]
  #   nb_sims = delta_hg.shape[0]
  #   tau_w = np.empty((nb_sims, N, nv-6))
  #   for sim in range(nb_sims):
  #     for time_idx in range(N):
  #       qk = q[time_idx]
  #       Jc_stacked = np.array([]).reshape(0, nv)
  #       for contact_idx, logic in enumerate(contacts_logic[time_idx]):
  #           if logic:
  #             foot_idx = rmodel.getFrameId(ee_frame_names[contact_idx])
  #             foot_jacobian_local = pin.getFrameJacobian(rmodel, rdata, foot_idx, pin.ReferenceFrame.LOCAL)
  #             world_R_foot = pin.SE3(rdata.oMf[foot_idx].rotation, np.zeros(3))
  #             Jc_stacked = np.vstack([Jc_stacked, world_R_foot.action.dot(foot_jacobian_local)[:3]]) 
  #       M = pin.crba(rmodel, rdata, qk)
  #       Ag = pin.computeCentroidalMap(rmodel, rdata, qk)
  #       null_space_projector = np.eye(nv) - (np.linalg.pinv(np.asarray(Jc_stacked)) @ np.asarray(Jc_stacked))
  #       tau_w[sim, time_idx, :] = M[6:] @ (null_space_projector @ (np.linalg.pinv(Ag) @ delta_hg[sim, time_idx, :])) 
  #   return tau_w
 
  def run(self, des_traj, terminal_goal, GAIT, nb_sims=1, tilde=None, SOLVER='NOMINAL'): 
      pin_robot, rmodel, rdata = self.robot.pin_robot, self.robot.pin_robot.model, self.robot.pin_robot.data
      centroidal_des, tau_ff = des_traj['X'], des_traj['U']
      q_des, qdot_des = des_traj['q'], des_traj['qdot']
      K = self.centroidal_gains
      force_tilde = tilde['force_uncertainties_total']
      t_force = tilde['t']
      nq, nv = rmodel.nq, rmodel.nv  
      Nu, Nx, N_inner = tau_ff.shape[0], q_des.shape[0], int(self.dt_plan/self.dt_ctrl)
      contact_sequence = self.contact_sequence
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
      constraint_violations_N = 0.
      contact_forces_sim = []
      contact_positions_sim = []
      constraint_violations_sim = 0.
      # fill initial states
      centroidal_dynamics_sim[:, :N_inner, :3] = pin.centerOfMass(rmodel, rdata, q0, dq0)
      pin_robot.centroidalMomentum(q0, dq0)    
      q_sim[:, :N_inner, :], qdot_sim[:, :N_inner, :] = q0, dq0 
      centroidal_dynamics_sim[:, :N_inner, 3:9] = np.array(rdata.hg)
      # PD gains
      pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
      # tau_feedfwd = 0.
      # tau_feedback = 0.
      if GAIT=='TROT':
       # -----------
       # Trot debris
       # -----------
        Kp = 4.0*np.eye(12)
        Kd = 0.2*np.eye(12)
      elif GAIT=='TROT_ON_DEBRI':
        Kp = 5.*np.eye(12)
        Kd = 0.2*np.eye(12)
        self.build_one_stepstone(start_pos=(0.2, 0.15, 0.01), orientation=(0.1, -0., 0, 1))
        self.build_one_stepstone(start_pos=(0.2, -0.15, 0.01), orientation=(-0.1, -0., 0, 1))
        # self.build_one_stepstone(start_pos=(0.55, 0.15, 0.02), orientation=(0., 0.1, 0, 1))
        # self.build_one_stepstone(start_pos=(0.55, -0.15, 0.02), orientation=(-0., -0.1, 0, 1))
        self.build_one_stepstone(start_pos=(0.45, 0.15, 0.02), orientation=(0.15, 0., 0, 1))
        self.build_one_stepstone(start_pos=(0.44, -0.15, 0.02), orientation=(-0.15, 0., 0, 1))
      elif GAIT=='BOUND':
        # -----------
        # Bound debris
        # ----------- 
        Kp = 3*np.eye(12)
        Kd = 0.2*np.eye(12)
      elif GAIT=='BOUND_ON_DEBRI':
        Kp = 4.7*np.eye(12)
        Kd = 0.2*np.eye(12)
        self.build_one_stepstone(start_pos=(-0.15, 0.15, 0.02), orientation=(0.3, -0., 0, 1))
        self.build_one_stepstone(start_pos=(-0.15, -0.15, 0.02), orientation=(-0.3, -0., 0, 1))
        self.build_one_stepstone(start_pos=(0.12, 0.15, 0.02), orientation=(0.3, -0., 0, 1))
        self.build_one_stepstone(start_pos=(0.12, -0.15, 0.02), orientation=(-0.3, -0., 0, 1))
        self.build_one_stepstone(start_pos=(0.45, 0.15, 0.02), orientation=(-0.1, -0., 0, 1))
        self.build_one_stepstone(start_pos=(0.45, -0.15, 0.02), orientation=(0.1, -0., 0, 1))
        self.build_one_stepstone(start_pos=(0.75, -0.15, 0.02), orientation=(0., 0, 0, 1))
        self.build_one_stepstone(start_pos=(0.75, 0.15, 0.02), orientation=(-0., 0, 0, 1))  
      elif GAIT=='PACE':
        # -----------
        # pace debris
        # -----------
        self.build_one_stepstone(start_pos=(0.15, 0.15, 0.02), orientation=(0.05, -0., 0, 1))
        self.build_one_stepstone(start_pos=(0.15, -0.15, 0.02), orientation=(-0.05, -0., 0, 1))
        self.build_one_stepstone(start_pos=(-0.25, 0.15, 0.02), orientation=(0.05, -0., 0, 1))
        self.build_one_stepstone(start_pos=(-0.25, -0.15, 0.02), orientation=(-0.05, -0., 0, 1))
      # add terminal goals
      # self.build_one_sphere(terminal_goal)
      # simulation loop
      for sim in range(nb_sims):
        # if SOLVER=='NOMINAL' and GAIT =='TROT':
        #   logger = pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, 'trot_nom.mp4')
        # elif SOLVER=='NOMINAL' and GAIT =='TROT_ON_DEBRI':
        #   logger = pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, 'trot_nom_with_deris.mp4')  
        # elif SOLVER=='NOMINAL' and GAIT =='BOUND':
        #   logger = pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, 'bound_nom.mp4')
        # elif SOLVER=='NOMINAL' and GAIT =='BOUND_ON_DEBRI':
        #   logger = pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, 'bound_nom_with_debris.mp4')
        # elif SOLVER=='STOCHASTIC' and GAIT =='TROT':
        #   logger = pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, 'trot_stoch.mp4')  
        # elif SOLVER=='STOCHASTIC' and GAIT =='TROT_ON_DEBRI':
        #   logger = pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, 'trot_stoch_with_debris.mp4')
        # elif SOLVER=='STOCHASTIC' and GAIT =='BOUND':
        #   logger = pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, 'bound_stoch.mp4')
        # elif SOLVER=='STOCHASTIC' and GAIT =='BOUND_ON_DEBRI':
        #   logger = pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, 'bound_stoch_with_debris.mp4')
        # get sampled random force disturbance at a random time instance
        t = int(t_force[sim])
        f_tilde_t = force_tilde[sim, t]
        # trajectory loop
        for time_idx in range(Nu):
          # get robot state
          q, dq = self.robot.get_state()
          # apply random force disturbances at the center of the
          # robot base links for 200 ms
          if time_idx >= t and time_idx <= t+200:
            if time_idx == t: 
              print('force disturbance = ', f_tilde_t[1])
              pybullet.addUserDebugLine(q[:3], [q[0],  np.sign(f_tilde_t[1])*0.3, q[2]], [1,0,0], 20, 0.2)      
            pybullet.applyExternalForce(self.robot.robot_id, -1, 
                  [0.,f_tilde_t[1], 0.], [0., 0.,0.], pybullet.LINK_FRAME)  
          # compute centroidal LQR control from stochastic SCP
          Jc = self.get_contact_jacobians(q, contact_sequence[time_idx])
          pin_robot.centroidalMomentum(q, dq)
          delta_f = (K[time_idx, :, 18:24] @ (centroidal_des[time_idx, 3:] - np.array(rdata.hg)))      
          delta_tau = -Jc.T @ delta_f      
          # joint-torque controller + centroidal LQR control
          # tau = tau_ff + Kp(q_des - q) + Kd(qdot_des - qdot) + tau_tilde
          # Kp = K[time_idx, :, 6:18]
          # Kd = K[time_idx, :, 24::]  
          tau = tau_ff[time_idx, :] + Kp @ (q_des[time_idx][7:] - q[7:]) +\
                      Kd @ (qdot_des[time_idx][6:]- dq[6:]) + delta_tau[6:]
          # tau_feedfwd += np.linalg.norm(tau_ff[time_idx, :])
          # tau_feedback += np.linalg.norm(Kp @ (q_des[time_idx][7:] - q[7:]) +\
          #             Kd @ (qdot_des[time_idx][6:]- dq[6:]) + delta_tau[6:])
          # apply joint torques 
          self.robot.send_joint_command(tau)
          # step simulation 
          self.env.step(sleep=False) 
          # get robot state after applying disturbance
          q_tilde, dq_tilde = self.robot.get_state()
          pin_robot.centroidalMomentum(q_tilde, dq_tilde)
          hg_tilde = np.array(rdata.hg)
          com_tilde = pin.centerOfMass(rmodel, rdata, q_tilde, dq_tilde)
          p_k, _, _ = self.get_contact_positions_and_forces()
          # save data 
          centroidal_dynamics_sim[sim, time_idx+N_inner, :3] = com_tilde
          centroidal_dynamics_sim[sim, time_idx+N_inner, 3:9] = hg_tilde 
          q_sim[sim, time_idx+N_inner,:] = q_tilde
          qdot_sim[sim, time_idx+N_inner, :] = dq_tilde
          contact_positions_N += [p_k]
          # if time_idx == Nu-1:
          #   if SOLVER == 'NOMINAL':
          #     nom_end = self.build_one_sphere(com_tilde, 'ORANGE')
          #   elif SOLVER == 'STOCHASTIC':
          #     stoch_end = self.build_one_sphere(com_tilde, 'GREEN')
            # if SOLVER=='STOCHASTIC' and GAIT =='BOUND':
            #   pybullet.removeBody(nom_end)
            #   pybullet.removeBody(stoch_end)  
        # reset robot to original state for the new simulation
        self.robot.reset_state(q0, dq0) 
        self.robot.pin_robot.framesForwardKinematics(q0)
        contact_positions_sim += [contact_positions_N]
        # contact_forces_sim += [contact_forces_N]
        constraint_violations_sim += constraint_violations_N
        # contact_forces_N = []
        contact_positions_N = []
        # constraint_violations_N = 0.
      # print('average contribution of feedfwd torques = ', tau_feedfwd/N)
      # print('average contribution of feedback torques = ', tau_feedback/N)
      # pybullet.stopStateLogging(logger)
      return dict(centroidal=centroidal_dynamics_sim, q=q_sim, qdot=qdot_sim, 
             contact_positions=contact_positions_sim, contact_forces=contact_forces_sim,
             constraint_violations=constraint_violations_sim) 

# if __name__ == "__main__":
#     import matplotlib.pylab as plt
#     # import conf_solo12_trot as conf
#     import conf_solo12_bound as conf  
#     from centroidal_model import Centroidal_model
#     env = BulletEnvWithGround()
#     # load optimized nominal trajectories
#     centroidal_des = np.load('wholeBody_interpolated_traj.npz')['X']
#     des_traj_nom = np.load('wholeBody_interpolated_traj.npz')
#     ddp_gains = np.load('wholeBody_interpolated_traj.npz')['gains']
#     x_ref_nom = np.load('scp_sol_interpol_nom.npz')['X']
#     x_ref_nom_final = x_ref_nom[:3, -1]
#     # load optimized stochastic trajectories
#     centroidal_des_stoch = np.load('wholeBody_interpolated_traj_stoch.npz')['X']
#     des_traj_stoch = np.load('wholeBody_interpolated_traj_stoch.npz')
#     x_ref_stoch = np.load('scp_sol_interpol_stoch.npz')['X']
#     x_ref_stoch_final = x_ref_stoch[:3, -1]
#     # gains = np.load('wholeBody_interpolated_traj_stoch.npz')['gains'] 
#     # get contact sequence
#     model = Centroidal_model(conf)
#     contact_sequence = model._contact_data['contacts_logic']
#     # create a pybullet simulation environment and
#     # sample pseudorandom force disturbances 
#     nb_sims = 5
#     init_force_seed_1 = jax.random.PRNGKey(np.random.randint(0, 1000))
#     init_time_seed_1 = jax.random.PRNGKey(np.random.randint(0, 1000))
#     init_force_seed_2 = jax.random.PRNGKey(np.random.randint(0, 1000))
#     init_time_seed_2 = jax.random.PRNGKey(np.random.randint(0, 1000))
#     simulator = Simulator(env, Solo12Robot(), conf, contact_sequence, ddp_gains)

#     force_tilde_1 = simulator.sample_pseudorandom_force_uncertainties_total(init_time_seed_1, 
#                                                                   init_force_seed_1, nb_sims) 
#     force_tilde_2 = simulator.sample_pseudorandom_force_uncertainties_total(init_time_seed_2, 
#                                                                   init_force_seed_2, nb_sims)                                                             
#     # run monte-carlo simulations 
#     pybullet.changeDynamics(env.objects[0], -1, lateralFriction=0.4)
#     # trot
#     # data_nom_1 = simulator.run(des_traj_nom, x_ref_nom_final, 'TROT', nb_sims, force_tilde_1, 'NOMINAL')
#     # data_stoch_1 = simulator.run(des_traj_stoch, x_ref_stoch_final,'TROT', nb_sims, force_tilde_1, 'STOCHASTIC')
#     # data_nom_2 = simulator.run(des_traj_nom, x_ref_nom_final, 'TROT_ON_DEBRI', nb_sims, force_tilde_2, 'NOMINAL')
#     # data_stoch_2 = simulator.run(des_traj_stoch, x_ref_stoch_final, 'TROT_ON_DEBRI', nb_sims, force_tilde_2, 'STOCHASTIC')
#     # bound
#     data_nom_1 = simulator.run(des_traj_nom, x_ref_nom_final, 'BOUND', nb_sims, force_tilde_1, 'NOMINAL')
#     data_stoch_1 = simulator.run(des_traj_stoch,x_ref_nom_final, 'BOUND', nb_sims, force_tilde_1, 'STOCHASTIC')
#     data_nom_2 = simulator.run(des_traj_nom, x_ref_nom_final,'BOUND_ON_DEBRI', nb_sims, force_tilde_2, 'NOMINAL')
#     data_stoch_2 = simulator.run(des_traj_stoch,x_ref_nom_final, 'BOUND_ON_DEBRI', nb_sims, force_tilde_2, 'STOCHASTIC')
#     # compute centroidal tracking cost performance
#     N = x_ref_nom.shape[1]
#     Q = conf.state_cost_weights              
#     J_nom_1 = np.zeros((nb_sims, N))
#     J_stoch_1 = np.zeros((nb_sims, N))
#     J_nom_2 = np.zeros((nb_sims, N))
#     J_stoch_2 = np.zeros((nb_sims, N))
#     J_nom_stats_1 = np.zeros((N, 2))
#     J_stoch_stats_1 = np.zeros((N, 2))
#     J_nom_stats_2 = np.zeros((N, 2))
#     J_stoch_stats_2 = np.zeros((N, 2))
#     for sim in range(nb_sims):
#       for time_idx in range(N):
#         delta_x_nom_1 = x_ref_nom[:, time_idx] - data_nom_1['centroidal'][sim, time_idx]
#         delta_x_stoch_1 = x_ref_stoch[:, time_idx] - data_stoch_1['centroidal'][sim, time_idx]
#         delta_x_nom_2 = x_ref_nom[:, time_idx] - data_nom_2['centroidal'][sim, time_idx]
#         delta_x_stoch_2 = x_ref_stoch[:, time_idx] - data_stoch_2['centroidal'][sim, time_idx]
#         J_nom_1[sim, time_idx] = (delta_x_nom_1.T @ Q) @ delta_x_nom_1
#         J_stoch_1[sim, time_idx]= (delta_x_stoch_1.T @ Q) @ delta_x_stoch_1
#         J_nom_2[sim, time_idx] = (delta_x_nom_2.T @ Q) @ delta_x_nom_2
#         J_stoch_2[sim, time_idx]= (delta_x_stoch_2.T @ Q) @ delta_x_stoch_2
#     for time_idx in range(N):
#       if time_idx < 1:
#         J_nom_stats_1[time_idx, 0] = np.mean(J_nom_1[:, time_idx])
#         J_nom_stats_1[time_idx, 1] = np.std(J_nom_1[:, time_idx])
#         J_stoch_stats_1[time_idx, 0] = np.mean(J_stoch_1[:, time_idx])
#         J_stoch_stats_1[time_idx, 1] = np.std(J_stoch_1[:, time_idx])
#         J_nom_stats_2[time_idx, 0] = np.mean(J_nom_2[:, time_idx])
#         J_nom_stats_2[time_idx, 1] = np.std(J_nom_2[:, time_idx])
#         J_stoch_stats_2[time_idx, 0] = np.mean(J_stoch_2[:, time_idx])
#         J_stoch_stats_2[time_idx, 1] = np.std(J_stoch_2[:, time_idx])
#       else:
#         J_nom_stats_1[time_idx, 0] = np.mean(J_nom_1[:, time_idx]) + J_nom_stats_1[time_idx-1, 0]
#         J_nom_stats_1[time_idx, 1] = np.std(J_nom_1[:, time_idx]) + J_nom_stats_1[time_idx-1, 1]
#         J_stoch_stats_1[time_idx, 0] = np.mean(J_stoch_1[:, time_idx]) + J_stoch_stats_1[time_idx-1, 0]
#         J_stoch_stats_1[time_idx, 1] = np.std(J_stoch_1[:, time_idx]) + J_stoch_stats_1[time_idx-1, 1]
#         J_nom_stats_2[time_idx, 0] = np.mean(J_nom_2[:, time_idx]) + J_nom_stats_2[time_idx-1, 0]
#         J_nom_stats_2[time_idx, 1] = np.std(J_nom_2[:, time_idx]) + J_nom_stats_2[time_idx-1, 1]
#         J_stoch_stats_2[time_idx, 0] = np.mean(J_stoch_2[:, time_idx]) + J_stoch_stats_2[time_idx-1, 0]
#         J_stoch_stats_2[time_idx, 1] = np.std(J_stoch_2[:, time_idx]) + J_stoch_stats_2[time_idx-1, 1]
#     fig, ax = plt.subplots(1, 1, sharex=True)
#     time = np.arange(0, np.round(N*conf.dt_ctrl, 2),conf.dt_ctrl)
#     ax.plot(time, J_nom_stats_1[:, 0], label='nominal without debris')
#     ax.fill_between(time, J_nom_stats_1[:, 0]+J_nom_stats_1[:, 1], J_nom_stats_1[:, 0]-J_nom_stats_1[:, 1], alpha=0.2)
#     ax.plot(time, J_stoch_stats_1[:, 0], label='stochastic without debris')
#     ax.fill_between(time, J_stoch_stats_1[:, 0]+J_stoch_stats_1[:, 1], J_stoch_stats_1[:, 0]-J_stoch_stats_1[:, 1], alpha=0.2)
#     ax.plot(time, J_nom_stats_2[:, 0], label='nominal with debris')
#     ax.fill_between(time, J_nom_stats_2[:, 0]+J_nom_stats_2[:, 1], J_nom_stats_2[:, 0]-J_nom_stats_2[:, 1], alpha=0.2)
#     ax.plot(time, J_stoch_stats_2[:, 0], label='stochastic with debris')
#     ax.fill_between(time, J_stoch_stats_2[:, 0]+J_stoch_stats_2[:, 1], J_stoch_stats_2[:, 0]-J_stoch_stats_2[:, 1], alpha=0.2)
#     ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#     ax.legend()
#     ax.set_xlabel('Time (s)')
#     ax.set_ylabel('Centroidal tracking cost')
#     # --------------------------------------------------
#     # plot desired and simulated centroidal trajectories  
#     # --------------------------------------------------
#     X_des = centroidal_des 
#     centroidal_sim_nom_1 = data_nom_1['centroidal']
#     centroidal_sim_stoch_1 = data_stoch_1['centroidal']
#     centroidal_sim_nom_2 = data_nom_1['centroidal']
#     centroidal_sim_stoch_2 = data_stoch_1['centroidal']
#     fig1, (comx, comy, comz) = plt.subplots(3, 1, sharex=True)
#     fig2, (lx, ly, lz) = plt.subplots(3, 1, sharex=True)
#     fig3,  (kx, ky, kz) = plt.subplots(3, 1, sharex=True)
#     time = np.arange(0, np.round((X_des.shape[0])*conf.dt_ctrl, 2),conf.dt_ctrl)
#     # comx.plot(time, X_des[:, 0], label='des', color='red')
#     comx.set_title('x', fontsize=12)
#     comx.set_ylabel('(m)', fontsize=12)
#     plt.setp(comx, ylabel='(m)')
#     # comy.plot(time, X_des[:,1], label='des', color='red')
#     comy.set_title('y', fontsize=12)
#     comy.set_ylabel('(m)', fontsize=12)
#     # comz.plot(time, X_des[:,2], label='des', color='red')
#     comz.set_title('z', fontsize=12)
#     comz.set_ylabel('(m)', fontsize=12)
#     comz.set_xlabel('Time (s)', fontsize=12)
#     # lx.plot(time, X_des[:,3], label='des', color='red')
#     lx.set_title('x', fontsize=12)
#     lx.set_ylabel('(Kg/s)', fontsize=12)
#     # ly.plot(time, X_des[:,4], label='des', color='red')
#     ly.set_title('y', fontsize=12)
#     ly.set_ylabel('(Kg/s)', fontsize=12)
#     # lz.plot(time, X_des[:,5], label='des', color='red')
#     lz.set_title('z', fontsize=14)
#     lz.set_ylabel('(Kg/s)', fontsize=12)
#     lz.set_xlabel('Time (s)', fontsize=12)
#     # kx.plot(time, X_des[:,6], label='des', color='red')
#     kx.set_title('x', fontsize=12)
#     kx.set_ylabel('(Kg.m$^2$/s)', fontsize=12)
#     # ky.plot(time, X_des[:,7], label='des', color='red')
#     ky.set_title('y', fontsize=12)
#     ky.set_ylabel('(Kg.m$^2$/s)', fontsize=12)
#     # kz.plot(time, X_des[:,8], label='des', color='red')
#     kz.set_title('z', fontsize=12)
#     kz.set_ylabel('(Kg.m$^2$/s)', fontsize=12)
#     kz.set_xlabel('Time (s)', fontsize=12)
#     fig1.suptitle('CoM trajectories')
#     fig2.suptitle('Linear momentum trajectories')
#     fig3.suptitle('Angular momentum trajectories')
#     ang_mom_threshold = 0.06
#     for sim in range(nb_sims):
#       # nominal 
#       comx.plot(time, centroidal_sim_nom_1[sim, :, 0], color='red', alpha=0.6)
#       comy.plot(time, centroidal_sim_nom_1[sim, :, 1], color='red',alpha=0.6)
#       comz.plot(time, centroidal_sim_nom_1[sim, :, 2],color='red',alpha=0.6)
#       lx.plot(time, centroidal_sim_nom_1[sim, :, 3], color='red',alpha=0.6)
#       ly.plot(time, centroidal_sim_nom_1[sim, :, 4], color='red',alpha=0.6)
#       lz.plot(time, centroidal_sim_nom_1[sim, :, 5], color='red',alpha=0.6)
#       kx.plot(time, centroidal_sim_nom_1[sim, :, 6], color='red',alpha=0.6)
#       ky.plot(time, centroidal_sim_nom_1[sim, :, 7], color='red',alpha=0.6)
#       kz.plot(time, centroidal_sim_nom_1[sim, :, 8], color='red',alpha=0.6)
#       comx.plot(time, centroidal_sim_nom_2[sim, :, 0], color='red', alpha=0.6)
#       comy.plot(time, centroidal_sim_nom_2[sim, :, 1], color='red',alpha=0.6)
#       comz.plot(time, centroidal_sim_nom_2[sim, :, 2],color='red',alpha=0.6)
#       lx.plot(time, centroidal_sim_nom_2[sim, :, 3], color='red',alpha=0.6)
#       ly.plot(time, centroidal_sim_nom_2[sim, :, 4], color='red',alpha=0.6)
#       lz.plot(time, centroidal_sim_nom_2[sim, :, 5], color='red',alpha=0.6)
#       kx.plot(time, centroidal_sim_nom_2[sim, :, 6], color='red',alpha=0.6)
#       ky.plot(time, centroidal_sim_nom_2[sim, :, 7], color='red',alpha=0.6)
#       kz.plot(time, centroidal_sim_nom_2[sim, :, 8], color='red',alpha=0.6)
#       # stochastic
#       comx.plot(time, centroidal_sim_stoch_1[sim, :, 0], color='green',alpha=0.6)
#       comy.plot(time, centroidal_sim_stoch_1[sim, :, 1],color='green',alpha=0.6)
#       comz.plot(time, centroidal_sim_stoch_1[sim, :, 2],color='green',alpha=0.6)
#       lx.plot(time, centroidal_sim_stoch_1[sim, :, 3], color='green',alpha=0.6)
#       ly.plot(time, centroidal_sim_stoch_1[sim, :, 4], color='green',alpha=0.6)
#       lz.plot(time, centroidal_sim_stoch_1[sim, :, 5], color='green',alpha=0.6)
#       kx.plot(time, centroidal_sim_stoch_1[sim, :, 6], color='green',alpha=0.6)
#       ky.plot(time, centroidal_sim_stoch_1[sim, :, 7], color='green',alpha=0.6)
#       kz.plot(time, centroidal_sim_stoch_1[sim, :, 8], color='green',alpha=0.6)
#       comx.plot(time, centroidal_sim_stoch_2[sim, :, 0], color='green',alpha=0.6)
#       comy.plot(time, centroidal_sim_stoch_2[sim, :, 1],color='green',alpha=0.6)
#       comz.plot(time, centroidal_sim_stoch_2[sim, :, 2],color='green',alpha=0.6)
#       lx.plot(time, centroidal_sim_stoch_2[sim, :, 3], color='green',alpha=0.6)
#       ly.plot(time, centroidal_sim_stoch_2[sim, :, 4], color='green',alpha=0.6)
#       lz.plot(time, centroidal_sim_stoch_2[sim, :, 5], color='green',alpha=0.6)
#       kx.plot(time, centroidal_sim_stoch_2[sim, :, 6], color='green',alpha=0.6)
#       ky.plot(time, centroidal_sim_stoch_2[sim, :, 7], color='green',alpha=0.6)
#       kz.plot(time, centroidal_sim_stoch_2[sim, :, 8], color='green',alpha=0.6)
#       if sim == nb_sims-1:
#         # nominal 
#         comx.plot(time, centroidal_sim_nom_1[sim, :, 0],color='red', label='nominal',alpha=0.6)
#         comy.plot(time, centroidal_sim_nom_1[sim, :, 1],color='red', label='nominal',alpha=0.6)
#         comz.plot(time, centroidal_sim_nom_1[sim, :, 2],color='red', label='nominal',alpha=0.6)
#         lx.plot(time, centroidal_sim_nom_1[sim, :, 3], color='red', label='nominal',alpha=0.6)
#         ly.plot(time, centroidal_sim_nom_1[sim, :, 4], color='red', label='nominal',alpha=0.6)
#         lz.plot(time, centroidal_sim_nom_1[sim, :, 5], color='red', label='nominal',alpha=0.6)
#         kx.plot(time, centroidal_sim_nom_1[sim, :, 6], color='red', label='nominal',alpha=0.6)
#         ky.plot(time, centroidal_sim_nom_1[sim, :, 7], color='red', label='nominal',alpha=0.6)
#         kz.plot(time, centroidal_sim_nom_1[sim, :, 8], color='red', label='nominal',alpha=0.6)
#         # stochastic
#         comx.plot(time, centroidal_sim_stoch_1[sim, :, 0], color='green', label='stochastic',alpha=0.6)
#         comy.plot(time, centroidal_sim_stoch_1[sim, :, 1],color='green', label='stochastic',alpha=0.6)
#         comz.plot(time, centroidal_sim_stoch_1[sim, :, 2],color='green', label='stochastic',alpha=0.6)
#         lx.plot(time, centroidal_sim_stoch_1[sim, :, 3], color='green', label='stochastic',alpha=0.6)
#         ly.plot(time, centroidal_sim_stoch_1[sim, :, 4], color='green', label='stochastic',alpha=0.6)
#         lz.plot(time, centroidal_sim_stoch_1[sim, :, 5], color='green', label='stochastic',alpha=0.6)
#         kx.plot(time, centroidal_sim_stoch_1[sim, :, 6], color='green', label='stochastic',alpha=0.6)
#         ky.plot(time, centroidal_sim_stoch_1[sim, :, 7], color='green', label='stochastic',alpha=0.6)
#         kz.plot(time, centroidal_sim_stoch_1[sim, :, 8], color='green', label='stochastic',alpha=0.6)
#         comx.legend()
#         lx.legend()
#         kx.legend()
#     # -----------------------
#     # plot contact positions
#     # -----------------------
#     ee_frame_names = [conf.ee_frame_names[1], conf.ee_frame_names[0],
#                       conf.ee_frame_names[2], conf.ee_frame_names[3]]
#     contact_pos_nom1_all = np.asarray(data_nom_1['contact_positions'])          
#     contact_pos_stoch1_all = np.asarray(data_stoch_1['contact_positions'])
#     contact_pos_nom2_all = np.asarray(data_nom_2['contact_positions'])          
#     contact_pos_stoch2_all = np.asarray(data_stoch_2['contact_positions'])
#     time = time[:contact_pos_nom1_all.shape[1]]
#     contact_dev_nom1 = np.zeros((nb_sims, contact_pos_nom1_all.shape[1]))
#     contact_dev_stoch1 = np.zeros((nb_sims, contact_pos_stoch1_all.shape[1]))
#     contact_dev_nom2 = np.zeros((nb_sims, contact_pos_nom2_all.shape[1]))
#     contact_dev_stoch2 = np.zeros((nb_sims, contact_pos_stoch2_all.shape[1]))
#     stat_nom_1 = np.zeros((contact_pos_nom1_all.shape[1], 2))
#     stat_stoch_1 = np.zeros((contact_pos_stoch1_all.shape[1], 2))
#     stat_nom_2 = np.zeros((contact_pos_nom2_all.shape[1], 2))
#     stat_stoch_2 = np.zeros((contact_pos_stoch2_all.shape[1], 2))
#     # ----------------------------------------------------------------------    
#     # plot the normalized cumulative sum of the contact positions deviations
#     # ----------------------------------------------------------------------
#     for sim in range(nb_sims):
#       # if sim == 0:
#       #   fig, (ax) = plt.subplots(1, 1, sharex=True)
#       for contact_idx, contact in enumerate (conf.ee_frame_names): 
#         contact_pos_nom1 = contact_pos_nom1_all[sim, :, contact_idx]
#         contact_dev_nom1[sim, :] = compute_norm_contact_slippage(contact_pos_nom1)
#         contact_pos_stoch1 = contact_pos_stoch1_all[sim,:, contact_idx]
#         contact_dev_stoch1[sim, :] = compute_norm_contact_slippage(contact_pos_stoch1)
        
#         contact_pos_nom2 = contact_pos_nom2_all[sim, :, contact_idx]
#         contact_dev_nom2[sim, :] = compute_norm_contact_slippage(contact_pos_nom2)
#         contact_pos_stoch2 = contact_pos_stoch2_all[sim,:, contact_idx]
#         contact_dev_stoch2[sim, :] = compute_norm_contact_slippage(contact_pos_stoch2)
#       # ax.plot(time, contact_dev_nom[sim, :])
#       # ax.plot(time, contact_dev_stoch[sim, :])
#     for time_idx in range(contact_pos_nom1_all.shape[1]):
#       if time_idx < 1:
#         stat_nom_1[time_idx, 0] = np.mean(contact_dev_nom1[:, time_idx])
#         stat_nom_1[time_idx, 1] = np.std(contact_dev_nom1[:, time_idx])
#         stat_stoch_1[time_idx, 0] = np.mean(contact_dev_stoch1[:, time_idx])
#         stat_stoch_1[time_idx, 1] = np.std(contact_dev_stoch1[:, time_idx])

#         stat_nom_2[time_idx, 0] = np.mean(contact_dev_nom2[:, time_idx])
#         stat_nom_2[time_idx, 1] = np.std(contact_dev_nom2[:, time_idx])
#         stat_stoch_2[time_idx, 0] = np.mean(contact_dev_stoch2[:, time_idx])
#         stat_stoch_2[time_idx, 1] = np.std(contact_dev_stoch2[:, time_idx])
#       else:
#         curr_mean_nom1 = np.mean(contact_dev_nom1[:, time_idx])
#         curr_std_nom1 = np.std(contact_dev_nom1[:, time_idx])
#         curr_mean_stoch1 = np.mean(contact_dev_stoch1[:, time_idx])
#         curr_std_stoch1 = np.std(contact_dev_stoch1[:, time_idx])
#         stat_nom_1[time_idx, 0] = stat_nom_1[time_idx-1, 0] + curr_mean_nom1-(stat_nom_1[time_idx-1, 0]/(time_idx))
#         stat_nom_1[time_idx, 1] = stat_nom_1[time_idx-1, 1] + curr_std_nom1-(stat_nom_1[time_idx-1, 1]/(time_idx))
#         stat_stoch_1[time_idx, 0] = stat_stoch_1[time_idx-1, 0] + curr_mean_stoch1-(stat_stoch_1[time_idx-1, 0]/(time_idx))
#         stat_stoch_1[time_idx, 1] = stat_stoch_1[time_idx-1, 1] + curr_std_stoch1-(stat_stoch_1[time_idx-1, 1]/(time_idx))

#         curr_mean_nom2 = np.mean(contact_dev_nom2[:, time_idx])
#         curr_std_nom2 = np.std(contact_dev_nom2[:, time_idx])
#         curr_mean_stoch2 = np.mean(contact_dev_stoch2[:, time_idx])
#         curr_std_stoch2 = np.std(contact_dev_stoch2[:, time_idx])
#         stat_nom_2[time_idx, 0] = stat_nom_2[time_idx-1, 0] + curr_mean_nom2-(stat_nom_2[time_idx-1, 0]/(time_idx))
#         stat_nom_2[time_idx, 1] = stat_nom_2[time_idx-1, 1] + curr_std_nom2-(stat_nom_2[time_idx-1, 1]/(time_idx))
#         stat_stoch_2[time_idx, 0] = stat_stoch_2[time_idx-1, 0] + curr_mean_stoch2-(stat_stoch_2[time_idx-1, 0]/(time_idx))
#         stat_stoch_2[time_idx, 1] = stat_stoch_2[time_idx-1, 1] + curr_std_stoch2-(stat_stoch_2[time_idx-1, 1]/(time_idx))  
#     fig, ax = plt.subplots(1, 1, sharex=True)
#     ax.plot(time, stat_nom_1[:, 0], label='nominal without debris')
#     ax.fill_between(time, stat_nom_1[:, 0]+stat_nom_1[:, 1], stat_nom_1[:, 0]-stat_nom_1[:, 1], alpha=0.2)
#     ax.plot(time, stat_stoch_1[:, 0], label='stochastic without debris')
#     ax.fill_between(time, stat_stoch_1[:, 0]+stat_stoch_1[:, 1], stat_stoch_1[:, 0]-stat_stoch_1[:, 1], alpha=0.2)
#     ax.plot(time, stat_nom_2[:, 0], label='nominal with debris')
#     ax.fill_between(time, stat_nom_2[:, 0]+stat_nom_2[:, 1], stat_nom_2[:, 0]-stat_nom_2[:, 1], alpha=0.2)
#     ax.plot(time, stat_stoch_2[:, 0], label='stochastic with debris')
#     ax.fill_between(time, stat_stoch_2[:, 0]+stat_stoch_2[:, 1], stat_stoch_2[:, 0]-stat_stoch_2[:, 1], alpha=0.2)
#     ax.legend()
#     ax.set_xlabel('Time (s)')
#     ax.set_ylabel('Normalized integral of feet slippage norm (m)')
        
#     # save simulated data
#     slippage_diff_1 = ((stat_nom_1[-1, 0]- stat_stoch_1[-1, 0])/ stat_nom_1[-1, 0])*100
#     slippage_diff_2 = ((stat_nom_2[-1, 0]- stat_stoch_2[-1, 0])/ stat_nom_2[-1, 0])*100
#     performance_diff_1 = ((J_nom_stats_1[-1, 0]- J_stoch_stats_1[-1, 0])/ J_nom_stats_1[-1, 0])*100
#     performance_diff_2 = ((J_nom_stats_2[-1, 0]- J_stoch_stats_2[-1, 0])/ J_nom_stats_2[-1, 0])*100
#     print('slippage_stoch/slippage_nom without debri = ', slippage_diff_1)
#     print('J_stoch/J_nom without debris = ', performance_diff_1)
#     print('slippage_stoch/slippage_nom with debri = ', slippage_diff_2)
#     print('J_stoch/J_nom with debri = ', performance_diff_2)
    
#     np.savez('data_nom_1.npz', slippage_stats=stat_nom_1, cost_stats = J_nom_stats_1, 
#                                                     centoridal= centroidal_sim_nom_1)
#     np.savez('data_stoch_1.npz', slippage_stats=stat_stoch_1, cost_stats= J_stoch_stats_1,
#                                                        centoridal= centroidal_sim_stoch_1)
#     np.savez('data_nom_2.npz', slippage_stats=stat_nom_2, cost_stats = J_nom_stats_2, 
#                                                     centoridal= centroidal_sim_nom_2)
#     np.savez('data_stoch_2.npz', slippage_stats=stat_stoch_2, cost_stats= J_stoch_stats_2,
#                                                        centoridal= centroidal_sim_stoch_2)
#     np.savez('difference_stats.npz', slippage_diff_without_debri=slippage_diff_1,
#                                      slippage_diff_with_debri=slippage_diff_2,
#                                      performance_diff_without_debri=performance_diff_1,
#                                      performance_diff_with_debri=performance_diff_2)                                                                                                      
#     plt.show()    