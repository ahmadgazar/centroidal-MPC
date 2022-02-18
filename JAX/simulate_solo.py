from robot_properties_solo.solo12wrapper import Solo12Robot
from bullet_utils.env import BulletEnvWithGround
from typing import Sequence, Tuple
import jax.numpy as jnp 
import pinocchio as pin
import numpy as np
import pybullet
import jax 

GRAY = (0.3, 0.3, 0.3, 1)
GREEN = [60/255, 186/255, 84/255, 1]
YELLOW = [244/255, 194/255, 13/255, 1]
RED = [219/255, 50/255, 54/255, 1]
BLUE = [72/255, 133/255, 237/255, 1]

def compute_norm_contact_slippage(contact_position):
  j = 0
  N = 0
  # first time instance the robot touches the ground
  for p in contact_position:
    if np.linalg.norm(p) > -1e-8 and  np.linalg.norm(p) < 1e-8:
      continue
    else:
      contact_pos_ref = p 
    break   
  contact_dev_norm = 0.
  # print(contact_pos_ref)
  for time_idx in range(len(contact_position)-1):
    # if j == 1500:
    #   contact_pos_ref = contact_position_k
    #   print(contact_pos_ref)
      # print(contact_pos_ref)
      # j = 0
    # ignore contact samples in the air  
    if np.linalg.norm(contact_position[time_idx], 2) > -1e-8 and np.linalg.norm(contact_position[time_idx], 2) < 1e-8:
      contact_pos_ref = contact_position[time_idx+1]
      # print('time_knot = ', time_idx, ', contact_ref = ', contact_pos_ref)
      # continue
    else:
      # print('time_knot = ', time_idx, ', contact_ref = ', contact_pos_ref)
      N += 1 # include as a sample
      contact_dev_norm += np.linalg.norm((contact_position[time_idx]-contact_pos_ref), 2)
    # j += 1
  return contact_dev_norm/N 

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
    """Loads a visible and tangible box.
    Args:
      half_extents: Half the dimension of the box in meters in each direction.
      position: Global coordinates of the center of the box.
      orientation: As a quaternion.
      rgba_color: The color and transparency of the box, each in the range
        [0,1]. Defaults to opaque gray.
      mass: Mass in kg. Mass 0 fixes the box in place.
    Returns:
      Unique integer referring to the loaded box.
    """
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
    """Generates one stepstone.
    Args:
      pybullet_client: The pybullet client instance.
      start_pos: The starting position (the midpoint of top-left edge) of the
        stepstone.
      stone_length: The length of the stepstone in meters.
      stone_height: The height of the stepstone in meters.
      stone_width: The width of the stepstone in meters.
      gap_length: The distance in meters between two adjacent stepstones.
      height_offset: The height difference in meters between two adjacent
        stepstones.
      rgba_color: The color and transparency of the object, each in the range
        [0,1]. Defaults to opaque gray.
    Returns:
      The position of the mid point of the right-top edge of the stepstone.
      The pybullet id of the stepstone.
    """
    half_length = stone_length / 2.0
    half_width = stone_width / 2.0
    half_height = stone_height / 2.0
    start_pos = np.asarray(start_pos) + np.array([gap_length, 0, height_offset])
    step_stone_id = self.load_box(half_extents=[half_length, half_width, half_height],
        position=start_pos + np.array([half_length, 0, -half_height]),
        orientation=orientation,
        rgba_color=rgba_color,
        mass=0)
    pybullet.changeDynamics(step_stone_id, -1, lateralFriction=0.7)    
    end_pos = start_pos + np.array([stone_length, 0, 0])
    return end_pos, step_stone_id   
  
  def sample_pseudorandom_force_uncertainties(self, key):
    N = self.N
    curr = dict(key=key, force_uncertainties=jnp.empty((N, 3)))
    def contact_loop(time_idx, curr):
        new_key, subkey = jax.random.split(curr['key'])
        force_sample_k = jax.random.multivariate_normal(subkey, np.zeros(3), 20*np.eye(3))
        curr['force_uncertainties'] = jax.ops.index_update(curr['force_uncertainties'], 
                                      jax.ops.index[time_idx, :], force_sample_k) 
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
      t = jax.random.randint(time_subkey, shape=(1,), minval=1000, maxval=N-199)
      curr['t'] = jax.ops.index_update(curr['t'], jax.ops.index[sim], t[0])
      curr['force_uncertainties_total'] = jax.ops.index_update(curr['force_uncertainties_total'], 
                                          jax.ops.index[sim, :, :], x['force_uncertainties'])
      curr['force_key'] = new_force_key
      curr['time_key'] = new_time_key
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
        if force_norm >= 0.5:
        #   for idx in range(self.pyramid_constraint_matrix.shape[0]):
        #     if self.pyramid_constraint_matrix[idx, :] @ force <= 5e-5:
        #       continue
              # pybullet.addUserDebugLine(lineFromXYZ=unused_pos_on_a, lineToXYZ=force, 
                # parentLinkIndex=link_a_id, lifeTime=0, lineColorRGB=[0, 1, 0], lineWidth=5.0)
            # else:
            #   friction_cone_violations[toe_link_order]+= 1
                # pybullet.addUserDebugLine(lineFromXYZ=unused_pos_on_a, lineToXYZ=force, 
                # parentLinkIndex=link_a_id, lifeTime=0, lineColorRGB=[1, 0, 0], lineWidth=5.0)
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
        tau_w[sim, time_idx, :] = M[6:] @ (null_space_projector @ (np.linalg.pinv(Ag) @ delta_hg[sim, time_idx, :])) 
    return tau_w
 
  def run(self, des_traj, nb_sims=1, tilde=None): 
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
      Kp = 35*np.eye(12)
      Kd = 0.4*np.eye(12)
      pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
      # tau_feedfwd = 0.
      # tau_feedback = 0.
      j = 0.
      # simulation loop
      for sim in range(nb_sims):
        # logger = pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, 'video.mp4')
        # get sampled random force disturbance at a random time instance
        t = int(t_force[sim])
        f_tilde_t = force_tilde[sim, t]
        # trajectory loop
        for time_idx in range(Nu):
          # get robot state
          q, dq = self.robot.get_state()
          # apply random force disturbances at the center of the base
          # for 200 ms
          if time_idx >= t and time_idx <= t+200:
            # print('pushing at t = ', time_idx)
            pybullet.applyExternalForce(self.robot.robot_id, -1, 
                [0.,f_tilde_t[1], 0.], [0., 0.,0.], pybullet.LINK_FRAME)
            # print("time knot = ", time_idx, " force = ", f_tilde_t[1])      
          # compute centroidal LQR control from stochastic SCP
          Jc = self.get_contact_jacobians(q, contact_sequence[time_idx])
          pin_robot.centroidalMomentum(q, dq)
          delta_f = (K[time_idx, :, 3:] @ (centroidal_des[time_idx, 3:] - np.array(rdata.hg)))      
          delta_tau = -Jc.T @ delta_f      
          # joint-torque controller + centroidal LQR control
          # tau = tau_ff + Kp(q_des - q) + Kd(qdot_des - qdot) + tau_tilde
          tau = tau_ff[time_idx, :] + Kp @ (q_des[time_idx][7:] - q[7:]) +\
                      Kd @ (qdot_des[time_idx][6:]- dq[6:]) + delta_tau[6:]
          # tau_feedfwd += np.linalg.norm(tau_ff[time_idx, :])
          # tau_feedback += np.linalg.norm(Kp @ (q_des[time_idx][7:] - q[7:]) +\
          #             Kd @ (qdot_des[time_idx][6:]- dq[6:]) + delta_tau[6:])
          # apply joint torques 
          self.robot.send_joint_command(tau)
          # step simulation 
          env.step(sleep=False) 
          # get robot state after applying disturbance
          q_tilde, dq_tilde = self.robot.get_state()
          pin_robot.centroidalMomentum(q_tilde, dq_tilde)
          hg_tilde = np.array(rdata.hg)
          com_tilde = pin.centerOfMass(rmodel, rdata, q_tilde, dq_tilde)
          p_k, _, _ = self.get_contact_positions_and_forces()
          # if hg_tilde[3] > 0.06 or hg_tilde[3] < -0.06:
          #   j += 1
          #   if j==30:
              # constraint_violations_N += 1
              # centroidal_dynamics_sim[sim, time_idx::, :3] = com_tilde
              # centroidal_dynamics_sim[sim, time_idx::, 3:9] = hg_tilde
              # q_sim[sim, time_idx::, :] = q_tilde
              # qdot_sim[sim, time_idx::, :] = dq_tilde
              # j = -100
              # break
          # else:
            # j = 0
            # pybullet.setDebugObjectColor(1, -1, RED)
          # save data 
          centroidal_dynamics_sim[sim, time_idx+N_inner, :3] = com_tilde
          centroidal_dynamics_sim[sim, time_idx+N_inner, 3:9] = hg_tilde 
          q_sim[sim, time_idx+N_inner,:] = q_tilde
          qdot_sim[sim, time_idx+N_inner, :] = dq_tilde
          contact_positions_N += [p_k]
          # contact_forces_N += [f_k]
          # constraint_violations_N +=[violations]
        # reset robot to original state for the new simulation
        self.robot.reset_state(q0, dq0) 
        self.robot.pin_robot.framesForwardKinematics(q0)
        contact_positions_sim += [contact_positions_N]
        # contact_forces_sim += [contact_forces_N]
        constraint_violations_sim += constraint_violations_N
        # contact_forces_N = []
        contact_positions_N = []
        # constraint_violations_N = 0.
        # j = 0
      # print('average contribution of feedfwd torques = ', tau_feedfwd/Nu)
      # print('average contribution of feedback torques = ', tau_feedback/Nu)
      # pybullet.stopStateLogging(logger)
      return dict(centroidal=centroidal_dynamics_sim, q=q_sim, qdot=qdot_sim, 
             contact_positions=contact_positions_sim, contact_forces=contact_forces_sim,
             constraint_violations=constraint_violations_sim) 

if __name__ == "__main__":
    import matplotlib.pylab as plt
    from scipy import sparse
    # import conf_solo12_trot as conf 
    # import conf_solo12_pace as conf
    import conf_solo12_bound as conf
    from centroidal_model import Centroidal_model
    env = BulletEnvWithGround()
    # load optimized nominal trajectories
    centroidal_des = np.load('wholeBody_interpolated_traj.npz')['X']
    des_traj_nom = np.load('wholeBody_interpolated_traj.npz')
    # load optimized stochastic trajectories
    centroidal_des_stoch = np.load('wholeBody_interpolated_traj_stoch.npz')['X']
    des_traj_stoch = np.load('wholeBody_interpolated_traj_stoch.npz')
    gains = np.load('wholeBody_interpolated_traj_stoch.npz')['gains'] 
    # get contact sequence
    model = Centroidal_model(conf)
    contact_sequence = model._contact_data['contacts_logic']
    # create a pybullet simulation environment and
    # sample pseudorandom force disturbances 
    nb_sims = 2
    init_force_seed = jax.random.PRNGKey(np.random.randint(0, 1000))
    init_time_seed = jax.random.PRNGKey(np.random.randint(0, 1000))
    simulator = Simulator(env, Solo12Robot(), conf, contact_sequence, gains)
    force_tilde = simulator.sample_pseudorandom_force_uncertainties_total(init_time_seed, 
                                                                init_force_seed, nb_sims)
    # -----------
    # Trot debris
    # -----------
    # simulator.build_one_stepstone(start_pos=(0.3, 0.15, 0.02), orientation=(-0.1, -0., 0, 1))
    # simulator.build_one_stepstone(start_pos=(0.3, -0.15, 0.02), orientation=(0.1, -0., 0, 1))
    # simulator.build_one_stepstone(start_pos=(0.55, 0.15, 0.02), orientation=(0., 0.1, 0, 1))
    # simulator.build_one_stepstone(start_pos=(0.55, -0.15, 0.02), orientation=(-0., -0.1, 0, 1))
    # simulator.build_one_stepstone(start_pos=(0.75, 0.15, 0.02), orientation=(0.2, 0., 0, 1))
    # simulator.build_one_stepstone(start_pos=(0.75, -0.15, 0.02), orientation=(-0., 0., 0, 1))
    # -----------
    # Bound debris
    # -----------
    simulator.build_one_stepstone(start_pos=(-0.1, 0.15, 0.02), orientation=(0.3, -0., 0, 1))
    simulator.build_one_stepstone(start_pos=(-0.1, -0.15, 0.02), orientation=(-0.3, -0., 0, 1))
    simulator.build_one_stepstone(start_pos=(0.15, 0.15, 0.02), orientation=(0.4, -0., 0, 1))
    simulator.build_one_stepstone(start_pos=(0.15, -0.15, 0.02), orientation=(-0.3, -0., 0, 1))
    simulator.build_one_stepstone(start_pos=(0.25, 0.15, 0.02), orientation=(-0., -0., 0, 1))
    simulator.build_one_stepstone(start_pos=(0.25, -0.15, 0.02), orientation=(0., -0., 0, 1))
    simulator.build_one_stepstone(start_pos=(0.5, 0.15, 0.02), orientation=(0.2, 0., 0, 1))
    simulator.build_one_stepstone(start_pos=(0.5, -0.15, 0.02), orientation=(-0.2, 0., 0, 1))
    simulator.build_one_stepstone(start_pos=(0.7, -0.15, 0.02), orientation=(0.2, 0, 0, 1))
    simulator.build_one_stepstone(start_pos=(0.7, 0.15, 0.02), orientation=(-0.2, 0, 0, 1))
    # run monte-carlo simulations 
    pybullet.changeDynamics(env.objects[0], -1, lateralFriction=0.5)
    data_nom = simulator.run(des_traj_nom, nb_sims, force_tilde)
    data_stoch = simulator.run(des_traj_stoch, nb_sims, force_tilde)
    # compute avg. cost performance
    x_ref = np.load('interpolated_centroidal_warm_start.npz')['X']
    N = x_ref.shape[0]
    Q = sparse.block_diag([sparse.kron(np.eye(N), 
                      model._state_cost_weights)])              
    x_ref = x_ref.flatten()
    delta_x_optimal = centroidal_des.flatten()-x_ref
    J_optimal = (delta_x_optimal.T @ Q) @ delta_x_optimal
    J_nom = 0.
    J_stoch = 0.
    # for sim in range(nb_sims):
    #   delta_x_nom = x_ref - data_nom['centroidal'][sim].flatten() 
    #   delta_x_stoch = x_ref - data_stoch['centroidal'][sim].flatten()
    #   J_nom += (delta_x_nom.T @ Q) @ delta_x_nom
    #   J_stoch += (delta_x_stoch.T @ Q) @ delta_x_stoch
    # J_nom_avg_ratio = (J_nom/nb_sims)/J_optimal
    # J_stoch_avg_ratio = (J_stoch/nb_sims)/J_optimal  
    # print('average nominal cost w.r.t. nominal cost fucntion = ', J_nom_avg_ratio)
    # print('average stochastic cost w.r.t. nominal cost fucntion = ', J_stoch_avg_ratio)
    # print('number of friction pyramid constraint violations nominal  = ', np.sum(data_nom['constraint_violations']))
    # print('number of friction pyramid constraint violations stochastic  = ', np.sum(data_stoch['constraint_violations']))
    # --------------------------------------------------
    # plot desired and simulated centroidal trajectories  
    # --------------------------------------------------
    X_des = centroidal_des
    centroidal_sim_nom = data_nom['centroidal']
    centroidal_sim_stoch = data_stoch['centroidal']
    fig1, (comx, comy, comz) = plt.subplots(3, 1, sharex=True)
    fig2, (lx, ly, lz) = plt.subplots(3, 1, sharex=True)
    fig3,  (kx, ky, kz) = plt.subplots(3, 1, sharex=True)
    time = np.arange(0, np.round((X_des.shape[0])*conf.dt_ctrl, 2),conf.dt_ctrl)
    # comx.plot(time, X_des[:, 0], label='des', color='red')
    comx.set_title('x', fontsize=12)
    comx.set_ylabel('(m)', fontsize=12)
    plt.setp(comx, ylabel='(m)')
    # comy.plot(time, X_des[:,1], label='des', color='red')
    comy.set_title('y', fontsize=12)
    comy.set_ylabel('(m)', fontsize=12)
    # comz.plot(time, X_des[:,2], label='des', color='red')
    comz.set_title('z', fontsize=12)
    comz.set_ylabel('(m)', fontsize=12)
    comz.set_xlabel('Time (s)', fontsize=12)
    # lx.plot(time, X_des[:,3], label='des', color='red')
    lx.set_title('x', fontsize=12)
    lx.set_ylabel('(Kg/s)', fontsize=12)
    # ly.plot(time, X_des[:,4], label='des', color='red')
    ly.set_title('y', fontsize=12)
    ly.set_ylabel('(Kg/s)', fontsize=12)
    # lz.plot(time, X_des[:,5], label='des', color='red')
    lz.set_title('z', fontsize=14)
    lz.set_ylabel('(Kg/s)', fontsize=12)
    lz.set_xlabel('Time (s)', fontsize=12)
    # kx.plot(time, X_des[:,6], label='des', color='red')
    kx.set_title('x', fontsize=12)
    kx.set_ylabel('(Kg.m$^2$/s)', fontsize=12)
    # ky.plot(time, X_des[:,7], label='des', color='red')
    ky.set_title('y', fontsize=12)
    ky.set_ylabel('(Kg.m$^2$/s)', fontsize=12)
    # kz.plot(time, X_des[:,8], label='des', color='red')
    kz.set_title('z', fontsize=12)
    kz.set_ylabel('(Kg.m$^2$/s)', fontsize=12)
    kz.set_xlabel('Time (s)', fontsize=12)
    fig1.suptitle('CoM trajectories')
    fig2.suptitle('Linear momentum trajectories')
    fig3.suptitle('Angular momentum trajectories')
    ang_mom_threshold = 0.06
    for sim in range(nb_sims):
      # nominal 
      comx.plot(time, centroidal_sim_nom[sim, :, 0], color='red', alpha=0.6)
      comy.plot(time, centroidal_sim_nom[sim, :, 1], color='red',alpha=0.6)
      comz.plot(time, centroidal_sim_nom[sim, :, 2],color='red',alpha=0.6)
      lx.plot(time, centroidal_sim_nom[sim, :, 3], color='red',alpha=0.6)
      ly.plot(time, centroidal_sim_nom[sim, :, 4], color='red',alpha=0.6)
      lz.plot(time, centroidal_sim_nom[sim, :, 5], color='red',alpha=0.6)
      kx.plot(time, centroidal_sim_nom[sim, :, 6], color='red',alpha=0.6)
      ky.plot(time, centroidal_sim_nom[sim, :, 7], color='red',alpha=0.6)
      kz.plot(time, centroidal_sim_nom[sim, :, 8], color='red',alpha=0.6)
      # stochastic
      comx.plot(time, centroidal_sim_stoch[sim, :, 0], color='green',alpha=0.6)
      comy.plot(time, centroidal_sim_stoch[sim, :, 1],color='green',alpha=0.6)
      comz.plot(time, centroidal_sim_stoch[sim, :, 2],color='green',alpha=0.6)
      lx.plot(time, centroidal_sim_stoch[sim, :, 3], color='green',alpha=0.6)
      ly.plot(time, centroidal_sim_stoch[sim, :, 4], color='green',alpha=0.6)
      lz.plot(time, centroidal_sim_stoch[sim, :, 5], color='green',alpha=0.6)
      kx.plot(time, centroidal_sim_stoch[sim, :, 6], color='green',alpha=0.6)
      ky.plot(time, centroidal_sim_stoch[sim, :, 7], color='green',alpha=0.6)
      kz.plot(time, centroidal_sim_stoch[sim, :, 8], color='green',alpha=0.6)
      if sim == nb_sims-1:
        # nominal 
        comx.plot(time, centroidal_sim_nom[sim, :, 0],color='red', label='nominal',alpha=0.6)
        comy.plot(time, centroidal_sim_nom[sim, :, 1],color='red', label='nominal',alpha=0.6)
        comz.plot(time, centroidal_sim_nom[sim, :, 2],color='red', label='nominal',alpha=0.6)
        lx.plot(time, centroidal_sim_nom[sim, :, 3], color='red', label='nominal',alpha=0.6)
        ly.plot(time, centroidal_sim_nom[sim, :, 4], color='red', label='nominal',alpha=0.6)
        lz.plot(time, centroidal_sim_nom[sim, :, 5], color='red', label='nominal',alpha=0.6)
        kx.plot(time, centroidal_sim_nom[sim, :, 6], color='red', label='nominal',alpha=0.6)
        ky.plot(time, centroidal_sim_nom[sim, :, 7], color='red', label='nominal',alpha=0.6)
        kz.plot(time, centroidal_sim_nom[sim, :, 8], color='red', label='nominal',alpha=0.6)
        # kx.plot(time, ang_mom_threshold*np.ones(time.shape[0]), color='red', linestyle='dashed')
        # kx.plot(time, -ang_mom_threshold*np.ones(time.shape[0]), color='red', linestyle='dashed')
        # stochastic
        comx.plot(time, centroidal_sim_stoch[sim, :, 0], color='green', label='stochastic',alpha=0.6)
        comy.plot(time, centroidal_sim_stoch[sim, :, 1],color='green', label='stochastic',alpha=0.6)
        comz.plot(time, centroidal_sim_stoch[sim, :, 2],color='green', label='stochastic',alpha=0.6)
        lx.plot(time, centroidal_sim_stoch[sim, :, 3], color='green', label='stochastic',alpha=0.6)
        ly.plot(time, centroidal_sim_stoch[sim, :, 4], color='green', label='stochastic',alpha=0.6)
        lz.plot(time, centroidal_sim_stoch[sim, :, 5], color='green', label='stochastic',alpha=0.6)
        kx.plot(time, centroidal_sim_stoch[sim, :, 6], color='green', label='stochastic',alpha=0.6)
        ky.plot(time, centroidal_sim_stoch[sim, :, 7], color='green', label='stochastic',alpha=0.6)
        kz.plot(time, centroidal_sim_stoch[sim, :, 8], color='green', label='stochastic',alpha=0.6)
        comx.legend()
        lx.legend()
        kx.legend()
    # -----------------------
    # plot contact positions
    # -----------------------
    ee_frame_names = [conf.ee_frame_names[1], conf.ee_frame_names[0],
                      conf.ee_frame_names[2], conf.ee_frame_names[3]]
    contact_pos_nom_all = np.asarray(data_nom['contact_positions'])          
    contact_pos_stoch_all = np.asarray(data_stoch['contact_positions'])
    time = time[:contact_pos_nom_all.shape[1]]
    contact_dev_nom = np.empty((nb_sims, 4))
    contact_dev_stoch = np.empty((nb_sims, 4))
    stat_nom = np.empty((4, 2))
    stat_stoch = np.empty((4, 2))
    for sim in range(nb_sims):
      for contact_idx, contact in enumerate (conf.ee_frame_names):
        # if sim == 0:
        #   fig, (px, py, pz) = plt.subplots(3, 1, sharex=True)
        #   px.plot(time, contact_trajectory_interpol[contact[:2]][:, 0], color='red') 
        #   py.plot(time, contact_trajectory_interpol[contact[:2]][:, 1], color='red') 
        #   pz.plot(time, contact_trajectory_interpol[contact[:2]][:, 2], color='red') 
        contact_pos_nom = contact_pos_nom_all[sim, :, contact_idx]
        contact_dev_nom[sim, contact_idx] = compute_norm_contact_slippage(contact_pos_nom)
        contact_pos_stoch = contact_pos_stoch_all[sim,:, contact_idx]
        contact_dev_stoch[sim, contact_idx] = compute_norm_contact_slippage(contact_pos_stoch)
        # px.scatter(time, contact_pos_nom[:, 0], color='blue', s=0.5)
        # py.scatter(time, contact_pos_nom[:, 1], color='blue', s=0.5)
        # pz.scatter(time, contact_pos_nom[:, 2], color='blue', s=0.5)
        # px.scatter(time, contact_pos_stoch[:, 0], color='green', s=0.5)
        # py.scatter(time, contact_pos_stoch[:, 1], color='green', s=0.5)
        # pz.scatter(time, contact_pos_stoch[:, 2], color='green', s=0.5)
        # px.plot(time, contact_pos_nom[:, 0], color='blue')
        # py.plot(time, contact_pos_nom[:, 1], color='blue')
        # pz.plot(time, contact_pos_nom[:, 2], color='blue')
        # px.plot(time, contact_pos_stoch[:, 0], color='green')
        # py.plot(time, contact_pos_stoch[:, 1], color='green')
        # pz.plot(time, contact_pos_stoch[:, 2], color='green')
        # plt.xlabel('time (s)', fontsize=14)
        # fig.suptitle('contact positions of  '+ str(contact[0:2]))  
    for contact_idx in range(4):
      stat_nom[contact_idx, 0] = 1e3*np.mean(contact_dev_nom[:, contact_idx])
      stat_nom[contact_idx, 1] = 1e3*np.std(contact_dev_nom[:, contact_idx])
      stat_stoch[contact_idx, 0] = 1e3*np.mean(contact_dev_stoch[:, contact_idx])
      stat_stoch[contact_idx, 1] = 1e3*np.std(contact_dev_stoch[:, contact_idx])
    print('mean and std-dev of contact slippage nominal =' , stat_nom)  
    print('mean and std-dev of contact slippage stochastic =' , stat_stoch)  
    # plot average contact slippage
    fig, ax = plt.subplots()
    labels = ['FL', 'FR', 'HL', 'HR']
    xpos = np.arange(len(labels))
    nom_bars = ax.bar(xpos-0.2, stat_nom[:,0], 0.4, yerr=stat_nom[:, 1], ecolor='black', capsize=10, label='nominal')
    stoch_bars = ax.bar(xpos+0.2, stat_stoch[:,0], 0.4,  yerr=stat_stoch[:, 1], ecolor='black', capsize=10, label='stochastic') 
    ax.set_ylabel('(mm)')
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_title('Average contact position slippage norm')
    ax.legend((nom_bars, stoch_bars), ('nominal', 'stochastic'))
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_with_error_bars.png')
    plt.show()    