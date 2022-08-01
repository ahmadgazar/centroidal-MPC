import numpy as np
from casadi import *
import matplotlib.pylab as plt
#TODO change to jnp and jit

"""
inner approximation
"""
def construct_friction_pyramid_constraint_matrix(model):
    mu_linear = model._linear_friction_coefficient/np.sqrt(2)
    pyramid_constraint_matrix = np.array([[1. ,  0., -mu_linear], 
                                          [-1.,  0., -mu_linear],                                     
                                          [0. ,  1., -mu_linear], 
                                          [0. , -1., -mu_linear],
                                          [0. ,  0., -1.]])
    return pyramid_constraint_matrix

def compute_centroid(vertices):
    centroid = [0., 0., 0.]
    n = len(vertices)
    centroid[0] = np.sum(np.asarray(vertices)[:, 0])/n
    centroid[1] = np.sum(np.asarray(vertices)[:, 1])/n
    centroid[2] = np.sum(np.asarray(vertices)[:, 2])/n
    return centroid

def interpolate_centroidal_traj(conf, data):
    N = conf.N
    N_ctrl = conf.N_ctrl   
    N_inner = int(N_ctrl/N)
    result = {'state':np.zeros((conf.n_x, N_ctrl+N_inner)), 
                    'control':np.zeros((conf.n_u, N_ctrl)),
            'contact_sequence':np.array([]).reshape(0, 4)}
    for outer_time_idx in range(N+1):
        inner_time_idx = outer_time_idx*N_inner
        result['state'][:, inner_time_idx:inner_time_idx+N_inner] = np.tile(data['state'][:, outer_time_idx], (N_inner,1)).T
        if outer_time_idx < N:
            result['contact_sequence'] = np.vstack([result['contact_sequence'], 
                np.tile(data['contact_sequence'][outer_time_idx], (N_inner, 1))])  
            result['control'][:, inner_time_idx:inner_time_idx+N_inner] = \
                    np.tile(data['control'][:, outer_time_idx], (N_inner,1)).T
    return result 

# Generate trajectory using 3rd order polynomial with following constraints:
# x(0)=x0, x(T)=x1, dx(0)=dx(T)=0
# x(t) = a + b t + c t^2 + d t^3
# x(0) = a = x0
# dx(0) = b = 0
# dx(T) = 2 c T + 3 d T^2 = 0 => c = -3 d T^2 / (2 T) = -(3/2) d T
# x(T) = x0 + c T^2 + d T^3 = x1
#        x0 -(3/2) d T^3 + d T^3 = x1
#        -0.5 d T^3 = x1 - x0
#        d = 2 (x0-x1) / T^3
# c = -(3/2) T 2 (x0-x1) / (T^3) = 3 (x1-x0) / T^2
def compute_3rd_order_poly_traj(x0, x1, T, dt):
    a = x0
    b = np.zeros_like(x0)
    c = 3*(x1-x0) / (T**2)
    d = 2*(x0-x1) / (T**3)
    N = int(T/dt)
    n = x0.shape[0]
    x = np.zeros((n,N))
    dx = np.zeros((n,N))
    ddx = np.zeros((n,N))
    for i in range(N):
        t = i*dt
        x[:,i]   = a + b*t + c*t**2 + d*t**3
        dx[:,i]  = b + 2*c*t + 3*d*t**2
        ddx[:,i] = 2*c + 6*d*t
    return x, dx, ddx

def compute_5th_order_poly_traj(x0, x1, T, dt):
    # x(0)=x0, x(T)=x1, dx(0)=dx(T)=0
    # x(t) = a + b t + c t^2 + d t^3 + e t^4 + f t^5
    # x(0) = a = x0 
    # dx(0) = b = 0
    a = x0
    b = np.zeros_like(x0) 
    c = np.zeros_like(x0)
    f = np.zeros_like(x0)
    d = 2*(x1-x0 )/ (T**3)
    e = (x0-x1) / (T**4)
    N = int(T/dt)
    n = x0.shape[0]
    x = np.zeros((n,N))
    dx = np.zeros((n,N))
    ddx = np.zeros((n,N))
    for i in range(N):
        t = i*dt
        x[:,i]   = a + b*t + c*t**2 + d*t**3 + e*t**4 + f*t**5
        dx[:,i]  = b + 2*c*t + 3*d*t**2 + 4*e*t**3 + 5*f*t**4
        ddx[:,i] = 2*c + 6*d*t + 12*e*t**2 + 20*f*t**3
    return x, dx, ddx

def compute_norm_contact_slippage(contact_position):
  # first time instance the robot touches the ground
  for p in contact_position:
    if np.linalg.norm(p) > -1e-8 and  np.linalg.norm(p) < 1e-8:
      continue
    else:
      contact_pos_ref = p 
    break   
  contact_dev_norm = np.zeros(contact_position.shape[0])
  for time_idx in range(len(contact_position)-1):
    # ignore contact samples in the air  
    if np.linalg.norm(contact_position[time_idx], 2) > -1e-8 and np.linalg.norm(contact_position[time_idx], 2) < 1e-8:
      contact_pos_ref = contact_position[time_idx+1]
    else:
      slippage_norm = np.linalg.norm((contact_pos_ref-contact_position[time_idx]), 2)
      # ignore simulation spikes 
      if slippage_norm > 0.015:
        contact_dev_norm[time_idx] = contact_dev_norm[time_idx-1]
      else:
        contact_dev_norm[time_idx] = slippage_norm
  return contact_dev_norm    

def plot_contact_forces(conf, scp_sol, scp_sol_stoch):
  U_ref_nom = scp_sol['control'][-1].T
  U_ref_stoch = scp_sol_stoch['control'][-1].T
  time = np.arange(0, np.round((U_ref_nom.shape[0])*0.01, 2),0.01)
  mu_vector = conf.mu*np.ones(time.shape[0])
  fig1, (FL, FR, HL, HR) = plt.subplots(4, 1, sharex=True)
  FL.plot(time, mu_vector, linestyle='dashed', color='black', label='$\mu$') 
  FR.plot(time, mu_vector, linestyle='dashed', color='black') 
  HL.plot(time, mu_vector, linestyle='dashed', color='black') 
  HR.plot(time, mu_vector, linestyle='dashed', color='black') 
  fig1.subplots_adjust(hspace=0.4)
  nb_steps = conf.gait['nbSteps']
  nb_ss_phases = nb_steps 
  nb_ds_phases = (nb_ss_phases*2)  + 1
  ss_knots = conf.gait['stepKnots'] 
  ds_knots = conf.gait['supportKnots']
  dt = conf.dt
  t_min = 0.
  # color step phases
  # DS phases 
  for i in range(nb_ds_phases):
      t_max = t_min + dt*ds_knots
      FR.axvspan(t_min, t_max, facecolor ='b', alpha = 0.1)
      HL.axvspan(t_min, t_max, facecolor ='b', alpha = 0.1)
      HR.axvspan(t_min, t_max, facecolor ='b', alpha = 0.1)
      if i == 0:
          FL.axvspan(t_min, t_max, facecolor ='b', label= 'DS phase', alpha = 0.1)
          t_min = t_max + dt*(ss_knots-1)
      else:
          FL.axvspan(t_min, t_max, facecolor ='b', alpha = 0.1)
          t_min = t_max + dt*ss_knots
  t_min = dt*ds_knots
  # SS phases
  if conf.gait['type'] == 'BOUND':
      for i in range(nb_ss_phases):
          if i == 0 :
              t_max = t_min + dt*(ss_knots-1)
              HL.axvspan(t_min, t_max, facecolor ='g', label= 'SS phase', alpha = 0.1)
          else:
              t_max = t_min + dt*(ss_knots)
              HL.axvspan(t_min, t_max, facecolor ='g', alpha = 0.1)
          HR.axvspan(t_min, t_max, facecolor ='g', alpha = 0.1)
          t_min = t_max + dt*(ss_knots + 2*ds_knots)

      t_min = dt*(ds_knots*2 + ss_knots-1)
      for i in range(nb_ss_phases):
          t_max = t_min + dt*(ss_knots)    
          FR.axvspan(t_min, t_max, facecolor ='g', alpha = 0.1)
          FL.axvspan(t_min, t_max, facecolor ='g', alpha = 0.1)
          t_min = t_max + dt*(ss_knots + 2*ds_knots)
  if conf.gait['type'] == 'TROT':
      for i in range(nb_ss_phases):
          if i == 0 :
              t_max = t_min + dt*(ss_knots-1)
              FL.axvspan(t_min, t_max, facecolor ='g', label= 'SS phase', alpha = 0.1)
          else:
              t_max = t_min + dt*(ss_knots)
              FL.axvspan(t_min, t_max, facecolor ='g', alpha = 0.1)
          HR.axvspan(t_min, t_max, facecolor ='g', alpha = 0.1)
          t_min = t_max + dt*(ss_knots + 2*ds_knots)

      t_min = dt*(ds_knots*2 + ss_knots-1)
      for i in range(nb_ss_phases):
          t_max = t_min + dt*(ss_knots)    
          FR.axvspan(t_min, t_max, facecolor ='g', alpha = 0.1)
          HL.axvspan(t_min, t_max, facecolor ='g', alpha = 0.1)
          t_min = t_max + dt*(ss_knots + 2*ds_knots)
  # compare SCP
  for contact in conf.ee_frame_names:
      plt.rc('font', family ='serif')
      contact_name = contact[0:2]
      if contact_name == 'FL':
          f_scp_nom = U_ref_nom[:, 3:6]
          f_scp_stoch = U_ref_stoch[:, 3:6] 
      elif contact_name == 'FR':
          f_scp_nom = U_ref_nom[:, 0:3] 
          f_scp_stoch = U_ref_stoch[:, 0:3]
      elif contact_name == 'HL':
          f_scp_nom = U_ref_nom[:, 9:12] 
          f_scp_stoch = U_ref_stoch[:, 9:12] 
      elif contact_name == 'HR':
          f_scp_nom = U_ref_nom[:, 6:9]
          f_scp_stoch = U_ref_stoch[:, 6:9]
      fig2, (fx, fy, fz) = plt.subplots(3, 1, sharex=True) 
      fx.step(time, f_scp_nom[:, 0], label=' SCP forces nom (N)')
      fx.step(time, f_scp_stoch[:, 0], label=' SCP forces stoch (N)')
      fx.legend()
      fx.set_title('force x')
      fy.step(time, f_scp_nom[:, 1])
      fy.step(time, f_scp_stoch[:, 1])
      fy.set_title('force y')
      fz.step(time, f_scp_nom[:, 2])
      fz.step(time, f_scp_stoch[:, 2])
      fz.set_title('force z')
      fz.set_xlabel('Time (s)', fontsize=14)
      fig2.suptitle('contact forces of '+ str(contact[0:2]))
      f_scp_nom_norm = np.zeros(U_ref_nom.shape[0])
      f_scp_stoch_norm = np.zeros(U_ref_nom.shape[0])
      # plot tangential vs vertical forces 
      for time_idx in range(U_ref_nom.shape[0]):
          # foot not on the ground
          if  np.linalg.norm(f_scp_stoch[time_idx], 2) == 0:
              continue
          else:
              f_scp_nom_norm[time_idx] =  np.linalg.norm(f_scp_nom[time_idx, :2], 2)/f_scp_nom[time_idx, 2]
              f_scp_stoch_norm[time_idx] = np.linalg.norm(f_scp_stoch[time_idx, :2], 2)/f_scp_stoch[time_idx, 2]
      if contact_name == 'FL':
          FL.step(time, f_scp_nom_norm, label='nominal')
          FL.step(time, f_scp_stoch_norm, label='stochastic')
          FL.set_title('FL', fontsize=12) 
      elif contact_name == 'FR':
          FR.step(time, f_scp_nom_norm)
          FR.step(time, f_scp_stoch_norm)
          FR.set_title('FR', fontsize=12)
      elif contact_name == 'HL':
          HL.step(time, f_scp_nom_norm)
          HL.step(time, f_scp_stoch_norm)
          HL.set_title('HL', fontsize=12)
          HL.set_ylabel('||Tangential forces|| / Vertical force', fontsize=12)
      elif contact_name == 'HR':
          HR.step(time, f_scp_nom_norm)
          HR.step(time, f_scp_stoch_norm)
          HR.set_title('HR', fontsize=12)
          HR.set_xlabel('Time (s)', fontsize=12)   
  if conf.gait['type'] == 'BOUND':
      fig1.legend(loc=7, fontsize="small")
  elif conf.gait['type'] == 'TROT':
      fig1.legend(loc=7, fontsize="small")

def plot_centroidal_tracking_cost(nb_sims, Q, dt_ctrl, x_ref_nom, x_ref_stoch, 
                         data_nom_1, data_nom_2, data_stoch_1, data_stoch_2):
  N = x_ref_nom.shape[1]
  J_nom_1 = np.zeros((nb_sims, N))
  J_stoch_1 = np.zeros((nb_sims, N))
  J_nom_2 = np.zeros((nb_sims, N))
  J_stoch_2 = np.zeros((nb_sims, N))
  J_nom_stats_1 = np.zeros((N, 2))
  J_stoch_stats_1 = np.zeros((N, 2))
  J_nom_stats_2 = np.zeros((N, 2))
  J_stoch_stats_2 = np.zeros((N, 2))
  for sim in range(nb_sims):
    for time_idx in range(N):
      delta_x_nom_1 = x_ref_nom[:, time_idx] - data_nom_1['centroidal'][sim, time_idx]
      delta_x_stoch_1 = x_ref_stoch[:, time_idx] - data_stoch_1['centroidal'][sim, time_idx]
      delta_x_nom_2 = x_ref_nom[:, time_idx] - data_nom_2['centroidal'][sim, time_idx]
      delta_x_stoch_2 = x_ref_stoch[:, time_idx] - data_stoch_2['centroidal'][sim, time_idx]
      J_nom_1[sim, time_idx] = (delta_x_nom_1.T @ Q) @ delta_x_nom_1
      J_stoch_1[sim, time_idx]= (delta_x_stoch_1.T @ Q) @ delta_x_stoch_1
      J_nom_2[sim, time_idx] = (delta_x_nom_2.T @ Q) @ delta_x_nom_2
      J_stoch_2[sim, time_idx]= (delta_x_stoch_2.T @ Q) @ delta_x_stoch_2
  for time_idx in range(N):
    if time_idx == 0:
      J_nom_stats_1[time_idx, 0] = np.mean(J_nom_1[:, time_idx])
      J_nom_stats_1[time_idx, 1] = np.std(J_nom_1[:, time_idx])
      J_stoch_stats_1[time_idx, 0] = np.mean(J_stoch_1[:, time_idx])
      J_stoch_stats_1[time_idx, 1] = np.std(J_stoch_1[:, time_idx])
      J_nom_stats_2[time_idx, 0] = np.mean(J_nom_2[:, time_idx])
      J_nom_stats_2[time_idx, 1] = np.std(J_nom_2[:, time_idx])
      J_stoch_stats_2[time_idx, 0] = np.mean(J_stoch_2[:, time_idx])
      J_stoch_stats_2[time_idx, 1] = np.std(J_stoch_2[:, time_idx])
    else:
      J_nom_stats_1[time_idx, 0] = np.mean(J_nom_1[:, time_idx]) + J_nom_stats_1[time_idx-1, 0]
      J_nom_stats_1[time_idx, 1] = np.std(J_nom_1[:, time_idx]) + J_nom_stats_1[time_idx-1, 1]
      J_stoch_stats_1[time_idx, 0] = np.mean(J_stoch_1[:, time_idx]) + J_stoch_stats_1[time_idx-1, 0]
      J_stoch_stats_1[time_idx, 1] = np.std(J_stoch_1[:, time_idx]) + J_stoch_stats_1[time_idx-1, 1]
      J_nom_stats_2[time_idx, 0] = np.mean(J_nom_2[:, time_idx]) + J_nom_stats_2[time_idx-1, 0]
      J_nom_stats_2[time_idx, 1] = np.std(J_nom_2[:, time_idx]) + J_nom_stats_2[time_idx-1, 1]
      J_stoch_stats_2[time_idx, 0] = np.mean(J_stoch_2[:, time_idx]) + J_stoch_stats_2[time_idx-1, 0]
      J_stoch_stats_2[time_idx, 1] = np.std(J_stoch_2[:, time_idx]) + J_stoch_stats_2[time_idx-1, 1]
  fig, ax = plt.subplots(1, 1, sharex=True)
  time = np.arange(0, np.round(N*dt_ctrl, 2), dt_ctrl)
  ax.plot(time, J_nom_stats_1[:, 0], label='nominal without debris')
  ax.fill_between(time, J_nom_stats_1[:, 0]+J_nom_stats_1[:, 1], 
            J_nom_stats_1[:, 0]-J_nom_stats_1[:, 1], alpha=0.2)
  ax.plot(time, J_stoch_stats_1[:, 0], label='stochastic without debris')
  ax.fill_between(time, J_stoch_stats_1[:, 0]+J_stoch_stats_1[:, 1],
             J_stoch_stats_1[:, 0]-J_stoch_stats_1[:, 1], alpha=0.2)
  ax.plot(time, J_nom_stats_2[:, 0], label='nominal with debris')
  ax.fill_between(time, J_nom_stats_2[:, 0]+J_nom_stats_2[:, 1],
             J_nom_stats_2[:, 0]-J_nom_stats_2[:, 1], alpha=0.2)
  ax.plot(time, J_stoch_stats_2[:, 0], label='stochastic with debris')
  ax.fill_between(time, J_stoch_stats_2[:, 0]+J_stoch_stats_2[:, 1],
             J_stoch_stats_2[:, 0]-J_stoch_stats_2[:, 1], alpha=0.2)
  ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
  ax.legend()
  ax.set_xlabel('Time (s)')
  ax.set_ylabel('Centroidal tracking cost')  

def plot_contact_slippage(nb_sims, dt_ctrl, data_nom_1, data_nom_2, data_stoch_1, data_stoch_2):
  contact_pos_nom1_all = np.asarray(data_nom_1['contact_positions'])          
  contact_pos_stoch1_all = np.asarray(data_stoch_1['contact_positions'])
  contact_pos_nom2_all = np.asarray(data_nom_2['contact_positions'])          
  contact_pos_stoch2_all = np.asarray(data_stoch_2['contact_positions'])
  # time = time[:contact_pos_nom1_all.shape[1]]
  contact_dev_nom1 = np.zeros((nb_sims, contact_pos_nom1_all.shape[1]))
  contact_dev_stoch1 = np.zeros((nb_sims, contact_pos_stoch1_all.shape[1]))
  contact_dev_nom2 = np.zeros((nb_sims, contact_pos_nom2_all.shape[1]))
  contact_dev_stoch2 = np.zeros((nb_sims, contact_pos_stoch2_all.shape[1]))
  stat_nom_1 = np.zeros((contact_pos_nom1_all.shape[1], 2))
  stat_stoch_1 = np.zeros((contact_pos_stoch1_all.shape[1], 2))
  stat_nom_2 = np.zeros((contact_pos_nom2_all.shape[1], 2))
  stat_stoch_2 = np.zeros((contact_pos_stoch2_all.shape[1], 2))
  N = contact_pos_nom1_all.shape[1]
  time = np.arange(0, np.round(N*dt_ctrl, 2), dt_ctrl)

  for sim in range(nb_sims):
    for contact_idx in range(4): 
      contact_pos_nom1 = contact_pos_nom1_all[sim, :, contact_idx]
      contact_dev_nom1[sim, :] = compute_norm_contact_slippage(contact_pos_nom1)
      contact_pos_stoch1 = contact_pos_stoch1_all[sim,:, contact_idx]
      contact_dev_stoch1[sim, :] = compute_norm_contact_slippage(contact_pos_stoch1)
      
      contact_pos_nom2 = contact_pos_nom2_all[sim, :, contact_idx]
      contact_dev_nom2[sim, :] = compute_norm_contact_slippage(contact_pos_nom2)
      contact_pos_stoch2 = contact_pos_stoch2_all[sim,:, contact_idx]
      contact_dev_stoch2[sim, :] = compute_norm_contact_slippage(contact_pos_stoch2)

  for time_idx in range(contact_pos_nom1_all.shape[1]):
    if time_idx == 0:
      stat_nom_1[time_idx, 0] = np.mean(contact_dev_nom1[:, time_idx])
      stat_nom_1[time_idx, 1] = np.std(contact_dev_nom1[:, time_idx])
      stat_stoch_1[time_idx, 0] = np.mean(contact_dev_stoch1[:, time_idx])
      stat_stoch_1[time_idx, 1] = np.std(contact_dev_stoch1[:, time_idx])

      stat_nom_2[time_idx, 0] = np.mean(contact_dev_nom2[:, time_idx])
      stat_nom_2[time_idx, 1] = np.std(contact_dev_nom2[:, time_idx])
      stat_stoch_2[time_idx, 0] = np.mean(contact_dev_stoch2[:, time_idx])
      stat_stoch_2[time_idx, 1] = np.std(contact_dev_stoch2[:, time_idx])
    else:
      curr_mean_nom1 = np.mean(contact_dev_nom1[:, time_idx])
      curr_std_nom1 = np.std(contact_dev_nom1[:, time_idx])
      curr_mean_stoch1 = np.mean(contact_dev_stoch1[:, time_idx])
      curr_std_stoch1 = np.std(contact_dev_stoch1[:, time_idx])
      stat_nom_1[time_idx, 0] = stat_nom_1[time_idx-1, 0] + \
               curr_mean_nom1-(stat_nom_1[time_idx-1, 0]/(time_idx))
      stat_nom_1[time_idx, 1] = stat_nom_1[time_idx-1, 1] + \
                curr_std_nom1-(stat_nom_1[time_idx-1, 1]/(time_idx))
      stat_stoch_1[time_idx, 0] = stat_stoch_1[time_idx-1, 0] + \
           curr_mean_stoch1-(stat_stoch_1[time_idx-1, 0]/(time_idx))
      stat_stoch_1[time_idx, 1] = stat_stoch_1[time_idx-1, 1] + \
            curr_std_stoch1-(stat_stoch_1[time_idx-1, 1]/(time_idx))

      curr_mean_nom2 = np.mean(contact_dev_nom2[:, time_idx])
      curr_std_nom2 = np.std(contact_dev_nom2[:, time_idx])
      curr_mean_stoch2 = np.mean(contact_dev_stoch2[:, time_idx])
      curr_std_stoch2 = np.std(contact_dev_stoch2[:, time_idx])
      stat_nom_2[time_idx, 0] = stat_nom_2[time_idx-1, 0] + \
            curr_mean_nom2-(stat_nom_2[time_idx-1, 0]/(time_idx))
      stat_nom_2[time_idx, 1] = stat_nom_2[time_idx-1, 1] + \
             curr_std_nom2-(stat_nom_2[time_idx-1, 1]/(time_idx))
      stat_stoch_2[time_idx, 0] = stat_stoch_2[time_idx-1, 0] + \
        curr_mean_stoch2-(stat_stoch_2[time_idx-1, 0]/(time_idx))
      stat_stoch_2[time_idx, 1] = stat_stoch_2[time_idx-1, 1] + \
          curr_std_stoch2-(stat_stoch_2[time_idx-1, 1]/(time_idx))  
  fig, ax = plt.subplots(1, 1, sharex=True)
  ax.plot(time, stat_nom_1[:, 0], label='nominal without debris')
  ax.fill_between(time, stat_nom_1[:, 0]+stat_nom_1[:, 1], 
            stat_nom_1[:, 0]-stat_nom_1[:, 1], alpha=0.2)
  ax.plot(time, stat_stoch_1[:, 0], label='stochastic without debris')
  ax.fill_between(time, stat_stoch_1[:, 0]+stat_stoch_1[:, 1], 
             stat_stoch_1[:, 0]-stat_stoch_1[:, 1], alpha=0.2)
  ax.plot(time, stat_nom_2[:, 0], label='nominal with debris')
  ax.fill_between(time, stat_nom_2[:, 0]+stat_nom_2[:, 1],
             stat_nom_2[:, 0]-stat_nom_2[:, 1], alpha=0.2)
  ax.plot(time, stat_stoch_2[:, 0], label='stochastic with debris')
  ax.fill_between(time, stat_stoch_2[:, 0]+stat_stoch_2[:, 1],
             stat_stoch_2[:, 0]-stat_stoch_2[:, 1], alpha=0.2)
  ax.legend()
  ax.set_xlabel('Time (s)')
  ax.set_ylabel('Normalized integral of feet slippage norm (m)')