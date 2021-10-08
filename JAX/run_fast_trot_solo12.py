from typing import Sequence
from centroidal_model import Centroidal_model
from utils import interpolate_centroidal_traj 
from contact_plan import compute_foot_traj
import conf_solo12_fast_trot as conf
from scp_solver import solve_scp
import matplotlib.pyplot as plt
import numpy as np

dt_dyn = conf.dt
dt_IK = conf.dt_ctrl 
N = dt_dyn/dt_IK 
if not conf.DYNAMICS_FIRST:
    # load centroidal IK trajectories
    com_pos_IK_data = np.loadtxt('com_position.dat')
    lin_mom_IK_data = np.loadtxt('linear_momentum.dat')
    ang_mom_IK_data = np.loadtxt('angular_momentum.dat')
    # read centroidal IK trajectories 
    IK_to_dyn = []
    for i in range(len(lin_mom_IK_data)):
        if i==0 or i%N==0:
            IK_to_dyn.append(np.concatenate([com_pos_IK_data[i, 1::], lin_mom_IK_data[i, 1::], ang_mom_IK_data[i, 1::]], axis=0))
    # save centroidal IK trajectories
    np.savez('IK_to_dyn_centroidal_traj', X=IK_to_dyn)               

# construct model and solve centoidal SCP
model = Centroidal_model(conf) 
solution = solve_scp(model, conf.scp_params)

# interpolate centroidal dynamics for tracking controller
interpolated_centroidal_traj = interpolate_centroidal_traj(conf, dict(contact_sequence=model._contact_data['contacts_logic'],
                                                                                                 state=solution['state'][-1],
                                                                                            control=solution['control'][-1]))
# construct swing-foot trajectories based on the contact plan in conf
swing_foot_traj  = compute_foot_traj(conf)  

# save everything
np.savez('dynamic_plan', CENTROIDAL_PLAN=interpolated_centroidal_traj['state'],
                                FORCES=interpolated_centroidal_traj['control'], 
                                CONTACT_SEQUENCE=interpolated_centroidal_traj['contact_sequence'],
                              FR_SWING_FOOT_TRAJ=np.concatenate([swing_foot_traj['FR']['x'],swing_foot_traj['FR']['x_dot'], swing_foot_traj['FR']['x_ddot']], axis=0),
                              FL_SWING_FOOT_TRAJ=np.concatenate([swing_foot_traj['FL']['x'],swing_foot_traj['FL']['x_dot'], swing_foot_traj['FL']['x_ddot']], axis=0),
                              HR_SWING_FOOT_TRAJ=np.concatenate([swing_foot_traj['HR']['x'],swing_foot_traj['HR']['x_dot'], swing_foot_traj['HR']['x_ddot']], axis=0),
                              HL_SWING_FOOT_TRAJ=np.concatenate([swing_foot_traj['HL']['x'],swing_foot_traj['HL']['x_dot'], swing_foot_traj['HL']['x_ddot']], axis=0))               


# plot final solution
# X = solution['state'][-1]
X = interpolated_centroidal_traj['state']
if conf.DYNAMICS_FIRST:
    X_ref = model._init_trajectories['state'].T  
else:
    X_ref = model._init_trajectories['state']    
plt.rc('text', usetex = True)
plt.rc('font', family ='serif')
dt = model._dt
fig, (comx, comy, comz, lx, ly, lz, kx, ky, kz) = plt.subplots(9, 1, sharex=True)
time = np.arange(0, np.round((X.shape[1])*dt, 2),dt)
# comx.plot(time, X_ref[:, 0])
comx.plot(time, X[0,:])
comx.set_title('CoM$_x$')
plt.setp(comx, ylabel=r'\textbf(m)')
# comy.plot(time, X_ref[:,1])
comy.plot(time, X[1,:])
comy.set_title('CoM$_y$')
plt.setp(comy, ylabel=r'\textbf(m)')
# comz.plot(time, X_ref[:,2])
comz.plot(time, X[2,:])
comz.set_title('CoM$_z$')
plt.setp(comz, ylabel=r'\textbf(m)')
# lx.plot(time, X_ref[:,3])
lx.plot(time, X[3,:])
lx.set_title('lin. mom$_x$')
plt.setp(lx, ylabel=r'\textbf(kg.m/s)')
# ly.plot(time, X_ref[:,4])
ly.plot(time, X[4,:])
ly.set_title('lin. mom$_y$')
plt.setp(ly, ylabel=r'\textbf(kg.m/s)')
# lz.plot(time, X_ref[:,5])
lz.plot(time, X[5,:])
lz.set_title('lin. mom$_z$')
plt.setp(lz, ylabel=r'\textbf(kg.m/s)')
# kx.plot(time, X_ref[:,6])
kx.plot(time, X[6,:])
kx.set_title('ang. mom$_x$')
plt.setp(kx, ylabel=r'\textbf(kg.m$^2$/s)')
# ky.plot(time, X_ref[:,7])
ky.plot(time, X[7,:])
ky.set_title('ang. mom$_y$')
plt.setp(ky, ylabel=r'\textbf(kg.m$^2$/s)')
# kz.plot(time, X_ref[:,8])
kz.plot(time, X[8,:])
kz.set_title('ang. mom$_z$')
plt.setp(kz, ylabel=r'\textbf(kg.m/s)')
plt.xlabel(r'\textbf{time} (s)', fontsize=14)
fig.suptitle('state trajectories')

nb_contacts = len(model._contact_trajectory)
nu = model._n_u 
for contact_idx, contact in enumerate (model._contact_trajectory):
    cop_idx_0 = int(contact_idx*nu/nb_contacts)
    plt.rc('text', usetex = True)
    plt.rc('font', family ='serif')
    # U = solution['control'][-1][cop_idx_0:cop_idx_0+int(nu/nb_contacts)] 
    U = interpolated_centroidal_traj['control'][cop_idx_0:cop_idx_0+int(nu/nb_contacts)] 
    time = np.arange(0, np.round((U.shape[1])*dt, 2),dt)
    if model._robot == 'solo12':
        fig, (fx, fy, fz) = plt.subplots(3, 1, sharex=True) 
        fx.plot(time, U[0,:], label=r'\textbf{z} (N)')
        fx.set_title('F$_x$')
        plt.setp(fx, ylabel=r'\textbf(N)')
        fy.plot(time, U[1,:])
        fy.set_title('F$_y$')
        plt.setp(fy, ylabel=r'\textbf(N)')
        fz.plot(time, U[2,:])
        fz.set_title('F$_z$')
        plt.setp(fz, ylabel=r'\textbf(N)')
        plt.xlabel(r'\textbf{time} (s)', fontsize=14)
        fig.suptitle('control trajectories of '+contact)
    elif model._robot == 'TALOS':
        fig, (copx, copy, fx, fy, fz, tauz) = plt.subplots(6, 1, sharex=True) 
        copx.plot(time, U[0,:])
        copx.set_title('CoP$_x$')
        plt.setp(copx, ylabel=r'\textbf(m)')
        copy.plot(time, U[1,:])
        copy.set_title('CoP$_y$')
        plt.setp(copy, ylabel=r'\textbf(m)')
        fx.plot(time, U[2,:], label=r'\textbf{z} (N)')
        fx.set_title('F$_x$')
        plt.setp(fx, ylabel=r'\textbf(N)')
        fy.plot(time, U[3,:])
        fy.set_title('F$_y$')
        plt.setp(fy, ylabel=r'\textbf(N)')
        fz.plot(time, U[4,:])
        fz.set_title('F$_z$')
        plt.setp(fz, ylabel=r'\textbf(N)')
        tauz.plot(time, U[5,:])
        tauz.set_title('M$_x$')
        plt.setp(tauz, ylabel=r'\textbf(N.m)')
        plt.xlabel(r'\textbf{time} (s)', fontsize=14)
        fig.suptitle('control trajectories of '+contact)

# visualize 
plt.show()
