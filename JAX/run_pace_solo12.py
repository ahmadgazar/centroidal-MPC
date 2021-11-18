from whole_body_control import WholeBodyModel, plotSolution 
from centroidal_model import Centroidal_model
from scp_solver import solve_scp
import conf_solo12_pace as conf
import matplotlib.pylab as plt
import numpy as np
import crocoddyl

# create and solve whole-body shooting problem 
wbd_model = WholeBodyModel(conf, TRACK_CENTROIDAL=False)
problem = wbd_model.createPaceShootingProblem()
solver = crocoddyl.SolverFDDP(problem)
xs = [wbd_model.rmodel.defaultState] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([wbd_model.rmodel.defaultState] * solver.problem.T)
solver.solve(xs, us, 100, False, 0.1)
# save centroidal trajectories from whole-body solution
ddp_sol_1 = wbd_model.get_solution_trajectories(solver)
np.savez('wholeBody_to_centroidal_traj.npz', X=ddp_sol_1['centroidal'])

# create and solve centroidal scp problem
model = Centroidal_model(conf) 
scp_sol = solve_scp(model, conf.scp_params)                
np.savez('centroidal_to_wholeBody_traj', X=scp_sol['state'][-1], U=scp_sol['control'][-1])               

# create and solve whole-body shooting problem
wbd_model = WholeBodyModel(conf, TRACK_CENTROIDAL=True)
problem = wbd_model.createPaceShootingProblem()
solver = crocoddyl.SolverFDDP(problem)
xs = [wbd_model.rmodel.defaultState] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([wbd_model.rmodel.defaultState] * solver.problem.T)
solver.solve(xs, us, 100, False, 0.1)
# interpolate final whole-body solution and save in dat files 
ddp_sol_2 = wbd_model.get_solution_trajectories(solver)
ddp_interpolated_solution = wbd_model.interpolate_whole_body_solution(ddp_sol_2)
np.savez('wholeBody_to_centroidal_traj.npz', X=ddp_interpolated_solution['centroidal'], 
                                          U=ddp_interpolated_solution['jointTorques'],
                                          q=ddp_interpolated_solution['jointPos'],
                                          qdot=ddp_interpolated_solution['jointVel'])
if conf.SAVEDAT:
    wbd_model.save_solution_dat(ddp_interpolated_solution)

contact_positions, contact_forces = wbd_model.get_contact_positions_and_forces_solution(solver)

# Added the callback functions
if conf.WITHDISPLAY and conf.WITHPLOT:
    display = crocoddyl.GepettoDisplay(conf.robot, 4, 4, conf.cameraTF, frameNames=conf.ee_frame_names)
    solver.setCallbacks(
        [crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackDisplay(display)])
elif conf.WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(conf.robot, 4, 4, conf.cameraTF, frameNames=conf.ee_frame_names)
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif conf.WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

# Display the entire motion
if conf.WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(conf.robot, frameNames=conf.ee_frame_names)
    display.displayFromSolver(solver)

# Plotting the entire motion
if conf.WITHPLOT:
    log = solver.getCallbacks()[0]
    plotSolution(solver, figIndex=1, show=True)

# plot contact positions 
for contact_idx, contact in enumerate (conf.ee_frame_names):
    force_idx_0 = contact_idx*3
    plt.rc('font', family ='serif')
    time = np.arange(0, np.round((contact_positions.shape[0])*0.01, 2),0.01)
    p_foot = contact_positions[:, force_idx_0:force_idx_0+3] 
    fig, (px, py, pz) = plt.subplots(3, 1, sharex=True) 
    px.plot(time, p_foot[:, 0], label=r'\textbf{z} (N)')
    px.set_title('px')
    plt.setp(px, ylabel=r'\textbf(m)')
    py.plot(time, p_foot[:, 1])
    py.set_title('py')
    plt.setp(py, ylabel=r'\textbf(m)')
    pz.plot(time, p_foot[:, 2])
    pz.set_title('pz')
    plt.setp(pz, ylabel=r'\textbf(m)')
    plt.xlabel('time (s)', fontsize=14)
    fig.suptitle('swing foot trajectories of '+ str(contact[0:2]))   

# plot contact forces 
U_ref = scp_sol['control'][-1].T
for contact_idx, contact in enumerate (conf.ee_frame_names):
    plt.rc('font', family ='serif')
    contact_name = contact[0:2]
    time = np.arange(0, np.round((U_ref.shape[0])*0.01, 2),0.01)
    if contact_name == 'FL':
        f_scp = U_ref[:, 3:6]
        f_croccodyl = contact_forces[:, 0:3]
    elif contact_name == 'FR':
        f_scp = U_ref[:, 0:3]
        f_croccodyl = contact_forces[:, 3:6]
    elif contact_name == 'HL':
        f_scp = U_ref[:, 9:12]
        f_croccodyl = contact_forces[:, 6:9]
    elif contact_name == 'HR':
        f_scp = U_ref[:, 6:9]    
        f_croccodyl = contact_forces[:, 9:12]
    fig, (fx, fy, fz) = plt.subplots(3, 1, sharex=True) 
    fx.plot(time, f_scp[:, 0], label=' SCP forces (N)')
    fx.plot(time, f_croccodyl[:, 0], label=' DDP forces (N)')
    fx.legend()
    fx.set_title('force x')
    fy.plot(time, f_scp[:, 1])
    fy.plot(time, f_croccodyl[:, 1])
    fy.set_title('force y')
    fz.plot(time, f_scp[:, 2])
    fz.plot(time, f_croccodyl[:, 2])
    fz.set_title('force z')
    plt.xlabel('time (s)', fontsize=14)
    fig.suptitle('swing foot trajectories of '+ str(contact[0:2]))   

# plot DDP and SCP centroidal solutions 
X_ddp1 = ddp_sol_1['centroidal']
X_ddp2 = ddp_sol_2['centroidal']
X_scp = scp_sol['state'][-1]
fig, (comx, comy, comz, lx, ly, lz, kx, ky, kz) = plt.subplots(9, 1, sharex=True)
time = np.arange(0, np.round((X_ddp1.shape[0])*conf.dt, 2),conf.dt)
comx.plot(time, X_ddp1[:, 0], label='DDP_1')
comx.plot(time, X_scp[0,:], label='SCP')
comx.plot(time, X_ddp2[:, 0], label='DDP_2')
comx.set_title('CoM$_x$')
comx.legend()
plt.setp(comx, ylabel=r'\textbf(m)')
comy.plot(time, X_ddp1[:,1], label='DDP_1')
comy.plot(time, X_scp[1,:], label='SCP')
comy.plot(time, X_ddp2[:,1], label='DDP_2')
comy.set_title('CoM$_y$')
plt.setp(comy, ylabel=r'\textbf(m)')
comz.plot(time, X_ddp1[:,2], label='DDP_1')
comz.plot(time, X_scp[2,:], label='SCP')
comz.plot(time, X_ddp2[:,2], label='DDP_2')
comz.set_title('CoM$_z$')
plt.setp(comz, ylabel=r'\textbf(m)')
lx.plot(time, X_ddp1[:,3], label='DDP_1')
lx.plot(time, X_scp[3,:], label='SCP')
lx.plot(time, X_ddp2[:,3], label='DDP_2')
lx.set_title('lin. mom$_x$')
plt.setp(lx, ylabel=r'\textbf(kg.m/s)')
ly.plot(time, X_ddp1[:,4], label='DDP_1')
ly.plot(time, X_scp[4,:], label='SCP')
ly.plot(time, X_ddp2[:,4], label='DDP_2')
ly.set_title('lin. mom$_y$')
plt.setp(ly, ylabel=r'\textbf(kg.m/s)')
lz.plot(time, X_ddp1[:,5], label='DDP_1')
lz.plot(time, X_scp[5,:], label='SCP')
lz.plot(time, X_ddp2[:,5], label='DDP_2')
lz.set_title('lin. mom$_z$')
plt.setp(lz, ylabel=r'\textbf(kg.m/s)')
kx.plot(time, X_ddp1[:,6], label='DDP_1')
kx.plot(time, X_scp[6,:], label='SCP')
kx.plot(time, X_ddp2[:,6], label='DDP_2')
kx.set_title('ang. mom$_x$')
plt.setp(kx, ylabel=r'\textbf(kg.m$^2$/s)')
ky.plot(time, X_ddp1[:,7], label='DDP_1')
ky.plot(time, X_scp[7,:], label='SCP')
ky.plot(time, X_ddp2[:,7], label='DDP_2')
ky.set_title('ang. mom$_y$')
plt.setp(ky, ylabel=r'\textbf(kg.m$^2$/s)')
kz.plot(time, X_ddp1[:,8], label='DDP_1')
kz.plot(time, X_scp[8,:], label='SCP')
kz.plot(time, X_ddp2[:,8], label='DDP_2')
kz.set_title('ang. mom$_z$')
plt.setp(kz, ylabel=r'\textbf(kg.m/s)')
plt.xlabel(r'\textbf{time} (s)', fontsize=14)
fig.suptitle('state trajectories')
plt.show()    