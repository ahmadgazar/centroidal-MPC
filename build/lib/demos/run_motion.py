from whole_body_control import WholeBodyModel, plotSolution 
from scp_solver import solve_scp, interpolate_SCP_solution
from centroidal_model import Centroidal_model
# import conf_solo12_trot as conf
import conf_solo12_bound as conf
# import conf_solo12_pace as conf
import matplotlib.pylab as plt
import numpy as np
import crocoddyl

# Iteration 1: whole-body DDP (used to warm-start SCP)
# ---------------------------------------------------
# tracks a CoM forward direction based on the  
# number of foot steps.
# create and solve whole-body shooting problem 
print('running whole-Body DDP to warm start centroidal SCP ..', '\n')
wbd_model = WholeBodyModel(conf, TRACK_CENTROIDAL=False)
if conf.gait['type'] == 'TROT':
    problem = wbd_model.createTrotShootingProblem()
elif conf.gait['type'] == 'PACE':    
    problem = wbd_model.createPaceShootingProblem()
elif conf.gait['type'] == 'BOUND':    
    problem = wbd_model.createBoundShootingProblem()
solver = crocoddyl.SolverFDDP(problem)
xs = [wbd_model.rmodel.defaultState] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([wbd_model.rmodel.defaultState] * solver.problem.T)
solver.solve(xs, us, 100, False, 0.1)
# save centroidal trajectories from whole-body solution
ddp_sol_1 = wbd_model.get_solution_trajectories(solver)
np.savez('wholeBody_to_centroidal_traj.npz', X=ddp_sol_1['centroidal'])
#----------------------------------------------------------------------------
#                          NOMINAL TRAJ OPT
# ---------------------------------------------------------------------------
# Iteration 2: centroidal SCP tracking centroidal states from DDP iteration 1
# ---------------------------------------------------------------------------
# create and solve centroidal scp problem
print('running centroidal SCP ..', '\n')
model = Centroidal_model(conf) 
# save centroidal trajectories from whole-body solution
scp_sol = solve_scp(model, conf.scp_params) 
scp_sol_interpol_nom = interpolate_SCP_solution(scp_sol)
np.savez('scp_sol_interpol_nom.npz', X=scp_sol_interpol_nom['X'], U=scp_sol_interpol_nom['U'])                
np.savez('centroidal_to_wholeBody_traj.npz', X=scp_sol['state'][-1], U=scp_sol['control'][-1])               
# -----------------------------------------------------------------------------
# Iteration 3: whole-body DDP tracking centroidal SCP states and contact forces
# -----------------------------------------------------------------------------
# create and solve whole-body shooting problem
print('running whole-Body DDP to track centorial SCP ..', '\n')
wbd_model = WholeBodyModel(conf, TRACK_CENTROIDAL=True)
if conf.gait['type'] == 'TROT':
    problem = wbd_model.createTrotShootingProblem()
elif conf.gait['type'] == 'PACE':    
    problem = wbd_model.createPaceShootingProblem()
elif conf.gait['type'] == 'BOUND':    
    problem = wbd_model.createBoundShootingProblem()
solver = crocoddyl.SolverFDDP(problem)
xs = [wbd_model.rmodel.defaultState] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([wbd_model.rmodel.defaultState] * solver.problem.T)
solver.setCallbacks([crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackVerbose()])
solver.solve(xs, us, 100, False, 1)
# --------------------------------------------------------
# Interpolate and save final solution for simulation later
# --------------------------------------------------------
# interpolate final whole-body solution and save in dat files 
ddp_sol_2 = wbd_model.get_solution_trajectories(solver)
ddp_interpolated_solution_nom = wbd_model.interpolate_whole_body_solution(ddp_sol_2)
np.savez('wholeBody_interpolated_traj.npz', X=ddp_interpolated_solution_nom['centroidal'], 
                                          U=ddp_interpolated_solution_nom['jointTorques'],
                                          q=ddp_interpolated_solution_nom['jointPos'],
                                          qdot=ddp_interpolated_solution_nom['jointVel'],
                                          gains=ddp_interpolated_solution_nom['gains'])

# get predicted contact positions, forces and jacobians from the last iteration of whole-body DDP
contact_positions_nom, ddp_forces_nom, contact_jacobians_nom = wbd_model.get_contact_positions_and_forces_solution(solver)

# Added the callback functions
# if conf.WITHDISPLAY and conf.WITHPLOT:
#     display = crocoddyl.GepettoDisplay(conf.robot, 4, 4, conf.cameraTF, frameNames=conf.ee_frame_names)
#     solver.setCallbacks(
#         [crocoddyl.CallbackLogger(),
#             crocoddyl.CallbackVerbose(),
#             crocoddyl.CallbackDisplay(display)])
# elif conf.WITHDISPLAY:
#     display = crocoddyl.GepettoDisplay(conf.robot, 4, 4, conf.cameraTF, frameNames=conf.ee_frame_names)
#     solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
# elif conf.WITHPLOT:
#     solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

# # Display the entire motion in gepetto GUI
# if conf.WITHDISPLAY:
#     display = crocoddyl.GepettoDisplay(conf.robot, frameNames=conf.ee_frame_names)
#     display.displayFromSolver(solver, factor=5)

# Plot the CoM, joint states and torques
# if conf.WITHPLOT:
#     log = solver.getCallbacks()[0]
#     plotSolution(solver, figIndex=1, show=True)

#---------------------------------------------------------------------------------------
#                               STOCHASTIC TRAJ OPT
# --------------------------------------------------------------------------------------
# Iteration 2: stochastic centroidal SCP tracking centroidal states from DDP iteration 1
# --------------------------------------------------------------------------------------
# create and solve centroidal scp problem
print('running stochastic centroidal SCP ..', '\n')
model = Centroidal_model(conf, STOCHASTIC_OCP=True) 
# save centroidal trajectories from whole-body solution
scp_sol_stoch = solve_scp(model, conf.scp_params)         
scp_sol_interpol_stoch = interpolate_SCP_solution(scp_sol)
np.savez('scp_sol_interpol_stoch.npz', X=scp_sol_interpol_stoch['X'], U=scp_sol_interpol_stoch['U'])              
np.savez('centroidal_to_wholeBody_traj.npz', X=scp_sol_stoch['state'][-1], U=scp_sol_stoch['control'][-1])               
# -----------------------------------------------------------------------------
# Iteration 3: whole-body DDP tracking centroidal SCP states and contact forces
# -----------------------------------------------------------------------------
# create and solve whole-body shooting problem
print('running whole-Body DDP to track stochastic centorial SCP ..', '\n')
wbd_model = WholeBodyModel(conf, TRACK_CENTROIDAL=True)
if conf.gait['type'] == 'TROT':
    problem = wbd_model.createTrotShootingProblem()
elif conf.gait['type'] == 'PACE':    
    problem = wbd_model.createPaceShootingProblem()
elif conf.gait['type'] == 'BOUND':    
    problem = wbd_model.createBoundShootingProblem()
solver = crocoddyl.SolverFDDP(problem)
xs = [wbd_model.rmodel.defaultState] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([wbd_model.rmodel.defaultState] * solver.problem.T)
solver.setCallbacks([crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose()])
solver.solve(xs, us, 100, False, 1)
# --------------------------------------------------------
# Interpolate and save final solution for simulation later
# --------------------------------------------------------
# interpolate final whole-body solution and save in dat files 
ddp_sol_2 = wbd_model.get_solution_trajectories(solver)
ddp_interpolated_solution_stoch = wbd_model.interpolate_whole_body_solution(ddp_sol_2)
np.savez('wholeBody_interpolated_traj_stoch.npz', X=ddp_interpolated_solution_stoch['centroidal'], 
                                          U=ddp_interpolated_solution_stoch['jointTorques'],
                                          q=ddp_interpolated_solution_stoch['jointPos'],
                                          qdot=ddp_interpolated_solution_stoch['jointVel'],
                                          gains=ddp_interpolated_solution_stoch['gains'])
# get predicted contact positions, forces and jacobians from the last iteration of whole-body DDP
contact_positions_stoch, ddp_forces_stoch, contact_jacobians_stoch = wbd_model.get_contact_positions_and_forces_solution(solver) 

# # Added the callback functions
# if conf.WITHDISPLAY and conf.WITHPLOT:
#     display = crocoddyl.GepettoDisplay(conf.robot, 4, 4, conf.cameraTF, frameNames=conf.ee_frame_names)
#     solver.setCallbacks(
#         [crocoddyl.CallbackLogger(),
#             crocoddyl.CallbackVerbose(),
#             crocoddyl.CallbackDisplay(display)])
# elif conf.WITHDISPLAY:
#     display = crocoddyl.GepettoDisplay(conf.robot, 4, 4, conf.cameraTF, frameNames=conf.ee_frame_names)
#     solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
# elif conf.WITHPLOT:
#     solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

# # Display the entire motion in gepetto GUI
# if conf.WITHDISPLAY:
#     display = crocoddyl.GepettoDisplay(conf.robot, frameNames=conf.ee_frame_names)
#     display.displayFromSolver(solver, factor=5)

# Plot the CoM, joint states and torques
# if conf.WITHPLOT:
#     log = solver.getCallbacks()[0]
#     plotSolution(solver, figIndex=1, show=True)
#-----------------------------------------------------------------------------------------------------------
#   PLOT STUFF
# ---------------------------------------------------------------------------------------------------------
# plot contact positions 
# for contact_idx, contact in enumerate (conf.ee_frame_names):
#     force_idx_0 = contact_idx*3
#     plt.rc('font', family ='serif')
#     time = np.arange(0, np.round((contact_positions_nom.shape[0])*0.01, 2),0.01)
#     p_foot = contact_positions_nom[:, force_idx_0:force_idx_0+3] 
#     fig, (px, py, pz) = plt.subplots(3, 1, sharex=True) 
#     px.plot(time, p_foot[:, 0], label=r'\textbf{z} (N)')
#     px.set_title('px')
#     plt.setp(px, ylabel=r'\textbf(m)')
#     py.plot(time, p_foot[:, 1])
#     py.set_title('py')
#     plt.setp(py, ylabel=r'\textbf(m)')
#     pz.plot(time, p_foot[:, 2])
#     pz.set_title('pz')
#     plt.setp(pz, ylabel=r'\textbf(m)')
#     plt.xlabel('time (s)', fontsize=14)
#     fig.suptitle('swing foot trajectories of '+ str(contact[0:2]))   

# plot contact forces 
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


# contact forces 
for contact_idx, contact in enumerate (conf.ee_frame_names):
    plt.rc('font', family ='serif')
    contact_name = contact[0:2]
    if contact_name == 'FL':
        f_scp_nom = U_ref_nom[:, 3:6]
        f_scp_stoch = U_ref_stoch[:, 3:6] 
        f_ddp_nom = ddp_forces_nom[:, 0:3]
        f_ddp_stoch = ddp_forces_stoch[:, 0:3]
    elif contact_name == 'FR':
        f_scp_nom = U_ref_nom[:, 0:3] 
        f_scp_stoch = U_ref_stoch[:, 0:3]
        f_ddp_nom = ddp_forces_nom[:, 3:6]
        f_ddp_stoch = ddp_forces_stoch[:, 3:6] 
    elif contact_name == 'HL':
        f_scp_nom = U_ref_nom[:, 9:12] 
        f_scp_stoch = U_ref_stoch[:, 9:12] 
        f_ddp_nom = ddp_forces_nom[:, 6:9]
        f_ddp_stoch = ddp_forces_stoch[:, 6:9]
    elif contact_name == 'HR':
        f_scp_nom = U_ref_nom[:, 6:9]
        f_scp_stoch = U_ref_stoch[:, 6:9]
        f_ddp_nom = ddp_forces_nom[:, 9:12]
        f_ddp_stoch = ddp_forces_stoch[:, 9:12] 
    fig2, (fx, fy, fz) = plt.subplots(3, 1, sharex=True) 
    fx.step(time, f_scp_nom[:, 0], label=' SCP forces nom (N)')
    fx.step(time, f_scp_stoch[:, 0], label=' SCP forces stoch (N)')
    # fx.plot(time, f_ddp_nom[:, 0], label=' DDP forces nom (N)')
    # fx.plot(time, f_ddp_stoch[:, 0], label=' DDP forces stoch (N)')
    fx.legend()
    fx.set_title('force x')
    fy.step(time, f_scp_nom[:, 1])
    fy.step(time, f_scp_stoch[:, 1])
    # fy.plot(time, f_ddp_nom[:, 1])
    # fy.plot(time, f_ddp_stoch[:, 1])
    fy.set_title('force y')
    fz.step(time, f_scp_nom[:, 2])
    fz.step(time, f_scp_stoch[:, 2])
    # fz.plot(time, f_ddp_nom[:, 2])
    # fz.plot(time, f_ddp_stoch[:, 2])
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
# # plot DDP and SCP centroidal solutions 
# X_ddp1 = ddp_sol_1['centroidal']
# X_ddp2 = ddp_sol_2['centroidal']
# X_scp = scp_sol['state'][-1]
# fig, (comx, comy, comz, lx, ly, lz, kx, ky, kz) = plt.subplots(9, 1, sharex=True)
# time = np.arange(0, np.round((X_ddp1.shape[0])*conf.dt, 2),conf.dt)
# comx.plot(time, X_ddp1[:, 0], label='DDP_1')
# comx.plot(time, X_scp[0,:], label='SCP')
# comx.plot(time, X_ddp2[:, 0], label='DDP_2')
# comx.set_title('CoM$_x$')
# comx.legend()
# plt.setp(comx, ylabel=r'\textbf(m)')
# comy.plot(time, X_ddp1[:,1], label='DDP_1')
# comy.plot(time, X_scp[1,:], label='SCP')
# comy.plot(time, X_ddp2[:,1], label='DDP_2')
# comy.set_title('CoM$_y$')
# plt.setp(comy, ylabel=r'\textbf(m)')
# comz.plot(time, X_ddp1[:,2], label='DDP_1')
# comz.plot(time, X_scp[2,:], label='SCP')
# comz.plot(time, X_ddp2[:,2], label='DDP_2')
# comz.set_title('CoM$_z$')
# plt.setp(comz, ylabel=r'\textbf(m)')
# lx.plot(time, X_ddp1[:,3], label='DDP_1')
# lx.plot(time, X_scp[3,:], label='SCP')
# lx.plot(time, X_ddp2[:,3], label='DDP_2')
# lx.set_title('lin. mom$_x$')
# plt.setp(lx, ylabel=r'\textbf(kg.m/s)')
# ly.plot(time, X_ddp1[:,4], label='DDP_1')
# ly.plot(time, X_scp[4,:], label='SCP')
# ly.plot(time, X_ddp2[:,4], label='DDP_2')
# ly.set_title('lin. mom$_y$')
# plt.setp(ly, ylabel=r'\textbf(kg.m/s)')
# lz.plot(time, X_ddp1[:,5], label='DDP_1')
# lz.plot(time, X_scp[5,:], label='SCP')
# lz.plot(time, X_ddp2[:,5], label='DDP_2')
# lz.set_title('lin. mom$_z$')
# plt.setp(lz, ylabel=r'\textbf(kg.m/s)')
# kx.plot(time, X_ddp1[:,6], label='DDP_1')
# kx.plot(time, X_scp[6,:], label='SCP')
# kx.plot(time, X_ddp2[:,6], label='DDP_2')
# kx.set_title('ang. mom$_x$')
# plt.setp(kx, ylabel=r'\textbf(kg.m$^2$/s)')
# ky.plot(time, X_ddp1[:,7], label='DDP_1')
# ky.plot(time, X_scp[7,:], label='SCP')
# ky.plot(time, X_ddp2[:,7], label='DDP_2')
# ky.set_title('ang. mom$_y$')
# plt.setp(ky, ylabel=r'\textbf(kg.m$^2$/s)')
# kz.plot(time, X_ddp1[:,8], label='DDP_1')
# kz.plot(time, X_scp[8,:], label='SCP')
# kz.plot(time, X_ddp2[:,8], label='DDP_2')
# kz.set_title('ang. mom$_z$')
# plt.setp(kz, ylabel=r'\textbf(kg.m/s)')
# plt.xlabel(r'\textbf{time} (s)', fontsize=14)
# fig.suptitle('state trajectories')
plt.show()    
