from constraints import evaluate_friction_pyramid_constraints
from whole_body_control import WholeBodyModel 
from centroidal_model import Centroidal_model
from scp_solver import solve_scp
# import conf_solo12_trot as conf
# import conf_solo12_pace as conf
import conf_solo12_bound as conf
import matplotlib.pylab as plt
import numpy as np
import crocoddyl
#-------------------------------------------------------------------
#                    NOMINAL SCP
# ------------------------------------------------------------------
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
# save centroidal reference trajectories from whole-body solution
ddp_sol_1 = wbd_model.get_solution_trajectories(solver)
interpolated_centroidal_ref = wbd_model.interpolate_whole_body_solution(ddp_sol_1)
np.savez('wholeBody_to_centroidal_traj.npz', X=ddp_sol_1['centroidal'])
np.savez('interpolated_centroidal_warm_start.npz', X=interpolated_centroidal_ref['centroidal'])

# create and solve centroidal SCP problem
print('running centroidal SCP ..', '\n')
model = Centroidal_model(conf) 
scp_sol_nom = solve_scp(model, conf.scp_params)  
np.savez('centroidal_to_wholeBody_traj.npz', X=scp_sol_nom['state'][-1], U=scp_sol_nom['control'][-1], 
                                                                       gains=scp_sol_nom['gains'][-1])      

# create and solve whole-body shooting problem to track SCP trajectories 
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
solver.solve(xs, us, 100, False, 0.1)
contact_positions_nom, contact_forces_nom = wbd_model.get_contact_positions_and_forces_solution(solver)
# reorder croccodyl forces to match SCP ones
contact_forces_nom = np.concatenate([contact_forces_nom[:,3:6], contact_forces_nom[:,0:3],
                                     contact_forces_nom[:,9:12], contact_forces_nom[:,6:9]], axis=1)
ddp_sol_1 = wbd_model.get_solution_trajectories(solver)
ddp_interpolated_solution = wbd_model.interpolate_whole_body_solution(ddp_sol_1)
np.savez('wholeBody_interpolated_traj.npz', X=ddp_interpolated_solution['centroidal'], 
                                          U=ddp_interpolated_solution['jointTorques'],
                                          q=ddp_interpolated_solution['jointPos'],
                                          qdot=ddp_interpolated_solution['jointVel'],
                                          gains=ddp_interpolated_solution['gains'])                                     
print("evaluating if SCP solution is satisfying friction cones ..")
evaluate_friction_pyramid_constraints(model, scp_sol_nom['control'][-1])
print("evaluating if DDP solution is satisfying friction cones ..")
evaluate_friction_pyramid_constraints(model, contact_forces_nom.T)
# #-------------------------------------------------------------------
# #                    STOCHASTIC SCP
# # ------------------------------------------------------------------
# create and solve whole-body shooting problem 
print('running whole-Body DDP to warm start chance-constrained centroidal SCP ..', '\n')
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

# create and solve centroidal stochastic SCP problem
print('running stochastic centroidal SCP ..', '\n')
model = Centroidal_model(conf, STOCHASTIC_OCP=True) 
scp_sol_stoch = solve_scp(model, conf.scp_params)                
np.savez('centroidal_to_wholeBody_traj', X=scp_sol_stoch['state'][-1], U=scp_sol_stoch['control'][-1], 
                                     gains=scp_sol_stoch['gains'][-1], covs=scp_sol_stoch['covs'][-1])     

# create and solve whole-body shooting problem to track stochastic SCP 
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
solver.solve(xs, us, 100, False, 0.1)
contact_positions_stoch, contact_forces_stoch = wbd_model.get_contact_positions_and_forces_solution(solver)
# reorder croccodyl forces to match SCP ones
contact_forces_stoch = np.concatenate([contact_forces_stoch[:,3:6], contact_forces_stoch[:,0:3],
                                     contact_forces_stoch[:,9:12], contact_forces_stoch[:,6:9]], axis=1)
ddp_sol_2 = wbd_model.get_solution_trajectories(solver)
ddp_interpolated_solution = wbd_model.interpolate_whole_body_solution(ddp_sol_2, scp_sol_stoch['gains'][-1])
np.savez('wholeBody_interpolated_traj_stoch.npz', X=ddp_interpolated_solution['centroidal'], 
                                          U=ddp_interpolated_solution['jointTorques'],
                                          q=ddp_interpolated_solution['jointPos'],
                                          qdot=ddp_interpolated_solution['jointVel'],
                                          gains=ddp_interpolated_solution['gains'])
print("evaluating if chance-constrained SCP solution is satisfying friction cones ..")
evaluate_friction_pyramid_constraints(model, scp_sol_stoch['control'][-1])
print("evaluating if DDP solution is satisfying friction cones ..")
evaluate_friction_pyramid_constraints(model, contact_forces_stoch.T)

# plot CoM

x_stoch = scp_sol_stoch['state'][-1]
# nb_friction_constraints = 5
# xi = norm.ppf(1-(conf.beta_u/nb_friction_constraints*3))
time = np.arange(0, np.round((x_stoch.shape[1])*conf.dt, 2),conf.dt)
# fig1, ax = plt.subplots()
# for time_idx in range(x_stoch.shape[1]):
#     com_x, com_y = x_stoch[0, time_idx], x_stoch[1, time_idx]
#     com_cov = scp_sol_stoch['covs'][-1][time_idx, :2, :2]
#     w, v = np.linalg.eig(com_cov)
#     order = w.argsort()[::-1]
#     w, v = w[order], v[:, order]
#     vx, vy = v[:,0][0], v[:,0][1]
#     theta = np.arctan2(vy, vx)
#     width, height = 2 * xi * np.sqrt(w)
#     # cov_ellipse = Ellipse(xy=(com_x, com_y), width=width, 
#     #              height=height, angle=np.degrees(theta))
#     # ax.plt.plot(com_x, com_y)
    # ax.add_artist(cov_ellipse)
# plot linear momentum
# fig2, (lin_mom_x, lin_mom_y, lin_mom_z) = plt.subplots(3, 1, sharex=True)
# lin_mom_x.plot(time, x_stoch[3, :])
# lin_mom_y.plot(time, x_stoch[4, :])
# lin_mom_z.plot(time, x_stoch[5, :])
# # plot angular momentum
# fig3, (ang_mom_x, ang_mom_y, ang_mom_z) = plt.subplots(3, 1, sharex=True)
# ang_mom_x.plot(time, x_stoch[6, :])
# ang_mom_y.plot(time, x_stoch[7, :])
# ang_mom_z.plot(time, x_stoch[8, :])
# plot DDP forces 
for contact_idx, contact in enumerate (conf.ee_frame_names):
    plt.rc('font', family ='serif')
    contact_name = contact[0:2]
    time = np.arange(0, np.round((contact_forces_stoch.shape[0])*0.01, 2),0.01)
    if contact_name == 'FL':
        f_nom = contact_forces_nom[:, 0:3]
        f_stoch = contact_forces_stoch[:, 0:3]
    elif contact_name == 'FR':
        f_nom = contact_forces_nom[:, 3:6]
        f_stoch = contact_forces_stoch[:, 3:6]
    elif contact_name == 'HL':
        f_nom = contact_forces_nom[:, 6:9]
        f_stoch = contact_forces_stoch[:, 6:9]
    elif contact_name == 'HR':
        f_nom = contact_forces_nom[:, 9:12]    
        f_stoch = contact_forces_stoch[:, 9:12]
    fig, (fx, fy, fz) = plt.subplots(3, 1, sharex=True) 
    fx.plot(time, f_nom[:, 0], label=' DDP tracking nominal SCP forces (N)')
    fx.plot(time, f_stoch[:, 0], label=' DDP tracking stochastic SCP (N)')
    fx.legend()
    fx.set_title('force x')
    fy.plot(time, f_nom[:, 1])
    fy.plot(time, f_stoch[:, 1])
    fy.set_title('force y')
    fz.plot(time, f_nom[:, 2])
    fz.plot(time, f_stoch[:, 2])
    fz.set_title('force z')
    plt.xlabel('time (s)', fontsize=14)
    fig.suptitle('DDP contact forces of '+ str(contact[0:2]))   

# plot SCP forces 
scp_nom = scp_sol_nom['control'][-1].T    
scp_stoch = scp_sol_stoch['control'][-1].T
for contact_idx, contact in enumerate (conf.ee_frame_names):
    plt.rc('font', family ='serif')
    contact_name = contact[0:2]
    time = np.arange(0, np.round((scp_nom.shape[0])*0.01, 2),0.01)
    if contact_name == 'FL':
        f_nom = scp_nom[:, 0:3]
        f_stoch = scp_stoch[:, 0:3]
    elif contact_name == 'FR':
        f_nom = scp_nom[:, 3:6]
        f_stoch = scp_stoch[:, 3:6]
    elif contact_name == 'HL':
        f_nom = scp_nom[:, 6:9]
        f_stoch = scp_stoch[:, 6:9]
    elif contact_name == 'HR':
        f_nom = scp_nom[:, 9:12]    
        f_stoch = scp_stoch[:, 9:12]
    fig, (fx, fy, fz) = plt.subplots(3, 1, sharex=True) 
    fx.plot(time, f_nom[:, 0], label=' nominal SCP ')
    fx.plot(time, f_stoch[:, 0], label=' stochastic SCP ')
    fx.legend()
    fx.set_title('x', fontsize=12)
    fx.set_ylabel('(N)', fontsize=12)

    fy.plot(time, f_nom[:, 1])
    fy.plot(time, f_stoch[:, 1])
    fy.set_ylabel('(N)', fontsize=12)

    fy.set_title('y', fontsize=12)
    fz.plot(time, f_nom[:, 2])
    fz.plot(time, f_stoch[:, 2])
    fz.set_ylabel('(N)', fontsize=12)

    fz.set_title('z', fontsize=12)
    plt.xlabel('Time (s)', fontsize=12)
    fig.suptitle('SCP contact forces of '+ str(contact[0:2] + ' foot'))   

# plot SCP forces norms and their norm errors
fig, (FL, FR, HL, HR) = plt.subplots(4, 1, sharex=True) 
for contact_idx, contact in enumerate (conf.ee_frame_names):
    plt.rc('font', family ='serif')
    contact_name = contact[0:2]
    time = np.arange(0, np.round((scp_nom.shape[0])*0.01, 2),0.01)
    if contact_name == 'FL':
        f_nom = scp_nom[:, 0:3]
        f_stoch = scp_stoch[:, 0:3]
    elif contact_name == 'FR':
        f_nom = scp_nom[:, 3:6]
        f_stoch = scp_stoch[:, 3:6]
    elif contact_name == 'HL':
        f_nom = scp_nom[:, 6:9]
        f_stoch = scp_stoch[:, 6:9]
    elif contact_name == 'HR':
        f_nom = scp_nom[:, 9:12]    
        f_stoch = scp_stoch[:, 9:12]
    f_norm_ratio = 0.
    f_nom_norm = np.zeros(scp_nom.shape[0])
    f_stoch_norm = np.zeros(scp_nom.shape[0])
    counter = 0
    for time_idx in range(scp_nom.shape[0]):
        # foot not on the ground
        if  np.linalg.norm(f_stoch[time_idx], 2) == 0:
            continue
        else:
            counter += 1
            f_nom_norm_k = np.linalg.norm(f_nom[time_idx], 2)
            f_stoch_norm_k = np.linalg.norm(f_stoch[time_idx], 2)
            f_nom_norm[time_idx] = f_nom_norm_k
            f_stoch_norm[time_idx] = f_stoch_norm_k
            f_norm_ratio += f_stoch_norm_k/f_nom_norm_k
    if contact_name == 'FL':
        FL.plot(time, f_nom_norm, label='nominal')
        FL.plot(time, f_stoch_norm, label='stochastic')
        FL.set_title('FL', fontsize=12)
        FL.set_ylabel('$||f||$', fontsize=12)
        # FL.legend(title="||f_stoch||/||f_nom|| = " +str(f_norm_ratio/counter))
    elif contact_name == 'FR':
        FR.plot(time, f_nom_norm, label='nominal')
        FR.plot(time, f_stoch_norm, label='stochastic')
        FR.set_title('FR', fontsize=12)
        FR.set_ylabel('$||f||$', fontsize=12)
        # FR.legend(title="||f_stoch||/||f_nom|| = " +str(f_norm_ratio/counter))

    elif contact_name == 'HL':
        HL.plot(time, f_nom_norm, label="nominal")
        HL.plot(time, f_stoch_norm, label="stochastic")
        HL.set_title('HL', fontsize=12)
        HL.set_ylabel('$||f||$', fontsize=12)
        # HL.legend(title="||f_stoch||/||f_nom|| = " +str(f_norm_ratio/counter))

    elif contact_name == 'HR':
        HR.plot(time, f_nom_norm, label="nominal")
        HR.plot(time, f_stoch_norm, label="stochastic")
        HR.set_title('HR', fontsize=12)
        HR.set_ylabel('$||f||$', fontsize=12)
        # HR.legend(title="||f_stoch||/||f_nom|| = " +str(f_norm_ratio/counter))
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('$||f||$', fontsize=12)
    # print('norm of f_nom = ', f_nom_norm, 'norm of f_stoch = ', f_stoch_norm)
    print('average norm ratio betwen stochastic SCP forces and nominal SCP forces = ', f_norm_ratio/counter)    
    # ax.plot(time, f_norm_ratio, label='str(contact[0:2]')
    # ax.set_title('Ratio of stochastic SCP forces /nominal SCP forces', fontsize=12)


# Plotting interpolated DDP joint torques
# plt.figure()
# legJointNames = ['HAA', 'HFE', 'KFE']
# # LF foot
# plt.subplot(4, 3, 3)
# plt.title('joint torque [Nm]')
# [plt.plot(ddp_interpolated_solution['jointTorques'][:, k], label=legJointNames[i]) for i, k in enumerate(range(0, 3))]
# plt.ylabel('LF')
# plt.legend()

# # LH foot
# plt.subplot(4, 3, 6)
# [plt.plot(ddp_interpolated_solution['jointTorques'][:, k], label=legJointNames[i]) for i, k in enumerate(range(3, 6))]
# plt.ylabel('LH')
# plt.legend()

# # RF foot
# plt.subplot(4, 3, 9)
# [plt.plot(ddp_interpolated_solution['jointTorques'][:, k], label=legJointNames[i]) for i, k in enumerate(range(6, 9))]
# plt.ylabel('RF')
# plt.legend()

# # RH foot
# plt.subplot(4, 3, 12)
# [plt.plot(ddp_interpolated_solution['jointTorques'][:, k], label=legJointNames[i]) for i, k in enumerate(range(9, 12))]
# plt.ylabel('RH')
# plt.legend()
# plt.xlabel('knots')

# from whole_body_control import plotSolution
# plotSolution(solver)
plt.show()

