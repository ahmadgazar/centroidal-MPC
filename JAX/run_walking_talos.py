from centroidal_model import Centroidal_model
from collections import namedtuple 
from scp_solver import solve_scp
import matplotlib.pyplot as plt
import conf_talos as conf
import numpy as np
import pickle 

# construct model and solve SCP
model = Centroidal_model(conf) 
solution = solve_scp(model, conf.scp_params)

if model._robot == 'TALOS':
    Controls = namedtuple('Controls', 'cop_x, cop_y, fx, fy, fz, tau_z')
elif model._robot == 'solo12':
    Controls = namedtuple('Controls', 'fx, fy, fz')

# plot final solution
X = solution['state'][-1]
plt.rc('text', usetex = True)
plt.rc('font', family ='serif')
dt = model._dt
fig, (comx, comy, comz, lx, ly, lz, kx, ky, kz) = plt.subplots(9, 1, sharex=True)
time = np.arange(0, np.round((X.shape[1])*dt, 2),dt)
comx.plot(time, X[0,:])
comx.set_title('CoM$_x$')
plt.setp(comx, ylabel=r'\textbf(m)')
comy.plot(time, X[1,:])
comy.set_title('CoM$_y$')
plt.setp(comy, ylabel=r'\textbf(m)')
comz.plot(time, X[2,:])
comz.set_title('CoM$_z$')
plt.setp(comz, ylabel=r'\textbf(m)')
lx.plot(time, X[3,:])
lx.set_title('lin. mom$_x$')
plt.setp(lx, ylabel=r'\textbf(kg.m/s)')
ly.plot(time, X[4,:])
ly.set_title('lin. mom$_y$')
plt.setp(ly, ylabel=r'\textbf(kg.m/s)')
lz.plot(time, X[5,:])
lz.set_title('lin. mom$_z$')
plt.setp(lz, ylabel=r'\textbf(kg.m/s)')
kx.plot(time, X[6,:])
kx.set_title('ang. mom$_x$')
plt.setp(kx, ylabel=r'\textbf(kg.m$^2$/s)')
ky.plot(time, X[7,:])
ky.set_title('ang. mom$_y$')
plt.setp(ky, ylabel=r'\textbf(kg.m$^2$/s)')
kz.plot(time, X[8,:])
kz.set_title('ang. mom$_z$')
plt.setp(kz, ylabel=r'\textbf(kg.m/s)')
plt.xlabel(r'\textbf{time} (s)', fontsize=14)
fig.suptitle('state trajectories')

nb_contacts = len(model._contact_trajectory)
nu = model._n_u 
U_dict = {}
for contact_idx, contact in enumerate (model._contact_trajectory):
    cop_idx_0 = int(contact_idx*nu/nb_contacts)
    plt.rc('text', usetex = True)
    plt.rc('font', family ='serif')
    U = solution['control'][-1][cop_idx_0:cop_idx_0+int(nu/nb_contacts)] 
    time = np.arange(0, np.round((U.shape[1])*dt, 2),dt)
    if model._robot == 'solo12':
        u = Controls(fx=U[0,:], fy=U[1,:], fz=U[2,:])
        U_dict[contact] = u._asdict()
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
        u = Controls(cop_x=U[0,:], cop_y=U[1,:], fx=U[2,:], fy=U[3,:], fz=U[4,:], tau_z=U[5,:])
        U_dict[contact] = u._asdict()
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

# save pickled solution data
data_dict = dict(X=X, U=[U_dict], CONTACT_SEQUENCE_TRAJECTORY=model._contact_trajectory)
with open('DYN_TO_IK.pickle', 'wb') as filename:
    pickle.dump(data_dict, filename)
# visualize 
plt.show()
