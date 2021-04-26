import numpy as np
import matplotlib.pyplot as plt 
# helper functions
def normalize(v):
    assert(v.shape[0] == 3)
    return v/np.linalg.norm(v)

def construct_friction_pyramid_constraint_matrix(self):
    pyramid_constraint_matrix = np.zeros((3,3))
    pyramid_constraints_vector = normalize(np.array([0.5*np.sqrt(2), 0.5*np.sqrt(2), -self.__linear_friction_coefficient]))
    pyramid_rotation_matrix = np.array([[np.cos(2*np.pi/4), -np.sin(2*np.pi/4), 0],
                                        [np.sin(2*np.pi/4), np.cos(2*np.pi/4) , 0],
                                        [0                , 0                 , 1]])
    rotated_pyramid_vector = pyramid_rotation_matrix @ pyramid_constraints_vector
    for i in range(3):
        pyramid_constraint_matrix[i,::] = rotated_pyramid_vector 
    return pyramid_constraint_matrix

def plot_state(self):
    X = self.all_solution['state'][0]
    plt.rc('text', usetex = True)
    plt.rc('font', family ='serif')
    dt = self.model._dt
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
    fig.suptitle('state trajectories', fontsize=20)
    #plt.show()

def plot_rf_controls(self):
    U = self.all_solution['control'][0][0:int(self.n_u/2), :]
    plt.rc('text', usetex = True)
    plt.rc('font', family ='serif')
    dt = self.model._dt
    fig, (copx, copy, fx, fy, fz, tauz) = plt.subplots(6, 1, sharex=True)
    time = np.arange(0, np.round((U.shape[1])*dt, 2),dt)
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
    fig.suptitle('control trajectories of the right foot', fontsize=20)

def plot_lf_controls(self):
    U = self.all_solution['control'][0][int(self.n_u/2):, :]
    plt.rc('text', usetex = True)
    plt.rc('font', family ='serif')
    dt = self.model._dt
    fig, (copx, copy, fx, fy, fz, tauz) = plt.subplots(6, 1, sharex=True)
    time = np.arange(0, np.round((U.shape[1])*dt, 2),dt)
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
    fig.suptitle('control trajectories of the left foot', fontsize=20)
    plt.show()