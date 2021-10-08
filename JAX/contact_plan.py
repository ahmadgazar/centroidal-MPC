import numpy as np
import pinocchio as pin
import jax.numpy as jnp 
import matplotlib.pyplot as plt
from collections import namedtuple

class Debris():
    def __init__(self, CONTACT, t_start=0.0, t_end=1.0, x=None, y=None, z=None, axis=None, angle=None, ACTIVE=False):
        """
        Minimal helper function: return the SE3 configuration of a stepstone, with some
        ad-hoc configuration.
        """
        if ACTIVE:
            STEP = 1.0
            axis = np.array(axis, np.float64)
            axis /= np.linalg.norm(axis)
            self.axis = axis
            self.pose = pin.SE3(pin.AngleAxis(angle, np.concatenate([axis, [0]])).matrix(),
                            np.array([x * STEP, y * STEP, z]))
        self.t_start = t_start 
        self.t_end = t_end
        self.CONTACT = CONTACT
        self.ACTIVE = ACTIVE 
        self.__fill_contact_idx()

    def __fill_contact_idx(self):
        if self.CONTACT == 'RF' or self.CONTACT == 'FR':
            self.idx = 0
        elif self.CONTACT == 'LF' or self.CONTACT == 'FL':
            self.idx = 1
        elif self.CONTACT == 'HR':
            self.idx = 2
        elif self.CONTACT == 'HL':
            self.idx = 3                                     
    
# given a contact plan, fill a contact trajectory    
def create_contact_trajectory(conf):
    contact_sequence = conf.contact_sequence
    contact_trajectory = dict([(foot.CONTACT, []) for foot in  contact_sequence[0]])
    for contacts in contact_sequence:
        for contact in contacts:
            contact_duration = int(round((contact.t_end-contact.t_start)/conf.dt))  
            for time in range(contact_duration):
                contact_trajectory[contact.CONTACT].append(contact)  
    return contact_trajectory                

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

def compute_foot_traj(conf):
    step_height = conf.step_height
    dt_ctrl = conf.dt_ctrl
    contact_sequence = conf.contact_sequence
    N_ctrl = conf.N_ctrl
    foot_traj_dict = dict([(foot.CONTACT, dict(x=np.zeros((3, N_ctrl)), x_dot=np.zeros((3, N_ctrl)), 
                                     x_ddot=np.zeros((3, N_ctrl)))) for foot in  contact_sequence[0]])
    previous_contact_sequence = contact_sequence[0]
    for i, contacts in enumerate(contact_sequence):
        if i < len(contact_sequence)-1:
            next_contact_sequence = contact_sequence[i+1]
        else:
           next_contact_sequence = contact_sequence[i]     
        for contact in contacts:
            t_start_idx = int(contact.t_start/dt_ctrl)
            t_end_idx = int(contact.t_end/dt_ctrl)
            N_contact = int((contact.t_end-contact.t_start)/dt_ctrl)
            # foot is in contact 
            if contact.ACTIVE:
                foot_traj_dict[contact.CONTACT]['x'][:, t_start_idx:t_end_idx] = np.tile(contact.pose.translation, (N_contact,1)).T 
            # foot is in the air
            elif not contact.ACTIVE:
                x0 = previous_contact_sequence[contact.idx].pose.translation 
                x1 = next_contact_sequence[contact.idx].pose.translation
                # x and y directions
                x, xdot, xddot = compute_3rd_order_poly_traj(x0[:2], x1[:2], (contact.t_end-contact.t_start), dt_ctrl)
                foot_traj_dict[contact.CONTACT]['x'][:2, t_start_idx:t_end_idx] = x
                foot_traj_dict[contact.CONTACT]['x_dot'][:2, t_start_idx:t_end_idx] = xdot
                foot_traj_dict[contact.CONTACT]['x_ddot'][:2, t_start_idx:t_end_idx] = xddot
                # z direction (interpolate half way from zero to a step height)
                x_up, xdot_up, xddot_up = compute_3rd_order_poly_traj(np.array([0.]), np.array([step_height]), 0.5*(contact.t_end-contact.t_start), dt_ctrl)
                foot_traj_dict[contact.CONTACT]['x'][2, t_start_idx:t_start_idx+int(0.5*N_contact)] = x_up
                foot_traj_dict[contact.CONTACT]['x_dot'][2, t_start_idx:t_start_idx+int(0.5*N_contact)] = xdot_up
                foot_traj_dict[contact.CONTACT]['x_ddot'][2, t_start_idx:t_start_idx+int(0.5*N_contact)] = xddot_up
                # z direction (interpolate half way back from a step height to the ground)
                x_down, xdot_down, xddot_down = compute_3rd_order_poly_traj(np.array([step_height]), np.array([0.]), 0.5*(contact.t_end-contact.t_start), dt_ctrl)
                foot_traj_dict[contact.CONTACT]['x'][2, t_start_idx+int(0.5*N_contact):t_end_idx] = x_down 
                foot_traj_dict[contact.CONTACT]['x_dot'][2, t_start_idx+int(0.5*N_contact):t_end_idx] = xdot_down 
                foot_traj_dict[contact.CONTACT]['x_ddot'][2, t_start_idx+int(0.5*N_contact):t_end_idx] = xddot_down 
        previous_contact_sequence = contact_sequence[i]        
    return foot_traj_dict 

def plot_swing_foot_traj(swing_foot_dict, conf):
    dt = conf.dt_ctrl
    for contact in swing_foot_dict:
        plt.rc('text', usetex = True)
        plt.rc('font', family ='serif')   
        fig, (p_x, v_x, a_x, p_y, v_y, a_y, p_z, v_z, a_z) = plt.subplots(9, 1, sharex=True) 
        px = swing_foot_dict[contact]['x'][0, :]
        py = swing_foot_dict[contact]['x'][1, :]
        pz = swing_foot_dict[contact]['x'][2, :]
        vx = swing_foot_dict[contact]['x_dot'][0, :]
        vy = swing_foot_dict[contact]['x_dot'][1, :]
        vz = swing_foot_dict[contact]['x_dot'][2, :]
        ax = swing_foot_dict[contact]['x_ddot'][0, :]
        ay = swing_foot_dict[contact]['x_ddot'][1, :]
        az = swing_foot_dict[contact]['x_ddot'][2, :]
        time = np.arange(0, np.round((px.shape[0])*dt, 2), dt)
        # end-effector positions
        p_x.plot(time, px)
        p_x.set_title('p$_x$')
        p_y.plot(time, py)
        p_y.set_title('p$_y$')
        p_z.plot(time, pz)
        p_z.set_title('p$_z$')
        # end-effector velocities
        v_x.plot(time, vx)
        v_x.set_title('v$_x$')
        v_y.plot(time, vy)
        v_y.set_title('v$_y$')
        v_z.plot(time, vz)
        v_z.set_title('v$_z$')
        # end-effector accelerations
        a_x.plot(time, ax)
        a_x.set_title('a$_x$')
        a_y.plot(time, ay)
        a_y.set_title('a$_y$')
        a_z.plot(time, az)
        a_z.set_title('a$_z$')
    plt.show()
 
def fill_debris_list(conf):
    Debri = namedtuple('Debris', 'LOGIC, R, p')  
    outer_tuple_list = []
    contact_trajectory = create_contact_trajectory(conf)
    for time_idx in range(conf.N):
        contacts_logic_k = []
        contacts_position_k = []
        contacts_orientation_k = [] 
        inner_tuple_list = []
        for contact in contact_trajectory:
            if contact_trajectory[contact][time_idx].ACTIVE:
                contact_logic = 1
                R = contact_trajectory[contact][time_idx].pose.rotation
                p = contact_trajectory[contact][time_idx].pose.translation
            else:
                contact_logic = 0
                R = jnp.zeros((3,3))
                p = jnp.zeros(3)
            contacts_logic_k.append(contact_logic)
            contacts_orientation_k.append(R)
            contacts_position_k.append(p) 
            inner_tuple_list.append(Debri(contacts_logic_k, contacts_orientation_k, contacts_position_k))                                
        outer_tuple_list.append(inner_tuple_list)
    return outer_tuple_list

if __name__=='__main__':
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    import conf_solo12_fast_trot as conf
    contact_trajectory = create_contact_trajectory(conf)
    swing_foot_traj = compute_foot_traj(conf)
    plot_swing_foot_traj(swing_foot_traj, conf)

