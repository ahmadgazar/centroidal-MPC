import numpy as np
import pinocchio as pin
import jax.numpy as jnp 
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
    print(contact_trajectory)
