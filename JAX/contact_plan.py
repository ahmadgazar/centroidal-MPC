import numpy as np
import sympy as sp
import pinocchio as pin
from numpy.linalg import norm
import jax.numpy as jnp 
from collections import namedtuple

class Contact:
    def __init__(self, NAME, TYPE):
       self._type = TYPE
       self._name = NAME
       self._active = sp.symbols(NAME+'_ACTIVE')
       self._u = sp.Matrix(['cop_x_'+NAME, 'cop_y_'+NAME, 'fx_'+NAME, 
                                'fy_'+NAME, 'fz_'+NAME, 'tau_z_'+NAME])             
       self._orientation = sp.MatrixSymbol('R_'+NAME,3,3) 
       self._position = sp.MatrixSymbol('p_'+NAME,3,1)
       self._optimizers_indices = {'cops':None, 'forces':None, 'moment':None}
       self._idx = None     

class Debris():
    def __init__(self, CONTACT, x=None, y=None, z=None, axis=None, angle=None, ACTIVE=False):
        """
        Minimal helper function: return the SE3 configuration of a stepstone, with some
        ad-hoc configuration.
        """
        if ACTIVE:
            STEP = 1.0
            axis = np.array(axis, np.float64)
            axis /= norm(axis)
            self.axis = axis
            self.pose = pin.SE3(pin.AngleAxis(angle, np.concatenate([axis, [0]])).matrix(),
                            np.array([x * STEP, y * STEP, z]))
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
    feet = conf.feet
    contact_trajectory = dict([(foot, []) for foot in feet ])
    for contacts in conf.contact_sequence:
        for contact in contacts: 
            for time in range(conf.contact_knots):
                for foot in feet:  
                    if contact.CONTACT==foot: 
                        contact_trajectory[foot].append(contact)
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
    # terminal print settings
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    #np.set_printoptions(linewidth=500)
    import conf_talos as conf
    contact_trajectory = create_contact_trajectory(conf)
    contact_dic = fill_debris_list(conf)
    print(contact_dic)

    for time_idx in range(conf.N):
        debris = []
        for contact_idx, contact_name in enumerate(contact_trajectory): 
            debris.append(contact_trajectory[contact_name][time_idx])
        #print(debris)
