import numpy as np
import sympy as sp
import pinocchio as pin
from numpy.linalg import norm

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
    contact_trajectory = dict([(foot._name, []) for foot in feet ])
    for contacts in conf.contact_sequence:
        for contact in contacts: 
            for time in range(conf.contact_knots):
                for foot in feet:  
                    if contact.CONTACT==foot._name: 
                        contact_trajectory[foot._name].append(contact)
    return contact_trajectory                

if __name__=='__main__':
    import conf_talos as conf
    contact_trajectory = create_contact_trajectory(conf)
    for time_idx in range(conf.N):
        debris = []
        for contact_idx, contact_name in enumerate(contact_trajectory): 
            debris.append(contact_trajectory[contact_name][time_idx])
        print(debris)