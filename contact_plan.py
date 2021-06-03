import numpy as np
import sympy as sp
import pinocchio as pin
from numpy.linalg import norm
import conf

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
    contact_trajectory = create_contact_trajectory(conf)
    for contact in contact_trajectory['RF']: 
        if contact.ACTIVE:
            print(contact.pose.rotation)
        else:
            print(contact.ACTIVE)
   

