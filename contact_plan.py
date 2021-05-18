import example_robot_data as robex
import hppfcl
import numpy as np
import pinocchio as pin
from numpy.linalg import norm
import conf

class Debris():
    def __init__(self, i, j, altitude, axis, angle, ACTIVE_CONTACT):
        """
        Minimal helper function: return the SE3 configuration of a stepstone, with some
        ad-hoc configuration.
        """
        # i,j are integers (x, y)
        # altitude is the altitude in meter (z)
        # axis is 2D, rotation axis in the plane
        # angle is the angle of inclination, in radian
        STEP = 1.0
        axis = np.array(axis, np.float64)
        axis /= norm(axis)
        self.axis = axis
        self.pose = pin.SE3(pin.AngleAxis(angle, np.concatenate([axis, [0]])).matrix(),
                           np.array([i * STEP, j * STEP, altitude]))
        self.ACTIVE_CONTACT = ACTIVE_CONTACT                     

def addTerrainsToGeomModel(gmodel, terrain, obstacles):
    """
    Add a list of stepstones and obstacles to the robot geometry object.
    Each step stone is defined by its SE3 placement. It is added as a red disk of 20cm radius.
    Each obstacles is defined by its 3d position. It is added as a white sphere of radius 20cm.
    - gmodel is a pinocchio geometry model
    - terrain is a list of SE3 placement of the step stones
    - obstacles is a list of 3d position of the sphere centers.
    """
    # Create pinocchio 3d objects for each step stone
    for i, d in enumerate(terrain):
        # The step stones have name "debris0X" and are attached to the world (jointId=0).
        g2 = pin.GeometryObject("debris%02d" % i, 0, hppfcl.Cylinder(.2, .01), d.pose)
        g2.meshColor = np.array([1, 0, 0, 1.])
        gmodel.addGeometryObject(g2)

        # Create Pinocchio 3d objects for the obstacles.
        for i, obs in enumerate(obstacles):
        # The obstacles have name "obs0X" and are attached to the world (jointId=0).
            g2 = pin.GeometryObject("obs%02d" % i, 0, hppfcl.Sphere(.2), pin.SE3(np.eye(3), obs))
            g2.meshColor = np.array([1, 1, 1, 1.])
            gmodel.addGeometryObject(g2)

        # Add the collision pair to check the robot collision against the step stones and the obstacles.
        # For simplicity, does not activate the self-collision pairs.
        ngeomRobot = len(gmodel.geometryObjects) - len(terrain) - len(obstacles)
        for irobot in range(ngeomRobot):
            for ienv in range(len(terrain) + len(obstacles)):
                gmodel.addCollisionPair(pin.CollisionPair(irobot, ngeomRobot + ienv))
    
# given a contact plan, fill a contact trajectory    
def create_contact_trajectory(conf):
    contact_sequence = conf.contact_sequence 
    contact_duration = conf.contact_knots
    contact_trajectory = {'RF':[], 'LF':[]}
    for contacts in contact_sequence:
        for contact_idx, contact in enumerate (contacts): 
            for time in range(contact_duration):
                if not contact:
                    if contact_idx == 0:
                        contact_trajectory['LF'].append(None)
                    elif contact_idx == 1:
                        contact_trajectory['RF'].append(None)    
                elif contact.ACTIVE_CONTACT=='RF': 
                    contact_trajectory['RF'].append(contact)
                elif contact.ACTIVE_CONTACT=='LF':
                    contact_trajectory['LF'].append(contact)
    return contact_trajectory                

if __name__=='__main__':
    
    # Obstacle is a list of 3D sphere obstacle. They are defined by their sphere centers.
    # obstacles = [
    #     np.array([.5, .3, 1.1]),
    #     np.array([-.3, -.6, 1.4]),
    #     ]
    # The viewer should be initialized after adding the terrain to the robot
    # otherwise, the terrain will not be displayed.
    # --- Load robot model
    # robot = robex.load('talos')
    # addTerrainsToGeomModel(robot.collision_model, conf.terrain, obstacles)
    # addTerrainsToGeomModel(robot.visual_model, conf.terrain, obstacles)
    # Viewer = pin.visualize.GepettoVisualizer
    # viz = Viewer(robot.model, robot.collision_model, robot.visual_model)
    # viz.initViewer(loadModel=True)
    # viz.display(robot.q0)
    contact_trajectory = create_contact_trajectory(conf)
    for contact in contact_trajectory['RF']: 
        if contact:
            print(contact.pose.rotation)
        else:
            print(contact)
   

