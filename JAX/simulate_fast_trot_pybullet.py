from pickle import TRUE
from mim_control.robot_centroidal_controller import RobotCentroidalController
from mim_control.robot_impedance_controller import RobotImpedanceController
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from bullet_utils.env import BulletEnvWithGround
import conf_solo12_fast_trot as conf
import numpy as np
import argparse

def demo(robot_name):
    # load centroidal dynamics plan based on the 
    plan = np.load('dynamic_plan.npz')
    # load centroidal wrench  and swing foot desired trajectories 
    com_position_des_all = plan['CENTROIDAL_PLAN'][0:3, :]
    com_velocity_des_all = plan['CENTROIDAL_PLAN'][3:6, :]/conf.robot_mass
    base_orie_des_all = np.tile(np.array([[0.0], [0.0], [0.0], [1.0]]), com_position_des_all.shape[1])
    base_ang_velocity_des_all = np.linalg.inv(conf.robot_inertia) @ plan['CENTROIDAL_PLAN'][6:9, :]
    cnt_sequence_all = []
    ee_position_des_all = np.array([]).reshape(0, 12)
    ee_velocity_des_all = np.array([]).reshape(0, 12)
    ee_acceleration_des_all = np.array([]).reshape(0, 12)
    for time_idx in range(plan['CONTACT_SEQUENCE'].shape[0]):
        cnt_sequence_all.append([plan['CONTACT_SEQUENCE'][time_idx,1], plan['CONTACT_SEQUENCE'][time_idx,0], 
                                 plan['CONTACT_SEQUENCE'][time_idx,3], plan['CONTACT_SEQUENCE'][time_idx,2]])
        ee_position_des_all = np.vstack([ee_position_des_all,
                                    np.hstack([plan['FL_SWING_FOOT_TRAJ'][:3, time_idx],
                                    plan['FR_SWING_FOOT_TRAJ'][:3, time_idx],
                                    plan['HL_SWING_FOOT_TRAJ'][:3, time_idx],
                                    plan['HR_SWING_FOOT_TRAJ'][:3, time_idx]])])
        ee_velocity_des_all = np.vstack([ee_velocity_des_all,  
                                    np.hstack([plan['FL_SWING_FOOT_TRAJ'][3:6, time_idx],
                                    plan['FR_SWING_FOOT_TRAJ'][3:6, time_idx],
                                    plan['HL_SWING_FOOT_TRAJ'][3:6, time_idx],
                                    plan['HR_SWING_FOOT_TRAJ'][3:6, time_idx]])])
        ee_acceleration_des_all = np.vstack([ee_acceleration_des_all,  
                            np.hstack([plan['FL_SWING_FOOT_TRAJ'][6:9, time_idx],
                            plan['FR_SWING_FOOT_TRAJ'][6:9, time_idx],
                            plan['HL_SWING_FOOT_TRAJ'][6:9, time_idx],
                            plan['HR_SWING_FOOT_TRAJ'][6:9, time_idx]])])                             

    # Create a Pybullet simulation environment
    env = BulletEnvWithGround()

    # Create a robot instance in the simulator.
    if robot_name == "solo":
        robot = Solo12Robot()
        robot = env.add_robot(robot)
        robot_config = Solo12Config()
        mu = 0.2
        kc = [100., 100., 100.]
        dc = [15., 15., 15.]
        kb = [12, 35, 24]
        db = [1.0, 11.0, 11.0]
        qp_penalty_lin = [5e5, 5e5, 5e5]
        qp_penalty_ang = [1e6, 1e6, 1e6]
    else:
        raise RuntimeError(
            "Robot name [" + str(robot_name) + "] unknown. "
            "Try 'solo"
        )
    # Initialize control
    tau = np.zeros(robot.nb_dof)

    # Reset the robot to some initial state.
    q0 = np.matrix(robot_config.initial_configuration).T
    q0[0] = 0.0
    dq0 = np.matrix(robot_config.initial_velocity).T
    robot.reset_state(q0, dq0)

    # Impedance controller gains
    kp = robot.nb_ee * [50.5, 50.5, 50.5]  
    kd = robot.nb_ee * [0.5, 0.5, 0.5]
    config_file = "impedance_ctrl.yaml"
    robot_cent_ctrl = RobotCentroidalController(
    robot_config,
    mu=mu,
    kc=kc,
    dc=dc,
    kb=kb,
    db=db,
    qp_penalty_lin=qp_penalty_lin,
    qp_penalty_ang=qp_penalty_ang)
    robot_leg_ctrl = RobotImpedanceController(robot, config_file)
    for time_idx in range(len(cnt_sequence_all)):
        # Step the simulator.
        env.step(
            sleep=True 
        )  # You can sleep here if you want to slow down the replay
        # Read the final state and forces after the stepping.
        q, dq = robot.get_state()
        # get desired references
        cnt_sequence = cnt_sequence_all[time_idx]
        # if time_idx > 1000:
        #     print(cnt_sequence)
        com_position_des = com_position_des_all[:, time_idx]
        com_velocity_des = com_velocity_des_all[:, time_idx] 
        base_orie_des = base_orie_des_all[:, time_idx]
        base_ang_velocity_des = base_ang_velocity_des_all[:, time_idx]
        ee_position_des = ee_position_des_all[time_idx]
        ee_velocity_des = ee_velocity_des_all[time_idx]

        # computing forces to be applied in the centroidal space
        w_com = robot_cent_ctrl.compute_com_wrench(
            q, dq, com_position_des, com_velocity_des, base_orie_des, base_ang_velocity_des
        )
        # distributing forces to the active end effectors
        F = robot_cent_ctrl.compute_force_qp(q, dq, cnt_sequence, w_com)
        # passing forces to the impedance controller
        tau = robot_leg_ctrl.return_joint_torques_world(
            q, dq, kp, kd, ee_position_des, ee_velocity_des, F
        )
        # passing torques to the robot
        robot.send_joint_command(tau)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solo", help="Demonstrate Solo.", action="store_true"
    )
    parser.add_argument(
        "--bolt", help="Demonstrate Bolt.", action="store_true"
    )
    args = parser.parse_args()
    if args.solo:
        robot_name = "solo"
    elif args.bolt:
        robot_name = "bolt"
    else:
        robot_name = "solo"

    demo(robot_name)