"""Agent based on PID controller"""
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p
import math
import env

class PIDAgent:
    def __init__(self):
        
        self.DT = 1./240.
        self.G = 9.81
        
        self.KF = 3.16e-10
        self.KM = 7.94e-12
        self.M = 0.027
        self.ARM = 0.0397

        self.W = self.M*self.G
        self.T2W = 2.5
        self.HOVER_RPM = np.sqrt(self.W/(4*self.KF))
        self.MAX_THRUST = self.T2W*self.W
        self.MAX_RPM = np.sqrt(self.MAX_THRUST/(4*self.KF))
        
        self.MAX_TORQUE_XY = self.ARM*self.KF*self.MAX_RPM**2
        self.MAX_TORQUE_Z = 2*self.KM*self.MAX_RPM**2
        
        
        self.KP_Force = np.array([.4, .4, 1.25])
        self.KI_Force = np.array([.05, .05, .05])
        self.KD_Force = np.array([.2, .2, .5])
        self.KP_Torque = np.array([50000., 50000., 30000.])
        self.KI_Torque = np.array([.0, .0, 500.])
        self.KD_Torque = np.array([20000., 20000., 12000.])
        
        self.MAX_ROLL_PITCH = np.pi/6
        
        self.integration_error_pos = np.zeros(3)
        self.integration_error_rot = np.zeros(3)

        self.last_rpy = np.zeros(3)
        
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.MIXER_MATRIX = np.array([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ])
        
    def calculate_rpm(self, pos, rot, desired_pos, desired_rot, vel):
        pos = np.array(pos)
        rot = np.array(rot)
        desired_pos = np.array(desired_pos)
        desired_rot = np.array(desired_rot)
        vel = np.array(vel)
        
        thrust, computed_target_rpy = self.PositionalControl(pos,rot, desired_pos, desired_rot, vel)
        computed_target_rpy[:2] = np.clip(computed_target_rpy[:2], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
        # print(computed_target_rpy)
        rpm = self.AngularControl(thrust, rot, computed_target_rpy)  
        # rpm = self.AngularControl(thrust, rot, np.array([0,-0.3,0]))    
        return rpm
    
    def PositionalControl(self,pos, rot, desired_pos, desired_rot, vel):
        
        cur_rotation = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rot))).reshape(3, 3)
        pos_e = desired_pos - pos
        vel_e = - vel
        self.integration_error_pos += pos_e*self.DT
        
        self.integration_error_pos = np.clip(self.integration_error_pos, -2., 2.)
        self.integration_error_pos[2] = np.clip(self.integration_error_pos[2], -0.15, .15)
        
        #### PID target thrust #####################################
        
        target_thrust = np.multiply(self.KP_Force, pos_e) \
                        + np.multiply(self.KI_Force, self.integration_error_pos) \
                        + np.multiply(self.KD_Force, vel_e) + np.array([0, 0, self.W])
                        
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(desired_rot[2]), math.sin(desired_rot[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler
    
    def AngularControl(self,thrust, rot, target_euler):
        
        cur_rotation = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rot))).reshape(3, 3)
        cur_rpy = rot
        
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = - (cur_rpy - self.last_rpy)/self.DT
        self.last_rpy = cur_rpy
        self.integration_error_rot = self.integration_error_rot - rot_e*self.DT
        
        self.integration_error_rot = np.clip(self.integration_error_rot, -1500., 1500.)
        self.integration_error_rot[0:2] = np.clip(self.integration_error_rot[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.KP_Torque, rot_e) \
                         + np.multiply(self.KD_Torque, rpy_rates_e) \
                         + np.multiply(self.KI_Torque, self.integration_error_rot)
                         
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    def reset(self):
        self.integration_pos = np.zeros(3)
        self.integration_rot = np.zeros(3)
        self.error_pos = np.zeros(3)
        self.error_rot = np.zeros(3)

        self.last_rpy = np.zeros(3)
        
if __name__ == "__main__":
    env = env.QuadEnv(REAL_TIME=False)
    agent = PIDAgent()
    
    obs = env._getObservation()
    while True:
        
        action = agent.calculate_rpm(obs.P, obs.O, [1,3,5], [0,0,np.pi/3], obs.V)
        obs, reward, done, info = env.step(np.array(action), rpm= True)

        # print(action)
        if(done): break
        
    env.visualize_logs()
    env.save_logs()
    env.close()