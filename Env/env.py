"""Environment Class for Quadrotor Simulation"""

import pybullet as p
import time
import pybullet_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import namedtuple
import gym
from stable_baselines3.common.env_checker import check_env

class QuadEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, EPS_TIME= 5, LOG_TIME= 0.1, REAL_TIME=False, GUI= True, FLOOR= True):
        
        self.FLOOR = FLOOR
        self.REAL_TIME = bool(REAL_TIME)
        self.DT = 1./240.
        self.G = 9.81
        self.EPS_TIME = EPS_TIME
        self.LOG_TIME = LOG_TIME
        
        self.LOG_STEPS = LOG_TIME//self.DT
        self.KF = 3.16e-10
        self.KM = 7.94e-12
        self.M = 0.027
        self.W = self.M*self.G
        self.HOVER_RPM = np.sqrt(self.W/(4*self.KF))
        # self.state_tuple = namedtuple('state_tuple', ['P', 'O', 'V', 'W', 'A'])
        if(GUI):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        
        self.DesiredPos = np.array([0,0,2])
        self.StartPos = np.array([0,0,0.5])

        self.current_step = 0
        
        self.logtime = []
        self.logstates = []
        self.logstates_normalized = []
        self.logactions = []
        self.logrewards = []
        
        self.MAX_VXY = 5
        self.MAX_VZ = 2
        self.MAX_XY = self.MAX_VXY*self.EPS_TIME
        self.MAX_Z = self.MAX_VZ*self.EPS_TIME
        
        self.MAX_ROLL_PITCH = np.pi
        
        self.action_space = gym.spaces.Box(low=-1*np.ones(4), high=np.ones(4), dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(low=np.array([-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1]), 
                                                high= np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1]), 
                                                dtype=np.float32)
        
        self.reset()
        
    def set_desired_pos(self, desired_pos):
        self.DesiredPos = desired_pos
        
    def reset(self):
        self.last_action = np.array([0,0,0,0])
        
        self.current_step = 0
        
        p.resetSimulation()
        p.setGravity(0,0,-self.G)
        if self.FLOOR:
            self.planeId = p.loadURDF("plane.urdf")
        
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.droneId = p.loadURDF("Env/cf2x.urdf",self.StartPos, startOrientation)
        
        self.done_premature = False

        # print(self.observation_space, '\n', self._normalize_Observation(self._getObservation()))
        
        return self._normalize_Observation(self._getObservation())
    
    def step(self, action: np.ndarray, rpm= False):
        if(rpm):
            F, T = self._calculate_F_T(action)
        else:
            F, T = self._calculate_F_T(self._calculate_rpm(action))
        self._apply_F_T(F, T)
        p.stepSimulation()
        
        obs = self._getObservation()
        obs_normalized = self._normalize_Observation(obs)
        done = self._calculate_done(obs)
        reward = self._calculate_reward(obs)
        self.last_action = action
        if self.REAL_TIME:
            time.sleep(self.DT)
        if self.current_step%self.LOG_STEPS == 0:
            self.logtime.append(self.current_step*self.DT)
            self.logstates.append(obs)
            self.logactions.append(action)
            self.logrewards.append(reward)
            self.logstates_normalized.append(obs_normalized)
            
        self.current_step += 1
        
        self.done_premature = False
        if(rpm):
            return obs, reward, done, {}
        
        return obs_normalized, reward, done, {}    
    
    def get_logs_raw(self):
        return self.logstates, self.logactions, self.logrewards
    
    def get_logs_df(self):
        x = [state[0] for state in self.logstates]
        y = [state[1] for state in self.logstates]
        z = [state[2] for state in self.logstates]
        
        vx = [state[6] for state in self.logstates]
        vy = [state[7] for state in self.logstates]
        vz = [state[8] for state in self.logstates]
        
        phi = [state[3] for state in self.logstates]
        theta = [state[4] for state in self.logstates]
        psi = [state[5] for state in self.logstates] 
        
        wx = [state[9] for state in self.logstates]
        wy = [state[10] for state in self.logstates]
        wz = [state[11] for state in self.logstates]
        
        
        a1 = [action[0] for action in self.logactions]
        a2 = [action[1] for action in self.logactions]
        a3 = [action[2] for action in self.logactions]
        a4 = [action[3] for action in self.logactions]
        
        df = pd.DataFrame(index= self.logtime)
        df['x'] = x
        df['y'] = y
        df['z'] = z
        df['vx'] = vx
        df['vy'] = vy
        df['vz'] = vz
        df['phi'] = phi
        df['theta'] = theta
        df['psi'] = psi
        df['wx'] = wx
        df['wy'] = wy
        df['wz'] = wz
        df['a1'] = a1
        df['a2'] = a2
        df['a3'] = a3
        df['a4'] = a4
        df['r'] = self.logrewards
        
        return df
    def save_logs(self):
        df = self.get_logs_df()
        df.to_csv('logs.csv')
        
    def visualize_logs(self):
        df = self.get_logs_df()
        plt.figure(figsize=(20,10))
        plt.subplot(2,2,1)
        plt.plot(df.index, df.x, label='x')   
        plt.plot(df.index, df.y, label='y') 
        plt.plot(df.index, df.z, label='z')  
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Position')
        #repeat for orientation
        plt.subplot(2,2,2)
        plt.plot(df.index, df.phi, label='phi')   
        plt.plot(df.index, df.theta, label='theta') 
        plt.plot(df.index, df.psi, label='psi')  
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Orientation')
        plt.title('Orientation')
        #repeat for velocity
        plt.subplot(2,2,3)
        plt.plot(df.index, df.vx, label='vx')   
        plt.plot(df.index, df.vy, label='vy') 
        plt.plot(df.index, df.vz, label='vz')  
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Velocity')
        #repeat for angular velocity
        plt.subplot(2,2,4)
        plt.plot(df.index, df.wx, label='wx')   
        plt.plot(df.index, df.wy, label='wy') 
        plt.plot(df.index, df.wz, label='wz')  
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Angular Velocity')
        plt.title('Angular Velocity')
        plt.show()
        
    def close(self):
        p.disconnect()
        
    def render(self, mode="human"):
        print('Rendering...')
        print(f'Step: {self.current_step}')
        print(f'Time: {self.current_step*self.DT}')
        print(f'Current State: {self._normalize_Observation(self._getObservation())}')
        
    def step_range(self, t):
        return range(int(t/self.DT))
    
    def _getObservation(self):
        dronePos, droneQuat = p.getBasePositionAndOrientation(self.droneId)
        droneEuler = p.getEulerFromQuaternion(droneQuat)
        droneV, droneW = p.getBaseVelocity(self.droneId)
        Obs= np.array([dronePos, droneEuler, droneV, droneW]).reshape(12,).astype(np.float32)
        return Obs
    
    def _normalize_Observation(self, Obs):
        
        PosXY = np.clip(Obs[0:2], -self.MAX_XY, self.MAX_XY)/self.MAX_XY
        PosZ = np.clip(Obs[2], 0, self.MAX_Z)/self.MAX_Z

        OrRP = np.clip(Obs[3:5], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)/self.MAX_ROLL_PITCH
        OrY = Obs[5]/np.pi
        
        VXY = np.clip(Obs[6:8], -self.MAX_VXY, self.MAX_VXY)/self.MAX_VXY
        VZ = np.clip(Obs[8], -self.MAX_VZ, self.MAX_VZ)/self.MAX_VZ
        
        W = Obs[9:12]
        if(any(W)):
            W = W/np.linalg.norm(W)
        
        return np.hstack([PosXY,PosZ, OrRP, OrY, VXY, VZ, W]).flatten().reshape(12,).astype(np.float32)

        
    
    def _calculate_F_T(self, w_rpm):
        w_rpm = np.array(w_rpm)
        F = np.array(w_rpm**2)*self.KF
        T = np.array(w_rpm**2)*self.KM
        Tz = (-T[0] + T[1] - T[2] + T[3])
        return F, np.array([0,0,Tz])
    
    def _apply_F_T(self, F, T):
        #applying forces on the propellers links in the body frame upwards
        for i in range(4):
            p.applyExternalForce(self.droneId, i, forceObj= [0,0,F[i]], posObj= [0,0,0], flags=p.LINK_FRAME)
            
        # applying Tz on the center of mass, the only one that depend on the drag and isn't simulated by the forces before
        p.applyExternalTorque(self.droneId, 4, torqueObj= T, flags=p.LINK_FRAME)
        
    def _calculate_rpm(self, action: np.ndarray):
        return self.HOVER_RPM*(1+0.05*action)
    
    def _calculate_reward(self, Obs):
        if(self.done_premature):
            return -1000
        
        Pos = Obs[0:3]
        return -np.linalg.norm(Pos-self.DesiredPos)**2
    
    def _calculate_done(self, Obs):
        t = self.current_step*self.DT
        if(t > self.EPS_TIME): 
            self.done_premature = False
            return True
        
        Pos = Obs[0:3]
        if(abs(Pos[0]-self.DesiredPos[0])>1):
            self.done_premature = True 
            return True
        if(abs(Pos[1]-self.DesiredPos[1])>1): 
            self.done_premature = True 
            return True
        
        Or = Obs[3:6]
        if (abs(Or[0])>self.MAX_ROLL_PITCH):
            self.done_premature = True 
            return True
        if(abs(Or[1])>self.MAX_ROLL_PITCH): 
            self.done_premature = True 
            return True
        
        return False
    
    
def test_env():
    env = QuadEnv(REAL_TIME=True)
    obs = env._getObservation()
    while True:
        obs, reward, done, info = env.step(np.array([0,0,0,0]))
        if(done): break
        
    env.visualize_logs()
    env.save_logs()
    env.close()
def check_gym_compatibility():
    env = QuadEnv(REAL_TIME=True)
    check_env(env)
    
if __name__ == "__main__":
    test_env()

    