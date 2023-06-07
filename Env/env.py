"""Environment Class for Quadrotor Simulation"""

import pybullet as p
import time
import pybullet_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import namedtuple

class QuadEnv:
    def __init__(self, EPS_TIME= 10, LOG_TIME= 0.1, REAL_TIME=False):
        
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
        self.state_tuple = namedtuple('state_tuple', ['P', 'O', 'V', 'W', 'A'])
        
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()
        
        self.DesiredPos = np.array([0,0,1])

        self.current_step = 0
        
        self.logtime = []
        self.logstates = []
        self.logactions = []
        self.logrewards = []
        
    def reset(self):
        self.last_action = np.array([0,0,0,0])
        
        self.current_step = 0
        
        p.resetSimulation()
        p.setGravity(0,0,-self.G)
        
        self.planeId = p.loadURDF("plane.urdf")
        
        startPos = [0,0,1]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.droneId = p.loadURDF("Env/cf2x.urdf",startPos, startOrientation)
    
    def step(self, action: np.ndarray, rpm= False):
        if(rpm):
            F, T = self._calculate_F_T(action)
        else:
            F, T = self._calculate_F_T(self._calculate_rpm(action))
        self._apply_F_T(F, T)
        p.stepSimulation()
        
        obs = self._getObservation()
        reward = self._calculate_reward(obs)
        done = self._calculate_done(obs)
        self.last_action = action
        if self.REAL_TIME:
            time.sleep(self.DT)
        if self.current_step%self.LOG_STEPS == 0:
            self.logtime.append(self.current_step*self.DT)
            self.logstates.append(obs)
            self.logactions.append(action)
            self.logrewards.append(reward)
        self.current_step += 1
        return obs, reward, done, {}    
    
    def get_logs_raw(self):
        return self.logstates, self.logactions, self.logrewards
    
    def get_logs_df(self):
        x = [state.P[0] for state in self.logstates]
        y = [state.P[1] for state in self.logstates]
        z = [state.P[2] for state in self.logstates]
        
        vx = [state.V[0] for state in self.logstates]
        vy = [state.V[1] for state in self.logstates]
        vz = [state.V[2] for state in self.logstates]
        
        phi = [state.O[0] for state in self.logstates]
        theta = [state.O[1] for state in self.logstates]
        psi = [state.O[2] for state in self.logstates] 
        
        wx = [state.W[0] for state in self.logstates]
        wy = [state.W[1] for state in self.logstates]
        wz = [state.W[2] for state in self.logstates]
        
        
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
    def step_range(self, t):
        return range(int(t/self.DT))
    
    def _getObservation(self):
        dronePos, droneQuat = p.getBasePositionAndOrientation(self.droneId)
        droneEuler = p.getEulerFromQuaternion(droneQuat)
        droneV, droneW = p.getBaseVelocity(self.droneId)
        return self.state_tuple(dronePos, droneEuler, droneV, droneW, self.last_action)
    
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
        Pos = Obs.P
        return -np.linalg.norm(Pos-self.DesiredPos)**2
    
    def _calculate_done(self, Obs):
        t = self.current_step*self.DT
        if(t > self.EPS_TIME): return True
        return False
    
    

if __name__ == "__main__":
    env = QuadEnv(REAL_TIME=False)
    obs = env._getObservation()
    while True:
        obs, reward, done, info = env.step(np.array([0,0,0,0]))
        if(done): break
        
    env.visualize_logs()
    env.save_logs()
    env.close()

    