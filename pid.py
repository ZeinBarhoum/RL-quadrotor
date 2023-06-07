from Agents.pid import PIDAgent
from Env.env import QuadEnv
import numpy as np

if __name__ == "__main__":
    env = QuadEnv(REAL_TIME=False)
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