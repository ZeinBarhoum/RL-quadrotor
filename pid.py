from Agents.pid import PIDAgent
from Env.env import QuadEnv
import numpy as np

if __name__ == "__main__":
    env = QuadEnv(REAL_TIME=False)
    agent = PIDAgent()
    
    obs = env._getObservation()
    while True:
        
        action = agent.calculate_rpm(obs[0:3], obs[3:6], [0.5,0,2], [0,0,np.pi/3], obs[6:9])
        obs, reward, done, info = env.step(np.array(action), rpm= True)

        # print(action)
        if(done): break
        
    env.visualize_logs()
    env.save_logs()
    env.close()