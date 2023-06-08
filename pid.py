from Agents.pid import PIDAgent
from Env.env import QuadEnv
import numpy as np

if __name__ == "__main__":
    env = QuadEnv(REAL_TIME=False, MODE= 'TakeOFF', EPS_TIME= 20)
    agent = PIDAgent()
    
    obs = env._getObservation()
    
    rewards = 0
    while True:
        
        action = agent.calculate_rpm(obs[0:3], obs[3:6], env.DesiredPos, [0,0,0], obs[6:9])
        obs, reward, done, info = env.step(np.array(action), rpm= True)
        rewards += reward

        # print(action)
        if(done): break
        
    env.visualize_logs()
    env.save_logs()
    env.close()
    
    print(f'{rewards=}')