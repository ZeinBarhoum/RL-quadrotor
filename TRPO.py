from Env.env import QuadEnv
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
import time
import numpy as np
import torch

def train():
    start_time = time.time()
    
    env_no_gui_train = QuadEnv(REAL_TIME= False, GUI= False, FLOOR = True, ONE_COMMAND= True, MODE= 'TakeOFF', EPS_TIME= 20)
    
    train_env = make_vec_env(QuadEnv, 
                             env_kwargs= dict(REAL_TIME= False, GUI= False, FLOOR = True, ONE_COMMAND= True, MODE= 'TakeOFF', EPS_TIME= 20), 
                             n_envs= 4)
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-0,
                                                     verbose=1
                                                     )
    eval_callback = EvalCallback(env_no_gui_train, #env_no_gui_eval
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 eval_freq=int(500),
                                 deterministic=True,
                                 render=False
                                 )
    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        #    net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
                           )
    model = TRPO("MlpPolicy", train_env, verbose= 1, policy_kwargs=onpolicy_kwargs)
    
    model.learn(total_timesteps=1e5, callback= eval_callback)
    
    model.save("TRPO_TakeOFF")

    del model
    
    env_no_gui_train.close()
    
    end_time = time.time()
    print(f'Total Training Time for {1e5} steps is {end_time - start_time}')

def test_n_times(n):
    model = TRPO.load("TRPO_TakeOFF")
    
    env = QuadEnv(REAL_TIME= False, GUI= False, ONE_COMMAND= True, MODE= 'TakeOFF', EPS_TIME= 20)
    reward_list = []
    for i in range(n):
        obs = env.reset()
        rewards = 0
        while True:
            action, _states = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            rewards += reward
            if(dones):
                break
        reward_list.append(rewards)
    env.close()
        
    print(f'{np.mean(reward_list)=}')
    
def test():
    model = TRPO.load("TRPO_TakeOFF")
    
    env_gui = QuadEnv(REAL_TIME= False, ONE_COMMAND= True, MODE= 'TakeOFF', EPS_TIME= 20)
    obs = env_gui.reset()
    rewards = 0
    while True:
        action, _states = model.predict(obs)
        obs, reward, dones, info = env_gui.step(action)
        rewards += reward
        if(dones):
            break
    env_gui.visualize_logs()
    env_gui.save_logs()
    env_gui.close()
    
    print(f'{rewards=}')
if __name__ == "__main__":
    # train()
    test()
    # test_n_times(10)
    
    
    
