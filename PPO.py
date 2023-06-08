from Env.env import QuadEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
import torch
import numpy as np

if __name__ == "__main__":
    env_no_gui_train = QuadEnv(REAL_TIME= False, GUI= False, FLOOR = True, ONE_COMMAND= True, MODE= 'TakeOFF')
    # env_no_gui_eval = QuadEnv(REAL_TIME= False, GUI= False, FLOOR = False)

    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-0,
                                                     verbose=1
                                                     )
    eval_callback = EvalCallback(env_no_gui_train, #env_no_gui_eval
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 eval_freq=int(2000),
                                 deterministic=True,
                                 render=False
                                 )
    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                           net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
                           )
    
    model = PPO(a2cppoMlpPolicy, env_no_gui_train, verbose= 1, policy_kwargs=onpolicy_kwargs,).learn(total_timesteps=100000, callback= eval_callback, log_interval= 100)
    
    model.save("PPO_TakeOFF")

    del model # remove to demonstrate saving and loading
    
    env_no_gui_train.close()
    # env_no_gui_eval.close()
    
    model = PPO.load("PPO_TakeOFF")
    
    env_gui = QuadEnv(REAL_TIME= True, ONE_COMMAND= True, MODE= 'TakeOFF')
    
    obs = env_gui.reset()
    
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env_gui.step(action)
        if(dones):
            break
        
    env_gui.visualize_logs()
    env_gui.save_logs()
    env_gui.close()
