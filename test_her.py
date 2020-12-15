import numpy as np, tensorflow as tf
import gym, gym_pomdps
from stable_baselines import HER, DQN

model_class = DQN
env = gym.make('MDP-hallwayonehotmdp-dense-episodic-v0', step_cap=50)
model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy='future',
            seed=0, tensorboard_log='./test_logs', verbose=1)

model.learn(200000)

obs = env.reset()
print(np.where(obs['observation'] == 1))
action, _ = model.predict(obs)
print(action)
obs, reward, done, _  = env.step(action)


