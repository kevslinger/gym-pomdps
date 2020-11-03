

import tensorflow as tf
import numpy as np

from stable_baselines import HER, DQN
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
#from stable_baselines.common.callbacks import BaseCallbacks
import gym, gym_pomdps

model_class = DQN
goal_selection_strategy = 'future'
#env = gym.make('MDP-hallwaymdp-episodic-v0')
env = gym.make('MDP-mitmdp-episodic-v0')

model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
            tensorboard_log='./logs/mit/', verbose=1)

model.learn(1000000)