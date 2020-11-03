
import argparse
import tensorflow as tf
import numpy as np

from stable_baselines import HER, DQN
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
#from stable_baselines.common.callbacks import BaseCallbacks
import gym, gym_pomdps


def main(args):
    model_class = DQN
    goal_selection_strategy = 'future'
    if args.env == 'hallway':
        env = gym.make('MDP-hallwaymdp-episodic-v0')
        model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                    tensorboard_log='./logs/hallway/', verbose=1)
    elif args.env == 'mit':
        env = gym.make('MDP-mitmdp-episodic-v0')
        model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                    tensorboard_log='./logs/mit/', verbose=1)
    elif args.env == 'cheese':
        env = gym.make('MDP-cheesemdp-episodic-v0')
        model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                tensorboard_log='./logs/cheese/', verbose=1)
    else:
        raise NotImplementedError('Environment not yet implemented. Current environment are [\'cheese\', \'mit\', and \'hallway\']')
    model.learn(200000)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env')

    main(parser.parse_args())
