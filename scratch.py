
import argparse
import tensorflow as tf
import numpy as np

from stable_baselines import HER, DQN
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
#from stable_baselines.common.callbacks import BaseCallbacks
import gym, gym_pomdps


def main(args):
    model_class = DQN
    if 'hallway' in args.env:
        env = gym.make('MDP-hallwaymdp-episodic-v0')
        model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=args.goal_selection_strategy,
                    tensorboard_log=args.hallwaylogdir, verbose=1)
        model.learn(200000)
    if 'mit' in args.env:
        env = gym.make('MDP-mitmdp-episodic-v0')
        model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=args.goal_selection_strategy,
                    tensorboard_log=args.logdir, verbose=1)
        model.learn(200000)
    if 'cheese' in args.env:
        env = gym.make('MDP-cheesemdp-episodic-v0')
        model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=args.goal_selection_strategy,
                tensorboard_log=args.logdir, verbose=1)
        model.learn(200000)
    #else:
    #    raise NotImplementedError('Environment not yet implemented. Current environment are [\'cheese\', \'mit\', and \'hallway\']')
    if 'cheeseonehot' in args.env:
        env = gym.make('MDP-cheeseonehotmdp-episodic-v0')
        model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=args.goal_selection_strategy,
                    tensorboard_log=args.logdir, verbose=1)
        model.learn(200000)
    if 'hallwayonehot' in args.env:
        env = gym.make('MDP-hallwayonehotmdp-episodic-v0')
        model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=args.goal_selection_strategy,
                    tensorboard_log=args.logdir, verbose=1)
        model.learn(200000)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', nargs='+', help='Environment you want to process')
    parser.add_argument('--hallway_logdir', type=str, default='./logs/hallway',
                        help='Where to store the logs for the hallway environment')
    parser.add_argument('--mit_logdir', type=str, default='./logs/mit',
                        help='Where to store the logs for the mit environment')
    parser.add_argument('--cheese_logdir', type=str, default='./logs/cheese',
                        help='Where to store the logs for the cheese environment')
    parser.add_argument('--cheeseonehot_logdir', type=str, default='./logs/cheeseonehot',
                        help='Where to store the logs for the cheese one hot environment')
    parser.add_argument('--hallwayonehot_logdir', type=str, default='./logs/hallwayonehot',
                        help='Where to store the logs for the hallway one hot environment.')
    parser.add_argument('--goal_selection_strategy', type=str, default='future',
                        help='Goal selection strategy (choose \'future\' or \'final\')')
    parser.add_argument('--layer-size', type=int, default=64,
                        help='Size of DQN architecture (e.g. [64, 64], [32, 32], ...')


    main(parser.parse_args())
