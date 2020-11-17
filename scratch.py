
import argparse
import tensorflow as tf
import numpy as np

from stable_baselines import HER, DQN
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
#from stable_baselines.common.callbacks import BaseCallbacks
import gym, gym_pomdps


def main(args):
    model_class = DQN
    env = gym.make('MDP-' + args.env + 'mdp-episodic-v0')
    model = HER('MlpPolicy', env, model_class, n_sample_goal=4, goal_selection_strategy=args.goal_selection_strategy,
                tensorboard_log=args.logdir, verbose=1)
    model.learn(200000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='Environment you want to process')
    parser.add_argument('--goal_selection_strategy', type=str, default='future',
                        help='Goal selection strategy (choose \'future\' or \'final\')')
    parser.add_argument('--layer-size', type=int, default=64,
                        help='Size of DQN architecture (e.g. [64, 64], [32, 32], ...')
    parser.add_argument('--logdir', type=str, default='./logs',
                        help='Where to store logs')


    main(parser.parse_args())
