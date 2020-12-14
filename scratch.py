
import argparse
import tensorflow as tf
import numpy as np

from stable_baselines import HER, DQN
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
import gym
import gym_pomdps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='Environment you want to process')
    parser.add_argument('--goal-selection-strategy', type=str, default='future',
                        help='Goal selection strategy (choose \'future\' or \'final\')')
    parser.add_argument('--step-cap', default=np.inf,
                        help='Number of timesteps the agent gets to solve the environment')
    parser.add_argument('--logdir', type=str, default='./logs',
                        help='Where to store logs')
    parser.add_argument('--seed', type=int,
                        help='Random seed')
    parser.add_argument('--reward-density', type=str, default='dense',
                        help='Whether to use sparse or dense rewards (Options: \'sparse\' or \'dense\'')
    args = parser.parse_args()

    model_class = DQN
    env = gym.make('MDP-' + args.env + 'mdp-' + args.reward_density + '-episodic-v0', step_cap=args.step_cap)
    model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=args.goal_selection_strategy,
                seed=args.seed, tensorboard_log=args.logdir, verbose=1)
    model.learn(200000)


if __name__ == '__main__':
    main()
