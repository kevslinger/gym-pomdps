import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


from rl_parsers.mdp import parse

__all__ = ['MDP', 'HallwayMDP', 'MitMDP', 'BigcheeseMDP', 'CheeseMDP', 'CitMDP']

# NOTE: Each domain must extend this
class MDP(gym.GoalEnv):  # pylint: disable=abstract-method
    """Environment specified by MDP file."""

    def __init__(self, text, *, seed=None, dense_reward=True, step_cap=np.inf, potential_goals=None):
        model = parse(text)
        self.seed(seed)

        if model.values == 'cost':
            raise ValueError('Unsupported `cost` values.')

        self.model = model
        self.discount = model.discount
        self.state_space = spaces.Discrete(len(model.states))
        self.action_space = spaces.Discrete(len(model.actions))
        self.observation_space = None
        self.goal = None
        self.reward_range = model.R.min(), model.R.max()

        self.steps = 0
        self.step_cap = step_cap

        self.rewards_dict = {r: i for i, r in enumerate(np.unique(model.R))}

        if model.start is None:
            self.start = np.full(self.state_space.n, 1 / self.state_space.n)
        else:
            self.start = model.start
        self.T = model.T.transpose(1, 0, 2).copy()
        # if model.flags['O_includes_state']:
        #    self.O = model.O.transpose(1, 0, 2, 3).copy()
        # else:
        #    self.O = np.expand_dims(model.O, axis=0).repeat(
        #        self.state_space.n, axis=0
        #    )
        # print(model.R)
        # print(type(model.R))
        # print(model.R.shape)
        # self.R = model.R.transpose(1, 0, 2, 3).copy()
        self.R = model.R.transpose(1, 0, 2).copy()
        # print(self.R)


        # NOTE currently elsewhere
        # self.TO = np.expand_dims(self.T, axis=-1) * self.O
        # self.EO = self.TO.sum(-2)
        # self.ER = (self.TO * self.R).sum((-2, -1))

        self.dense = dense_reward

        if potential_goals:
            self.potential_goals = potential_goals
        else:
            self.potential_goals = list(range(len(self.model.states)))

        self.state = -1

    def _get_obs(self):
        obs = self.get_state()
        achieved_goal = self.get_state()
        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': self.goal
        }


    def _sample_goal(self):
        raise NotImplementedError

    def get_state(self):
        return self.state

    def seed(self, seed):  # pylint: disable=signature-differs
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.state = self.reset_functional()
        self.steps = 0
        self.goal = self._sample_goal()
        return self._get_obs()

    def step(self, action):
        # self.state, *ret = self.step_functional(self.state, action)
        obs = self._get_obs()
        ret = self.step_functional(self.state, obs, action)
        self.state = ret[0]
        obs = self._get_obs()
        return (obs, *ret[1:])

    def reset_functional(self):
        return self.np_random.multinomial(1, self.start).argmax().item()

    def step_functional(self, state, obs, action):
        if (state == -1) != (action == -1):
            raise ValueError(f'Invalid state-action pair ({state}, {action}).')

        if state == -1 and action == -1:
            return -1, -1, 0.0, True, None

        assert 0 <= state < self.state_space.n
        assert 0 <= action < self.action_space.n

        state_next = (
            self.np_random.multinomial(1, self.T[state, action]).argmax().item()
        )
        # obs = (
        #    self.np_random.multinomial(1, self.O[state, action, state_next])
        #    .argmax()
        #    .item()
        # )
        # NOTE below is the same but unified in single sampling op; requires TO
        # state_next, obs = divmod(
        #     self.np_random.multinomial(
        #         1, self.TO[state, action].ravel()).argmax().item(),
        #     self.observation_space.n,
        # )

        # reward = self.R[state, action, state_next, obs].item()
        # reward = self.R[state, action, state_next].item()
        # reward = self.compute_reward()

        # done = self.D[state, action].item() if self.episodic else False

        info = {
            'is_success' : 1 if obs['achieved_goal'] == self.goal else 0,

        }

        reward = self.compute_reward(state_next, self.goal, info)
        done = self._is_success(state_next, self.goal)
        if done:
            state_next = -1

        self.steps += 1
        if self.steps >= self.step_cap:
            done = True

        # reward_cat = self.rewards_dict[reward]
        # info = dict(reward_cat=reward_cat)

        # return state_next, obs, reward, done, info
        return state_next, reward, done, info

    def compute_reward(self, achieved_goal, goal, info):
        if self.dense:
            if achieved_goal == goal:
                return 0
            else:
                return -1
        else:
            if achieved_goal == goal:
                return 1
            else:
                return 0

    def _is_success(self, achieved_goal, desired_goal):
        if achieved_goal == desired_goal:
            return True
        else:
            return False


class HallwayMDP(MDP):
    def __init__(self, text, *, seed=None, dense_reward=True, step_cap=np.inf):

        super().__init__(text, seed=seed, dense_reward=dense_reward, step_cap=step_cap)
        #self.goal = self._sample_goal()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Discrete(16),  # 44 to 59
            achieved_goal=spaces.Discrete(len(self.model.states)),
            observation=spaces.Discrete(len(self.model.states)),
        ))
        #self.step_cap = 15 #np.inf

    # any state between 44 and 59 can be a goal (4 orientations in one of the 4 boxes.)
    def _sample_goal(self):
        return np.random.randint(44, len(self.model.states)+1)


class MitMDP(MDP):
    def __init__(self, text, *, seed=None, dense_reward=True, step_cap=np.inf):
        super().__init__(text, seed=seed, dense_reward=dense_reward, step_cap=step_cap)
        #self.goal = self._sample_goal()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Discrete(len(self.model.states)),
            achieved_goal=spaces.Discrete(len(self.model.states)),
            observation=spaces.Discrete(len(self.model.states)),
        ))
        #self.step_cap = 50  #np.inf

    # any state can be a goal for now
    def _sample_goal(self):
        return np.random.randint(len(self.model.states)+1)

class BigcheeseMDP(MDP):
    def __init__(self, text, *, seed=None, step_cap=np.inf):
        super().__init__(text, seed=seed, step_cap=step_cap)

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Discrete(5),
            achieved_goal=spaces.Discrete(len(self.model.states)),
            observation=spaces.Discrete(len(self.model.states))  
        ))

    def _sample_goal(self):
        return np.random.choice([28, 29, 30, 31, 32])


class CheeseMDP(MDP):
    def __init__(self, text, *, seed=None, dense_reward=True, step_cap=np.inf):
        super().__init__(text, seed=seed, dense_reward=dense_reward, step_cap=step_cap)

        #self.goal = self._sample_goal()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Discrete(3),
            achieved_goal=spaces.Discrete(len(self.model.states)),
            observation=spaces.Discrete(len(self.model.states)),
        ))
        #self.step_cap = 10  # np.inf

    # only states 9, 10, and 11 can be goals for now
    def _sample_goal(self):
        return np.random.choice([8, 9, 10])


class CitMDP(MDP):
    def __init__(self, text, *, seed=None, dense_reward=True, step_cap=np.inf):
        super().__init__(text, seed=seed, dense_reward=dense_reward, step_cap=step_cap)

        #self.goal = self._sample_goal()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Discrete(len(self.model.states)),
            achieved_goal=spaces.Discrete(len(self.model.states)),
            observation=spaces.Discrete(len(self.model.states)),
        ))

    # Any state can be a goal for now
    def _sample_goal(self):
        return np.random.randint(len(self.model_states)+1)
