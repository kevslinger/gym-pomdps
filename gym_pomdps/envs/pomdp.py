import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from rl_parsers.pomdp import parse as pomdp_parse
from rl_parsers.mdp import parse as mdp_parse

__all__ = ['POMDP', 'MDP']


class POMDP(gym.Env):  # pylint: disable=abstract-method
    """Environment specified by POMDP file."""

    def __init__(self, text, *, episodic, seed=None):
        print('text is {}'.format(text))
        model = pomdp_parse(text)
        self.episodic = episodic
        self.seed(seed)

        if model.values == 'cost':
            raise ValueError('Unsupported `cost` values.')

        self.model = model
        self.discount = model.discount
        self.state_space = spaces.Discrete(len(model.states))
        self.action_space = spaces.Discrete(len(model.actions))
        self.observation_space = spaces.Discrete(len(model.observations))
        self.reward_range = model.R.min(), model.R.max()

        self.rewards_dict = {r: i for i, r in enumerate(np.unique(model.R))}

        if model.start is None:
            self.start = np.full(self.state_space.n, 1 / self.state_space.n)
        else:
            self.start = model.start
        self.T = model.T.transpose(1, 0, 2).copy()
        if model.flags['O_includes_state']:
            self.O = model.O.transpose(1, 0, 2, 3).copy()
        else:
            self.O = np.expand_dims(model.O, axis=0).repeat(
                self.state_space.n, axis=0
            )
        self.R = model.R.transpose(1, 0, 2, 3).copy()

        if episodic:
            self.D = model.reset.T.copy()  # only if episodic

        # NOTE currently elsewhere
        # self.TO = np.expand_dims(self.T, axis=-1) * self.O
        # self.EO = self.TO.sum(-2)
        # self.ER = (self.TO * self.R).sum((-2, -1))

        self.state = -1

    def seed(self, seed):  # pylint: disable=signature-differs
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.state = self.reset_functional()

    def step(self, action):
        self.state, *ret = self.step_functional(self.state, action)
        return ret

    def reset_functional(self):
        return self.np_random.multinomial(1, self.start).argmax().item()

    def step_functional(self, state, action):
        if (state == -1) != (action == -1):
            raise ValueError(f'Invalid state-action pair ({state}, {action}).')

        if state == -1 and action == -1:
            return -1, -1, 0.0, True, None

        assert 0 <= state < self.state_space.n
        assert 0 <= action < self.action_space.n

        state_next = (
            self.np_random.multinomial(1, self.T[state, action]).argmax().item()
        )
        obs = (
            self.np_random.multinomial(1, self.O[state, action, state_next])
            .argmax()
            .item()
        )
        # NOTE below is the same but unified in single sampling op; requires TO
        # state_next, obs = divmod(
        #     self.np_random.multinomial(
        #         1, self.TO[state, action].ravel()).argmax().item(),
        #     self.observation_space.n,
        # )

        reward = self.R[state, action, state_next, obs].item()

        done = self.D[state, action].item() if self.episodic else False
        if done:
            state_next = -1

        reward_cat = self.rewards_dict[reward]
        info = dict(reward_cat=reward_cat)

        return state_next, obs, reward, done, info


class MDP(gym.Env):  # pylint: disable=abstract-method
    """Environment specified by MDP file."""

    def __init__(self, text, *, episodic, seed=None):
        #print('text is {}'.format(text))
        model = mdp_parse(text)
        self.episodic = episodic
        self.seed(seed)

        if model.values == 'cost':
            raise ValueError('Unsupported `cost` values.')

        self.model = model
        self.discount = model.discount
        self.state_space = spaces.Discrete(len(model.states))
        self.action_space = spaces.Discrete(len(model.actions))
        #self.observation_space = spaces.Discrete(len(model.observations))
        self.reward_range = model.R.min(), model.R.max()

        self.rewards_dict = {r: i for i, r in enumerate(np.unique(model.R))}

        if model.start is None:
            self.start = np.full(self.state_space.n, 1 / self.state_space.n)
        else:
            self.start = model.start
        self.T = model.T.transpose(1, 0, 2).copy()
        #if model.flags['O_includes_state']:
        #    self.O = model.O.transpose(1, 0, 2, 3).copy()
        #else:
        #    self.O = np.expand_dims(model.O, axis=0).repeat(
        #        self.state_space.n, axis=0
        #    )
        print(model.R)
        print(type(model.R))
        print(model.R.shape)
        #self.R = model.R.transpose(1, 0, 2, 3).copy()
        self.R = model.R.transpose(1, 0, 2).copy()
        print(self.R)

        if episodic:
            self.D = model.reset.T.copy()  # only if episodic

        # NOTE currently elsewhere
        # self.TO = np.expand_dims(self.T, axis=-1) * self.O
        # self.EO = self.TO.sum(-2)
        # self.ER = (self.TO * self.R).sum((-2, -1))

        self.state = -1

    def _get_obs(self):
        obs = self.get_state()
        
        
    def get_state(self):
        return self.state
        
    def seed(self, seed):  # pylint: disable=signature-differs
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.state = self.reset_functional()

    def step(self, action):
        #self.state, *ret = self.step_functional(self.state, action)
        ret = self.step_functional(self.state, action)
        self.state = ret[0]
        return ret

    def reset_functional(self):
        return self.np_random.multinomial(1, self.start).argmax().item()

    def step_functional(self, state, action):
        if (state == -1) != (action == -1):
            raise ValueError(f'Invalid state-action pair ({state}, {action}).')

        if state == -1 and action == -1:
            return -1, -1, 0.0, True, None

        assert 0 <= state < self.state_space.n
        assert 0 <= action < self.action_space.n

        state_next = (
            self.np_random.multinomial(1, self.T[state, action]).argmax().item()
        )
        #obs = (
        #    self.np_random.multinomial(1, self.O[state, action, state_next])
        #    .argmax()
        #    .item()
        #)
        # NOTE below is the same but unified in single sampling op; requires TO
        # state_next, obs = divmod(
        #     self.np_random.multinomial(
        #         1, self.TO[state, action].ravel()).argmax().item(),
        #     self.observation_space.n,
        # )

        #reward = self.R[state, action, state_next, obs].item()
        reward = self.R[state, action, state_next].item()
        
        done = self.D[state, action].item() if self.episodic else False
        if done:
            state_next = -1

        reward_cat = self.rewards_dict[reward]
        info = dict(reward_cat=reward_cat)

        #return state_next, obs, reward, done, info
        return (state_next, reward, done, info)
