import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from rl_parsers.pomdp import parse as pomdp_parse

__all__ = ['OneHotConcatPOMDP', 'CheeseOneHotConcatPOMDP', 'HallwayOneHotConcatPOMDP']

ELEMENTS = ['observation', 'achieved_goal', 'desired_goal']
#update to only have one desired_goal
#ELEMENTS = ['observation', 'achieved_goal']

class OneHotConcatPOMDP(gym.GoalEnv):  # pylint: disable=abstract-method
    """Environment specified by POMDP file."""

    def __init__(self, text, *, seed=None, potential_goals=None, start=None,
                 start_to_obs=None):
        # print('text is {}'.format(text))
        model = pomdp_parse(text)
        self.seed(seed)

        if model.values == 'cost':
            raise ValueError('Unsupported `cost` values.')

        self.model = model
        self.discount = model.discount
        # Number of observations to concat
        self.concat_length = 4
        self.state_space = spaces.Discrete(len(model.states))
        self.action_space = spaces.Discrete(len(model.actions))
        self.obs_space = spaces.Discrete(len(model.observations))
        self.observation_space = None
        self.reward_range = model.R.min(), model.R.max()

        self.rewards_dict = {r: i for i, r in enumerate(np.unique(model.R))}

        if start:
            self.start = start
        else:
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

        # NOTE currently elsewhere
        # self.TO = np.expand_dims(self.T, axis=-1) * self.O
        # self.EO = self.TO.sum(-2)
        # self.ER = (self.TO * self.R).sum((-2, -1))

        # KEVIN ADDITIONS
        # self.state = -1
        self.state = np.zeros(len(model.states), dtype=np.int)
        self.obs = np.zeros(len(model.states), dtype=np.int)

        self.goal = None
        if potential_goals:
            self.potential_goals = potential_goals
        else:
            # self.potential_goals = list(range(len(model.states)))
            self.potential_goals = list(range(len(model.observations)))
        if start_to_obs:
            self.start_to_obs = start_to_obs
        else:
            self.start_to_obs = np.identity(len(model.observations))
        self.history_buffer = None

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.MultiBinary(len(self.model.observations) * self.concat_length),
            achieved_goal=spaces.MultiBinary(len(self.model.observations) * self.concat_length),
            observation=spaces.MultiBinary(len(self.model.observations) * self.concat_length)
        ))
        # Create a history of concat_length observations (times 3 since we have 3 components of our observation)
        self.history_buffer = dict()
        for element in ['observation', 'achieved_goal', 'desired_goal']:
            self.history_buffer[element] = np.zeros(len(self.model.observations) * self.concat_length, dtype=np.int)



    def _get_obs(self):
        obs = self.get_obs()
        achieved_goal = self.get_obs()
        desired_goal = self.get_goal()
        # return {
        #    'observation': obs,
        #    'achieved_goal': achieved_goal,
        #    'desired_goal': desired_goal
        # We're gonna go with obs, desired, achieved for now
        # Order doesn't matter, it's just a dict
        #    'observation': obs,
        #    'desired_goal': desired_goal,
        #    'achieved_goal': achieved_goal
        # }
        return self.update_concatenated_observation(obs, achieved_goal, desired_goal)

    def _sample_goal(self):
        # goal = np.zeros(len(self.model.states), dtype=np.int)
        # goal[np.random.choice(self.potential_goals)] = 1
        goal = np.zeros(len(self.model.observations), dtype=np.int)
        goal[np.random.choice(self.potential_goals)] = 1
        return goal

    def get_starting_obs(self, state):
        obs = self.start_to_obs[state.argmax()]
        # obs = np.zeros(len(self.model.observations), dtype=np.int)
        # obs[self.start_to_obs[state.argmax()]] = 1
        return obs

    def update_concatenated_observation(self, obs, achieved_goal, desired_goal):
        # Imagine we have [obs1, ag1, dg1, obs2, ag2, dg2, ..., obs4, ag4, dg4]
        # We want to push that to be [obs0, ag0, dg0, ..., obs3, ag3, dg3]
        # We know: len(self.history_buffer) / self.concat_length / 3 is the length of each piece
        # Although, right now we'll actually have 3 separate parts:
        # [obs1, obs2, obs3, obs4], [ag1, ag2, ag3, ag4], [dg1, dg2, dg3, dg4]
        for element_name, element in zip(ELEMENTS, [obs, achieved_goal, desired_goal]):
        #for element_name, element in zip(ELEMENTS, [obs, achieved_goal]):
            # UPDATE: now we only use 1 desired_goal, instead of having the same one 4 times.
            #if element_name == 'desired_goal':
                #self.history_buffer[element_name] = desired_goal
            #else:
            self.history_buffer[element_name] = np.append(element,
                                                     self.history_buffer[element_name][:len(self.history_buffer[element_name]) -
                                                                               self.obs_space.n])
        #self.history_buffer['desired_goal'] = desired_goal
        return self.history_buffer

    def get_concatenated_observation(self):
        return self.history_buffer

    def get_obs(self):
        return self.obs

    def get_state(self):
        return self.state

    def get_goal(self):
        return self.goal

    def seed(self, seed):  # pylint: disable=signature-differs
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    # Kevin changed
    def reset(self):
        self.state = self.reset_functional()
        self.goal = self._sample_goal().copy()
        self.obs = self.get_starting_obs(self.state)
        obs = achieved_goal = self.obs
        desired_goal = self.goal
        # obs = {
        #    'observation': self.get_obs(),
        #    'achieved_goal': self.get_state(),
        #    'desired_goal': self.get_goal()
        # removing state from obs.
        #    'observation': self.get_obs(),
        #    'achieved_goal': self.get_obs(),
        #    'desired_goal': self.get_goal()
        # }
        for element_name, element in zip(ELEMENTS, [obs, achieved_goal, desired_goal]):
        #for element_name, element in zip(ELEMENTS, [obs, achieved_goal]):
            self.history_buffer[element_name][:] = 0
            self.history_buffer[element_name][:self.obs_space.n] = element
        #self.history_buffer['desired_goal'] = desired_goal
        return self.history_buffer

    def step(self, action):
        self.state, self.obs, *ret = self.step_functional(self.state, action)
        obs = self._get_obs()
        return (obs, *ret)

    # Returns a random starting state in one-hot form
    def reset_functional(self):
        # return self.np_random.multinomial(1, self.start).argmax().item()
        return self.np_random.multinomial(1, self.start)

    def step_functional(self, state, action):
        if (np.array_equal(state, np.zeros(len(self.model.states), dtype=np.int))) != (action == -1):
            # if (state == -1) != (action == -1):
            raise ValueError(f'Invalid state-action pair ({state}, {action}).')

        if (np.array_equal(state, np.zeros(len(self.model.states), dtype=np.int))) and action == -1:
            # if state == -1 and action == -1:
            return np.zeros(len(self.model.states), dtype=np.int), np.zeros(len(self.mode.observations),
                                                                            dtype=np.int), -1, True, None
            # return -1, -1, 0.0, True, None

        # assert 0 <= state < self.state_space.n
        assert len(state) == len(self.model.states)
        assert 0 <= action < self.action_space.n

        state_next = (
            #    self.np_random.multinomial(1, self.T[state, action]).argmax().item()
            self.np_random.multinomial(1, self.T[state.argmax(), action])
        )
        obs = (
            self.np_random.multinomial(1, self.O[state.argmax(), action, state_next.argmax()])
            # .argmax()
            # .item()
        )
        # NOTE below is the same but unified in single sampling op; requires TO
        # state_next, obs = divmod(
        #     self.np_random.multinomial(
        #         1, self.TO[state, action].ravel()).argmax().item(),
        #     self.observation_space.n,
        # )

        # done = self._is_success(state_next, self.goal)
        done = self._is_success(obs, self.goal)
        info = {
            'is_success': 1 if done else 0,
        }

        # reward = self.R[state, action, state_next, obs].item()
        # reward = self.compute_reward(state_next, self.goal, info)
        reward = self.compute_reward(obs, self.goal, info)

        # done = self.D[state, action].item() if self.episodic else False
        if done:
            # state_next = -1
            state_next = np.zeros(len(self.model.states), dtype=np.int)


        # reward_cat = self.rewards_dict[reward]
        # info = dict(reward_cat=reward_cat)

        return state_next, obs, reward, done, info

    def compute_reward(self, achieved_goal: np.array, desired_goal: np.array, info: dict) -> int:
        # Negative Living Reward
        if np.array_equal(achieved_goal[:self.obs_space.n], desired_goal):
            return 0
        else:
            return -1
        # Positive Success Reward
        # if np.array_equal(achieved_goal, desired_goal):
        #     return 1
        # else:
        #     return 0

    def _is_success(self, achieved_goal: np.array, desired_goal: np.array) -> bool:
        if np.array_equal(achieved_goal[:self.obs_space.n], desired_goal):
            return True
        else:
            return False


class CheeseOneHotConcatPOMDP(OneHotConcatPOMDP):
    def __init__(self, text, *, seed=None):
        # Goals must be observations
        potential_goals = [5, 6, 7]
        start = None
        # start = [1/3, 0, 1/3, 0, 1/3, 0, 0, 0, 0, 0, 0]
        start_to_obs = {
            0: [1, 0, 0, 0, 0, 0, 0, 0],
            2: [0, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 1, 0, 0, 0, 0]
        }
        super().__init__(text, seed=seed, potential_goals=potential_goals,
                         start=start, start_to_obs=start_to_obs)

        #self.observation_space = spaces.Dict(dict(
        #    desired_goal=spaces.MultiBinary(len(self.model.observations) * self.concat_length),
        #    achieved_goal=spaces.MultiBinary(len(self.model.observations) * self.concat_length),
        #    observation=spaces.MultiBinary(len(self.model.observations) * self.concat_length)
        #))
        # Create a history of concat_length observations (times 3 since we have 3 components of our observation)
        #self.history_buffer = dict()
        #for element in ['observation', 'achieved_goal', 'desired_goal']:
        #    self.history_buffer[element] = np.zeros(len(self.model.observations) * self.concat_length, dtype=np.int)


class HallwayOneHotConcatPOMDP(OneHotConcatPOMDP):
    def __init__(self, text, *, seed=None):
        start_states = [0, 1, 2, 3, 40, 41, 42, 43]
        start = np.zeros(60, dtype=np.float32)
        start[start_states] = 1 / len(start_states)
        start_to_obs = {
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0],
            1: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0],
            2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0],
            3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0],
            40: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0],
            41: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0],
            42: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0],
            43: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0]
        }
        # potential_goals = list(range(44, 60))
        potential_goals = list(range(20, 36))
        super().__init__(text, seed=seed, potential_goals=potential_goals,
                         start_to_obs=start_to_obs)

        #self.observation_space = spaces.Dict(dict(
        #    desired_goal=spaces.MultiBinary(len(self.model.observations) * self.concat_length),
        #    achieved_goal=spaces.MultiBinary(len(self.model.observations) * self.concat_length),
        #    observation=spaces.MultiBinary(len(self.model.observations) * self.concat_length)
        #))
        # Create a history of concat_length observations (times 3 since we have 3 components of our observation)
        #self.history_buffer = dict()
        #for element in ['observation', 'achieved_goal', 'desired_goal']:
            # UPDATE: use only one desired_goal
            #if element == 'desired_goal':
            #    self.history_buffer[element] = np.zeros(len(self.model.observations), dtype=np.int)
            #else:
        #    self.history_buffer[element] = np.zeros(len(self.model.observations) * self.concat_length, dtype=np.int)
