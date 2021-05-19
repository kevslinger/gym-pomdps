import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from rl_parsers.pomdp import parse as pomdp_parse

__all__ = ['OneHotPOMDP', 'CheeseOneHotPOMDP', 'HallwayOneHotPOMDP']


class OneHotPOMDP(gym.Env):  # pylint: disable=abstract-method
    """Environment specified by POMDP file."""

    def __init__(self, text, *, seed=None, goal=None, start=None, start_to_obs=None):
        #print('text is {}'.format(text))
        model = pomdp_parse(text)
        self.seed(seed)

        if model.values == 'cost':
            raise ValueError('Unsupported `cost` values.')

        self.model = model
        self.discount = model.discount
        self.state_space = spaces.Discrete(len(model.states))
        self.action_space = spaces.Discrete(len(model.actions))
        #self.observation_space = spaces.Discrete(len(model.observations))
        self.observation_space = spaces.MultiBinary(len(self.model.observations))
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
        #self.state = -1
        self.state = np.zeros(len(model.states), dtype=np.int)
        self.obs = np.zeros(len(model.states), dtype=np.int)

        if goal is None:
            raise ValueError("POMDP must have at least one goal state")
        else:
            self.goal = goal
        if start_to_obs:
            self.start_to_obs = start_to_obs
        else:
            self.start_to_obs = np.identity(len(model.observations))

    def _sample_goal(self):
        raise NotImplementedError("We do not sample goal in the single goal case")

    def get_starting_obs(self, state):
        obs = self.start_to_obs[state.argmax()]
        #obs = np.zeros(len(self.model.observations), dtype=np.int)
        #obs[self.start_to_obs[state.argmax()]] = 1
        return obs

    def get_obs(self):
        return self.obs
    
    def get_state(self):
        return self.state

    def get_goal(self):
        return self.goal

    def seed(self, seed):  # pylint: disable=signature-differs
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.state = self.reset_functional()
        self.obs = self.get_starting_obs(self.state)
        #obs = {
        #    'observation': self.get_obs(),
        #    'achieved_goal': self.get_state(),
        #    'desired_goal': self.get_goal()
        # removing state from obs.
        #    'observation': self.get_obs(),
        #    'achieved_goal': self.get_obs(),
        #    'desired_goal': self.get_goal()
        #}
        return self.obs.copy()

    def step(self, action):
        self.state, self.obs, *ret = self.step_functional(self.state, action)
        return (self.obs.copy(), *ret)

    # Returns a random starting state in one-hot form
    def reset_functional(self):
        #return self.np_random.multinomial(1, self.start).argmax().item()
        return self.np_random.multinomial(1, self.start)
        
    def step_functional(self, state, action):
        if (np.array_equal(state, np.zeros(len(self.model.states), dtype=np.int))) != (action == -1):
        #if (state == -1) != (action == -1):
            raise ValueError(f'Invalid state-action pair ({state}, {action}).')

        if (np.array_equal(state, np.zeros(len(self.model.states), dtype=np.int))) and action == -1:
        #if state == -1 and action == -1:
            return np.zeros(len(self.model.states), dtype=np.int), np.zeros(len(self.model.observations), dtype=np.int), -1, True, None
            #return -1, -1, 0.0, True, None

        #assert 0 <= state < self.state_space.n
        assert len(state) == len(self.model.states)
        assert 0 <= action < self.action_space.n

        state_next = (
        #    self.np_random.multinomial(1, self.T[state, action]).argmax().item()
            self.np_random.multinomial(1, self.T[state.argmax(), action])
        )
        obs = (
            self.np_random.multinomial(1, self.O[state.argmax(), action, state_next.argmax()])
            #.argmax()
            #.item()
        )
        # NOTE below is the same but unified in single sampling op; requires TO
        # state_next, obs = divmod(
        #     self.np_random.multinomial(
        #         1, self.TO[state, action].ravel()).argmax().item(),
        #     self.observation_space.n,
        # )

        #done = self._is_success(state_next, self.goal)
        done = self._is_success(obs, self.goal)
        info = {
            'is_success': 1 if done else 0,
        }
        
        #reward = self.R[state, action, state_next, obs].item()
        #reward = self.compute_reward(state_next, self.goal, info)
        reward = self.compute_reward(obs, self.goal, info)

        #done = self.D[state, action].item() if self.episodic else False
        if done:
            #state_next = -1
            state_next = np.zeros(len(self.model.states), dtype=np.int)

            
        #reward_cat = self.rewards_dict[reward]
        #info = dict(reward_cat=reward_cat)

        return state_next, obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> int:
        # Negative Living Reward
        if np.array_equal(achieved_goal, desired_goal):
            return 0
        else:
            return -1
        #if np.array_equal(achieved_goal, desired_goal):
        #    return 0
        #else:
        #    return -1
        # Positive Success Reward
        # if np.array_equal(achieved_goal, desired_goal):
        #     return 1
        # else:
        #     return 0

    def _is_success(self, achieved_goal: np.array, desired_goal: np.array) -> bool:
        """Determines whether the agent achieved its goal"""
        if np.array_equal(achieved_goal, desired_goal):
                return True
        else:
            return False
        #if np.array_equal(achieved_goal, desired_goal):
        #    return True
        #else:
        #    return False

class CheeseOneHotPOMDP(OneHotPOMDP):
    def __init__(self, text, *, seed=None):
        # Goals must be observations
        goal = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.int)
        start = None
        #start = [1/3, 0, 1/3, 0, 1/3, 0, 0, 0, 0, 0, 0]
        start_to_obs = {
            0: [1, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 0, 0, 0, 0, 0],
            2: [0, 0, 1, 0, 0, 0, 0],
            3: [0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 1, 0, 0, 0],
            5: [0, 0, 0, 0, 1, 0, 0],
            6: [0, 0, 0, 0, 1, 0, 0],
            7: [0, 0, 0, 0, 1, 0, 0],
            8: [0, 0, 0, 0, 0, 1, 0],
            9: [0, 0, 0, 0, 0, 1, 0],
            10: [0, 0, 0, 0, 0, 0, 1],
        }
        super().__init__(text, seed=seed, goal=goal, start_to_obs=start_to_obs)

        #self.observation_space = spaces.Dict(dict(
        #    desired_goal=spaces.MultiBinary(len(self.model.observations)),
        #    achieved_goal=spaces.MultiBinary(len(self.model.observations)),
        #    observation=spaces.MultiBinary(len(self.model.observations))
        #))




class HallwayOneHotPOMDP(OneHotPOMDP):
    def __init__(self, text, *, seed=None):
        start_to_obs = {
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            40: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            41: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            42: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            43: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        start_to_obs = {
            0: [0.000949, 0.008549, 0.008549, 0.076949, 0.000049, 0.000449, 0.000449, 0.004049, 0.008549, 0.076949,
                0.076949, 0.692550, 0.000449, 0.004049, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            1: [0.000949, 0.008549, 0.008549, 0.076949, 0.008549, 0.076949, 0.076949, 0.692550, 0.000049, 0.000449,
                0.000449, 0.004049, 0.000449, 0.004049, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            2: [0.000949, 0.000049, 0.008549, 0.000449, 0.008549, 0.000449, 0.076949, 0.004049, 0.008549, 0.000449,
                0.076949, 0.004049, 0.076949, 0.004049, 0.692550, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            3: [0.000949, 0.008549, 0.000049, 0.000449, 0.008549, 0.076949, 0.000449, 0.004049, 0.008549, 0.076949,
                0.000449, 0.004049, 0.076949, 0.692550, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            4: [0.009024, 0.000474, 0.081225, 0.004275, 0.000474, 0.000024, 0.004275, 0.000225, 0.081225, 0.004275,
                0.731024, 0.038475, 0.004275, 0.000225, 0.038475, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            5: [0.009024, 0.081225, 0.000474, 0.004275, 0.081225, 0.731024, 0.004275, 0.038475, 0.000474, 0.004275,
                0.000024, 0.000225, 0.004275, 0.038475, 0.000225, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            6: [0.009024, 0.000474, 0.081225, 0.004275, 0.000474, 0.000024, 0.004275, 0.000225, 0.081225, 0.004275,
                0.731024, 0.038475, 0.004275, 0.000225, 0.038475, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            7: [0.009024, 0.081225, 0.000474, 0.004275, 0.081225, 0.731024, 0.004275, 0.038475, 0.000474, 0.004275,
                0.000024, 0.000225, 0.004275, 0.038475, 0.000225, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            8: [0.085737, 0.004512, 0.004512, 0.000237, 0.004512, 0.000237, 0.000237, 0.000012, 0.771637, 0.040612,
                0.040612, 0.002137, 0.040612, 0.002137, 0.002137, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            9: [0.085737, 0.771637, 0.004512, 0.040612, 0.004512, 0.040612, 0.000237, 0.002137, 0.004512, 0.040612,
                0.000237, 0.002137, 0.000237, 0.002137, 0.000012, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                 0.0],
            11: [0.085737, 0.004512, 0.004512, 0.000237, 0.771637, 0.040612, 0.040612, 0.002137, 0.004512, 0.000237,
                 0.000237, 0.000012, 0.040612, 0.002137, 0.002137, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            12: [0.009024, 0.000474, 0.081225, 0.004275, 0.000474, 0.000024, 0.004275, 0.000225, 0.081225, 0.004275,
                 0.731024, 0.038475, 0.004275, 0.000225, 0.038475, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            13: [0.009024, 0.081225, 0.000474, 0.004275, 0.081225, 0.731024, 0.004275, 0.038475, 0.000474, 0.004275,
                 0.000024, 0.000225, 0.004275, 0.038475, 0.000225, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            14: [0.009024, 0.000474, 0.081225, 0.004275, 0.000474, 0.000024, 0.004275, 0.000225, 0.081225, 0.004275,
                 0.731024, 0.038475, 0.004275, 0.000225, 0.038475, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            15: [0.009024, 0.081225, 0.000474, 0.004275, 0.081225, 0.731024, 0.004275, 0.038475, 0.000474, 0.004275,
                 0.000024, 0.000225, 0.004275, 0.038475, 0.000225, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            16: [0.085737, 0.004512, 0.004512, 0.000237, 0.004512, 0.000237, 0.000237, 0.000012, 0.771637, 0.040612,
                 0.040612, 0.002137, 0.040612, 0.002137, 0.002137, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            17: [0.085737, 0.771637, 0.004512, 0.040612, 0.004512, 0.040612, 0.000237, 0.002137, 0.004512, 0.040612,
                 0.000237, 0.002137, 0.000237, 0.002137, 0.000012, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            18: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                 0.0],
            19: [0.085737, 0.004512, 0.004512, 0.000237, 0.771637, 0.040612, 0.040612, 0.002137, 0.004512, 0.000237,
                 0.000237, 0.000012, 0.040612, 0.002137, 0.002137, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            20: [0.009024, 0.000474, 0.081225, 0.004275, 0.000474, 0.000024, 0.004275, 0.000225, 0.081225, 0.004275,
                 0.731024, 0.038475, 0.004275, 0.000225, 0.038475, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            21: [0.009024, 0.081225, 0.000474, 0.004275, 0.081225, 0.731024, 0.004275, 0.038475, 0.000474, 0.004275,
                 0.000024, 0.000225, 0.004275, 0.038475, 0.000225, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            22: [0.009024, 0.000474, 0.081225, 0.004275, 0.000474, 0.000024, 0.004275, 0.000225, 0.081225, 0.004275,
                 0.731024, 0.038475, 0.004275, 0.000225, 0.038475, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            23: [0.009024, 0.081225, 0.000474, 0.004275, 0.081225, 0.731024, 0.004275, 0.038475, 0.000474, 0.004275,
                 0.000024, 0.000225, 0.004275, 0.038475, 0.000225, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            24: [0.085737, 0.004512, 0.004512, 0.000237, 0.004512, 0.000237, 0.000237, 0.000012, 0.771637, 0.040612,
                 0.040612, 0.002137, 0.040612, 0.002137, 0.002137, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            25: [0.085737, 0.771637, 0.004512, 0.040612, 0.004512, 0.040612, 0.000237, 0.002137, 0.004512, 0.040612,
                 0.000237, 0.002137, 0.000237, 0.002137, 0.000012, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            26: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                 0.0],
            27: [0.085737, 0.004512, 0.004512, 0.000237, 0.771637, 0.040612, 0.040612, 0.002137, 0.004512, 0.000237,
                 0.000237, 0.000012, 0.040612, 0.002137, 0.002137, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            28: [0.009024, 0.000474, 0.081225, 0.004275, 0.000474, 0.000024, 0.004275, 0.000225, 0.081225, 0.004275,
                 0.731024, 0.038475, 0.004275, 0.000225, 0.038475, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            29: [0.009024, 0.081225, 0.000474, 0.004275, 0.081225, 0.731024, 0.004275, 0.038475, 0.000474, 0.004275,
                 0.000024, 0.000225, 0.004275, 0.038475, 0.000225, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            30: [0.009024, 0.000474, 0.081225, 0.004275, 0.000474, 0.000024, 0.004275, 0.000225, 0.081225, 0.004275,
                 0.731024, 0.038475, 0.004275, 0.000225, 0.038475, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            31: [0.009024, 0.081225, 0.000474, 0.004275, 0.081225, 0.731024, 0.004275, 0.038475, 0.000474, 0.004275,
                 0.000024, 0.000225, 0.004275, 0.038475, 0.000225, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            32: [0.085737, 0.004512, 0.004512, 0.000237, 0.004512, 0.000237, 0.000237, 0.000012, 0.771637, 0.040612,
                 0.040612, 0.002137, 0.040612, 0.002137, 0.002137, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            33: [0.085737, 0.771637, 0.004512, 0.040612, 0.004512, 0.040612, 0.000237, 0.002137, 0.004512, 0.040612,
                 0.000237, 0.002137, 0.000237, 0.002137, 0.000012, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            34: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                 0.0],
            35: [0.085737, 0.004512, 0.004512, 0.000237, 0.771637, 0.040612, 0.040612, 0.002137, 0.004512, 0.000237,
                 0.000237, 0.000012, 0.040612, 0.002137, 0.002137, 0.000120, 0.0, 0.0, 0.0, 0.0, 0.0],
            36: [0.009024, 0.000474, 0.081225, 0.004275, 0.000474, 0.000024, 0.004275, 0.000225, 0.081225, 0.004275,
                 0.731024, 0.038475, 0.004275, 0.000225, 0.038475, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            37: [0.009024, 0.081225, 0.000474, 0.004275, 0.081225, 0.731024, 0.004275, 0.038475, 0.000474, 0.004275,
                 0.000024, 0.000225, 0.004275, 0.038475, 0.000225, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            38: [0.009024, 0.000474, 0.081225, 0.004275, 0.000474, 0.000024, 0.004275, 0.000225, 0.081225, 0.004275,
                 0.731024, 0.038475, 0.004275, 0.000225, 0.038475, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            39: [0.009024, 0.081225, 0.000474, 0.004275, 0.081225, 0.731024, 0.004275, 0.038475, 0.000474, 0.004275,
                 0.000024, 0.000225, 0.004275, 0.038475, 0.000225, 0.002030, 0.0, 0.0, 0.0, 0.0, 0.0],
            40: [0.000949, 0.000049, 0.008549, 0.000449, 0.008549, 0.000449, 0.076949, 0.004049, 0.008549, 0.000449,
                 0.076949, 0.004049, 0.076949, 0.004049, 0.692550, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            41: [0.000949, 0.008549, 0.000049, 0.000449, 0.008549, 0.076949, 0.000449, 0.004049, 0.008549, 0.076949,
                 0.000449, 0.004049, 0.076949, 0.692550, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            42: [0.000949, 0.008549, 0.008549, 0.076949, 0.000049, 0.000449, 0.000449, 0.004049, 0.008549, 0.076949,
                 0.076949, 0.692550, 0.000449, 0.004049, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            43: [0.000949, 0.008549, 0.008549, 0.076949, 0.008549, 0.076949, 0.076949, 0.692550, 0.000049, 0.000449,
                 0.000449, 0.004049, 0.000449, 0.004049, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            44: [0.000949, 0.008549, 0.008549, 0.076949, 0.008549, 0.076949, 0.076949, 0.692550, 0.000049, 0.000449,
                 0.000449, 0.004049, 0.000449, 0.004049, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            45: [0.000949, 0.000049, 0.008549, 0.000449, 0.008549, 0.000449, 0.076949, 0.004049, 0.008549, 0.000449,
                 0.076949, 0.004049, 0.076949, 0.004049, 0.692550, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            46: [0.000949, 0.008549, 0.000049, 0.000449, 0.008549, 0.076949, 0.000449, 0.004049, 0.008549, 0.076949,
                 0.000449, 0.004049, 0.076949, 0.692550, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            47: [0.000949, 0.008549, 0.008549, 0.076949, 0.000049, 0.000449, 0.000449, 0.004049, 0.008549, 0.076949,
                 0.076949, 0.692550, 0.000449, 0.004049, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            48: [0.000949, 0.008549, 0.008549, 0.076949, 0.008549, 0.076949, 0.076949, 0.692550, 0.000049, 0.000449,
                 0.000449, 0.004049, 0.000449, 0.004049, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            49: [0.000949, 0.000049, 0.008549, 0.000449, 0.008549, 0.000449, 0.076949, 0.004049, 0.008549, 0.000449,
                 0.076949, 0.004049, 0.076949, 0.004049, 0.692550, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            50: [0.000949, 0.008549, 0.000049, 0.000449, 0.008549, 0.076949, 0.000449, 0.004049, 0.008549, 0.076949,
                 0.000449, 0.004049, 0.076949, 0.692550, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            51: [0.000949, 0.008549, 0.008549, 0.076949, 0.000049, 0.000449, 0.000449, 0.004049, 0.008549, 0.076949,
                 0.076949, 0.692550, 0.000449, 0.004049, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            52: [0.000949, 0.008549, 0.008549, 0.076949, 0.008549, 0.076949, 0.076949, 0.692550, 0.000049, 0.000449,
                 0.000449, 0.004049, 0.000449, 0.004049, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            53: [0.000949, 0.000049, 0.008549, 0.000449, 0.008549, 0.000449, 0.076949, 0.004049, 0.008549, 0.000449,
                 0.076949, 0.004049, 0.076949, 0.004049, 0.692550, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            54: [0.000949, 0.008549, 0.000049, 0.000449, 0.008549, 0.076949, 0.000449, 0.004049, 0.008549, 0.076949,
                 0.000449, 0.004049, 0.076949, 0.692550, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            55: [0.000949, 0.008549, 0.008549, 0.076949, 0.000049, 0.000449, 0.000449, 0.004049, 0.008549, 0.076949,
                 0.076949, 0.692550, 0.000449, 0.004049, 0.004049, 0.036464, 0.0, 0.0, 0.0, 0.0, 0.0],
            56: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 1.0],
            57: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 1.0],
            58: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 1.0],
            59: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 1.0]
        }
        #potential_goals = list(range(44, 60))
        #potential_goals = list(range(20, 36))
        goal = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1], dtype=np.int)
        super().__init__(text, seed=seed, goal=goal, start_to_obs=None)

        #self.observation_space = spaces.Dict(dict(
        #    desired_goal=spaces.MultiBinary(len(self.model.observations)),
        #    achieved_goal=spaces.MultiBinary(len(self.model.observations)),
        #    observation=spaces.MultiBinary(len(self.model.observations))
        #))
    def get_starting_obs(self, state):
        self.state, self.obs, *ret = self.step_functional(state, 0)
        obs = self.get_obs()
        #print(state)
        #obs, _, _, _ = self.step(0) # 0 is no-op
        #print(obs)
        #obs = np.zeros(len(self.model.observations), dtype=np.int)
        #obs[self.start_to_obs[state.argmax()]] = 1
        return obs
