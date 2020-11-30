import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


from rl_parsers.mdp import parse

__all__ = ['OneHotMDP', 'HallwayOneHotMDP', 'CheeseOneHotMDP', 'MITOneHotMDP', 'CITOneHotMDP']


# NOTE: Each domain must extend this
class OneHotMDP(gym.GoalEnv):  # pylint: disable=abstract-method
    """Environment specified by MDP file."""

    def __init__(self, text, *, episodic, seed=None, potential_goals=None):
        model = parse(text)
        self.episodic = episodic
        self.seed(seed)

        if model.values == 'cost':
            raise ValueError('Unsupported `cost` values.')

        self.model = model
        self.discount = model.discount
        self.state_space = spaces.MultiBinary(len(model.states))
        self.action_space = spaces.Discrete(len(model.actions))
        self.observation_space = None
        self.goal = None
        self.reward_range = model.R.min(), model.R.max()

        self.steps = 0
        self.step_cap = 0

        self.rewards_dict = {r: i for i, r in enumerate(np.unique(model.R))}

        if model.start is None:
            self.start = np.full(self.state_space.n, 1 / self.state_space.n)
        else:
            self.start = model.start
        self.T = model.T.transpose(1, 0, 2).copy()

        # self.R = model.R.transpose(1, 0, 2, 3).copy()
        self.R = model.R.transpose(1, 0, 2).copy()

        if episodic:
            self.D = model.reset.T.copy()  # only if episodic

        if potential_goals:
            self.potential_goals = potential_goals
        else:
            self.potential_goals = list(range(len(model.states)))

        #self.state = -1
        self.state = np.zeros(len(model.states), dtype=np.int)
        
    def _get_obs(self):
        obs = self.get_state()
        achieved_goal = self.get_state()
        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': self.goal
        }

    def _sample_goal(self):
        goal = np.zeros(len(self.model.states), dtype=np.int)
        goal[np.random.choice(self.potential_goals)] = 1
        return goal

    def get_state(self):
        return self.state

    def seed(self, seed):  # pylint: disable=signature-differs
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.state = self.reset_functional()
        self.steps = 0
        self.goal = self._sample_goal().copy()
        return self._get_obs()

    def step(self, action):
        # self.state, *ret = self.step_functional(self.state, action)
        obs = self._get_obs()
        ret = self.step_functional(self.state, obs, action)
        self.state = ret[0]
        obs = self._get_obs()
        return (obs, *ret[1:])

    def reset_functional(self):
        #return self.np_random.multinomial(1, self.start).argmax().item()
        return self.np_random.multinomial(1, self.start)
        
    def step_functional(self, state, obs, action):
        if (np.array_equal(state, np.zeros(len(self.model.states), dtype=np.int))) != (action == -1):
            raise ValueError(f'Invalid state-action pair ({state}, {action}).')
        if (np.array_equal(state, np.zeros(len(self.model.states), dtype=np.int))) and action == -1:
            return  np.zeros(len(self.model.states), dtype=np.int), -1, True, None
        #if (state == -1) != (action == -1):
        #    raise ValueError(f'Invalid state-action pair ({state}, {action}).')

        #if state == -1 and action == -1:
        #    return -1, -1, 0.0, True, None

        #assert 0 <= state < self.state_space.n
        assert 0 <= action < self.action_space.n

        #state_next = (
        #    self.np_random.multinomial(1, self.T[state, action]).argmax().item()
        #)
        state_next = (
            self.np_random.multinomial(1, self.T[state.argmax(), action])
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

        done = self._is_success(state, self.goal)
        info = {
            'is_success' : 1 if done else 0,

        }
        
        reward = self.compute_reward(state, self.goal, info)
       
        if done:
            # state_next = -1
            state_next = np.zeros(len(self.model.states), dtype=np.int)
            
        self.steps += 1
        if self.steps >= self.step_cap:
            done = True

        # reward_cat = self.rewards_dict[reward]
        # info = dict(reward_cat=reward_cat)

        # return state_next, obs, reward, done, info
        return state_next, reward, done, info

    def compute_reward(self, achieved_goal, goal, info):
        #if achieved_goal == goal:
        #    return 0
        #else:
        #    return -1
        if np.array_equal(achieved_goal, goal):
            return 0
        else:
            return -1
        
    def _is_success(self, achieved_goal, desired_goal):
        #if achieved_goal == desired_goal:
        #    return True
        #else:
        #    return False
        if np.array_equal(achieved_goal, desired_goal):
            return True
        else:
            return False


class HallwayOneHotMDP(OneHotMDP):
    def __init__(self, text, *, episodic, seed=None):
        super().__init__(text, episodic=episodic, seed=seed)

        #self.goal = self._sample_goal()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.MultiBinary(len(self.model.states)),
            achieved_goal=spaces.MultiBinary(len(self.model.states)),
            observation=spaces.MultiBinary(len(self.model.states))
        ))
        self.step_cap = 15 #np.inf

#    def _sample_goal(self):
#        possible_goals = list(range(44, 60))
#        goal = np.zeros(len(self.model.states), dtype=np.int)
#        goal[np.random.choice(possible_goals)] = 1
#        return goal


class CheeseOneHotMDP(OneHotMDP):
    def __init__(self, text, *, episodic, seed=None):
        super().__init__(text, episodic=episodic, seed=seed, potential_goals=[8, 9, 10])

        #self.goal = self._sample_goal()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.MultiBinary(len(self.model.states)),
            achieved_goal=spaces.MultiBinary(len(self.model.states)),
            observation=spaces.MultiBinary(len(self.model.states))
        ))
        self.step_cap = 10  # np.inf

  #  def _sample_goal(self):
  #      possible_goals = [8, 9, 10]
  #      goal = np.zeros(len(self.model.states), dtype=np.int)
  #      goal[np.random.choice(possible_goals)] = 1
  #      return goal


class MITOneHotMDP(OneHotMDP):
    def __init__(self, text, *, episodic, seed=None):
        super().__init__(text, episodic=episodic, seed=seed)

        #self.goal = self._sample_goal()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.MultiBinary(len(self.model.states)),
            achieved_goal=spaces.MultiBinary(len(self.model.states)),
            observation=spaces.MultiBinary(len(self.model.states))
        ))
        self.step_cap = 50

 #   def _sample_goal(self):
        # 168, 169, 170, 171
 #       possible_goals = list(range(len(self.model.states)))
 #       goal = np.zeros(len(self.model.states), dtype=np.int)
 #       goal[np.random.choice(possible_goals)] = 1
 #       return goal
        

class CITOneHotMDP(OneHotMDP):
    def __init__(self, text, *, episodic, seed=None):
        super().__init__(text, episodic=episodic, seed=seed)

        #self.goal = self._sample_goal()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.MultiBinary(len(self.model.states)),
            achieved_goal=spaces.MultiBinary(len(self.model.states)),
            observation=spaces.MultiBinary(len(self.model.states))
        ))
        self.step_cap = 50

  #  def _sample_goal(self):
  #      # 68, 69, 70, 71
  #      possible_goals = list(range(len(self.model.states)))
  #      goal = np.zeros(len(self.model.states), dtype=np.int)
  #      goal[np.random.choice(possible_goals)] = 1
  #      return goal
