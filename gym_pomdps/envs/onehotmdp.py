import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


from rl_parsers.mdp import parse

__all__ = ['OneHotMDP', 'HallwayOneHotMDP', 'CheeseOneHotMDP', 'BigcheeseOneHotMDP', 'MitOneHotMDP', 'CitOneHotMDP']


# NOTE: Each domain must extend this
class OneHotMDP(gym.Env):  # pylint: disable=abstract-method
    """Environment specified by MDP file."""

    def __init__(self, text, *, seed=None, goal: int=None):
        model = parse(text)
        self.seed(seed)

        if model.values == 'cost':
            raise ValueError('Unsupported `cost` values.')

        self.model = model
        self.discount = model.discount
        self.state_space = spaces.MultiBinary(len(model.states))
        self.action_space = spaces.Discrete(len(model.actions))
        self.observation_space = observation=spaces.MultiBinary(len(self.model.states))
        self.goal = None
        self.reward_range = model.R.min(), model.R.max()


        self.rewards_dict = {r: i for i, r in enumerate(np.unique(model.R))}

        if model.start is None:
            self.start = np.full(self.state_space.n, 1 / self.state_space.n)
        else:
            self.start = model.start
        self.T = model.T.transpose(1, 0, 2).copy()

        # self.R = model.R.transpose(1, 0, 2, 3).copy()
        self.R = model.R.transpose(1, 0, 2).copy()
        if goal is None:
            raise NotImplementedError("Need a goal")
        self.goal = np.zeros(len(self.model.states))
        self.goal[goal] = 1

        #self.state = -1
        self.state = np.zeros(len(model.states), dtype=np.int)

    def get_state(self):
        return self.state

    def seed(self, seed):  # pylint: disable=signature-differs
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.state = self.reset_functional()
        return self.get_state().copy()

    def step(self, action):
        ret = self.step_functional(self.state, action)
        self.state = ret[0]
        return ret

    def reset_functional(self):
        #return self.np_random.multinomial(1, self.start).argmax().item()
        return self.np_random.multinomial(1, self.start)
        
    def step_functional(self, state, action):
        if (np.array_equal(state, np.zeros(len(self.model.states), dtype=np.int))) != (action == -1):
            raise ValueError(f'Invalid state-action pair ({state}, {action}).')
        if (np.array_equal(state, np.zeros(len(self.model.states), dtype=np.int))) and action == -1:
            return np.zeros(len(self.model.states), dtype=np.int), -1, True, None
        #if (state == -1) != (action == -1):
        #    raise ValueError(f'Invalid state-action pair ({state}, {action}).')

        #if state == -1 and action == -1:
        #    return -1, -1, 0.0, True, None

        #assert 0 <= state < self.state_space.n
        assert len(state) == len(self.model.states)
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
        # TODO: Experimental, changing to state_next from state
        #done = self._is_success(state, self.goal)
        done = self._is_success(state_next, self.goal)
        info = {
            'is_success': 1 if done else 0,

        }

        reward = self.compute_reward(state_next, self.goal, info)

        if done:
            state_next = np.zeros(len(self.model.states), dtype=np.int)

        return state_next, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        if np.array_equal(achieved_goal, desired_goal):
            return 0
        else:
            return -1
        
    def _is_success(self, achieved_goal, desired_goal):
        if np.array_equal(achieved_goal, desired_goal):
            return True
        else:
            return False


class HallwayOneHotMDP(OneHotMDP):
    def __init__(self, text, *, seed=None):
        super().__init__(text, seed=seed, goal=58)

        #self.goal = self._sample_goal()



#    def _sample_goal(self):
#        possible_goals = list(range(44, 60))
#        goal = np.zeros(len(self.model.states), dtype=np.int)
#        goal[np.random.choice(possible_goals)] = 1
#        return goal


class CheeseOneHotMDP(OneHotMDP):
    def __init__(self, text, *, seed=None):
        super().__init__(text, seed=seed, goal=10)

        #self.goal = self._sample_goal()


  #  def _sample_goal(self):
  #      possible_goals = [8, 9, 10]
  #      goal = np.zeros(len(self.model.states), dtype=np.int)
  #      goal[np.random.choice(possible_goals)] = 1
  #      return goal


class BigcheeseOneHotMDP(OneHotMDP):
    def __init__(self, text, *, seed=None):
        super().__init__(text, seed=seed, goal=30)

class MitOneHotMDP(OneHotMDP):
    def __init__(self, text, *, seed=None):
        super().__init__(text, seed=seed,
                         potential_goals=[52, 53, 54, 55, 100, 101, 102, 103, 168, 169, 170, 171, 196, 197, 198, 199])



 #   def _sample_goal(self):
        # 168, 169, 170, 171
 #       possible_goals = list(range(len(self.model.states)))
 #       goal = np.zeros(len(self.model.states), dtype=np.int)
 #       goal[np.random.choice(possible_goals)] = 1
 #       return goal
        

class CitOneHotMDP(OneHotMDP):
    def __init__(self, text, *, seed=None):
        super().__init__(text, seed=seed)

        #self.goal = self._sample_goal()


  #  def _sample_goal(self):
  #      # 68, 69, 70, 71
  #      possible_goals = list(range(len(self.model.states)))
  #      goal = np.zeros(len(self.model.states), dtype=np.int)
  #      goal[np.random.choice(possible_goals)] = 1
  #      return goal
