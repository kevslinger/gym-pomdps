import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


from rl_parsers.mdp import parse

__all__ = ['OneHotGoalMDP', 'HallwayOneHotGoalMDP', 'CheeseOneHotGoalMDP', 'BigcheeseOneHotGoalMDP']


# NOTE: Each domain must extend this
class OneHotGoalMDP(gym.GoalEnv):  # pylint: disable=abstract-method
    """Environment specified by MDP file."""

    def __init__(self, text, *, seed=None, step_cap=np.inf, potential_goals=None):
        model = parse(text)
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
        self.step_cap = step_cap

        self.rewards_dict = {r: i for i, r in enumerate(np.unique(model.R))}

        if model.start is None:
            self.start = np.full(self.state_space.n, 1 / self.state_space.n)
        else:
            self.start = model.start
        self.T = model.T.transpose(1, 0, 2).copy()

        # self.R = model.R.transpose(1, 0, 2, 3).copy()
        self.R = model.R.transpose(1, 0, 2).copy()


        if potential_goals:
            self.potential_goals = potential_goals
        else:
            self.potential_goals = list(range(len(model.states)))

        #self.state = -1
        self.state = np.zeros(len(model.states), dtype=np.int)

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.MultiBinary(len(self.model.states)),
            achieved_goal=spaces.MultiBinary(len(self.model.states)),
            observation=spaces.MultiBinary(len(self.model.states))
        ))
        
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
            'is_success' : 1 if done else 0,

        }

        #TODO: Experimental, changing to state_next from state
        #reward = self.compute_reward(state, self.goal, info)
        reward = self.compute_reward(state_next, self.goal, info)


        if done:
            state_next = np.zeros(len(self.model.states), dtype=np.int)
            self.goal = np.zeros(len(self.model.states), dtype=np.int)
            
        self.steps += 1
        if self.steps >= self.step_cap:
            done = True

        return state_next, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.array_equal(achieved_goal, desired_goal):
            return 0
        else:
            return -1
        
    def _is_success(self, achieved_goal, desired_goal):
        if np.array_equal(achieved_goal, desired_goal):
            return True
        else:
            return False


class HallwayOneHotGoalMDP(OneHotGoalMDP):
    def __init__(self, text, *, seed=None, step_cap=np.inf):
        super().__init__(text, seed=seed, step_cap=step_cap, potential_goals=list(range(44, 60)))

        #self.goal = self._sample_goal()

        #self.step_cap = 15 #np.inf

#    def _sample_goal(self):
#        possible_goals = list(range(44, 60))
#        goal = np.zeros(len(self.model.states), dtype=np.int)
#        goal[np.random.choice(possible_goals)] = 1
#        return goal


class CheeseOneHotGoalMDP(OneHotGoalMDP):
    def __init__(self, text, *, seed=None, step_cap=np.inf):
        super().__init__(text, seed=seed, step_cap=step_cap, potential_goals=[8, 9, 10])

        #self.goal = self._sample_goal()

        #self.step_cap = 10  # np.inf

  #  def _sample_goal(self):
  #      possible_goals = [8, 9, 10]
  #      goal = np.zeros(len(self.model.states), dtype=np.int)
  #      goal[np.random.choice(possible_goals)] = 1
  #      return goal


class BigcheeseOneHotGoalMDP(OneHotGoalMDP):
    def __init__(self, text, *, seed=None, step_cap=np.inf):
        super().__init__(text, seed=seed, step_cap=step_cap,
                         potential_goals=[28, 29, 30, 31, 32])
