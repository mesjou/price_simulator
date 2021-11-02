import abc
import random
from typing import List, Tuple

import attr

import numpy as np


@attr.s
class AgentStrategy(metaclass=abc.ABCMeta):
    """Top-level interface for Price setting agents"""

    marginal_cost: float = attr.ib(default=1.0)
    quality: float = attr.ib(default=2.0)

    @marginal_cost.validator
    def check_marginal_costs(self, attribute, value):
        if not value >= 0.0:
            raise ValueError("Marginal costs must be positive")

    @quality.validator
    def check_quality_costs(self, attribute, value):
        if not self.marginal_cost <= value:
            raise ValueError("Quality must be at least as high as marginal costs to be active in market")

    @abc.abstractmethod
    def play_price(self, state, action_space, n_period, t):
        raise NotImplementedError

    @abc.abstractmethod
    def learn(
        self,
        previous_reward: float,
        reward: float,
        previous_action: float,
        action: float,
        action_space: List,
        previous_state: Tuple,
        state: Tuple,
        next_state: Tuple,
    ):
        raise NotImplementedError

    def who_am_i(self) -> str:
        return type(self).__name__


@attr.s
class AlwaysDefectAgent(AgentStrategy):
    """Agent that always defects"""

    def play_price(self, state: Tuple, action_space: List, n_period: int, t: int):
        """Always play the lowest possible price."""
        return min(action_space)

    def learn(
        self,
        previous_reward: float,
        reward: float,
        previous_action: float,
        action: float,
        action_space: List,
        previous_state: Tuple,
        state: Tuple,
        next_state: Tuple,
    ):
        pass


@attr.s
class RandomAgent(AlwaysDefectAgent):
    """Agent that plays random prices"""

    def play_price(self, state: Tuple, action_space: List, n_period: int, t: int):
        return random.choice(action_space)


@attr.s
class TitForTat(AlwaysDefectAgent):
    """
    Tit for Tat Agent.

    If opponent undercut last period play lowest price.
    Otherwise play opponent last periods price.
    Agent must be second in list. Only two agents possible.

    """

    def play_price(self, state: Tuple, action_space: List, n_period: int, t: int):
        if state[0] < state[1]:
            return min(action_space)
        else:
            return state[0]


@attr.s
class PremiumPricer(AlwaysDefectAgent):
    """
    Always plays a price one above the competitor prices.
    Agent must be last in list.

    """

    def play_price(self, state: Tuple, action_space: List, n_period: int, t: int):
        competitor_actions = state[:-1]
        max_competitor = int(np.where(np.array(action_space) == max(competitor_actions))[0])
        if max_competitor < len(action_space) - 1:
            return action_space[max_competitor + 1]
        else:
            return action_space[max_competitor]


@attr.s
class PenetrationPricer(AlwaysDefectAgent):
    """
    Always plays a price one below the competitor prices.
    Agent must be last in list.

    """

    def play_price(self, state: Tuple, action_space: List, n_period: int, t: int):
        competitor_actions = state[:-1]
        min_competitor = int(np.where(np.array(action_space) == min(competitor_actions))[0])
        if min_competitor > 0:
            return action_space[min_competitor - 1]
        else:
            return action_space[0]

 
@attr.s
class Follower(AlwaysDefectAgent):
    """
    Always plays the minimum price of last period.
    Agent must be last in list.
    
    """

    def play_price(self, state: Tuple, action_space: List, n_period: int, t: int):
        competitor_actions = state[:-1]
        min_competitor = int(np.where(np.array(action_space) == min(competitor_actions))[0])
        
        return action_space[min_competitor]
