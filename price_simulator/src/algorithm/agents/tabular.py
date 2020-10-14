import copy
import itertools
import random
from typing import Dict, List, Tuple

import attr

from price_simulator.src.algorithm.agents.simple import AgentStrategy
from price_simulator.src.algorithm.policies import EpsilonGreedy, ExplorationStrategy


@attr.s
class Qlearning(AgentStrategy):
    """Q learning agent with bounded memory of 1 period"""

    q_matrix: Dict = attr.ib(default=None)
    discount: float = attr.ib(default=0.95)
    learning_rate: float = attr.ib(default=0.1)
    decision: ExplorationStrategy = attr.ib(factory=EpsilonGreedy)

    @discount.validator
    def check_discount(self, attribute, value):
        if not 0 <= value <= 1:
            raise ValueError("Discount factor must lie in [0,1]")

    @learning_rate.validator
    def check_learning_rate(self, attribute, value):
        """For learning_rate = 0, the algorithm does not learn at all.
        For learning_rate = 1, it immediately forgets what it has learned in the past.
        """
        if not 0 <= value < 1:
            raise ValueError("Learning rate must lie in [0,1)")

    def who_am_i(self) -> str:
        return type(self).__name__ + " (gamma: {}, alpha: {}, policy: {}, quality: {}, mc: {})".format(
            self.discount, self.learning_rate, self.decision.who_am_i(), self.quality, self.marginal_cost
        )

    def play_price(self, state: Tuple, action_space: List, n_period: int, t: int):
        """Either experiment or play greedy action."""
        if not self.q_matrix:
            self.q_matrix = self.initialize_q_matrix(len(state), action_space)
        if self.decision.explore(n_period, t):
            return random.choice(action_space)
        else:
            return self.get_optimal_action(self.q_matrix, state)

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
        state_action_value = copy.deepcopy(self.q_matrix[state][action])
        future_next_state_action_value = self.q_matrix[next_state][self.get_optimal_action(self.q_matrix, next_state)]
        self.q_matrix[state][action] = (1 - self.learning_rate) * state_action_value + self.learning_rate * (
            reward + self.discount * future_next_state_action_value
        )

    @staticmethod
    def get_optimal_action(q_matrix: Dict, state: Tuple) -> float:
        """Return the action with highest value for given state.

        If there are two or more values with identical value,
        choose randomly.
        """
        optimal_actions = [
            action for action, value in q_matrix[state].items() if value == max(q_matrix[state].values())
        ]
        return random.choice(optimal_actions)

    @staticmethod
    def initialize_q_matrix(n_agents: int, actions_space: List) -> Dict:
        """Create dictionary where all actions are mapped to possible state combinations.

        Initial value for each state action combination is set to 0.
        Q-matrix = {
            state1: {
                action1: 0,
                action2: 0,
                ...
            },
            state2:{
                action1: 0,
                action2: 0,
                ...
            },
        }
        """
        q_matrix = {}
        for possible_state in itertools.product(actions_space, repeat=n_agents):
            q_matrix[possible_state] = dict((price, 0.0) for price in actions_space)
        return q_matrix


@attr.s
class SARSA(Qlearning):
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
        previous_state_action_value = copy.deepcopy(self.q_matrix[previous_state][previous_action])
        state_action_value = copy.deepcopy(self.q_matrix[state][action])
        self.q_matrix[previous_state][previous_action] = (
            1 - self.learning_rate
        ) * previous_state_action_value + self.learning_rate * (previous_reward + self.discount * state_action_value)
