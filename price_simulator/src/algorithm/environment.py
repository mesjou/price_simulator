import abc
import copy
import random
from typing import List, Tuple

import attr

import numpy as np
from price_simulator.src.algorithm.agents.simple import AgentStrategy
from price_simulator.src.algorithm.demand import LogitDemand, MarketDemandStrategy, PrisonersDilemmaDemand
from price_simulator.src.algorithm.equilibrium import EquilibriumCalculator
from price_simulator.src.utils.storage import Storage


@attr.s
class EnvironmentStrategy(metaclass=abc.ABCMeta):
    """Top-level interface for Environment."""

    agents: List[AgentStrategy] = attr.ib(factory=list)
    possible_prices: List[float] = attr.ib(factory=list)
    demand: MarketDemandStrategy = attr.ib(factory=LogitDemand)
    nash_prices: np.array = attr.ib(init=False)
    monopoly_prices: np.array = attr.ib(init=False)

    def __attrs_post_init__(self):
        """Compute Nash Price and Monopoly price after initialization."""
        if len(self.agents) > 0.0:
            if isinstance(self.demand, PrisonersDilemmaDemand):
                assert len(self.possible_prices) > 0.0, "Priosoners Dilemma needs two possible prices"
                self.monopoly_prices = [max(self.possible_prices), max(self.possible_prices)]
                self.nash_prices = np.array([min(self.possible_prices), min(self.possible_prices)])
            else:
                marginal_costs = [agent.marginal_cost for agent in self.agents]
                qualities = [agent.quality for agent in self.agents]
                self.monopoly_prices = EquilibriumCalculator(demand=self.demand).get_monopoly_outcome(
                    qualities, marginal_costs
                )
                self.nash_prices = EquilibriumCalculator(demand=self.demand).get_nash_equilibrium(
                    qualities, marginal_costs
                )

    @abc.abstractmethod
    def play_game(self):
        raise NotImplementedError


@attr.s
class DiscreteSynchronEnvironment(EnvironmentStrategy):
    """Environment for discrete states and prices.

     Before the first iteration, prices are randomly initialized.
     Agents set prices at the same time.
     After choosing prices, demand and rewards are calculated.
     Then agents have the opportunity to learn.
     """

    n_periods: int = attr.ib(default=1)
    markup: float = attr.ib(default=0.1)
    n_prices: int = attr.ib(default=15)
    convergence_after: int = attr.ib(default=np.inf)
    history_after: int = attr.ib(default=np.inf)
    price_history: List = attr.ib(factory=list)
    quantity_history: List = attr.ib(factory=list)
    reward_history: List = attr.ib(factory=list)
    storage: Storage = attr.ib(factory=Storage)

    @n_periods.validator
    def check_n_periods(self, attribute, value):
        if not 0 < value:
            raise ValueError("Number of periods must be strictly positive")

    @markup.validator
    def check_markup(self, attribute, value):
        if not 0 <= value:
            raise ValueError("Price markup must be positive")

    @n_prices.validator
    def check_n_prices(self, attribute, value):
        if not 0 < value:
            raise ValueError("Number of prices must be strictly positive")

    def play_game(self) -> int:

        qualities = tuple(agent.quality for agent in self.agents)
        marginal_costs = tuple(agent.marginal_cost for agent in self.agents)

        # initialize first rounds
        if len(self.possible_prices) == 0:
            self.possible_prices = self.get_price_range(
                min(self.nash_prices), max(self.monopoly_prices), self.markup, self.n_prices
            )
        previous_state = tuple(random.choices(self.possible_prices, k=len(self.agents)))
        state = tuple(
            agent.play_price(previous_state, self.possible_prices, self.n_periods, 0) for agent in self.agents
        )
        quantities = self.demand.get_quantities(state, qualities)
        previous_rewards = np.multiply(np.subtract(state, marginal_costs), quantities)

        # set up storage
        self.storage.set_up(len(self.agents), self.n_periods)

        for t in range(self.n_periods):

            # agents decide about there prices (hereafter is the state different)
            next_state = tuple(
                agent.play_price(state, self.possible_prices, self.n_periods, t) for agent in self.agents
            )

            # demand is estimated for prices
            quantities = self.demand.get_quantities(next_state, qualities)
            rewards = np.multiply(np.subtract(next_state, marginal_costs), quantities)

            # assert that everything is correct
            assert (np.array(quantities) >= 0.0).all(), "Quantities cannot be negative"
            assert (np.array(next_state) >= 0.0).all(), "Prices cannot be negative"

            # agents learn
            for agent, action, previous_action, reward, previous_reward in zip(
                self.agents, next_state, state, rewards, previous_rewards
            ):
                agent.learn(
                    previous_reward=previous_reward,
                    reward=reward,
                    previous_action=previous_action,
                    action=action,
                    action_space=self.possible_prices,
                    previous_state=previous_state,
                    state=state,
                    next_state=next_state,
                )

            # update variables
            previous_state = copy.deepcopy(state)
            state = copy.deepcopy(next_state)
            previous_rewards = copy.deepcopy(rewards)

            # save prices for the last periods
            if t > self.history_after:
                self.price_history.append(previous_state)
                self.quantity_history.append(quantities)
                self.reward_history.append(rewards)

            # Fill storage
            self.storage.observe(rewards, state, quantities)

        return t

    @staticmethod
    def get_price_range(nash_price: float, monopoly_price: float, markup: float, n_step: int) -> List:
        increase = (monopoly_price - nash_price) * markup
        return list(np.linspace(nash_price - increase, monopoly_price + increase, n_step))


@attr.s
class ReformulationEnvironment(DiscreteSynchronEnvironment):
    """Environment with reformulated state representation."""

    @staticmethod
    def reformulate(actions: Tuple) -> Tuple:
        return tuple([min(actions), max(actions), np.mean(actions)])

    def play_game(self) -> int:

        qualities = tuple(agent.quality for agent in self.agents)
        marginal_costs = tuple(agent.marginal_cost for agent in self.agents)

        # initialize first rounds
        if len(self.possible_prices) == 0:
            self.possible_prices = self.get_price_range(
                min(self.nash_prices), max(self.monopoly_prices), self.markup, self.n_prices
            )
        previous_state = self.reformulate(tuple(random.choices(self.possible_prices, k=len(self.agents))))
        previous_actions = tuple(
            agent.play_price(previous_state, self.possible_prices, self.n_periods, 0) for agent in self.agents
        )
        state = self.reformulate(previous_actions)
        quantities = self.demand.get_quantities(previous_actions, qualities)
        previous_rewards = np.multiply(np.subtract(previous_actions, marginal_costs), quantities)

        # set up storage
        self.storage.set_up(len(self.agents), self.n_periods)

        for t in range(self.n_periods):

            # agents decide about there prices (hereafter is the state different)
            actions = tuple(agent.play_price(state, self.possible_prices, self.n_periods, t) for agent in self.agents)
            next_state = self.reformulate(actions)

            # demand is estimated for prices
            quantities = self.demand.get_quantities(actions, qualities)
            rewards = np.multiply(np.subtract(actions, marginal_costs), quantities)

            # assert that everything is correct
            assert (np.array(quantities) >= 0.0).all(), "Quantities cannot be negative"
            assert (np.array(actions) >= 0.0).all(), "Prices cannot be negative"

            # agents learn
            for agent, action, previous_action, reward, previous_reward in zip(
                self.agents, actions, previous_actions, rewards, previous_rewards
            ):
                agent.learn(
                    previous_reward=previous_reward,
                    reward=reward,
                    previous_action=previous_action,
                    action=action,
                    action_space=self.possible_prices,
                    previous_state=previous_state,
                    state=state,
                    next_state=next_state,
                )

            # update variables
            previous_state = copy.deepcopy(state)
            state = copy.deepcopy(next_state)
            previous_rewards = copy.deepcopy(rewards)
            previous_actions = copy.deepcopy(actions)

            # save prices for the last periods
            if t > self.history_after:
                self.price_history.append(actions)
                self.quantity_history.append(quantities)
                self.reward_history.append(rewards)

            # Fill storage
            self.storage.observe(rewards, actions, quantities)

        return t
