import copy
from typing import List

import attr

import numpy as np
from price_simulator.src.algorithm.demand import MarketDemandStrategy
from scipy.optimize import fsolve, minimize


@attr.s
class EquilibriumCalculator(object):
    """Find equilibrium (Monopoly and Nash) for market environment specified by marginal costs, qualities and demand."""

    demand: MarketDemandStrategy = attr.ib()

    def get_nash_equilibrium(self, qualities: List, marginal_costs: List) -> np.array:
        """Calculate prices that makes market outcome an equilibrium"""
        param = (qualities, marginal_costs)
        p0 = np.array(marginal_costs)
        return fsolve(self.vector_reaction, p0, args=param)

    def profit(
        self, own_price: float, prices: np.array, qualities: np.array, marginal_costs: np.array, i: int
    ) -> float:
        """Calculate profit for ith firm if it sets his price to own_price given competitor prices."""
        temp_prices = copy.deepcopy(prices)
        temp_prices[i] = own_price
        return -1 * (temp_prices[i] - marginal_costs[i]) * self.demand.get_quantities(temp_prices, qualities)[i]

    def reaction_function(self, prices: np.array, qualities: np.array, marginal_costs: np.array, i: float) -> float:
        """Get price (optimal reaction) that maximizes own profit for given competitor prices."""
        return minimize(
            fun=self.profit,
            x0=np.array(marginal_costs[i]),
            args=(prices, qualities, marginal_costs, i),
            method="nelder-mead",
            options={"xatol": 1e-8},
        ).x[0]

    def vector_reaction(self, nash_prices: np.array, qualities: np.array, marginal_costs: np.array) -> np.array:
        """Vector representation of the fix-point for Nash prices."""
        return np.array(nash_prices) - np.array(
            [self.reaction_function(nash_prices, qualities, marginal_costs, i) for i in range(len(nash_prices))]
        )

    def get_monopoly_outcome(self, qualities: List, marginal_costs: List) -> np.array:
        """Get prices that maximize joint profit."""
        return minimize(
            fun=self.joint_profit,
            x0=np.array(qualities),
            args=(qualities, marginal_costs),
            method="nelder-mead",
            options={"xatol": 1e-8},
        ).x

    def joint_profit(self, prices: np.array, qualities: np.array, marginal_costs: np.array) -> float:
        """Return (negative) joint profit for prices."""
        return -1 * np.sum(
            np.multiply(np.subtract(prices, marginal_costs), self.demand.get_quantities(prices, qualities))
        )
