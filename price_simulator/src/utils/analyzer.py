from typing import Tuple

import numpy as np
from tabulate import tabulate


def prepare_profit_calculation(environment) -> Tuple[np.array, np.array]:
    qualities = tuple(agent.quality for agent in environment.agents)
    marginal_costs = tuple(agent.marginal_cost for agent in environment.agents)
    nash_quantities = environment.demand.get_quantities(environment.nash_prices, qualities)
    nash_profits = np.multiply(np.subtract(environment.nash_prices, marginal_costs), nash_quantities)
    monopoly_quantities = environment.demand.get_quantities(environment.monopoly_prices, qualities)
    monopoly_profits = np.multiply(np.subtract(environment.monopoly_prices, marginal_costs), monopoly_quantities)
    return nash_profits, monopoly_profits


def get_collusion_for(averages: np.array, nash_values: np.array, monopoly_values: np.array) -> np.array:
    return np.divide(np.subtract(averages, nash_values), np.subtract(monopoly_values, nash_values))


def analyze(environment):
    average_prices = np.array(environment.price_history).mean(axis=0)
    average_profits = np.array(environment.reward_history).mean(axis=0)
    nash_profits, monopoly_profits = prepare_profit_calculation(environment)
    collusion_profits = get_collusion_for(average_profits, nash_profits, monopoly_profits)

    info = [agent.who_am_i() for agent in environment.agents]

    print(
        tabulate(
            {
                "Agent": info,
                "Average Price": average_prices,
                "Nash Price": environment.nash_prices,
                "Monopoly Price": environment.monopoly_prices,
                "Average Profit Gain": collusion_profits,
                "Nash Profit": nash_profits,
                "Monopoly Profit": monopoly_profits,
            },
            headers="keys",
        )
    )
