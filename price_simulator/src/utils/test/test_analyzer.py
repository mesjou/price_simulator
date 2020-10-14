from statistics import mean

import numpy as np
import price_simulator.src.utils.analyzer as analyzer
from price_simulator.src.algorithm.agents.tabular import Qlearning
from price_simulator.src.algorithm.demand import ConstantDemand, LogitDemand
from price_simulator.src.algorithm.environment import DiscreteSynchronEnvironment


def test_prepare_profit_calculation():
    env = DiscreteSynchronEnvironment(
        n_periods=1, agents=[Qlearning(), Qlearning(), Qlearning(), Qlearning()], demand=LogitDemand(),
    )
    env.play_game()
    nash_profits, monopoly_profits = analyzer.prepare_profit_calculation(env)
    assert len(nash_profits) == len(env.agents)
    assert len(monopoly_profits) == len(env.agents)
    assert (nash_profits < monopoly_profits).all()


def test_get_collusion_for():
    assert (analyzer.get_collusion_for(np.array([2]), np.array([1]), np.array([4])) == np.array([1 / 3])).all()
    assert (
        analyzer.get_collusion_for(np.array([2, 2]), np.array([1, 1]), np.array([4, 4])) == np.array([1 / 3, 1 / 3])
    ).all()
    assert (analyzer.get_collusion_for(np.array([1, 4]), np.array([1, 1]), np.array([4, 4])) == np.array([0, 1])).all()


def test_analyze():
    env = DiscreteSynchronEnvironment(
        agents=[
            Qlearning(marginal_cost=0.0),
            Qlearning(marginal_cost=0.0),
            Qlearning(marginal_cost=1.0),
            Qlearning(marginal_cost=1.0),
        ],
        demand=ConstantDemand(),
    )
    env.nash_prices = [1.0, 4.0, 1.0, 4.0]
    env.monopoly_prices = [2.0, 6.0, 2.0, 6.0]
    env.agents[0].rewards = [2.0, 2.0, 3.0, 3.0, 1.0, 1.0]
    env.agents[1].rewards = [2.0, 2.0, 3.0, 3.0, 1.0, 1.0]
    env.agents[2].rewards = [2.0, 2.0, 3.0, 3.0, 1.0, 1.0]
    env.agents[3].rewards = [2.0, 2.0, 3.0, 3.0, 1.0, 1.0]
    average_profits = [mean(agent.rewards) for agent in env.agents]
    nash_profits, monopoly_profits = analyzer.prepare_profit_calculation(env)
    collusion_profits = analyzer.get_collusion_for(average_profits, nash_profits, monopoly_profits)
    assert (collusion_profits == np.array([1.0, -1.0, 2.0, -0.5])).all()

    env = DiscreteSynchronEnvironment(
        agents=[Qlearning(marginal_cost=1.0), Qlearning(marginal_cost=0.0)], demand=ConstantDemand()
    )
    env.nash_prices = [1.0, 1.0]
    env.monopoly_prices = [2.0, 4.0]
    env.agents[0].rewards = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    env.agents[1].rewards = [2.0, 2.0, 3.0, 3.0, 1.0, 1.0]
    average_profits = [mean(agent.rewards) for agent in env.agents]
    nash_profits, monopoly_profits = analyzer.prepare_profit_calculation(env)
    collusion_profits = analyzer.get_collusion_for(average_profits, nash_profits, monopoly_profits)
    assert (collusion_profits == np.array([1.5, 1 / 3])).all()
