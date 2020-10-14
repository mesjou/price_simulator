import pytest

import numpy as np
from price_simulator.src.algorithm.agents.approximate import DiffDQN
from price_simulator.src.algorithm.agents.simple import AlwaysDefectAgent
from price_simulator.src.algorithm.agents.tabular import Qlearning
from price_simulator.src.algorithm.demand import ConstantDemand, LogitDemand, PrisonersDilemmaDemand
from price_simulator.src.algorithm.environment import DiscreteSynchronEnvironment, ReformulationEnvironment
from price_simulator.src.algorithm.policies import DecreasingEpsilonGreedy, EpsilonGreedy


def test_environment_prisoners():
    test_1 = DiscreteSynchronEnvironment(
        n_periods=10000,
        possible_prices=[2, 3],
        demand=PrisonersDilemmaDemand(),
        agents=[
            Qlearning(discount=0.95, learning_rate=0.3, decision=EpsilonGreedy(eps=0.1)),
            Qlearning(discount=0.95, learning_rate=0.3, decision=EpsilonGreedy(eps=0.1)),
        ],
    )

    test_2 = DiscreteSynchronEnvironment(
        n_periods=10,
        possible_prices=[1, 2],
        demand=PrisonersDilemmaDemand(),
        agents=[Qlearning(discount=0.95, learning_rate=0.5, decision=DecreasingEpsilonGreedy()), AlwaysDefectAgent()],
    )

    test_3 = DiscreteSynchronEnvironment(
        n_periods=10000,
        possible_prices=[1, 2],
        demand=PrisonersDilemmaDemand(),
        agents=[
            Qlearning(discount=0.95, learning_rate=0.5, decision=DecreasingEpsilonGreedy()),
            Qlearning(discount=0.5, learning_rate=0.1, decision=DecreasingEpsilonGreedy()),
        ],
    )

    assert test_1.play_game()
    assert test_2.play_game()
    assert test_3.play_game()


def test_environment_advanced_qlearning():
    test_1 = DiscreteSynchronEnvironment(
        n_periods=10000,
        possible_prices=[2, 3],
        demand=LogitDemand(),
        agents=[
            Qlearning(discount=0.95, learning_rate=0.3, decision=EpsilonGreedy(eps=0.1)),
            Qlearning(
                discount=0.95, learning_rate=0.3, marginal_cost=4.0, quality=5.0, decision=EpsilonGreedy(eps=0.1)
            ),
            AlwaysDefectAgent(marginal_cost=0.1),
        ],
    )

    assert test_1.play_game()


def test_price_range():
    n = 10
    assert len(DiscreteSynchronEnvironment.get_price_range(0.5, 1.0, 0.1, n)) == n
    assert min(DiscreteSynchronEnvironment.get_price_range(0.5, 1.0, 0.1, 100)) == 0.45
    assert max(DiscreteSynchronEnvironment.get_price_range(0.5, 1.0, 0.1, n)) == 1.05


def test_init():
    env = DiscreteSynchronEnvironment(
        demand=LogitDemand(price_sensitivity=1.0, outside_quality=0.0),
        agents=[Qlearning(quality=1.0, marginal_cost=0.0), Qlearning(quality=1.0, marginal_cost=0.0)],
    )
    assert max(env.monopoly_prices) > min(env.nash_prices)
    assert sum(np.greater(env.nash_prices, env.monopoly_prices)) == 0

    env = DiscreteSynchronEnvironment(
        demand=LogitDemand(price_sensitivity=1.0, outside_quality=10.0),
        agents=[Qlearning(quality=10.0, marginal_cost=5.0), Qlearning(quality=10.0, marginal_cost=1.0)],
    )
    assert max(env.monopoly_prices) > min(env.nash_prices)
    assert sum(np.greater(env.nash_prices, env.monopoly_prices)) == 0

    env = DiscreteSynchronEnvironment(
        demand=PrisonersDilemmaDemand(),
        agents=[Qlearning(quality=10.0, marginal_cost=5.0), Qlearning(quality=10.0, marginal_cost=1.0)],
        possible_prices=[2, 3],
    )
    assert (env.monopoly_prices == np.array([3, 3])).all()
    assert (env.nash_prices == np.array([2, 2])).all()

    with pytest.raises(AssertionError):
        DiscreteSynchronEnvironment(
            demand=PrisonersDilemmaDemand(),
            agents=[Qlearning(quality=10.0, marginal_cost=5.0), Qlearning(quality=10.0, marginal_cost=1.0)],
        )


def test_play_game():
    env = DiscreteSynchronEnvironment(
        demand=LogitDemand(price_sensitivity=1.0, outside_quality=10.0),
        agents=[Qlearning(quality=10.0, marginal_cost=5.0), Qlearning(quality=10.0, marginal_cost=1.0)],
        markup=0.0,
        n_prices=10,
        n_periods=1,
    )
    env.play_game()
    assert len(env.possible_prices) == 10
    assert min(env.possible_prices) == min(env.nash_prices)
    assert max(env.possible_prices) == max(env.monopoly_prices)

    env = DiscreteSynchronEnvironment(
        demand=PrisonersDilemmaDemand(),
        agents=[Qlearning(quality=10.0, marginal_cost=5.0), Qlearning(quality=10.0, marginal_cost=1.0)],
        possible_prices=[3, 4],
        markup=0.1,
        n_prices=10,
        n_periods=1,
    )
    env.play_game()
    assert len(env.possible_prices) == 2
    assert min(env.possible_prices) == min(env.nash_prices)
    assert max(env.possible_prices) == max(env.monopoly_prices)


def test_reformulation():
    env = ReformulationEnvironment(
        demand=ConstantDemand(default_quantity=0.0), agents=[DiffDQN(), DiffDQN()], n_periods=5,
    )
    assert env.play_game() == 4

    env = ReformulationEnvironment(
        demand=ConstantDemand(default_quantity=0.0), agents=[DiffDQN(), DiffDQN(), DiffDQN()], n_periods=5,
    )
    assert env.play_game() == 4
