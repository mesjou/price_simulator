import numpy as np
from price_simulator.src.algorithm.agents.tabular import Qlearning
from price_simulator.src.algorithm.environment import DiscreteSynchronEnvironment
from price_simulator.src.algorithm.policies import DecreasingEpsilonGreedy, EpsilonGreedy, LinearDecreasing, Temperature


def test_explore():
    assert EpsilonGreedy(eps=0.5).explore(1, 1) in [True, False]
    assert EpsilonGreedy(eps=1).explore(1, 1) is True
    assert EpsilonGreedy(eps=0).explore(1, 1) is False
    assert DecreasingEpsilonGreedy().explore(1, 1) in [True, False]
    assert Temperature().explore(1, 1) in [True, False]
    assert Temperature(beta=0).explore(1, 1) is True


def test_epsilon_greedy():
    for e in [0.00001, 0.5, 0.1, 0.999999]:
        assert EpsilonGreedy(eps=e).epsilon(100000, 100000) == e


def test_decreasing_epsilon_greedy():
    assert DecreasingEpsilonGreedy().epsilon(10, 1) > DecreasingEpsilonGreedy().epsilon(10, 2)
    assert DecreasingEpsilonGreedy().epsilon(100000, 100000) < 0.001


def test_temperature():
    assert Temperature().epsilon(10000, 1) > Temperature().epsilon(10000, 2)
    assert Temperature(beta=0.00005).epsilon(10000, 10) < Temperature(beta=0.00004).epsilon(10000, 10)
    assert Temperature(beta=0.00005).epsilon(10000, 500000) < 0.001


def test_linear():
    assert LinearDecreasing().epsilon(10000, 1) > LinearDecreasing().epsilon(10000, 2)
    assert LinearDecreasing().epsilon(10000, 10000) >= 0.0001


def test_correct_init():
    env = DiscreteSynchronEnvironment(
        n_periods=100,
        n_prices=100,
        history_after=0,
        agents=[Qlearning(decision=EpsilonGreedy(eps=1.0)), Qlearning(decision=EpsilonGreedy(eps=1.0))],
    )
    env.play_game()
    prices = np.array(env.price_history)
    assert np.all(prices[:, 1] == prices[:, 0]) == False  # noqa E712
