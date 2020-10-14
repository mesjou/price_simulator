import copy

from price_simulator.src.algorithm.agents.tabular import Qlearning
from price_simulator.src.algorithm.policies import EpsilonGreedy


def test_play_price():
    agent = Qlearning(decision=EpsilonGreedy(eps=0.0))
    p = agent.play_price((1.0, 1.0), [1.0, 2.0], 0, 0)
    assert p == 1.0 or p == 2.0


def test_initialize_q_matrix():
    # 1 action
    q_matrix = Qlearning().initialize_q_matrix(n_agents=1, actions_space=[1.0])
    assert q_matrix == {(1.0,): {1.0: 0.0}}
    q_matrix = Qlearning().initialize_q_matrix(n_agents=2, actions_space=[1.0])
    assert q_matrix == {(1.0, 1.0): {1.0: 0.0}}
    q_matrix = Qlearning().initialize_q_matrix(n_agents=3, actions_space=[1.0])
    assert q_matrix == {(1.0, 1.0, 1.0): {1.0: 0.0}}

    # 2 actions
    q_matrix = Qlearning().initialize_q_matrix(n_agents=1, actions_space=[1.0, 2.0])
    assert q_matrix == {(1.0,): {1.0: 0.0, 2.0: 0.0}, (2.0,): {1.0: 0.0, 2.0: 0.0}}
    q_matrix = Qlearning().initialize_q_matrix(n_agents=2, actions_space=[1.0, 2.0])
    assert q_matrix == {
        (1.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (1.0, 2.0): {1.0: 0.0, 2.0: 0.0},
        (2.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (2.0, 2.0): {1.0: 0.0, 2.0: 0.0},
    }
    q_matrix = Qlearning().initialize_q_matrix(n_agents=3, actions_space=[1.0, 2.0])
    assert q_matrix == {
        (1.0, 1.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (1.0, 2.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (1.0, 2.0, 2.0): {1.0: 0.0, 2.0: 0.0},
        (1.0, 1.0, 2.0): {1.0: 0.0, 2.0: 0.0},
        (2.0, 1.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (2.0, 2.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (2.0, 2.0, 2.0): {1.0: 0.0, 2.0: 0.0},
        (2.0, 1.0, 2.0): {1.0: 0.0, 2.0: 0.0},
    }

    # 3 actions
    q_matrix = Qlearning().initialize_q_matrix(n_agents=1, actions_space=[1.0, 2.0, 3.0])
    assert q_matrix == {
        (1.0,): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0,): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0,): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
    }
    q_matrix = Qlearning().initialize_q_matrix(n_agents=2, actions_space=[1.0, 2.0, 3.0])
    assert q_matrix == {
        (1.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (1.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (1.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
    }
    q_matrix = Qlearning().initialize_q_matrix(n_agents=3, actions_space=[1.0, 2.0, 3.0])
    assert q_matrix == {
        (1.0, 1.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 1.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 1.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (1.0, 2.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 2.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 2.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (1.0, 3.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 3.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 3.0, 1.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (1.0, 1.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 1.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 1.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (1.0, 2.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 2.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 2.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (1.0, 3.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 3.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 3.0, 2.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (1.0, 1.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 1.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 1.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (1.0, 2.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 2.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 2.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (1.0, 3.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (2.0, 3.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
        (3.0, 3.0, 3.0): {1.0: 0.0, 2.0: 0.0, 3.0: 0.0},
    }


def test_learn():
    q_matrix = {
        (1.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (1.0, 2.0): {1.0: 0.0, 2.0: 0.0},
        (2.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (2.0, 2.0): {1.0: 0.0, 2.0: 0.0},
    }

    # no reward
    agent = Qlearning(q_matrix=copy.deepcopy(q_matrix), discount=0.95, learning_rate=0.1)
    agent.learn(
        reward=0.0,
        state=(1.0, 1.0),
        action=1.0,
        next_state=(1.0, 1.0),
        action_space=[],
        previous_reward=0.0,
        previous_action=0.0,
        previous_state=(None,),
    )
    assert agent.q_matrix == q_matrix

    # learned nothing
    agent = Qlearning(q_matrix=copy.deepcopy(q_matrix), discount=0.95, learning_rate=0.0)
    agent.learn(
        reward=10.0,
        state=(1.0, 1.0),
        action=1.0,
        next_state=(1.0, 1.0),
        action_space=[],
        previous_reward=0.0,
        previous_action=0.0,
        previous_state=(None,),
    )
    assert agent.q_matrix == q_matrix

    q_matrix = {
        (1.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (1.0, 2.0): {1.0: 5.0, 2.0: 0.0},
        (2.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (2.0, 2.0): {1.0: 0.0, 2.0: 0.0},
    }

    # future has no meaning
    agent = Qlearning(q_matrix=copy.deepcopy(q_matrix), discount=0.0, learning_rate=0.9)
    agent.learn(
        reward=10.0,
        state=(1.0, 1.0),
        action=1.0,
        next_state=(1.0, 2.0),
        action_space=[],
        previous_reward=0.0,
        previous_action=0.0,
        previous_state=(None,),
    )
    assert agent.q_matrix[(1.0, 1.0)][1.0] == 9.0
    assert agent.q_matrix == {
        (1.0, 1.0): {1.0: 9.0, 2.0: 0.0},
        (1.0, 2.0): {1.0: 5.0, 2.0: 0.0},
        (2.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (2.0, 2.0): {1.0: 0.0, 2.0: 0.0},
    }

    # future has meaning
    agent = Qlearning(q_matrix=copy.deepcopy(q_matrix), discount=1.0, learning_rate=0.5)
    agent.learn(
        reward=10.0,
        state=(1.0, 1.0),
        action=1.0,
        next_state=(1.0, 2.0),
        action_space=[],
        previous_reward=0.0,
        previous_action=0.0,
        previous_state=(None,),
    )
    assert agent.q_matrix[(1.0, 1.0)][1.0] == 7.5
    assert agent.q_matrix == {
        (1.0, 1.0): {1.0: 7.5, 2.0: 0.0},
        (1.0, 2.0): {1.0: 5.0, 2.0: 0.0},
        (2.0, 1.0): {1.0: 0.0, 2.0: 0.0},
        (2.0, 2.0): {1.0: 0.0, 2.0: 0.0},
    }
