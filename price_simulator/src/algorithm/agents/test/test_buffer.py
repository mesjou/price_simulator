import numpy as np
from price_simulator.src.algorithm.agents.approximate import DQN
from price_simulator.src.algorithm.agents.buffer import ReplayBuffer
from price_simulator.src.algorithm.environment import DiscreteSynchronEnvironment
from price_simulator.src.algorithm.policies import EpsilonGreedy


def test_init():
    buffer = ReplayBuffer(buffer_size=100)
    assert len(buffer) == 0


def test_add_and_sample():
    buffer = ReplayBuffer(buffer_size=100)
    buffer.add(state=1, action=2, reward=3, next_state=4)
    assert len(buffer) == 1
    assert buffer.sample(batch_size=1) == (1, 2, 3, 4)

    states = np.array([[1, 1, 1]] * 20)
    actions = np.array([[2, 2, 2]] * 20)
    rewards = np.array([[3, 3, 3]] * 20)
    next_states = np.array([[4, 4, 4]] * 20)
    buffer = ReplayBuffer(buffer_size=100)
    for _ in range(100):
        buffer.add(state=[1, 1, 1], action=[2, 2, 2], reward=[3, 3, 3], next_state=[4, 4, 4])
    assert len(buffer) == 100
    assert np.array(buffer.sample(batch_size=20) == np.array([states, actions, rewards, next_states])).all()

    buffer = ReplayBuffer(buffer_size=10)
    for _ in range(10):
        buffer.add(state=[1, 1, 1], action=[2, 2, 2], reward=[3, 3, 3], next_state=[4, 4, 4])
    buffer.add(state=[100, 1, 1], action=[2, 2, 2], reward=[3, 3, 3], next_state=[4, 4, 4])
    assert len(buffer) == 10
    assert np.array(buffer.sample(batch_size=10) != np.array([states, actions, rewards, next_states])).all()


def test_factory_init():
    env = DiscreteSynchronEnvironment(
        n_periods=1,
        n_prices=100,
        agents=[
            DQN(decision=EpsilonGreedy(eps=1.0), marginal_cost=0.0),
            DQN(decision=EpsilonGreedy(eps=1.0), marginal_cost=2.0),
        ],
    )
    env.play_game()
    assert np.array(env.agents[0].replay_memory.sample(1)[0] == env.agents[1].replay_memory.sample(1)[0]).all()
    assert np.array(env.agents[0].replay_memory.sample(1)[3] == env.agents[1].replay_memory.sample(1)[3]).all()
    assert np.array(env.agents[0].replay_memory.sample(1)[2] != env.agents[1].replay_memory.sample(1)[2]).all()
