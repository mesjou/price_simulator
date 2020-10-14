import copy
import random

import numpy as np
from price_simulator.src.algorithm.agents.approximate import DDQN, DiffDQN, DQN
from price_simulator.src.algorithm.agents.buffer import ReplayBuffer
from price_simulator.src.algorithm.policies import EpsilonGreedy


# TEST DQN
def test_deep_scalings():
    price_range = np.arange(1.4, 1.7, 0.001)
    scaled_prices = DQN().scale(tuple(price_range), list(price_range))
    assert scaled_prices.max() == 1 and scaled_prices.min() == 0


def test_play_price():
    agent = DQN(decision=EpsilonGreedy(eps=0.0))
    assert agent.play_price((1.0,), [1.0, 1.0], 0, 0) == 1.0

    agent = DiffDQN(decision=EpsilonGreedy(eps=0.0))
    assert agent.play_price((1.0,), [1.0, 1.0], 0, 0) == 1.0


def test_play_optimal_action():
    possible_prices = [1.0, 2.0]
    agent = DQN(decision=EpsilonGreedy(eps=0.0), replay_memory=ReplayBuffer(50), batch_size=1)
    state = tuple(random.choices(possible_prices, k=2))
    agent.play_price(state, possible_prices, 0, 0)
    for _ in range(10):
        agent.learn(
            previous_reward=1.0,
            reward=10.0,
            previous_action=0.0,
            action=np.float64(1.0),
            action_space=possible_prices,
            previous_state=state,
            state=state,
            next_state=state,
        )
        agent.learn(
            previous_reward=1.0,
            reward=-10.0,
            previous_action=0.0,
            action=np.float64(2.0),
            action_space=possible_prices,
            previous_state=state,
            state=state,
            next_state=state,
        )
    assert agent.play_price(state, possible_prices, 1, 1) == 1.0


def test_delayed_learning():
    possible_prices = [0.0, 1.0, 2.0, 3.0]
    agent = DQN(decision=EpsilonGreedy(eps=0.0), replay_memory=ReplayBuffer(50), batch_size=1)
    state = tuple(random.choices(possible_prices, k=2))
    agent.play_price(state, possible_prices, 0, 0)
    weights_before_learning = copy.deepcopy(agent.qnetwork_local.get_weights())
    agent.learn(
        previous_reward=1.0,
        reward=10.0,
        previous_action=0.0,
        action=np.float64(1.0),
        action_space=possible_prices,
        previous_state=state,
        state=state,
        next_state=state,
    )
    assert np.equal(np.array(weights_before_learning[0]), np.array(agent.qnetwork_local.get_weights()[0])).all()
    agent.learn(
        previous_reward=1.0,
        reward=10.0,
        previous_action=0.0,
        action=np.float64(1.0),
        action_space=possible_prices,
        previous_state=state,
        state=state,
        next_state=state,
    )
    assert (
        np.equal(np.array(weights_before_learning[0]), np.array(agent.qnetwork_local.get_weights()[0])).all()
        == False  # noqa E712
    )


def test_update_network():
    possible_prices = [0.0, 1.0, 2.0, 3.0]
    agent = DQN(decision=EpsilonGreedy(eps=0.0), replay_memory=ReplayBuffer(50), batch_size=1, update_target_after=10)
    state = tuple(random.choices(possible_prices, k=2))
    agent.play_price(state, possible_prices, 0, 0)
    assert np.isclose(
        np.array(agent.qnetwork_local.get_weights()[0]), np.array(agent.qnetwork_target.get_weights()[0])
    ).all()
    for _ in range(10):
        agent.learn(
            previous_reward=1.0,
            reward=10.0,
            previous_action=0.0,
            action=np.float64(1.0),
            action_space=possible_prices,
            previous_state=state,
            state=state,
            next_state=state,
        )
    assert (
        np.equal(
            np.array(agent.qnetwork_local.get_weights()[0]), np.array(agent.qnetwork_target.get_weights()[0])
        ).all()
        == False  # noqa E712, W503
    )
    agent.learn(
        previous_reward=1.0,
        reward=10.0,
        previous_action=0.0,
        action=np.float64(1.0),
        action_space=possible_prices,
        previous_state=state,
        state=state,
        next_state=state,
    )
    assert np.isclose(
        np.array(agent.qnetwork_local.get_weights()[0]), np.array(agent.qnetwork_target.get_weights()[0])
    ).all()


# TEST DiffDQN
def test_deep_scalings_diff():
    price_range = np.arange(1.4, 1.7, 0.001)
    scaled_prices = DiffDQN().scale(tuple(price_range), list(price_range))
    assert scaled_prices.max() == 1 and scaled_prices.min() == 0


def test_play_price_diff():
    agent = DiffDQN(decision=EpsilonGreedy(eps=0.0))
    assert agent.play_price((1.0,), [1.0, 1.0], 0, 0) == 1.0

    agent = DiffDQN(decision=EpsilonGreedy(eps=0.0))
    assert agent.play_price((1.0,), [1.0, 1.0], 0, 0) == 1.0


def test_play_optimal_action_diff():
    possible_prices = [1.0, 2.0]
    agent = DiffDQN(decision=EpsilonGreedy(eps=0.0), replay_memory=ReplayBuffer(50), batch_size=1)
    state = tuple(random.choices(possible_prices, k=2))
    agent.play_price(state, possible_prices, 0, 0)
    for _ in range(10):
        agent.learn(
            previous_reward=1.0,
            reward=10.0,
            previous_action=0.0,
            action=np.float64(1.0),
            action_space=possible_prices,
            previous_state=state,
            state=state,
            next_state=state,
        )
        agent.learn(
            previous_reward=1.0,
            reward=-10.0,
            previous_action=0.0,
            action=np.float64(2.0),
            action_space=possible_prices,
            previous_state=state,
            state=state,
            next_state=state,
        )
    assert agent.play_price(state, possible_prices, 1, 1) == 1.0


def test_delayed_learning_diff():
    possible_prices = [0.0, 1.0, 2.0, 3.0]
    agent = DiffDQN(decision=EpsilonGreedy(eps=0.0), replay_memory=ReplayBuffer(50), batch_size=1)
    state = tuple(random.choices(possible_prices, k=2))
    agent.play_price(state, possible_prices, 0, 0)
    weights_before_learning = copy.deepcopy(agent.qnetwork_local.get_weights())
    agent.learn(
        previous_reward=1.0,
        reward=10.0,
        previous_action=0.0,
        action=np.float64(1.0),
        action_space=possible_prices,
        previous_state=state,
        state=state,
        next_state=state,
    )
    assert np.isclose(np.array(weights_before_learning[0]), np.array(agent.qnetwork_local.get_weights()[0])).all()
    agent.learn(
        previous_reward=1.0,
        reward=10.0,
        previous_action=0.0,
        action=np.float64(1.0),
        action_space=possible_prices,
        previous_state=state,
        state=state,
        next_state=state,
    )
    assert (
        np.isclose(np.array(weights_before_learning[0]), np.array(agent.qnetwork_local.get_weights()[0])).all()
        == False  # noqa E712
    )


def test_update_network_diff():
    possible_prices = [0.0, 1.0, 2.0, 3.0]
    agent = DiffDQN(
        decision=EpsilonGreedy(eps=0.0), replay_memory=ReplayBuffer(50), batch_size=1, update_target_after=10
    )
    state = tuple(random.choices(possible_prices, k=2))
    agent.play_price(state, possible_prices, 0, 0)
    assert np.isclose(
        np.array(agent.qnetwork_local.get_weights()[4]), np.array(agent.qnetwork_target.get_weights()[4])
    ).all()
    for _ in range(10):
        agent.learn(
            previous_reward=1.0,
            reward=10.0,
            previous_action=0.0,
            action=np.float64(1.0),
            action_space=possible_prices,
            previous_state=state,
            state=state,
            next_state=state,
        )
    assert (
        np.isclose(
            np.array(agent.qnetwork_local.get_weights()[0]), np.array(agent.qnetwork_target.get_weights()[0])
        ).all()
        == False  # noqa E712, W503
    )
    agent.learn(
        previous_reward=1.0,
        reward=10.0,
        previous_action=0.0,
        action=np.float64(1.0),
        action_space=possible_prices,
        previous_state=state,
        state=state,
        next_state=state,
    )
    assert np.isclose(
        np.array(agent.qnetwork_local.get_weights()[0]), np.array(agent.qnetwork_target.get_weights()[0])
    ).all()


def test_random_action_selection():
    np.random.seed(1)
    possible_prices = [1.0, 2.0]
    agent = DiffDQN(decision=EpsilonGreedy(eps=0.0), replay_memory=ReplayBuffer(2), batch_size=1)
    state = tuple(random.choices(possible_prices, k=2))
    agent.play_price(state, possible_prices, 0, 0)

    # same action values (all weights are zreo)
    agent.play_price(state, possible_prices, 0, 0)
    weights = agent.qnetwork_local.get_weights()
    for w in weights:
        w[w != 0.0] = 0.0
    agent.qnetwork_local.set_weights(weights)
    played_prices = []
    for _ in range(10):
        played_prices.append(agent.play_price(state, possible_prices, 0, 0))
    assert len(set(played_prices)) == 2

    # learn that 1.0 is better
    for _ in range(10):
        agent.learn(
            previous_reward=1.0,
            reward=10.0,
            previous_action=0.0,
            action=np.float64(1.0),
            action_space=possible_prices,
            previous_state=state,
            state=state,
            next_state=state,
        )
    played_prices = []
    for _ in range(10):
        played_prices.append(agent.play_price(state, possible_prices, 0, 0))
    assert len(set(played_prices)) == 1
    assert list(set(played_prices))[0] == 1


def test_ddqn():
    possible_prices = [0.0, 1.0, 2.0, 3.0]
    agent = DDQN(decision=EpsilonGreedy(eps=0.0), replay_memory=ReplayBuffer(50), batch_size=2)
    state = tuple(random.choices(possible_prices, k=2))
    agent.play_price(state, possible_prices, 0, 0)
    weights_before_learning = copy.deepcopy(agent.qnetwork_local.get_weights())
    agent.learn(
        previous_reward=1.0,
        reward=10.0,
        previous_action=0.0,
        action=np.float64(1.0),
        action_space=possible_prices,
        previous_state=state,
        state=state,
        next_state=state,
    )
    assert np.isclose(np.array(weights_before_learning[0]), np.array(agent.qnetwork_local.get_weights()[0])).all()
    for i in range(5):
        agent.learn(
            previous_reward=1.0,
            reward=10.0,
            previous_action=0.0,
            action=np.float64(1.0),
            action_space=possible_prices,
            previous_state=state,
            state=state,
            next_state=state,
        )
    assert (
        np.isclose(np.array(weights_before_learning[0]), np.array(agent.qnetwork_local.get_weights()[0])).all()
        == False  # noqa E712
    )
