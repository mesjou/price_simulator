import numpy as np
from price_simulator.src.utils.storage import Storage


def test_set_up():
    storage = Storage()
    storage.set_up(3, 1000, 10)
    assert storage.update_steps == 100
    assert len(storage.running_rewards) == 3
    assert len(storage.running_rewards) == 3
    assert len(storage.running_rewards) == 3

    storage = Storage()
    storage.set_up(3, 10, 20)
    assert storage.update_steps == 1


def test_incremental_update():
    avg = np.array([0])
    for cnt in range(100):
        cnt += 1
        avg = Storage().incremental_update(np.array([10, 20]), avg, cnt)
    assert np.all(avg == np.array([10, 20]))


def test_observe():
    n_periods = 100
    desired_length = 10
    n_agents = 2
    storage = Storage()
    storage.set_up(n_agents, n_periods, desired_length)
    for _ in range(n_periods):
        storage.observe(np.array([10, 20]), np.array([30, 40]), np.array([50, 0]))

    assert storage.average_rewards.shape == (desired_length, n_agents)
    assert storage.average_actions.shape == (desired_length, n_agents)
    assert storage.average_quantities.shape == (desired_length, n_agents)

    assert np.all(storage.average_rewards == np.repeat(np.array([[10, 20]]), repeats=desired_length, axis=0))
    assert np.all(storage.average_actions == np.repeat(np.array([[30, 40]]), repeats=desired_length, axis=0))
    assert np.all(storage.average_quantities == np.repeat(np.array([[50, 0]]), repeats=desired_length, axis=0))
