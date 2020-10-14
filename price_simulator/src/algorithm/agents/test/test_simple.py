from price_simulator.src.algorithm.agents.simple import AlwaysDefectAgent, PenetrationPricer, PremiumPricer


def test_play_price():
    agent = AlwaysDefectAgent()
    assert agent.play_price((), [1.0, 2.0], 0, 0) == 1.0

    agent = PremiumPricer()
    assert agent.play_price((1.0, 1.0, 1.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 2.0
    assert agent.play_price((1.0, 2.0, 3.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 3.0
    assert agent.play_price((1.0, 2.0, 2.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 3.0
    assert agent.play_price((1.0, 2.0, 1.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 3.0
    assert agent.play_price((4.0, 2.0, 3.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 4.0
    assert agent.play_price((4.0, 2.0, 3.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 4.0
    assert agent.play_price((4.0, 4.0, 4.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 4.0
    assert agent.play_price((3.0, 2.0, 4.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 4.0

    agent = PenetrationPricer()
    assert agent.play_price((1.0, 1.0, 1.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 1.0
    assert agent.play_price((1.0, 1.0, 3.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 1.0
    assert agent.play_price((4.0, 4.0, 2.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 3.0
    assert agent.play_price((2.0, 3.0, 1.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 1.0
    assert agent.play_price((4.0, 4.0, 4.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 3.0
    assert agent.play_price((4.0, 1.0, 3.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 1.0
    assert agent.play_price((3.0, 4.0, 2.0), [1.0, 2.0, 3.0, 4.0], 0, 0) == 2.0
