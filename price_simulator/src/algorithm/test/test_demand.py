from price_simulator.src.algorithm.demand import LogitDemand, PrisonersDilemmaDemand


def test_logit_demand():
    assert all(
        q > 0.0 for q in LogitDemand().get_quantities((0.1, 0.3, 10.4), (0.5, 0.5, 0.5))
    ), "Negative quantities in logit demand"
    assert (
        LogitDemand().get_quantities((1, 2), (3, 2))[0] > LogitDemand().get_quantities((1, 2), (3, 2))[1]
    ), "First product should be bought more often than the second (is cheaper and better quality)"
    assert (
        LogitDemand().get_quantities((1, 1), (3, 2))[0] > LogitDemand().get_quantities((1, 1), (3, 2))[1]
    ), "First product should be bought more often than the second (better quality)"
    assert (
        LogitDemand().get_quantities((1, 2), (1, 1))[0] > LogitDemand().get_quantities((1, 2), (1, 1))[1]
    ), "First product should be bought more often than the second (cheaper)"


def test_prisoners():
    assert PrisonersDilemmaDemand().get_quantities((1, 1), (1.0, 1.0))[0] == 0.5
    assert PrisonersDilemmaDemand().get_quantities((1, 0), (100.0, 10.0))[0] == 0
    assert PrisonersDilemmaDemand().get_quantities((0.5, 1), (0.0, 0.0))[0] == 1
