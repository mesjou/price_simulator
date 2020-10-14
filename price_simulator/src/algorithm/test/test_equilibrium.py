import numpy as np
from price_simulator.src.algorithm.demand import LogitDemand, PrisonersDilemmaDemand
from price_simulator.src.algorithm.equilibrium import EquilibriumCalculator
from scipy.optimize import minimize


def test_profit():
    assert (
        EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.5, outside_quality=1.0)).profit(
            4.0, np.array([10.0, 10.0]), np.array([2.0, 1.0]), np.array([4.0, 1.0]), 0
        )
        == 0.0  # noqa W503
    )
    assert EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.5, outside_quality=1.0)).profit(
        4.1, np.array([10.0, 10.0]), np.array([2.0, 1.0]), np.array([4.0, 1.0]), 0
    ) < EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.5, outside_quality=1.0)).profit(
        4.0, np.array([10.0, 10.0]), np.array([2.0, 1.0]), np.array([4.0, 1.0]), 0
    )
    assert (
        EquilibriumCalculator(demand=PrisonersDilemmaDemand()).profit(
            5.0, np.array([10.0, 10.0]), np.array([2.0, 1.0]), np.array([4.0, 1.0]), 0
        )
        == -1.0  # noqa W503
    )


def test_reaction_function():
    assert (
        EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.5, outside_quality=1.0)).reaction_function(
            np.array([10.0, 10.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), 0
        )
        <= 10.0  # noqa W503
    )

    assert EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.5, outside_quality=1.0)).reaction_function(
        np.array([10.0, 10.0]), np.array([2.0, 1.0]), np.array([4.0, 1.0]), 0
    ) == EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.5, outside_quality=1.0)).reaction_function(
        np.array([10.0, 10.0]), np.array([1.0, 2.0]), np.array([1.0, 4.0]), 1
    )

    best_response = EquilibriumCalculator(
        demand=LogitDemand(price_sensitivity=0.5, outside_quality=1.0)
    ).reaction_function(np.array([10.0, 10.0]), np.array([2.0, 1.0]), np.array([4.0, 1.0]), 0)
    assert EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.5, outside_quality=1.0)).profit(
        best_response, np.array([10.0, 10.0]), np.array([1.0, 2.0]), np.array([1.0, 1.0]), 1
    ) > EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.5, outside_quality=1.0)).profit(
        best_response - 0.001, np.array([10.0, 10.0]), np.array([1.0, 2.0]), np.array([1.0, 1.0]), 1
    )


def test_vector_reaction():
    assert (
        np.round(
            EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.8, outside_quality=1.0)).get_nash_equilibrium(
                [1.2, 1.0, 0.8], [1.0, 0.9, 0.8]
            )[0],
            5,
        )
        == 1.88108  # noqa W503
    )
    assert (
        np.round(
            EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.8, outside_quality=1.0)).get_nash_equilibrium(
                [1.0, 1.0], [1000.0, 10000.0]
            )[0],
            5,
        )
        > 0.0  # noqa W503
    )
    assert (
        np.round(
            EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.5, outside_quality=1.0)).get_nash_equilibrium(
                [1.0, 1.0], [1.0, 1.0],
            )[0],
            4,
        )
        == 1.5227  # noqa W503
    )
    assert (
        np.round(
            EquilibriumCalculator(
                demand=LogitDemand(price_sensitivity=0.005, outside_quality=1.0)
            ).get_nash_equilibrium([1.0], [1.0],)[0],
            4,
        )
        == 1.005  # noqa W503
    )

    def profit(price, cost=1.0, quality=1.0):
        quantity = LogitDemand(price_sensitivity=0.005, outside_quality=1.0).get_quantities((price,), (quality,))[0]
        return -1 * (price - cost) * quantity

    assert np.round(
        EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.005, outside_quality=1.0)).get_nash_equilibrium(
            [1.0], [1.0],
        )[0],
        4,
    ) == np.round(  # noqa W503
        minimize(profit, np.array([1]), method="nelder-mead", options={"xatol": 1e-8}).x, 4
    )


def test_equilibrium_calculation():
    """See Anderson & de Palma (1992) for theoretical equilibrium as outside quality goes to -inf."""
    a0 = -1000000000
    mcs = [1.0, 1.0]
    mu = 0.5
    assert (
        np.around(
            EquilibriumCalculator(demand=LogitDemand(price_sensitivity=mu, outside_quality=a0)).get_nash_equilibrium(
                mcs, mcs
            ),
            4,
        )
        == np.around(np.asarray(mcs) + (mu * len(mcs)) / (len(mcs) - 1), 4)  # noqa W503
    ).all()

    mcs = [1.0, 1.0, 1.0, 1.0]
    mu = 0.1
    assert (
        np.around(
            EquilibriumCalculator(demand=LogitDemand(price_sensitivity=mu, outside_quality=a0)).get_nash_equilibrium(
                mcs, mcs
            ),
            4,
        )
        == np.around(np.asarray(mcs) + (mu * len(mcs)) / (len(mcs) - 1), 4)  # noqa W503
    ).all()

    # more loyal consumers thus price increase
    assert (
        EquilibriumCalculator(demand=LogitDemand(price_sensitivity=0.8, outside_quality=1.0)).get_nash_equilibrium(
            mcs, mcs
        )
        >= EquilibriumCalculator(  # noqa W503
            demand=LogitDemand(price_sensitivity=0.5, outside_quality=1.0)
        ).get_nash_equilibrium(mcs, mcs)
    ).all()

    # for mu -> inf consumers become equally distributed across products
    demand = np.around(
        EquilibriumCalculator(demand=LogitDemand(price_sensitivity=10000.0, outside_quality=1.0)).get_nash_equilibrium(
            mcs, mcs
        ),
        3,
    )
    assert np.all(demand == demand[0])
