import abc
import math
import random

import attr


@attr.s
class ExplorationStrategy(metaclass=abc.ABCMeta):
    """Top-level interface for Exploration decision."""

    def who_am_i(self) -> str:
        return type(self).__name__

    def epsilon(self, length: int, time: int) -> float:
        raise NotImplementedError

    def explore(self, n_period: int, t: int) -> bool:
        epsilon = self.epsilon(n_period, t)
        return random.choices([True, False], weights=[epsilon, 1 - epsilon])[0]


@attr.s
class EpsilonGreedy(ExplorationStrategy):
    """Exploration decision based on fixed epsilon greedy policy."""

    eps: float = attr.ib(default=0.1)

    @eps.validator
    def check_epsilon(self, attribute, value):
        if not 0 <= value <= 1:
            raise ValueError("Epsilon must lie in [0,1]")

    def who_am_i(self) -> str:
        return type(self).__name__ + " ({})".format(self.eps)

    def epsilon(self, length: int, time: int) -> float:
        return self.eps


@attr.s
class DecreasingEpsilonGreedy(ExplorationStrategy):
    """
    Exploration decision with decreasing epsilon.
    Adapts dynamically to different simulation lengths

    """

    beta: float = attr.ib(default=0.015)

    def epsilon(self, length: int, time: int) -> float:
        """Returns epsilon for time step, such that after half of the time epsilon is 0.001"""
        return (self.beta ** (1.0 / (length / 2))) ** time


@attr.s
class Temperature(ExplorationStrategy):
    """Exploration decision with decreasing epsilon."""

    beta: float = attr.ib(default=0.00001)

    @beta.validator
    def check_beta(self, attribute, value):
        if not 0 <= value:
            raise ValueError("Epsilon must lie in [0,1]")

    def who_am_i(self) -> str:
        return type(self).__name__ + " ({})".format(self.beta)

    def epsilon(self, length: int, time: int) -> float:
        return math.exp(-self.beta * time)


@attr.s
class LinearDecreasing(ExplorationStrategy):
    """Exploration decision with linear decreasing epsilon."""

    def epsilon(self, length: int, time: int) -> float:

        if time / length < 0.6:
            return 1 - (1.0 - 0.1) / (0.6 * length) * time
        else:
            return 0.1 - (0.1 - 0.0001) / (0.4 * length) * (time - (0.6 * length))
