import abc
import math
from typing import Tuple

import attr


@attr.s
class MarketDemandStrategy(metaclass=abc.ABCMeta):
    """Top-level interface for all market demand modulation."""

    @abc.abstractmethod
    def get_quantities(self, prices: Tuple, qualities: Tuple) -> Tuple:
        """Return demand quanities for each price"""
        raise NotImplementedError()


@attr.s
class PrisonersDilemmaDemand(MarketDemandStrategy):
    """Market demand modulation for prisoners dilemma"""

    def get_quantities(self, prices: Tuple, qualities: Tuple) -> Tuple:
        assert len(prices) == 2, "Prisoners dilemma could only be played with two agents"
        if prices[0] == prices[1]:
            return 0.5, 0.5
        elif prices[0] > prices[1]:
            return 0.0, 1.0
        else:
            return 1.0, 0.0


@attr.s
class LogitDemand(MarketDemandStrategy):
    """Market demand modulation for logit demand"""

    price_sensitivity: float = attr.ib(0.25)  # lower more sensitive
    outside_quality: float = attr.ib(0.0)

    @price_sensitivity.validator
    def check_price_sensitivity(self, attribute, value):
        if not 0.005 <= value:
            raise ValueError("Price Sensitivity must lie above 0.005")

    def get_quantities(self, prices: Tuple, qualities: Tuple) -> Tuple:
        denominator = sum((math.exp((a - p) / self.price_sensitivity) for a, p in zip(qualities, prices))) + math.exp(
            self.outside_quality / self.price_sensitivity
        )
        return tuple(math.exp((a - p) / self.price_sensitivity) / denominator for a, p in zip(qualities, prices))


@attr.s
class ConstantDemand(MarketDemandStrategy):
    """Constant demand module for testing only."""

    default_quantity: float = attr.ib(default=1.0)

    def get_quantities(self, prices: Tuple, qualities: Tuple) -> Tuple:
        return tuple(self.default_quantity for _ in prices)
