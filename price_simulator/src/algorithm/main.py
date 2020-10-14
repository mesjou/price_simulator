import price_simulator.src.utils.analyzer as Analyzer
from price_simulator.src.algorithm.agents.approximate import DiffDQN
from price_simulator.src.algorithm.agents.simple import AlwaysDefectAgent
from price_simulator.src.algorithm.agents.tabular import Qlearning
from price_simulator.src.algorithm.demand import LogitDemand
from price_simulator.src.algorithm.environment import DiscreteSynchronEnvironment
from price_simulator.src.algorithm.policies import DecreasingEpsilonGreedy


def run():
    dqn_env = DiscreteSynchronEnvironment(
        markup=0.1,
        n_periods=100,
        possible_prices=[],
        n_prices=15,
        demand=LogitDemand(outside_quality=0.0, price_sensitivity=0.25),
        history_after=50,
        agents=[
            DiffDQN(
                discount=0.95, learning_rate=0.001, decision=DecreasingEpsilonGreedy(), marginal_cost=1.0, quality=2.0,
            ),
            Qlearning(
                discount=0.95, learning_rate=0.125, decision=DecreasingEpsilonGreedy(), marginal_cost=1.0, quality=2.0,
            ),
            AlwaysDefectAgent(marginal_cost=1.0, quality=2.0),
        ],
    )
    dqn_env.play_game()
    Analyzer.analyze(dqn_env)


if __name__ == "__main__":
    run()
