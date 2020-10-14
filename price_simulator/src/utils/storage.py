import copy

import attr

import numpy as np


@attr.s
class Storage:
    counter = attr.ib(default=0)
    update_steps = attr.ib(init=False)
    running_rewards = attr.ib(init=False)
    running_quantities = attr.ib(init=False)
    running_actions = attr.ib(init=False)
    average_rewards = attr.ib(default=None)
    average_quantities = attr.ib(default=None)
    average_actions = attr.ib(default=None)

    def set_up(self, n_agents: int, n_periods: int, desired_length: int = 1000):
        self.reset_running_storage(n_agents)
        self.update_steps = max(1, np.round(n_periods / desired_length, 0))

    def reset_running_storage(self, n_agents: int):
        self.running_rewards = np.array([0] * n_agents)
        self.running_quantities = np.array([0] * n_agents)
        self.running_actions = np.array([0] * n_agents)

    def observe(self, rewards: np.array, actions: np.array, quantities: np.array):
        self.counter += 1
        self.running_rewards = self.incremental_update(rewards, self.running_rewards, self.counter)
        self.running_quantities = self.incremental_update(quantities, self.running_quantities, self.counter)
        self.running_actions = self.incremental_update(actions, self.running_actions, self.counter)

        if self.counter == self.update_steps:
            if self.average_actions is not None:
                self.average_rewards = np.vstack([self.average_rewards, self.running_rewards])
                self.average_actions = np.vstack([self.average_actions, self.running_actions])
                self.average_quantities = np.vstack([self.average_quantities, self.running_quantities])
            else:
                self.average_rewards = copy.deepcopy(self.running_rewards)
                self.average_actions = copy.deepcopy(self.running_actions)
                self.average_quantities = copy.deepcopy(self.running_quantities)

            self.reset_running_storage(len(rewards))
            self.counter = 0

    @staticmethod
    def incremental_update(observation: np.array, average: np.array, cnt: int) -> np.array:
        return average + (observation - average) / cnt

    def print(self):
        print("Rewards:", self.average_rewards)
        print("Prices:", self.average_actions)
        print("Quantities:", self.average_quantities)
