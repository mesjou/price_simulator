import random
from typing import List, Tuple

import attr

import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from price_simulator.src.algorithm.agents.buffer import ReplayBuffer
from price_simulator.src.algorithm.agents.simple import AgentStrategy
from price_simulator.src.algorithm.policies import EpsilonGreedy, ExplorationStrategy


@attr.s
class DQN(AgentStrategy):
    """Deep-Q-Netowrks Agent with discounted reward formulation"""

    # Q-Network
    qnetwork_target: keras.models = attr.ib(default=None)
    qnetwork_local: keras.models = attr.ib(default=None)
    update_target_after: int = attr.ib(default=100)
    replay_memory: ReplayBuffer = attr.ib(factory=ReplayBuffer)
    batch_size: int = attr.ib(default=32)
    update_counter: int = attr.ib(default=0)
    hidden_nodes: int = attr.ib(default=32)

    # General
    decision: ExplorationStrategy = attr.ib(factory=EpsilonGreedy)
    discount: float = attr.ib(default=0.95)
    learning_rate: float = attr.ib(default=0.1)

    @discount.validator
    def check_discount(self, attribute, value):
        if not 0 <= value <= 1:
            raise ValueError("Discount factor must lie in [0,1]")

    @learning_rate.validator
    def check_learning_rate(self, attribute, value):
        """For learning_rate = 0, the algorithm does not learn at all.
        For learning_rate = 1, it immediately forgets what it has learned in the past.
        """
        if not 0 <= value < 1:
            raise ValueError("Learning rate must lie in [0,1)")

    def who_am_i(self) -> str:
        # TODO better who am i
        return type(self).__name__ + " (gamma: {}, alpha: {}, policy: {}, quality: {}, mc: {})".format(
            self.discount, self.learning_rate, self.decision.who_am_i(), self.quality, self.marginal_cost
        )

    def play_price(self, state: Tuple[float], action_space: List[float], n_period: int, t: int) -> float:
        """Returns an action by either following greedy policy or experimentation."""

        # init q networks if necessary
        if not self.qnetwork_target or not self.qnetwork_local:
            self.qnetwork_target = self.initialize_network(len(state), len(action_space))
            self.qnetwork_local = self.initialize_network(len(state), len(action_space))
            self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())

        # play action
        if self.decision.explore(n_period, t):
            return random.choice(action_space)
        else:
            action_values = self.qnetwork_local.predict(np.expand_dims(self.scale(state, action_space), axis=0))
            if sum(np.isclose(action_values[0], action_values[0].max())) > 1:
                optimal_action_index = np.random.choice(
                    np.flatnonzero(np.isclose(action_values[0], action_values[0].max()))
                )
            else:
                optimal_action_index = np.argmax(action_values[0])
            return action_space[optimal_action_index]

    def learn(
        self,
        previous_reward: float,
        reward: float,
        previous_action: float,
        action: float,
        action_space: List,
        previous_state: Tuple,
        state: Tuple,
        next_state: Tuple,
    ):
        # store experience in buffer (action is converted to index)
        action = np.where(action_space == action)[0]
        state = self.scale(state, action_space)
        next_state = self.scale(next_state, action_space)
        self.replay_memory.add(state, action, reward, next_state)

        if len(self.replay_memory) > self.batch_size:

            # get training sample
            states, actions, rewards, next_states = self.replay_memory.sample(self.batch_size)

            # Get max predicted Q values (for next states) from target model
            next_optimal_q = np.amax(self.qnetwork_target.predict(next_states), axis=1, keepdims=True)

            # Compute Q targets for current states
            targets = rewards + self.discount * next_optimal_q

            # Get current Q values from local model and update them
            # with better estimates (target) for the played actions
            local_estimates = self.qnetwork_local.predict(states)
            local_estimates[np.arange(len(actions)), actions.flatten()] = targets.flatten()

            # perform gradient descent step on local network
            self.qnetwork_local.fit(states, local_estimates, epochs=1, verbose=0, batch_size=self.batch_size)

            # update target_qnetwork after some periods
            self.update_counter += 1
            if self.update_counter == self.update_target_after:
                self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())
                self.update_counter = 0
                print("I updated my target model")

    def initialize_network(self, n_agents: int, n_actions: int):
        """Create a neuronal network with one output node per possible action"""
        model = Sequential()
        model.add(Dense(int(self.hidden_nodes), input_dim=n_agents, activation="relu"))
        model.add(Dense(int(self.hidden_nodes), activation="relu"))
        model.add(Dense(n_actions, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    @staticmethod
    def scale(inputs: Tuple, action_space: List) -> np.array:
        """Scale float input to range from 0 to 1."""
        max_action = max(action_space)
        min_action = min(action_space)
        return np.multiply(np.divide(np.array(inputs) - min_action, max_action - min_action), 1)


@attr.s
class DiffDQN(DQN):
    """Deep-Q-Netowrks Agent with average reward as target baseline"""

    reward_step_size: float = attr.ib(default=0.01)
    average_reward: float = attr.ib(default=0.0)

    def who_am_i(self) -> str:
        # TODO better who am i
        return type(self).__name__ + " (lambda: {}, alpha: {}, policy: {}, quality: {}, mc: {})".format(
            self.reward_step_size, self.learning_rate, self.decision.who_am_i(), self.quality, self.marginal_cost
        )

    def learn(
        self,
        previous_reward: float,
        reward: float,
        previous_action: float,
        action: float,
        action_space: List,
        previous_state: Tuple,
        state: Tuple,
        next_state: Tuple,
    ):
        # store experience in buffer (action is converted to index)
        action = np.where(action_space == action)[0]
        state = self.scale(state, action_space)
        next_state = self.scale(next_state, action_space)
        self.replay_memory.add(state, action, reward, next_state)

        if len(self.replay_memory) > self.batch_size:

            # get training sample
            states, actions, rewards, next_states = self.replay_memory.sample(self.batch_size)

            # Get max predicted Q values (for next states) from target model
            next_optimal_q = np.amax(self.qnetwork_target.predict(next_states), axis=1, keepdims=True)

            # Compute Q targets for current states
            targets = rewards - self.average_reward + next_optimal_q

            # Get current Q values from local model and update them
            # with better estimates (target) for the played actions
            local_estimates = self.qnetwork_local.predict(states)
            local_estimates[np.arange(len(actions)), actions.flatten()] = targets.flatten()

            # perform gradient descent step on local network
            self.qnetwork_local.fit(states, local_estimates, epochs=1, verbose=0, batch_size=self.batch_size)

            # update average reward
            self.average_reward = self.average_reward + self.reward_step_size * (
                reward
                - self.average_reward  # noqa W503
                + np.amax(self.qnetwork_target.predict(np.array([next_state])))  # noqa W503
                - float(self.qnetwork_target.predict(np.array([state]))[0, action])  # noqa W503
            )

            # update target_qnetwork after some periods
            self.update_counter += 1
            if self.update_counter == self.update_target_after:
                self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())
                self.update_counter = 0
                print("I updated my target model")


@attr.s
class DiffDQN3(DiffDQN):
    def initialize_network(self, n_agents: int, n_actions: int):
        """Create a neuronal network with one output node per possible action"""
        model = Sequential()
        model.add(Dense(int(self.hidden_nodes), input_dim=n_agents, activation="relu"))
        model.add(Dense(int(self.hidden_nodes), activation="relu"))
        model.add(Dense(int(self.hidden_nodes), activation="relu"))
        model.add(Dense(n_actions, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model


@attr.s
class DDQN(DiffDQN):
    """Double-Q-Netowrks Agent that decouples optimal action selection from it's evaluation."""

    def learn(
        self,
        previous_reward: float,
        reward: float,
        previous_action: float,
        action: float,
        action_space: List,
        previous_state: Tuple,
        state: Tuple,
        next_state: Tuple,
    ):
        # store experience in buffer (action is converted to index)
        action = np.where(action_space == action)[0]
        state = self.scale(state, action_space)
        next_state = self.scale(next_state, action_space)
        self.replay_memory.add(state, action, reward, next_state)

        if len(self.replay_memory) > self.batch_size:

            # get training sample
            states, actions, rewards, next_states = self.replay_memory.sample(self.batch_size)

            # Get max predicted Q values (for next states) from target model

            # Get optimal action according to loc
            optimal_actions = np.argmax(self.qnetwork_local.predict(next_states), axis=1)
            next_optimal_q = self.qnetwork_target.predict(next_states)[
                np.arange(np.shape(states)[0]), optimal_actions.flatten()
            ]
            next_optimal_q = np.expand_dims(next_optimal_q, axis=1)

            # Compute Q targets for current states
            targets = rewards - self.average_reward + next_optimal_q

            # Get current Q values from local model and update them
            # with better estimates (target) for the played actions
            local_estimates = self.qnetwork_local.predict(states)
            local_estimates[np.arange(len(actions)), actions.flatten()] = targets.flatten()

            # perform gradient descent step on local network
            self.qnetwork_local.fit(states, local_estimates, epochs=1, verbose=0, batch_size=self.batch_size)

            # update average reward
            self.average_reward = self.average_reward + self.reward_step_size * (
                reward
                - self.average_reward  # noqa W503
                + np.amax(self.qnetwork_target.predict(np.array([next_state])))  # noqa W503
                - float(self.qnetwork_target.predict(np.array([state]))[0, action])  # noqa W503
            )

            # update target_qnetwork after some periods
            self.update_counter += 1
            if self.update_counter == self.update_target_after:
                self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())
                self.update_counter = 0
                print("I updated my target model")
