import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.01, discount_factor=0.99, exploration_rate=1.0):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state_index):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        return np.argmax(self.q_table[state_index])

    def learn(self, state_index, action, reward, next_state_index):
        assert 0 <= state_index < self.state_space, "Invalid state_index"
        assert 0 <= next_state_index < self.state_space, "Invalid next_state_index"

        predict = self.q_table[state_index, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state_index])
        self.q_table[state_index, action] += self.learning_rate * (target - predict)

        self.exploration_rate = max(self.exploration_rate * self.exploration_decay, self.exploration_min)
