import numpy as np
import random


class QLearningAgent:
    def __init__(
        self,
        state_space: int,
        action_space: int,
        learning_rate=0.01,
        discount_factor=0.99,
        exploration_rate=1.0,
    ):
        # Env
        self.state_space = state_space
        self.action_space = action_space

        # Learning
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.q_table = np.zeros((state_space, action_space))

        # Monitoring
        self.q_table_history = np.zeros((1, state_space, action_space))

    def __str__(self) -> str:
        info = """The agent is using Q-Learning algorithm\n
        It is working on Simplified Discrete Trading Environment (Experiment 1)\n
        The current Q Table values can be fetch by calling get_current_q_values() method\n
        The history of Q Table values can be fetch by calling get_history_q_values() method"""
        return info

    def choose_action(self, state_index: int) -> int:
        """
        Choose action according to current Q Table
        """
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        return np.argmax(self.q_table[state_index])

    def train(
        self, state_index: int, action: int, reward: float, next_state_index: int
    ) -> None:
        """
        Update Q values following Q Learning classical update
        """
        assert 0 <= state_index < self.state_space, "Invalid state_index"
        assert 0 <= next_state_index < self.state_space, "Invalid next_state_index"

        q_value = self.q_table[state_index, action]

        # Target = Rt + Gamma x max(Q[S(t+1), a])
        target = reward + self.discount_factor * np.max(self.q_table[next_state_index])

        # Q[S(t), action] =  Q[S(t), action] + alpha x (Rt + Gamma x max(Q[S(t+1), a]) - Q[S(t), action])
        self.q_table[state_index, action] += self.learning_rate * (target - q_value)

        # Store Q table
        self.q_table_history = np.concatenate((self.q_table_history, [self.q_table]))

        self.exploration_rate = max(
            self.exploration_rate * self.exploration_decay, self.exploration_min
        )

    def get_current_q_values(self) -> np.ndarray:
        """
        Fetch the current Q Table as a numpy array of shape:
            (number of possible states, number of possible actions)
        """
        return self.q_table

    def get_history_q_values(self) -> np.ndarray:
        """
        Fetch the history of Q Tables as a numpy array of shape:
            (number of episodes seen, number of possible states, number of possible actions)
        """
        return self.q_table_history
