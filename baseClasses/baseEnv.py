import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Tuple


class TradingEnvironment:
    def __init__(
        self, pepsi_file: str, cola_file: str, observation_dim: int, action_dim: int
    ):
        self.pepsi_data = pd.read_csv(pepsi_file)
        self.cola_data = pd.read_csv(cola_file)

        self.action_space = range(action_dim)
        self.state = np.zeros(observation_dim)

        self.current_step = 0
        self.portfolio_value = self._compute_portfolio_value()

    def step(self, action: int) -> Tuple[np.array, float, bool]:
        """
        Update the environment with action taken by the agent

        Args:
            action: int, The action taken by the agent

        Returns:
            next_state_index: int, The index of the next state
            reward: float, The reward returned by the environment
            done: bool, Is the episode terminated or truncated
        """
        self._check_action_validity(action)
        self.state = self._trade(action)
        self.current_step += 1
        done = self.current_step >= len(self.pepsi_data) - 1
        reward = self._compute_reward()

        return self.state, reward, done

    @abstractmethod
    def reset(self) -> None:
        # Not Implemented
        raise NotImplementedError

    @abstractmethod
    def _trade(self, action: int) -> np.array:
        # Not Implemented
        raise NotImplementedError

    @abstractmethod
    def _get_indicator(self, stock_data: pd.DataFrame) -> int | float:
        # Not Implemented
        raise NotImplementedError

    @abstractmethod
    def _check_action_validity(self, action: int) -> None:
        # Not Implemented
        raise NotImplementedError

    @abstractmethod
    def _compute_portfolio_value(self) -> float:
        # Not Implemented
        raise NotImplementedError

    def _get_stock_price(self, step: int, stock_data: pd.DataFrame) -> float:
        """
        Fetch the price for the given step and stock
        """
        return stock_data.iloc[step]["Close"]

    def _get_stock_trend(self, step: int, stock_data: pd.DataFrame) -> float:
        """
        Fetch the trend for the given stock between the given step and the previous one
        """
        return stock_data.iloc[step]["Close"] - stock_data.iloc[step - 1]["Close"]

    def _compute_reward(self) -> float:
        """
        Computes and updates the portfolio value and returns the reward associated
        The reward is the difference between the current portfolio value and the previous one
        """
        current_portfolio_value = self._calculate_portfolio_value()
        reward = current_portfolio_value - self.previous_portfolio_value
        self.previous_portfolio_value = current_portfolio_value
        return reward
