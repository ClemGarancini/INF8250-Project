import numpy as np
import pandas as pd
from abc import abstractmethod


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

    def step(self, action: int) -> tuple(list | int, float, bool):
        """
        Update the environment with action taken by the agent

        Args:
            action: int, The action taken by the agent

        Returns:
            next_state_index: int, The index of the next state
            reward: float, The reward returned by the environment
            done: bool, Is the episode terminated or truncated
        """
        self._trade(action)
        self.current_step += 1
        done = self.current_step >= len(self.pepsi_data) - 1
        reward = self._compute_reward()

        self._update_state()
        return self.state, reward, done

    @abstractmethod
    def _trade(self, action: int) -> tuple(int | float, int, int):
        # Not Implemented
        raise NotImplementedError

    @abstractmethod
    def _get_indicator(self, stock_data: pd.DataFrame) -> int | float:
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

    def _update_state(
        self, new_balance: int | float, new_shares_pepsi: int, new_shares_cola: int
    ) -> None:
        self.state = [
            new_balance,
            new_shares_pepsi,
            new_shares_cola,
            self._get_indicator(self.pepsi_data),
            self._get_indicator(self.cola_data),
        ]

    def _compute_reward(self) -> float:
        """
        Computes and updates the portfolio value and returns the reward associated
        The reward is the difference between the current portfolio value and the previous one
        """
        current_portfolio_value = self._calculate_portfolio_value()
        reward = current_portfolio_value - self.previous_portfolio_value
        self.previous_portfolio_value = current_portfolio_value
        return reward

    def _compute_portfolio_value(self) -> float:
        """
        Computes the current portfolio value as
            V = b + price_p * share_p + price_c * share_c
        Where
            V is the portfolio value
            b the remaining balance
            price_p and share_p (resp. price_c) the pepsi (resp. coca) stock price and held amount of shares
        """
        pepsi_price = self._get_stock_price(self.current_step, self.pepsi_data)
        cola_price = self._get_stock_price(self.current_step, self.cola_data)
        return self.state[0] + self.state[1] * pepsi_price + self.state[2] * cola_price

    @abstractmethod
    def reset(self) -> None:
        # Not Implemented
        raise NotImplementedError
