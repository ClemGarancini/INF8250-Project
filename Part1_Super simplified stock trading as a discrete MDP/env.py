import numpy as np
import pandas as pd
from ..baseClasses import baseEnv


class StockTradingEnvironment(baseEnv.TradingEnvironment):
    def __init__(self, pepsi_file: str, cola_file: str):
        self.observation_dim = 5
        self.action_dim = 4
        super.__init__(pepsi_file, cola_file, self.observation_dim, self.action)

        self.balance_unit = 10
        self.max_balance_units = 10
        self.max_shares_per_stock = 5

        self.state_space = (
            11 * 6 * 6 * 2 * 2
        )  # 11 balances, 6 shares each for Pepsi and Cola, 2 trends each

        self.state = np.array(
            [15, 0, 0, 0, 0]
        )  # Initial state: [Balance, Pepsi shares, Cola shares, Trend of Pepsi, Trend of Cola]

    def convert_state_to_index(self, state: np.array) -> int:
        balance_index, pepsi_shares, cola_shares, trend_pepsi, trend_cola = state
        index = balance_index
        index += pepsi_shares * 11
        index += cola_shares * 11 * 6
        index += trend_pepsi * 11 * 6 * 6
        index += trend_cola * 11 * 6 * 6 * 2
        return int(index)

    def convert_index_to_state(self, index: int) -> np.array:
        trend_cola = index // (11 * 6 * 6 * 2)
        index %= 11 * 6 * 6 * 2
        trend_pepsi = index // (11 * 6 * 6)
        index %= 11 * 6 * 6
        cola_shares = index // (11 * 6)
        index %= 11 * 6
        pepsi_shares = index // 11
        balance_index = index % 11
        return np.array(
            [balance_index, pepsi_shares, cola_shares, trend_pepsi, trend_cola]
        )

    def _get_indicator(self, step: int, stock_data: pd.DataFrame) -> int:
        trend = self._get_stock_trend(step, stock_data)
        return int(trend > 0)

    def _trade(self, action: int) -> tuple(int, int, int):
        """
        Trade the desired amount

        Args:
            action: int, The trade order, can be
                - 0: Sell all
                - 1: Hold
                - 2: Buy Pepsi
                - 3: Buy Cola
        """
        balance_units, shares_pepsi, shares_cola = (
            self.state[0],
            self.state[1],
            self.state[2],
        )
        balance = balance_units * self.balance_unit
        pepsi_price = self._get_stock_price(self.current_step, self.pepsi_data)
        cola_price = self._get_stock_price(self.current_step, self.cola_data)

        if action == 0:  # Sell all
            balance += shares_pepsi * pepsi_price + shares_cola * cola_price
            shares_pepsi, shares_cola = 0, 0
        elif action == 2:  # Buy Pepsi
            quantity = min(
                balance // pepsi_price, self.max_shares_per_stock - shares_pepsi
            )
            shares_pepsi += quantity
            balance -= quantity * pepsi_price
        elif action == 3:  # Buy Cola
            quantity = min(
                balance // cola_price, self.max_shares_per_stock - shares_cola
            )
            shares_cola += quantity
            balance -= quantity * cola_price

        # Update state with rounded balance
        new_balance = max(int(balance / self.balance_unit), 0), self.max_balance_units

        return new_balance, shares_pepsi, shares_cola

    def step(self, action: int) -> tuple(int, float, bool):
        """
        Update the environment with action taken by the agent

        Args:
            action: int, The action taken by the agent

        Returns:
            next_state_index: int, The index of the next state
            reward: float, The reward returned by the environment
            done: bool, Is the episode terminated or truncated
        """
        if action not in range(self.action_space):
            raise ValueError(
                "Invalid action. Action should be in {}".format(
                    list(range(self.action_space))
                )
            )

        new_balance, new_shares_pepsi, new_shares_cola = self._trade(action)
        self.current_step += 1

        # Update state
        self._update_state(new_balance, new_shares_pepsi, new_shares_cola)

        # Compute reward and update portfolio value
        reward = self._compute_reward()

        done = self.current_step >= len(self.pepsi_data) - 1
        next_state_index = self.convert_state_to_index(self.state)
        return next_state_index, reward, done

    def _compute_portfolio_value(self) -> float:
        balance = self.state[0] * self.balance_unit
        pepsi_holdings_value = self.state[1] * self._get_stock_price(
            self.current_step, self.pepsi_data
        )
        cola_holdings_value = self.state[2] * self._get_stock_price(
            self.current_step, self.cola_data
        )
        return balance + pepsi_holdings_value + cola_holdings_value

    def reset(self) -> None:
        self.state = np.array([15, 0, 0, 0, 0])  # Reset to initial state
        self.current_step = 0
        self.previous_portfolio_value = self._calculate_portfolio_value()
        return self.convert_state_to_index(self.state)
