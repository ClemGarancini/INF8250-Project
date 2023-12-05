import numpy as np
import pandas as pd

class StockTradingEnvironment:
    def __init__(self, pepsi_file, cola_file):
        self.pepsi_data = pd.read_csv(pepsi_file)
        self.cola_data = pd.read_csv(cola_file)

        self.balance_unit = 10
        self.max_balance_units = 10
        self.max_shares_per_stock = 5
        self.state_space = 11 * 6 * 6 * 2 * 2  # 11 balances, 6 shares each for Pepsi and Cola, 2 trends each
        self.action_space = 4  # Sell all, Hold, Buy Pepsi, Buy Cola

        self.state = [15, 0, 0, 0, 0]  # Initial state: [Balance, Pepsi shares, Cola shares, Trend of Pepsi, Trend of Cola]
        self.current_step = 0
        self.previous_portfolio_value = self._calculate_portfolio_value()

    def convert_state_to_index(self, state):
        balance_index, pepsi_shares, cola_shares, trend_pepsi, trend_cola = state
        index = balance_index
        index += pepsi_shares * 11
        index += cola_shares * 11 * 6
        index += trend_pepsi * 11 * 6 * 6
        index += trend_cola * 11 * 6 * 6 * 2
        return int(index)

    def convert_index_to_state(self, index):
        trend_cola = index // (11 * 6 * 6 * 2)
        index %= (11 * 6 * 6 * 2)
        trend_pepsi = index // (11 * 6 * 6)
        index %= (11 * 6 * 6)
        cola_shares = index // (11 * 6)
        index %= (11 * 6)
        pepsi_shares = index // 11
        balance_index = index % 11
        return [balance_index, pepsi_shares, cola_shares, trend_pepsi, trend_cola]

    def _get_stock_price(self, step, stock_data):
        return stock_data.iloc[step]['Close']

    def _get_stock_trend(self, step, stock_data):
        return 1 if stock_data.iloc[step]['Close'] > stock_data.iloc[step - 1]['Close'] else 0

    def _trade(self, action):
        balance_units, shares_pepsi, shares_cola = self.state[0], self.state[1], self.state[2]
        balance = balance_units * self.balance_unit
        pepsi_price = self._get_stock_price(self.current_step, self.pepsi_data)
        cola_price = self._get_stock_price(self.current_step, self.cola_data)

        if action == 0:  # Sell all
            balance += shares_pepsi * pepsi_price + shares_cola * cola_price
            shares_pepsi, shares_cola = 0, 0
        elif action == 2:  # Buy Pepsi
            quantity = min(balance // pepsi_price, self.max_shares_per_stock - shares_pepsi)
            shares_pepsi += quantity
            balance -= quantity * pepsi_price
        elif action == 3:  # Buy Cola
            quantity = min(balance // cola_price, self.max_shares_per_stock - shares_cola)
            shares_cola += quantity
            balance -= quantity * cola_price

        # Update state with rounded balance
        self.state[0] = min(max(int(balance / self.balance_unit), 0), self.max_balance_units)
        self.state[1] = shares_pepsi
        self.state[2] = shares_cola

    def step(self, action):
        if action not in range(self.action_space):
            raise ValueError("Invalid action.")

        self._trade(action)
        self.current_step += 1

        # Update trends for the new state
        self.state[3] = self._get_stock_trend(self.current_step, self.pepsi_data)
        self.state[4] = self._get_stock_trend(self.current_step, self.cola_data)

        # Calculate reward
        portfolio_value = self._calculate_portfolio_value()
        reward = portfolio_value - self.previous_portfolio_value
        self.previous_portfolio_value = portfolio_value

        done = self.current_step >= len(self.pepsi_data) - 1
        next_state_index = self.convert_state_to_index(self.state)
        return next_state_index, reward, done

    def _calculate_portfolio_value(self):
        balance = self.state[0] * self.balance_unit
        pepsi_holdings_value = self.state[1] * self._get_stock_price(self.current_step, self.pepsi_data)
        cola_holdings_value = self.state[2] * self._get_stock_price(self.current_step, self.cola_data)
        return balance + pepsi_holdings_value + cola_holdings_value

    def reset(self):
        self.state = [15, 0, 0, 0, 0]  # Reset to initial state
        self.current_step = 0
        self.previous_portfolio_value = self._calculate_portfolio_value()
        return self.convert_state_to_index(self.state)
