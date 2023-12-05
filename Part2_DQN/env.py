import numpy as np
import pandas as pd

class StockTradingEnvironmentDQN:
    def __init__(self, pepsi_file, cola_file):
        self.pepsi_data = pd.read_csv(pepsi_file)
        self.cola_data = pd.read_csv(cola_file)

        self.state_dimensions = 5  # balance, shares of Pepsi, shares of Cola, trend of Pepsi, trend of Cola
        self.action_space = 4  # Sell all, Hold, Buy Pepsi, Buy Cola

        self.state = np.array([10000.0, 0, 0, 0, 0])  # Initial state
        self.current_step = 0
        self.previous_portfolio_value = self._calculate_portfolio_value() 

    def step(self, action):
        self._trade(action)
        self.current_step += 1
        done = self.current_step >= len(self.pepsi_data) - 1
        reward = self._calculate_reward()

        self._update_state()
        return self.state, reward, done

    def _calculate_reward(self):
        # Simple reward function: change in portfolio value
        # More sophisticated reward functions can be used
        current_portfolio_value = self._calculate_portfolio_value()
        reward = current_portfolio_value - self.previous_portfolio_value
        self.previous_portfolio_value = current_portfolio_value  # Update the previous portfolio value
        return reward

    def _calculate_portfolio_value(self):
        # Calculate total portfolio value
        pepsi_price = self._get_stock_price(self.current_step, self.pepsi_data)
        cola_price = self._get_stock_price(self.current_step, self.cola_data)
        return self.state[0] + self.state[1] * pepsi_price + self.state[2] * cola_price

    def _trade(self, action):
        balance, shares_pepsi, shares_cola = self.state[0], self.state[1], self.state[2]
        pepsi_price = self._get_stock_price(self.current_step, self.pepsi_data)
        cola_price = self._get_stock_price(self.current_step, self.cola_data)

        if action == 0:  # Sell all
            balance += shares_pepsi * pepsi_price + shares_cola * cola_price
            shares_pepsi, shares_cola = 0, 0
        elif action == 2:  # Buy Pepsi
            quantity = balance / pepsi_price
            if quantity > 0:                
                shares_pepsi += quantity
                balance -= quantity * pepsi_price
        elif action == 3:  # Buy Cola
            quantity = balance / cola_price
            if quantity > 0:   
                shares_cola += quantity
                balance -= quantity * cola_price

        self.state[0] = balance
        self.state[1] = shares_pepsi
        self.state[2] = shares_cola

    def _update_state(self):
        pepsi_trend = self._get_stock_trend(self.current_step, self.pepsi_data)
        cola_trend = self._get_stock_trend(self.current_step, self.cola_data)
        self.state[3] = pepsi_trend
        self.state[4] = cola_trend

    def _get_stock_price(self, step, stock_data):
        return stock_data.iloc[step]['Close']

    def _get_stock_trend(self, step, stock_data):
        return 1 if stock_data.iloc[step]['Close'] > stock_data.iloc[step - 1]['Close'] else 0

    def reset(self):
        self.state = np.array([10000.0, 0, 0, 0, 0])  # Reset to initial state
        self.current_step = 0
        return self.state
