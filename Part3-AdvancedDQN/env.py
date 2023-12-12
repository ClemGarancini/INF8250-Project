import numpy as np
import pandas as pd
import talib

class AdvancedStockTradingEnvironment:
    def __init__(self, pepsi_file, cola_file, h_max):
        self.h_max = h_max
        self.action_space = (2 * h_max + 1) ** 2

        self.pepsi_data = self.load_data(pepsi_file)
        self.cola_data = self.load_data(cola_file)

        # Including closing price, MACD, MACD signal, RSI, CCI, ADX for each stock
        self.state_dimensions = 1 + 2 + 6 * 2
        self.state = self.initialize_state()
        self.current_step = 0

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        # Add technical indicators
        data['MACD'], data['MACDSignal'], _ = talib.MACD(data['Adj Close'])
        data['RSI'] = talib.RSI(data['Adj Close'])
        data['CCI'] = talib.CCI(data['High'], data['Low'], data['Adj Close'])
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Adj Close'])
        # Drop NaNs
        return data.dropna()

    def initialize_state(self):
        # State includes balance, shares of each stock, closing price, MACD, MACD signal, RSI, CCI, ADX
        return np.array([10000.0, 0, 0] +  
                        self.pepsi_data.iloc[0][['Adj Close', 'MACD', 'MACDSignal', 'RSI', 'CCI', 'ADX']].tolist() +
                        self.cola_data.iloc[0][['Adj Close', 'MACD', 'MACDSignal', 'RSI', 'CCI', 'ADX']].tolist())

    def step(self, action):
        done = self.current_step >= len(self.pepsi_data) - 1
        reward = 0

        if not done:
            pepsi_action, cola_action = self.decode_action(action)
            # Save the current portfolio value before executing trades
            prev_portfolio_value = self.calculate_portfolio_value()

            # Execute trades
            self.execute_trade('PEP', pepsi_action)
            self.execute_trade('KO', cola_action)

            # Calculate the new portfolio value and reward
            new_portfolio_value = self.calculate_portfolio_value()
            reward = new_portfolio_value - prev_portfolio_value

            # Update state with new market data
            self.update_state(pepsi_action, cola_action)
            self.current_step += 1

        return self.state, reward, done

    def decode_action(self, action):
        pepsi_action = action // (2 * self.h_max + 1) - self.h_max
        cola_action = action % (2 * self.h_max + 1) - self.h_max
        return pepsi_action, cola_action

    def execute_trade(self, stock, action):
        price_column = 'Adj Close'
        stock_data = self.pepsi_data if stock == 'PEP' else self.cola_data
        price = stock_data.iloc[self.current_step][price_column]
        if action > 0:  # Buy
            cost = min(self.state[0], price * action)
            self.state[0] -= cost
            self.state[1 if stock == 'PEP' else 2] += cost / price
        elif action < 0:  # Sell
            shares_to_sell = min(abs(action), self.state[1 if stock == 'PEP' else 2])
            self.state[0] += shares_to_sell * price
            self.state[1 if stock == 'PEP' else 2] -= shares_to_sell

    def update_state(self, pepsi_action, cola_action):
        self.state[3:9] = self.pepsi_data.iloc[self.current_step][['Adj Close', 'MACD', 'MACDSignal', 'RSI', 'CCI', 'ADX']].tolist()
        self.state[9:15] = self.cola_data.iloc[self.current_step][['Adj Close', 'MACD', 'MACDSignal', 'RSI', 'CCI', 'ADX']].tolist()

    def calculate_portfolio_value(self):
        pepsi_price = self.pepsi_data.iloc[self.current_step]['Adj Close']
        cola_price = self.cola_data.iloc[self.current_step]['Adj Close']
        return self.state[0] + self.state[1] * pepsi_price + self.state[2] * cola_price

    def reset(self):
        self.state = self.initialize_state()
        self.current_step = 0
        return self.state
