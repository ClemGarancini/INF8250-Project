import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_and_save_stock_data(stock_symbol, start_date, end_date, file_name):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data = preprocess_data(stock_data)
    stock_data.to_csv(file_name)
    print(f"Data for {stock_symbol} saved to {file_name}")

def preprocess_data(df):

    df.replace(0, np.nan, inplace=True)

    # Forward and backward fill to handle NaNs
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    return df
