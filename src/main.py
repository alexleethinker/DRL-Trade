from TradingEnv import StockTradingEnv
from pybroker import YFinance
import pandas as pd

yfinance = YFinance()
df = yfinance.query(['AAPL'], start_date='3/1/2021', end_date='3/1/2022')
df['date'] = pd.to_datetime(df['date']).dt.date

env = StockTradingEnv(df, initial_balance=100000, commission_fee=0.0001, slippage_cost=0.005)

observation = env.reset()
for _ in range(1, len(df['adj_close'])):
      # Display current state
    action = env.action_space.sample()  # Random action for demonstration
    observation, reward, done, info = env.step(action)
    
env.render_all()
