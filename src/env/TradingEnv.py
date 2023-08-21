import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StockTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, commission_fee=0.01, slippage_cost=0.1):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.stock_price_history = data['stock_price']
        self.moving_average_10 = data['moving_average_10']
        self.moving_average_30 = data['moving_average_30']
        self.relative_strength_index = data['relative_strength_index']
        self.commission_fee = commission_fee
        self.slippage_cost = slippage_cost
        
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), shape=(2,))  # (Amount, Action) where Action: -1: Buy, 0: Hold, 1: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.stock_owned = 0
        return self._get_observation()
    
    def step(self, action):
        assert self.action_space.contains(action)

        

        prev_portfolio_value = self.balance + self.stock_owned * self.stock_price_history[self.current_step - 1]
        current_price = self.stock_price_history[self.current_step]
        
        amount = int(self.initial_balance * action[1] / current_price)
        
        
        if action[0] > 0:  # Buy
            amount =  min( int(self.initial_balance * action[1] / current_price), int(self.balance / current_price * (1 + self.commission_fee + self.slippage_cost)))
            if self.balance >= current_price * amount * (1 + self.commission_fee + self.slippage_cost):
                self.stock_owned += amount
                self.balance -= current_price * amount * (1 + self.commission_fee + self.slippage_cost)
        elif action[0] < 0:  # Sell
            amount = min(amount, self.stock_owned)
            if self.stock_owned > 0:
                self.stock_owned -= amount
                self.balance += current_price * amount * (1 - self.commission_fee - self.slippage_cost)
        
        print( 'buy' if action[0] > 0 else 'sell',  amount)
        self.render()
        current_portfolio_value = self.balance + self.stock_owned * current_price
        
        excess_return = current_portfolio_value - prev_portfolio_value
        
        risk_free_rate = 0.02  # Example risk-free rate
        std_deviation = np.std(self.stock_price_history[:self.current_step+1])
        sharpe_ratio = (excess_return - risk_free_rate) / std_deviation if std_deviation != 0 else 0
        reward = sharpe_ratio
        
        self.current_step += 1
        
        if self.current_step == len(self.data['stock_price']):
            done = True
        else:
            done = False
        
        info = {}
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        return np.array([
            self.balance,
            self.stock_owned,
            self.stock_price_history[self.current_step],
            self.moving_average_10[self.current_step],
            self.relative_strength_index[self.current_step]
        ])
    
    def render(self, mode='human'):
        current_price = self.stock_price_history[self.current_step]
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Stock Owned: {self.stock_owned}, Stock Price: {current_price:.2f}")

# Example data for stock prices and technical indicators
data = {
    'stock_price': [50, 55, 60, 65, 70, 50, 55, 60, 65, 70],
    'moving_average_10': [52, 54, 58, 62, 66, 50, 55, 60, 65, 70],  # 10-day moving average
    'moving_average_30': [49, 51, 54, 57, 61, 50, 55, 60, 65, 70],  # 30-day moving average
    'relative_strength_index': [40, 50, 60, 70, 80, 50, 55, 60, 65, 70]  # RSI values
}

env = StockTradingEnv(data, initial_balance=100000, commission_fee=0.0001, slippage_cost=0.005)

observation = env.reset()
for _ in range(len(data['stock_price']) - 1):
      # Display current state
    action = env.action_space.sample()  # Random action for demonstration
    observation, reward, done, info = env.step(action)

