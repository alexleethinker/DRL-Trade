from enum import Enum
import pandas as pd
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt


from base import TransactionCost


class TradingAction(Enum):
    Buy = 1
    Hold = 0
    Sell = -1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):
    
    def __init__(self,
                 data_provider: BaseDataProvider,
                 reward_strategy: BaseRewardStrategy = IncrementalProfit,
                 transaction_cost: TransactionCost = TransactionCost,
                 initial_balance: int = 100000,
                 commissionPercent: float = 0.25,
                 maxSlippagePercent: float = 2.0,
                 **kwargs):
        
        self.logger = kwargs.get('logger', init_logger(__name__, show_debug=kwargs.get('show_debug', True)))

        self.base_precision: int = kwargs.get('base_precision', 2)
        self.asset_precision: int = kwargs.get('asset_precision', 8)
        self.min_cost_limit: float = kwargs.get('min_cost_limit', 1E-3)
        self.min_amount_limit: float = kwargs.get('min_amount_limit', 1E-3)

        self.initial_balance = round(initial_balance, self.base_precision)
        self.commissionPercent = commissionPercent
        self.maxSlippagePercent = maxSlippagePercent

        self.data_provider = data_provider
        self.reward_strategy = reward_strategy()
        self.transaction_cost = TransactionCost(commissionPercent=self.commissionPercent,
                                             maxSlippagePercent=self.maxSlippagePercent,
                                             base_precision=self.base_precision,
                                             asset_precision=self.asset_precision,
                                             min_cost_limit=self.min_cost_limit,
                                             min_amount_limit=self.min_amount_limit)

        self.render_benchmarks: List[Dict] = kwargs.get('render_benchmarks', [])
        self.normalize_obs: bool = kwargs.get('normalize_obs', True)
        self.stationarize_obs: bool = kwargs.get('stationarize_obs', True)
        self.normalize_rewards: bool = kwargs.get('normalize_rewards', False)
        self.stationarize_rewards: bool = kwargs.get('stationarize_rewards', True)

        self.n_discrete_actions: int = kwargs.get('n_discrete_actions', 24)
        self.action_space = spaces.Discrete(self.n_discrete_actions)

        self.n_features = 6 + len(self.data_provider.columns)
        self.obs_shape = (1, self.n_features)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float16)

        self.observations = pd.DataFrame(None, columns=self.data_provider.columns)
       
