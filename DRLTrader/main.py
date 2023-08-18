import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from finrl.agents.stablebaselines3.models import DRLEnsembleAgent

from Config import INDICATORS
from Config import TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE
from Config import A2C_PARAMS, PPO_PARAMS, DDPG_PARAMS,TIMESTEPS
import DataLoader
from FeatureEngineer import feature_engieering



df = DataLoader.download()
processed = feature_engieering(df)



stock_dimension = len(processed.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension

env_kwargs = {
    "initial_amount": 1000000, 
    "buy_cost_pct": 0.001, 
    "sell_cost_pct": 0.001, 
    "stock_dim": stock_dimension, 
    "action_space": stock_dimension, 
    "state_space": state_space, 
    "tech_indicator_list": INDICATORS,
    "hmax": 100, 
    "reward_scaling": 1e-4,
    "print_verbosity":1   
}

rebalance_window = 63 # rebalance_window is the number of days to retrain the model
validation_window = 63 # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)


ensemble_agent = DRLEnsembleAgent(df=processed,
                 train_period=(TRAIN_START_DATE,TRAIN_END_DATE),
                 val_test_period=(TEST_START_DATE,TEST_END_DATE),
                 rebalance_window=rebalance_window, 
                 validation_window=validation_window, 
                 **env_kwargs)



df_summary = ensemble_agent.run_ensemble_strategy(A2C_PARAMS,
                                                 PPO_PARAMS,
                                                 DDPG_PARAMS,
                                                 TIMESTEPS)








unique_trade_date = processed[(processed.date > TEST_START_DATE)&(processed.date <= TEST_END_DATE)].date.unique()


df_trade_date = pd.DataFrame({'datadate':unique_trade_date})

df_account_value=pd.DataFrame()
for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
    temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format('ensemble',i))
    df_account_value = pd.concat([df_account_value,temp],ignore_index=True)
sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
print('Sharpe Ratio: ',sharpe)
df_account_value=df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))