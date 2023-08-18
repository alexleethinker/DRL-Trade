from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from Config import  DATA_SAVE_DIR, TRAIN_START_DATE, TEST_END_DATE
 

def download():
    df = YahooDownloader(start_date = TRAIN_START_DATE,
                        end_date = TEST_END_DATE,
                        ticker_list = DOW_30_TICKER).fetch_data()
    return df