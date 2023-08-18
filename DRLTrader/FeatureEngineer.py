
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from Config import INDICATORS, DATA_SAVE_DIR
import numpy as np


def feature_engieering(df):

    fe = FeatureEngineer(use_technical_indicator=True,
                        tech_indicator_list = INDICATORS,
                        use_turbulence=True,
                        user_defined_feature = False)
    processed = fe.preprocess_data(df)
    processed = processed.copy()
    processed = processed.fillna(0)
    processed = processed.replace(np.inf,0)

    return processed
