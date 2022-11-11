import unittest
import numpy as np
import pandas as pd

from type_infer.infer import infer_types

from data_insights.cleaners import cleaner
from data_insights.imputers import NumericalImputer, CategoricalImputer


class TestCleaner(unittest.TestCase):
    def test_0_airline_sentiment(self):
        df = pd.read_csv("tests/data/airline_sentiment_sample.csv")
        inferred_types = infer_types(df, pct_invalid=0)
        target = 'airline_sentiment'
        tss = {
            'is_timeseries': False,
        }
        cdf = cleaner(data=df,
                      dtype_dict=inferred_types.dtypes,
                      pct_invalid=0.01,
                      identifiers={},
                      target=target,
                      mode='train',
                      timeseries_settings=tss,
                      anomaly_detection=False,
                      imputers={},
                      custom_cleaning_functions={})

        self.assertTrue(isinstance(cdf, pd.DataFrame))  # TODO: better asserts here

    def test_1_hdi(self):
        df = pd.read_csv("tests/data/hdi.csv")
        inferred_types = infer_types(df, pct_invalid=0)
        target = 'Development Index'
        tss = {
            'is_timeseries': False,
        }
        cdf = cleaner(data=df,
                      dtype_dict=inferred_types.dtypes,
                      pct_invalid=0.01,
                      identifiers={},
                      target=target,
                      mode='train',
                      timeseries_settings=tss,
                      anomaly_detection=False,
                      imputers={},
                      custom_cleaning_functions={})
        self.assertTrue(isinstance(cdf, pd.DataFrame))

    def test_2_imputers(self):
        df = pd.read_csv("tests/data/hdi.csv")
        df = df.rename(columns={'GDP ($ per capita)': 'GDP', 'Area (sq. mi.)': 'Area', 'Literacy (%)': 'Literacy'})
        df['Infant mortality '] = df['Infant mortality '].apply(lambda x: 'High' if x >= 20 else 'Low')

        inferred_types = infer_types(df, pct_invalid=0)
        target = 'Development Index'
        tss = {
            'is_timeseries': False,
        }

        cat_mode_impute_col = 'Infant mortality '
        num_mean_impute_col = 'Population'
        num_mode_impute_col = 'GDP'
        num_zero_impute_col = 'Pop. Density '
        num_median_impute_col = 'Area'

        cols = [num_mean_impute_col, num_mode_impute_col, num_zero_impute_col, num_median_impute_col,
                cat_mode_impute_col]

        for col in cols:
            df[col].iloc[0] = np.nan  # replace first row values with nans

        imputers = {
            num_mean_impute_col: NumericalImputer(value='mean', target=num_mean_impute_col),
            num_mode_impute_col: NumericalImputer(value='mode', target=num_mode_impute_col),
            num_median_impute_col: NumericalImputer(value='median', target=num_median_impute_col),
            num_zero_impute_col: NumericalImputer(value='zero', target=num_zero_impute_col),
            cat_mode_impute_col: CategoricalImputer(value='mode', target=cat_mode_impute_col)

        }

        num_mean_target_value = df[num_mean_impute_col].iloc[1:].mean()
        num_mode_target_value = df[num_mode_impute_col].iloc[1:].mode().iloc[0]
        num_median_target_value = df[num_median_impute_col].iloc[1:].median()
        num_zero_target_value = 0.0
        cat_mode_target_value = df[cat_mode_impute_col].iloc[1:].mode().iloc[0]

        cdf = cleaner(data=df,
                      dtype_dict=inferred_types.dtypes,
                      pct_invalid=0.01,
                      identifiers={},
                      target=target,
                      mode='train',
                      timeseries_settings=tss,
                      anomaly_detection=False,
                      imputers=imputers,
                      custom_cleaning_functions={})

        assert cdf[num_mean_impute_col].iloc[0] == num_mean_target_value
        assert cdf[num_mode_impute_col].iloc[0] == num_mode_target_value
        assert cdf[num_zero_impute_col].iloc[0] == num_zero_target_value
        assert cdf[num_median_impute_col].iloc[0] == num_median_target_value
        assert cdf[cat_mode_impute_col].iloc[0] == cat_mode_target_value
