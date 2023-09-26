import unittest
import numpy as np
import pandas as pd

from type_infer.infer import infer_types

from dataprep_ml.cleaners import cleaner
from dataprep_ml.imputers import NumericalImputer, CategoricalImputer


class TestCleaners(unittest.TestCase):

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

    def test_3_timeseries(self):
        """ Unit test for time series cleaner.

            This test checks that duplicated time-stamps are properly
            handled by the cleaner.
        """
        # setup correct dataframe
        x_correct = np.asarray([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            dtype=int)
        y_correct = np.asarray([
            'a', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            'a', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            'a', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        z_correct = np.asarray([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            dtype=float)
        g_correct = np.asarray([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            dtype=int)
        df_correct = pd.DataFrame.from_records({
            'x': x_correct,
            'y': y_correct,
            'z': z_correct,
            'group_id': g_correct})

        # setup corrupted DataFrame
        x_corrupt = np.asarray([
            1, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 10, 10,
            1, 2, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10, 10,
            1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10],
            dtype=int)
        y_corrupt = np.asarray([
            'a', '0', '1', '2', '3', '4', '0', '5', '6', '7', '8', '9', '10', '0', '0',
            'a', '2', '0', '0', '3', '4', '5', '0', '0', '6', '7', '8', '9', '10', '0',
            'a', '2', '3', '0', '0', '0', '4', '5', '6', '7', '8', '9', '0', '0', '10'])
        z_corrupt = np.asarray([
            1, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 10, 10,
            1, 2, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10, 10,
            1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10],
            dtype=float)
        g_corrupt = np.asarray([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            dtype=int)
        df_corrupt = pd.DataFrame.from_records({
            'x': x_corrupt,
            'y': y_corrupt,
            'z': z_corrupt,
            'group_id': g_corrupt})

        # inferred types are the same for both DataFrames
        inferred_types = infer_types(df_correct, pct_invalid=0)
        target = 'z'
        tss = {
            'is_timeseries': True,
            'order_by': 'x',
            'group_by': 'group_id'
        }
        df_correct_clean = cleaner(data=df_correct,
                                   dtype_dict=inferred_types.dtypes,
                                   pct_invalid=0.1,
                                   identifiers={},
                                   target=target,
                                   mode='train',
                                   timeseries_settings=tss,
                                   anomaly_detection=False,
                                   imputers={},
                                   custom_cleaning_functions={})
        df_clean = cleaner(data=df_corrupt,
                           dtype_dict=inferred_types.dtypes,
                           pct_invalid=0.1,
                           identifiers={},
                           target=target,
                           mode='train',
                           timeseries_settings=tss,
                           anomaly_detection=False,
                           imputers={},
                           custom_cleaning_functions={})
        df_clean = df_clean.drop('__mdb_original_index', axis=1)
        df_correct_clean = df_correct_clean.drop('__mdb_original_index',
                                                 axis=1)
        # TODO: better asserts here
        self.assertTrue(isinstance(df_clean, pd.DataFrame))
        # df_clean comes with an auxiliary column, wtf
        self.assertTrue(df_clean.equals(df_correct_clean))
