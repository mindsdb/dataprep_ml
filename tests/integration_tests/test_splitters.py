import unittest
import pandas as pd

from type_infer.api import infer_types

from dataprep_ml.splitters import splitter


class TestSplitters(unittest.TestCase):
    def test_0_hdi(self):
        df = pd.read_csv("tests/data/hdi.csv")
        inferred_types = infer_types(df, config={'pct_invalid': 0})
        target = 'Development Index'

        train_pct, dev_pct, test_pct = 0.8, 0.1, 0.1
        splits = splitter(data=df,
                          tss={},
                          dtype_dict=inferred_types.dtypes,
                          seed=0,
                          pct_train=train_pct,
                          pct_dev=dev_pct,
                          pct_test=test_pct,
                          target=target)
        train = splits['train']
        dev = splits['dev']
        test = splits['test']
        stratified_on = splits['stratified_on']

        self.assertTrue(isinstance(train, pd.DataFrame))
        self.assertTrue(isinstance(dev, pd.DataFrame))
        self.assertTrue(isinstance(test, pd.DataFrame))
        self.assertTrue(isinstance(stratified_on, list))

        train_len = round(len(df) * train_pct)
        dev_len = round(0.1 + (len(df) * dev_pct))  # 0.1 to bypass bankers round behavior
        self.assertTrue(len(train) == train_len)
        self.assertTrue(len(dev) == dev_len)
        self.assertTrue(len(test) == len(df) - (train_len + dev_len))

    # TODO add time series splitter test
