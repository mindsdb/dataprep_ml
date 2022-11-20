import unittest
import pandas as pd

from type_infer.infer import infer_types

from dataprep_ml.splitters import splitter


class TestSplitters(unittest.TestCase):
    def test_0_hdi(self):
        df = pd.read_csv("tests/data/hdi.csv")
        inferred_types = infer_types(df, pct_invalid=0)
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

        self.assertTrue(len(train) == round(len(df)*train_pct))
        self.assertTrue(len(dev) == round(len(df)*dev_pct))
        self.assertTrue(len(test) == round(len(df)*test_pct))
