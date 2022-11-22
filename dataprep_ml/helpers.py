import os
import random
import logging
from typing import Iterable

import colorlog
import numpy as np
import pandas as pd

from type_infer.helpers import is_nan_numeric


def initialize_log():
    pid = os.getpid()

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter())

    logging.basicConfig(handlers=[handler])
    log = logging.getLogger(f'dataprep_ml-{pid}')
    log_level = os.environ.get('DATAPREP_ML_LOG', 'DEBUG')
    log.setLevel(log_level)
    return log


log = initialize_log()


def seed(seed_nr: int) -> None:
    np.random.seed(seed_nr)
    random.seed(seed_nr)


def filter_nan_and_none(series: Iterable) -> list:
    return [x for x in series if not is_nan_numeric(x) and x is not None]


def get_ts_groups(df: pd.DataFrame, tss) -> list:
    ##############################
    # TODO - Should refactor this
    ##############################
    group_combinations = ['__default']
    if tss.get('group_by', False):
        groups = [tuple([g]) if not isinstance(g, tuple) else g
                  for g in list(df.groupby(by=tss['group_by']).groups.keys())]
        group_combinations.extend(groups)
    return group_combinations
