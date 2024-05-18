import abc
import pprint
from typing import Union, Text, Optional
import numpy as np
import pandas as pd

from qlib.utils.data import robust_zscore, zscore
from qlib.constant import EPS
from qlib.data.dataset.utils import fetch_df_by_index
from qlib.utils.serial import Serializable
from qlib.utils.paral import datetime_groupby_apply
from qlib.data.inst_processor import InstProcessor
from qlib.data import D
import logging


def get_group_columns(df: pd.DataFrame, group: Union[Text, None]):
    """
    get a group of columns from multi-index columns DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        with multi of columns.
    group : str
        the name of the feature group, i.e. the first level value of the group index.
    """
    if group is None:
        return df.columns
    else:
        return df.columns[df.columns.get_loc(group)]


class Processor(Serializable):
    def fit(self, df: pd.DataFrame = None):
        """
        learn data processing parameters

        Parameters
        ----------
        df : pd.DataFrame
            When we fit and process data with processor one by one. The fit function reiles on the output of previous
            processor, i.e. `df`.

        """

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame):
        """
        process the data

        NOTE: **The processor could change the content of `df` inplace !!!!! **
        User should keep a copy of data outside

        Parameters
        ----------
        df : pd.DataFrame
            The raw_df of handler or result from previous processor.
        """

    def is_for_infer(self) -> bool:
        """
        Is this processor usable for inference
        Some processors are not usable for inference.

        Returns
        -------
        bool:
            if it is usable for infenrece.
        """
        return True

    def readonly(self) -> bool:
        """
        Does the processor treat the input data readonly (i.e. does not write the input data) when processing

        Knowning the readonly information is helpful to the Handler to avoid uncessary copy
        """
        return False

    def config(self, **kwargs):
        attr_list = {"fit_start_time", "fit_end_time"}
        for k, v in kwargs.items():
            if k in attr_list and hasattr(self, k):
                setattr(self, k, v)

        for attr in attr_list:
            if attr in kwargs:
                kwargs.pop(attr)
        super().config(**kwargs)


class DropnaProcessor(Processor):
    def __init__(self, fields_group=None):
        super().__init__()
        self.fields_group = fields_group

    def __call__(self, df):
        return df.dropna(subset=get_group_columns(df, self.fields_group))

    def readonly(self):
        return False


class DropnaLabel(DropnaProcessor):
    def __init__(self, fields_group="label"):
        super().__init__(fields_group=fields_group)

    def is_for_infer(self) -> bool:
        """The samples are dropped according to label. So it is not usable for inference"""
        return True


class DropExtremeLabel(Processor):
    def __init__(self, fields_group='label', percentile: float = 0.975):
        super().__init__()
        self.fields_group = fields_group
        assert 0 < percentile < 1, "percentile not allowed"
        self.percentile = percentile

    def forward(self, df):
        rank_pct = df['label'].groupby(level='datetime').rank(pct=True)
        df.loc[:,'rank_pct'] = rank_pct
        trimmed_df = df[df['rank_pct'].between(1 - self.percentile, self.percentile, inclusive='both')]
        return trimmed_df.drop(columns=['rank_pct'])

    def __call__(self, df):
        return self.forward(df)

    def is_for_infer(self) -> bool:
        """The samples are dropped according to label. So it is not usable for inference"""
        return True

    def readonly(self):
        return False


def func_drop(df, percentile: float = 0.975):
    # 计算每行的百分位排名
    rank_pct = df['label'].groupby(level='datetime').rank(pct=True)

    # 计算需要删除的行
    df['rank_pct'] = rank_pct
    trimmed_df = df[df['rank_pct'].between(0.025, 0.975, inclusive='neither')]

    # 使用drop函数删除行
    return trimmed_df.drop(columns=['rank_pct'])
