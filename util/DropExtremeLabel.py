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
from qlib.data.dataset.processor import Processor, get_group_columns
import logging


# def get_group_columns(df: pd.DataFrame, group: Union[Text, None]):
#     """
#     get a group of columns from multi-index columns DataFrame
#     从多重索引列DataFrame中获取一组列

#     Parameters
#     ----------
#     df : pd.DataFrame
#         with multi of columns.
#         具有多重列的DataFrame
#     group : str
#         the name of the feature group, i.e. the first level value of the group index.
#         特征组的名称，即组索引的第一级值
#     """
#     if group is None:
#         return df.columns
#     else:
#         return df.columns[df.columns.get_loc(group)]


# class Processor(Serializable):
#     def fit(self, df: pd.DataFrame = None):
#         """
#         learn data processing parameters
#         学习数据处理参数

#         Parameters
#         ----------
#         df : pd.DataFrame
#             When we fit and process data with processor one by one. The fit function reiles on the output of previous
#             processor, i.e. `df`.
#             当我们逐个使用处理器拟合和处理数据时，fit函数依赖于前一个处理器的输出，即`df`
#         """

#     @abc.abstractmethod
#     def __call__(self, df: pd.DataFrame):
#         """
#         process the data
#         处理数据

#         NOTE: **The processor could change the content of `df` inplace !!!!! **
#         注意：**处理器可能会就地更改`df`的内容！！！！**
#         User should keep a copy of data outside
#         用户应该在外部保留数据副本

#         Parameters
#         ----------
#         df : pd.DataFrame
#             The raw_df of handler or result from previous processor.
#             处理器的原始df或来自前一个处理器的结果
#         """

#     def is_for_infer(self) -> bool:
#         """
#         Is this processor usable for inference
#         此处理器是否可用于推理
#         Some processors are not usable for inference.
#         某些处理器不可用于推理

#         Returns
#         -------
#         bool:
#             if it is usable for infenrece.
#             如果它可用于推理则返回True
#         """
#         return True

#     def readonly(self) -> bool:
#         """
#         Does the processor treat the input data readonly (i.e. does not write the input data) when processing
#         处理器在处理时是否将输入数据视为只读（即不写入输入数据）

#         Knowning the readonly information is helpful to the Handler to avoid uncessary copy
#         了解只读信息有助于Handler避免不必要的复制
#         """
#         return False

#     def config(self, **kwargs):
#         attr_list = {"fit_start_time", "fit_end_time"}
#         for k, v in kwargs.items():
#             if k in attr_list and hasattr(self, k):
#                 setattr(self, k, v)

#         for attr in attr_list:
#             if attr in kwargs:
#                 kwargs.pop(attr)
#         super().config(**kwargs)


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
        self.fields_group = fields_group  # 特征组名称,默认为'label'
        assert 0 < percentile < 1, "percentile not allowed"  # 检查百分位数是否在0-1之间
        self.percentile = percentile  # 存储百分位数阈值

    def forward(self, df):
        # 对每个时间点的label进行百分位数排名
        rank_pct = df['label'].groupby(level='datetime').rank(pct=True)
        # 将排名结果添加到DataFrame中
        df.loc[:,'rank_pct'] = rank_pct
        # 筛选出在指定百分位范围内的数据
        # 例如percentile=0.975时,保留2.5%到97.5%之间的数据
        trimmed_df = df[df['rank_pct'].between(1 - self.percentile, self.percentile, inclusive='both')]
        # 删除临时的rank_pct列并返回结果
        return trimmed_df.drop(columns=['rank_pct'])

    def __call__(self, df):
        # 调用实例时执行forward方法
        return self.forward(df)

    def is_for_infer(self) -> bool:
        """The samples are dropped according to label. So it is not usable for inference"""
        # 由于是基于label过滤样本,所以不适用于推理阶段
        return True

    def readonly(self):
        # 表示处理过程会修改输入数据
        return False


def func_drop(df, percentile: float = 0.975):
    # 计算每行的百分位排名
    rank_pct = df['label'].groupby(level='datetime').rank(pct=True)

    # 计算需要删除的行
    df['rank_pct'] = rank_pct
    trimmed_df = df[df['rank_pct'].between(0.025, 0.975, inclusive='neither')]

    # 使用drop函数删除行
    return trimmed_df.drop(columns=['rank_pct'])
