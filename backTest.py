import pickle
from pprint import pprint

import qlib
import pandas as pd
from qlib.utils.time import Freq
from qlib.utils import flatten_dict
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.report import analysis_position
from qlib.contrib.report import analysis_model
import qlib.contrib.report as qcr
from qlib.constant import REG_CN

qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)
CSI300_BENCH = "SH000300"  # CSI300
CSI300_BENCH = "SH000906"  # CSI800
seed = 18032
universe = "csi800"
# 从pickle文件中加载预测分数
with open(f"./label_pred/{universe}/{universe}_pred_{seed}.pkl", "rb") as f:
    pred_score = pickle.load(f)
# 从pickle文件中加载标签数据    
with open(f"./label_pred/{universe}/{universe}_labels_{seed}.pkl", "rb") as f:
    label = pickle.load(f)

# 将预测分数转换为DataFrame格式
pred_score_df = pd.DataFrame(pred_score, index=pred_score.index)
# 将标签数据转换为DataFrame格式
labels_df = pd.DataFrame(label, index=label.index)

# 设置回测频率为每天
FREQ = "day"
# 策略配置：选择前30支股票，每次更换30支
STRATEGY_CONFIG = {
    "topk": 30,        # 选择排名前30的股票
    "n_drop": 30,      # 每次更换30支股票
    "signal": pred_score_df,  # 使用预测分数作为信号
}

# 执行器配置
EXECUTOR_CONFIG = {
    "time_per_step": "day",   # 每天执行一次
    "generate_portfolio_metrics": True,  # 生成投资组合指标
}

# 回测配置参数
backtest_config = {
    "start_time": "2020-07-01",    # 回测开始时间
    "end_time": "2023-12-31",      # 回测结束时间
    "account": 100000000,          # 初始资金1亿
    "benchmark": CSI300_BENCH,      # 基准指数
    "exchange_kwargs": {
        "freq": FREQ,              # 交易频率
        "limit_threshold": 0.095,   # 涨跌停限制
        "deal_price": "close",      # 使用收盘价交易
        "open_cost": 0.0005,        # 开仓成本
        "close_cost": 0.0015,       # 平仓成本
        "min_cost": 5,             # 最小交易成本
    },
}

# 创建策略对象
strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
# 创建执行器对象
executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
# 执行回测
portfolio_metric_dict, indicator_dict = backtest(
    executor=executor_obj, strategy=strategy_obj, **backtest_config)
# 解析回测频率
analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
# 获取回测结果
report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

analysis = dict()
# 计算不考虑成本的超额收益
analysis["excess_return_without_cost"] = risk_analysis(
    report_normal["return"] - report_normal["bench"], freq=analysis_freq
)
# 计算考虑成本的超额收益
analysis["excess_return_with_cost"] = risk_analysis(
    report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq
)

# 将分析结果转换为DataFrame
analysis_df = pd.concat(analysis)
# 将分析结果展平为字典
analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())

# 打印基准收益分析结果
pprint(
    f"The following are analysis results of benchmark return({analysis_freq}).")
pprint(risk_analysis(report_normal["bench"], freq=analysis_freq))
# 打印不考虑成本的超额收益分析结果
pprint(
    f"The following are analysis results of the excess return without cost({analysis_freq}).")
pprint(analysis["excess_return_without_cost"])
pprint(f"The following are analysis results of the excess return with cost({analysis_freq}).")
pprint(analysis["excess_return_with_cost"])
# 将不考虑成本的超额收益分析结果保存到CSV文件
analysis["excess_return_without_cost"].to_csv(
    f"./label_pred/{universe}/{seed}.csv", header=True, index=True)
