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

with open(f"./label_pred/{universe}/{universe}_pred_{seed}.pkl", "rb") as f:
    pred_score = pickle.load(f)
with open(f"./label_pred/{universe}/{universe}_labels_{seed}.pkl", "rb") as f:
    label = pickle.load(f)

pred_score_df = pd.DataFrame(pred_score, index=pred_score.index)
labels_df = pd.DataFrame(label, index=label.index)

FREQ = "day"
STRATEGY_CONFIG = {
    "topk": 30,
    "n_drop": 30,
    "signal": pred_score_df,
}

EXECUTOR_CONFIG = {
    "time_per_step": "day",
    "generate_portfolio_metrics": True,
}

backtest_config = {
    "start_time": "2020-07-01",
    "end_time": "2023-12-31",
    "account": 100000000,
    "benchmark": CSI300_BENCH,
    "exchange_kwargs": {
        "freq": FREQ,
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    },
}

# strategy object
strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
# executor object
executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
# backtest
portfolio_metric_dict, indicator_dict = backtest(
    executor=executor_obj, strategy=strategy_obj, **backtest_config)
analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
# backtest info
report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

# analysis
analysis = dict()
analysis["excess_return_without_cost"] = risk_analysis(
    report_normal["return"] - report_normal["bench"], freq=analysis_freq
)
analysis["excess_return_with_cost"] = risk_analysis(
    report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq
)

analysis_df = pd.concat(analysis)  # type: pd.DataFrame
# log metrics
analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())

pprint(
    f"The following are analysis results of benchmark return({analysis_freq}).")
pprint(risk_analysis(report_normal["bench"], freq=analysis_freq))
pprint(
    f"The following are analysis results of the excess return without cost({analysis_freq}).")
pprint(analysis["excess_return_without_cost"])
# pprint(f"The following are analysis results of the excess return with cost({analysis_freq}).")
# pprint(analysis["excess_return_with_cost"])
analysis["excess_return_without_cost"].to_csv(
    f"./label_pred/{universe}/{seed}.csv", header=True, index=True)
