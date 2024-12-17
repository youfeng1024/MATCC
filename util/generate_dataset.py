import pickle
import torch
import pandas as pd
import numpy as np
import pprint as pp
import os
import argparse
import yaml
from qlib.data.dataset.handler import DataHandlerLP
from qlib.tests.data import GetData
# from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.constant import REG_CN
import qlib
import sys
from pathlib import Path
# import DropExtremeLabel

# DIRNAME = Path(__file__).absolute().resolve().parent
# sys.path.append(str(DIRNAME))
# sys.path.append(str(DIRNAME.parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="csi300",
                        help="dataset type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    # 使用qlib默认数据
    provider_uri = "~/.qlib/qlib_data/cn_data"  # 目标目录
    # GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    
    # 读取config文件
    with open(f"./2024.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # 获取处理器配置，构造处理器文件路径
    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    h_path = f'handler_{config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")}' \
                       f'_{config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")}.pkl'
    
    # # 如果处理器文件不存在，则暂存并保存
    # if not h_path.exists():
    #     h = init_instance_by_config(h_conf)
    #     h.to_pickle(h_path, dump_all=True)
    #     print('Save preprocessed data to', h_path)
    
    # # 更新配置中的处理器路径
    # config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"

    print(config)
    print("\n" + "==" * 20 + "\n")

    # 初始化数据集对象
    dataset = init_instance_by_config(config['task']["dataset"])

    # 准备测试、验证和训练数据
    dl_test = dataset.prepare(
        "test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dl_valid = dataset.prepare(
        "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dl_train = dataset.prepare(
        "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)

    # 创建数据集目录（如果不存在）
    if not os.path.exists("../dataset/csi300"):
        os.makedirs("../dataset/csi300")

    if not os.path.exists("../dataset/csi800"):
        os.makedirs("../dataset/csi800")

    # 保存测试、验证和训练数据到文件
    with open(f"../dataset/data/{args.universe}/{args.universe}_dl_test.pkl", "wb") as f:
        pickle.dump(dl_test, f)

    with open(f"../dataset/data/{args.universe}/{args.universe}_dl_valid.pkl", "wb") as f:
        pickle.dump(dl_valid, f)

    with open(f"../dataset/data/{args.universe}/{args.universe}_dl_train.pkl", "wb") as f:
        pickle.dump(dl_train, f)

    # # 删除处理器文件
    # if os.path.exists(f"{h_path}"):
    #     os.remove(f"{h_path}")
