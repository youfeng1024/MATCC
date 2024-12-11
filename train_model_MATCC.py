import copy
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter

from src.MATCC import MATCC
from my_lr_scheduler import ChainedScheduler

# os.environ['CUDA_VISIBLE_DEVICES']="7" # 程序可见的GPU
# 设置并限制CPU线程数,避免过度使用系统资源
cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


class TrainConfig:
    # 模型和训练的基本配置
    model_name = "MATCC_csi300"  # 模型名称
    GPU = 0  # 使用的GPU编号
    universe = 'csi300'  # 股票池
    seed = 11032  # 随机种子
    
    # 文件路径配置
    dataset_dir_path = "./dataset"  # 数据集路径
    model_save_path = f"./model_params/{universe}/{model_name}_{seed}"  # 模型保存路径
    metrics_loss_path = f"./metrics/{universe}/{model_name}_{seed}"  # 指标和损失保存路径
    log_dir = f"./logs/{model_name}_{seed}"  # 日志保存路径

    # 创建必要的目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.exists(metrics_loss_path):
        os.makedirs(metrics_loss_path)

    # 配置日志记录
    logging.basicConfig(filename=os.path.join(log_dir, f"{model_name}_seed_{seed}.log"), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(f"Train {model_name}")
    writer = SummaryWriter(
        log_dir=log_dir, filename_suffix=f"{model_name}_{seed}_")  # tensorboard可视化

    # 训练超参数配置
    n_epoch = 75  # 总训练轮数
    lr = 3e-4  # 初始学习率
    gamma = 1.0  # 学习率衰减因子
    coef = 1.0  # 学习率调整系数
    cosine_period = 4  # 余弦周期
    T_0 = 15  # 重启周期
    T_mult = 1  # 周期倍增因子
    warmUp_epoch = 10  # 预热轮数
    eta_min = 2e-5  # 最小学习率
    weight_decay = 0.001  # 权重衰减

    # 模型结构参数
    seq_len = 8  # 序列长度(时间窗口)
    d_feat = 158  # 特征维度
    d_model = 256  # 模型隐藏层维度
    n_head = 4  # 注意力头数
    dropout = 0.5  # dropout率
    gate_input_start_index = 158  # 门控输入起始索引
    gate_input_end_index = 221  # 门控输入结束索引
    train_stop_loss_threshold = 0.95  # 训练停止损失阈值
    
    # 设备配置
    device = torch.device(
        f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = MATCC(d_model=d_model, d_feat=d_feat, seq_len=seq_len,
                  t_nhead=n_head, S_dropout_rate=dropout).to(device)

    # 优化器配置
    train_optimizer = optim.Adam(model.parameters(), lr=lr, betas=(
        0.9, 0.999), weight_decay=weight_decay)

    # 学习率调度器配置 - 使用自定义的ChainedScheduler
    lr_scheduler = ChainedScheduler(train_optimizer, T_0=T_0, T_mul=T_mult, eta_min=eta_min,
                                    last_epoch=-1, max_lr=lr, warmup_steps=warmUp_epoch,
                                    gamma=gamma, coef=coef, step_size=3, cosine_period=cosine_period)

    # 记录训练配置到tensorboard
    writer.add_text("train_optimizer", train_optimizer.__str__())
    writer.add_text("lr_scheduler", lr_scheduler.__str__())

    # 记录详细配置到日志
    logger.info(msg=f"\n===== Model {model_name} =====\n"
                    f"n_epochs: {n_epoch}\n"
                    f"start_lr: {lr}\n"
                    f"T_0: {T_0}\n"
                    f"T_mult: {T_mult}\n"
                    f"gamma: {gamma}\n"
                    f"coef: {coef}\n"
                    f"cosine_period: {cosine_period}\n"
                    f"eta_min: {eta_min}\n"
                    f"seed: {seed}\n"
                    f"optimizer: {train_optimizer}\n"
                    f"lr_scheduler: {lr_scheduler}\n"
                    f"description: train {model_name}\n\n")

    # 记录关键参数到tensorboard
    writer.add_text("model_name", model_name)
    writer.add_text("seed", str(seed))
    writer.add_text("n_head", str(n_head))
    writer.add_text("learning rate", str(lr))
    writer.add_text("T_0", str(T_0))
    writer.add_text("T_mult", str(T_mult))
    writer.add_text("gamma", str(gamma))
    writer.add_text("coef", str(coef))
    writer.add_text("eta_min", str(eta_min))
    writer.add_text("weight_decay", str(weight_decay))
    writer.add_text("cosine_period", str(cosine_period))


def calc_ic(pred, label):
    """计算IC和RankIC
    
    Args:
        pred: 模型预测值
        label: 真实标签值
    
    Returns:
        ic: 信息系数(Information Coefficient) - 预测值和真实值的相关系数
        ric: 排序信息系数(Rank Information Coefficient) - 预测值和真实值的斯皮尔曼相关系数
    """
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])  # 皮尔逊相关系数
    ric = df['pred'].corr(df['label'], method='spearman')  # 斯皮尔曼秩相关系数
    return ic, ric


class DailyBatchSamplerRandom(Sampler):
    """按日期批量采样器
    将同一天的数据作为一个batch,可选是否随机打乱日期顺序
    """
    def __init__(self, data_source, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        # 计算每天的样本数量
        self.daily_count = pd.Series(index=self.data_source.get_index(), dtype=np.float64).groupby(
            "datetime").size().values
        # 计算每个batch的起始索引
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            # 随机打乱日期顺序
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            # 按时间顺序遍历
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


def loss_fn(pred, label):
    """计算MSE损失
    
    忽略标签中的NaN值,只计算有效值的均方误差
    """
    mask = ~torch.isnan(label)  # 创建非NaN值的掩码
    loss = (pred[mask] - label[mask]) ** 2  # 计算均方误差
    return torch.mean(loss)


def _init_data_loader(data, shuffle=True, drop_last=True):
    """初始化数据加载器
    
    Args:
        data: 数据集
        shuffle: 是否打乱日期顺序
        drop_last: 是否丢弃最后一个不完整的batch
    
    Returns:
        DataLoader对象
    """
    sampler = DailyBatchSamplerRandom(data, shuffle)
    data_loader = DataLoader(
        data, sampler=sampler, drop_last=drop_last, 
        num_workers=10,  
        pin_memory=True  
    )
    return data_loader


def train_epoch(data_loader, train_optimizer, lr_scheduler, model, device):
    """训练一个epoch的函数"""
    model.train()  # 设置模型为训练模式
    losses = []    # 用于记录每个batch的损失

    for data in data_loader:
        data = torch.squeeze(data, dim=0)
        '''
        数据维度说明:
        data.shape: (N, T, F)
        N - 股票数量
        T - 回看窗口长度, 8天
        F - 158个因子特征 + 63个市场信息 + 1个标签
        '''
        # 提取特征和标签并移至GPU
        feature = data[:, :, 0:-1].to(device)  # 所有特征
        label = data[:, -1, -1].to(device)     # 最后一天的标签

        # 模型前向传播
        pred = model(feature.float())

        # 计算损失
        loss = loss_fn(pred, label)
        losses.append(loss.item())

        # 反向传播和优化
        train_optimizer.zero_grad()  # 清空梯度
        loss.backward()              # 计算梯度
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)  # 梯度裁剪,防止梯度爆炸
        train_optimizer.step()       # 更新参数
    lr_scheduler.step()             # 学习率调整

    return float(np.mean(losses))   # 返回平均损失


def valid_epoch(data_loader, model, device):
    """验证/测试一个epoch的函数"""
    model.eval()   # 设置模型为评估模式
    losses = []    # 记录损失
    ic = []        # 记录IC值
    ric = []       # 记录RankIC值

    with torch.no_grad():  # 不计算梯度
        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            # 提取特征和标签
            feature = data[:, :, 0:-1].to(device)
            label = data[:, -1, -1].to(device)
            
            # 模型预测
            pred = model(feature.float())
            
            # 计算损失
            loss = loss_fn(pred, label)
            losses.append(loss.item())

            # 计算IC和RankIC
            daily_ic, daily_ric = calc_ic(
                pred.detach().cpu().numpy(), label.detach().cpu().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

    # 计算评估指标
    metrics = {
        'IC': np.mean(ic),          # IC均值
        'ICIR': np.mean(ic) / np.std(ic),    # IC比率
        'RIC': np.mean(ric),        # RankIC均值
        'RICIR': np.mean(ric) / np.std(ric)  # RankIC比率
    }

    return float(np.mean(losses)), metrics


def train():
    # 检查数据集路径是否存在
    if not os.path.exists(TrainConfig.dataset_dir_path):
        raise FileExistsError("Data dir not exists")

    universe = TrainConfig.universe  # or 'csi800'

    # 加载训练集、验证集、测试集数据
    # 数据格式为pickle文件,包含2020-2023年的数据
    with open(f'{TrainConfig.dataset_dir_path}/{universe}/{universe}_dl_train_2020_2023.pkl', 'rb') as f:
        dl_train = pickle.load(f)
    with open(f'{TrainConfig.dataset_dir_path}/{universe}/{universe}_dl_valid_2020_2023.pkl', 'rb') as f:
        dl_valid = pickle.load(f)
    with open(f'{TrainConfig.dataset_dir_path}/{universe}/{universe}_dl_test_2020_2023.pkl', 'rb') as f:
        dl_test = pickle.load(f)
    print("Data Loaded.")

    # 创建数据加载器
    # train_loader随机打乱数据,验证和测试集保持原顺序
    train_loader = _init_data_loader(dl_train, shuffle=True, drop_last=True)
    valid_loader = _init_data_loader(dl_valid, shuffle=False, drop_last=False)
    test_loader = _init_data_loader(dl_test, shuffle=False, drop_last=False)

    # 获取设备和tensorboard writer
    device = TrainConfig.device
    writer = TrainConfig.writer

    # 获取模型实例
    model = TrainConfig.model

    # 获取优化器和学习率调度器
    train_optimizer = TrainConfig.train_optimizer
    lr_scheduler = TrainConfig.lr_scheduler

    # 记录最佳验证集损失,用于模型保存
    best_valid_loss = np.Inf

    print("==" * 10 +
          f" Now is Training {TrainConfig.model_name}_{TrainConfig.seed} " + "==" * 10 + "\n")

    # 训练循环
    for step in range(TrainConfig.n_epoch):
        # 训练一个epoch,返回训练损失
        train_loss = train_epoch(train_loader, train_optimizer=train_optimizer, lr_scheduler=lr_scheduler, model=model,
                                 device=device)
        # 在验证集和测试集上评估,返回损失和评价指标(IC、ICIR等)
        val_loss, valid_metrics = valid_epoch(valid_loader, model, device)
        test_loss, test_metrics = valid_epoch(test_loader, model, device)

        # 记录训练过程到tensorboard
        if writer is not None:
            writer.add_scalars(
                "Valid metrics", valid_metrics, global_step=step)
            writer.add_scalars("Test metrics", test_metrics, global_step=step)
            writer.add_scalar("Train loss", train_loss, global_step=step)
            writer.add_scalar("Valid loss", val_loss, global_step=step)
            writer.add_scalar("Test loss", test_loss, global_step=step)
            writer.add_scalars("All loss Comparison",
                               {"train loss": train_loss,
                                   "val loss": val_loss, "test loss": test_loss},
                               global_step=step)
            writer.add_scalar(
                "Learning rate", train_optimizer.param_groups[0]['lr'], global_step=step)

        # 打印当前epoch的训练信息
        print(
            "==" * 10 + f" {TrainConfig.model_name}_{TrainConfig.seed} Epoch {step} " + "==" * 10)
        print("Epoch %d, train_loss %.6f, valid_loss %.6f, test_loss %.6f " %
              (step, train_loss, val_loss, test_loss))
        print("Valid Dataset Metrics performance:{}\n".format(valid_metrics))
        print("Test Dataset Metrics performance:{}\n".format(test_metrics))
        print("Learning rate :{}\n\n".format(
            train_optimizer.param_groups[0]['lr']))

        # 记录训练日志
        TrainConfig.logger.info(msg=f"\n===== Epoch {step} =====\ntrain loss:{train_loss}, "
                                    f"valid loss:{val_loss},test loss:{test_loss}\n"
                                    f"valid metrics:{valid_metrics}\n"
                                    f"test metrics:{test_metrics}\n"
                                    f"learning rate:{train_optimizer.param_groups[0]['lr']}\n")

        # 模型保存策略:
        # 1. 前10个epoch不保存
        # 2. 从第11个epoch开始,每15个epoch保存一次
        if step <= 10:
            continue

        if (step-10) % 15 == 0:
            best_valid_loss = val_loss
            model_param = copy.deepcopy(model.state_dict())
            torch.save(model_param,
                       f'{TrainConfig.model_save_path}/{TrainConfig.model_name}_model_params_epoch_{step}_seed_{TrainConfig.seed}.pth')

    # 保存最后一个epoch的模型作为测试结果
    print("SAVING LAST EPOCH RESULT AS THE TEST RESULT!")
    torch.save(model.state_dict(),
               f'{TrainConfig.model_save_path}/TEST_{TrainConfig.model_name}_model_params_seed_{TrainConfig.seed}.pth')
    print("\n" + "==" * 10 + " Training Over " + "==" * 10)
    writer.close()


if __name__ == "__main__":
    train()
