import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import random
import os
import numpy as np
import gc


# ====================================================================
# 1. 随机种子设置 (Set Seed) (保持不变)
# ====================================================================

def set_seed(seed):
    """
    固定所有随机种子以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ====================================================================
# 2. 指数移动平均 (EMA) 类
# ====================================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # 初始化影子参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        # 更新 EMA 参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name].copy_(new_average)

    def apply_shadow(self):
        # 将 EMA 参数应用到模型
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        # 恢复原始参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}


# ====================================================================
# 3. 训练函数 (Train Function) - 使用 MSE Loss
# ====================================================================

def train(model, train_loader, optimizer, scheduler, ema, device):
    model.train()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    for batch in tqdm(train_loader, desc=f"Training"):
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(input_values, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if ema is not None:
            ema.update()
    scheduler.step()
    return total_loss / len(train_loader)


# ====================================================================
# 4. 评估函数 (Evaluate Function) - 返回 MSE, MAE, Corr
# ====================================================================

def evaluate(model, test_loader, ema, device):  # 移除 num_labels
    model.eval()
    if ema is not None:
        ema.apply_shadow()
    total_loss = 0
    all_preds = []
    all_labels = []
    # 使用 MSE Loss 进行损失计算
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)  # torch.float32, 形状 [B, 1]
            logits = model(input_values, attention_mask=attention_mask)  # 形状 [B, 1]
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    avg_loss = total_loss / len(test_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # 计算回归指标 MSE, MAE, Corr
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    # Corr (皮尔逊相关系数)
    if len(all_labels) > 1 and all_labels.std() != 0 and all_preds.std() != 0:
        corr, _ = pearsonr(all_labels, all_preds)
    else:
        corr = 0.0
    if ema is not None:
        ema.restore()
    # 清理显存
    del all_preds, all_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return avg_loss, mse, mae, corr
