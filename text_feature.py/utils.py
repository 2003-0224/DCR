import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix  # 引入 sklearn 的 f1_score
import random  # 引入 random 模块
import os  # 引入 os 模块以设置环境变量
import numpy as np

# 改进的设置随机种子函数
def set_seed(seed):
    random.seed(seed)  # 固定 Python 的 random 模块
    np.random.seed(seed)  # 固定 NumPy
    torch.manual_seed(seed)  # 固定 PyTorch CPU
    torch.cuda.manual_seed(seed)  # 固定 PyTorch GPU（单 GPU）
    torch.cuda.manual_seed_all(seed)  # 固定 PyTorch GPU（多 GPU）
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定 Python 哈希种子
    torch.backends.cudnn.deterministic = True  # 确保 CuDNN 确定性
    torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 优化


# EMA 类定义
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
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # 将 EMA 参数应用到模型
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        # 恢复原始参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 训练函数
def train(model, train_loader, optimizer, scheduler, ema, device, version= None):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        if version == 1 or version==7_1:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            target_start_pos = batch["target_start_pos"].to(device)  # 添加词级特征所需字段
            target_end_pos = batch["target_end_pos"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, target_start_pos, target_end_pos)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if ema is not None:
            ema.update()  # 每次优化后更新 EMA 参数

    scheduler.step()
    return total_loss / len(train_loader)

# 评估函数
def evaluate(model, test_loader, ema, device, num_labels=7, version = None):
    model.eval()
    if ema is not None:
        ema.apply_shadow()  # 应用 EMA 参数
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []  # 存储所有预测值
    all_labels = []  # 存储所有真实标签
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if version == 1 or version == 7_1:
                input_values = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(input_values, attention_mask=attention_mask)
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                target_start_pos = batch["target_start_pos"].to(device)  # 添加词级特征所需字段
                target_end_pos = batch["target_end_pos"].to(device)
                
                logits = model(input_ids, attention_mask, target_start_pos, target_end_pos)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 收集预测值和真实标签
            all_preds.extend(preds.cpu().numpy())  # 转换为 numpy 并添加到列表
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    # 计算 Weighted-F1 分数
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    # 使用混淆矩阵计算每个类别的准确率
    conf_mat = confusion_matrix(y_true=all_labels, y_pred=all_preds, labels=range(num_labels))
    class_totals = conf_mat.sum(axis=1)  # 每个类别的真实样本数
    class_correct = conf_mat.diagonal()  # 每个类别的正确预测数
    class_accuracies = {}
    for i in range(num_labels):
        if class_totals[i] > 0:  # 避免除以零
            class_accuracies[i] = class_correct[i] / class_totals[i]
        else:
            class_accuracies[i] = 0.0  # 如果类别无样本，准确率为 0
    
    if ema is not None:
        ema.restore()  # 恢复原始参数
    return total_loss / len(test_loader), accuracy, weighted_f1, class_accuracies