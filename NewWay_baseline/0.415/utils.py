import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix  # 引入 sklearn 的 f1_score
import random  # 引入 random 模块
import os  # 引入 os 模块以设置环境变量
import numpy as np


# Random seed setting
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# EMA class
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# Training function
def train(model, train_loader, optimizer, scheduler, ema, device, use_cam_loss, use_raw_audio, cam_type):
    """
    Train the model for one epoch.
    
    Args:
        model: The multimodal fusion model.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        ema: Exponential Moving Average (optional).
        device: Device to run the model on.
    
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0

    # Mapping from modality codes to batch keys
    modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}

    for batch in tqdm(train_loader, desc="Training"):
        # Extract features and labels from batch dictionary
        inputs = {}
        for mod in model.modalities:
            key = modality_map[mod]
            if key == 'text':
                # 特殊处理 text 模态的字典
                inputs[mod] = {
                    'input_ids': batch[key]['input_ids'].to(device),
                    'attention_mask': batch[key]['attention_mask'].to(device),
                    'target_start_pos': batch[key]['target_start_pos'].to(device),
                    'target_end_pos': batch[key]['target_end_pos'].to(device)
                }
            elif key == 'audio' and use_raw_audio:
                inputs[mod] = {
                    'input_values': batch[key]['input_values'].to(device),
                    'attention_mask': batch[key]['attention_mask'].to(device)
                }
            else:
                # 音频和视频直接移动张量
                inputs[mod] = batch[key].to(device)
        labels = batch['label'].squeeze().to(device)  # Remove extra dimension [batch_size, 1] -> [batch_size]

        optimizer.zero_grad()
        if use_cam_loss:
            # 简化CAM损失处理 - 只支持T_to_CAM
            if cam_type == 'T_to_CAM':
                outputs, cam_loss = model(inputs, labels)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss_all = loss + 0.1 * cam_loss  # 降低CAM损失权重
            else:
                # 其他CAM类型使用简化处理
                outputs = model(inputs, labels)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss_all = loss
            
            loss_all.backward()
        else:
            outputs = model(inputs, labels)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
        
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if ema is not None:
            ema.update()

    scheduler.step()
    return total_loss / len(train_loader)


# Evaluation function
def evaluate(model, test_loader, ema, device, num_labels=7, use_cam_loss=False, use_raw_audio=False,
             cam_type="T_to_CAM"):
    """
    Evaluate the model on the test set.
    
    Args:
        model: The multimodal fusion model.
        test_loader: DataLoader for test data.
        ema: Exponential Moving Average (optional).
        device: Device to run the model on.
        num_labels: Number of emotion classes (default: 7 for MELD).
    
    Returns:
        Tuple of (average loss, accuracy, weighted F1 score, class accuracies).
    """
    model.eval()
    if ema is not None:
        ema.apply_shadow()

    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Mapping from modality codes to batch keys
    modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Extract features and labels from batch dictionary
            # Extract features and labels from batch dictionary
            inputs = {}
            for mod in model.modalities:
                key = modality_map[mod]
                if key == 'text':
                    # 特殊处理 text 模态的字典
                    inputs[mod] = {
                        'input_ids': batch[key]['input_ids'].to(device),
                        'attention_mask': batch[key]['attention_mask'].to(device),
                        'target_start_pos': batch[key]['target_start_pos'].to(device),
                        'target_end_pos': batch[key]['target_end_pos'].to(device)
                    }
                elif key == 'audio' and use_raw_audio:
                    inputs[mod] = {
                        'input_values': batch[key]['input_values'].to(device),
                        'attention_mask': batch[key]['attention_mask'].to(device)
                    }
                else:
                    # 音频和视频直接移动张量
                    inputs[mod] = batch[key].to(device)
            labels = batch['label'].squeeze().to(device)  # Remove extra dimension [batch_size, 1] -> [batch_size]

            if use_cam_loss:
                # 简化评估时的CAM损失处理
                if cam_type == 'T_to_CAM':
                    outputs, _ = model(inputs, labels=None)
                else:
                    outputs = model(inputs, labels=None)
            else:
                outputs = model(inputs, labels=None)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            total_loss += loss.item()
            # audio_loss += audio_loss.item()
            # video_loss += video_loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_mat = confusion_matrix(all_labels, all_preds, labels=range(num_labels))
    class_totals = conf_mat.sum(axis=1)
    class_correct = conf_mat.diagonal()
    class_accuracies = {i: (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0) for i in
                        range(num_labels)}

    if ema is not None:
        ema.restore()

    return avg_loss, accuracy, weighted_f1, class_accuracies
