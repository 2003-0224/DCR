# import torch
# from tqdm import tqdm
# from sklearn.metrics import f1_score, confusion_matrix  # 引入 sklearn 的 f1_score
# import random  # 引入 random 模块
# import os  # 引入 os 模块以设置环境变量
# import numpy as np
#
#
# # Random seed setting
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# # EMA class
# class EMA:
#     def __init__(self, model, decay=0.999):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()
#
#     def update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
#                 self.shadow[name] = new_average.clone()
#
#     def apply_shadow(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.backup[name] = param.data.clone()
#                 param.data = self.shadow[name]
#
#     def restore(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#         self.backup = {}
#
#
# # Training function
# def train(model, train_loader, optimizer, scheduler, ema, device, use_cam_loss, use_raw_audio, cam_type):
#     """
#     Train the model for one epoch.
#
#     Args:
#         model: The multimodal fusion model.
#         train_loader: DataLoader for training data.
#         optimizer: Optimizer for training.
#         scheduler: Learning rate scheduler.
#         ema: Exponential Moving Average (optional).
#         device: Device to run the model on.
#
#     Returns:
#         Average training loss for the epoch.
#     """
#     model.train()
#     total_loss = 0
#
#     # Mapping from modality codes to batch keys
#     modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
#
#     for batch in tqdm(train_loader, desc="Training"):
#         # Extract features and labels from batch dictionary
#         inputs = {}
#         for mod in model.modalities:
#             key = modality_map[mod]
#             if key == 'text':
#                 # 特殊处理 text 模态的字典
#                 inputs[mod] = {
#                     'input_ids': batch[key]['input_ids'].to(device),
#                     'attention_mask': batch[key]['attention_mask'].to(device),
#                     'target_start_pos': batch[key]['target_start_pos'].to(device),
#                     'target_end_pos': batch[key]['target_end_pos'].to(device)
#                 }
#             elif key == 'audio' and use_raw_audio:
#                 inputs[mod] = {
#                     'input_values': batch[key]['input_values'].to(device),
#                     'attention_mask': batch[key]['attention_mask'].to(device)
#                 }
#             else:
#                 # 音频和视频直接移动张量
#                 inputs[mod] = batch[key].to(device)
#         labels = batch['label'].squeeze().to(device)  # Remove extra dimension [batch_size, 1] -> [batch_size]
#
#         optimizer.zero_grad()
#         if use_cam_loss:
#             # T_to_CAM:
#             if cam_type == 'T_to_CAM' or cam_type == 'Tcam_to_CAM':
#                 outputs, cam_loss, text_loss = model(inputs, labels)  # Model expects a dictionary of inputs
#                 loss = torch.nn.functional.cross_entropy(outputs, labels)
#                 loss_all = loss + 0.2 * cam_loss + 0.2 * text_loss
#
#             # AV_to_CAM:
#             elif cam_type == 'AV_to_CAM' or cam_type == 'AVcam_to_CAM':
#                 outputs, cam_loss, audio_loss, video_loss = model(inputs,
#                                                                   labels)  # Model expects a dictionary of inputs
#                 loss = torch.nn.functional.cross_entropy(outputs, labels)
#                 loss_all = loss + 0.05 * cam_loss + 0.2 * audio_loss + 0.2 * video_loss
#                 # loss_all = loss
#
#             loss_all.backward()
#         else:
#             outputs = model(inputs, labels)
#             loss = torch.nn.functional.cross_entropy(outputs, labels)
#             loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#
#         if ema is not None:
#             ema.update()
#
#     scheduler.step()
#     return total_loss / len(train_loader)
#
#
# # Evaluation function
# def evaluate(model, test_loader, ema, device, num_labels=7, use_cam_loss=False, use_raw_audio=False,
#              cam_type="T_to_CAM"):
#     """
#     Evaluate the model on the test set.
#
#     Args:
#         model: The multimodal fusion model.
#         test_loader: DataLoader for test data.
#         ema: Exponential Moving Average (optional).
#         device: Device to run the model on.
#         num_labels: Number of emotion classes (default: 7 for MELD).
#
#     Returns:
#         Tuple of (average loss, accuracy, weighted F1 score, class accuracies).
#     """
#     model.eval()
#     if ema is not None:
#         ema.apply_shadow()
#
#     total_loss = 0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
#
#     # Mapping from modality codes to batch keys
#     modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
#
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Evaluating"):
#             # Extract features and labels from batch dictionary
#             # Extract features and labels from batch dictionary
#             inputs = {}
#             for mod in model.modalities:
#                 key = modality_map[mod]
#                 if key == 'text':
#                     # 特殊处理 text 模态的字典
#                     inputs[mod] = {
#                         'input_ids': batch[key]['input_ids'].to(device),
#                         'attention_mask': batch[key]['attention_mask'].to(device),
#                         'target_start_pos': batch[key]['target_start_pos'].to(device),
#                         'target_end_pos': batch[key]['target_end_pos'].to(device)
#                     }
#                 elif key == 'audio' and use_raw_audio:
#                     inputs[mod] = {
#                         'input_values': batch[key]['input_values'].to(device),
#                         'attention_mask': batch[key]['attention_mask'].to(device)
#                     }
#                 else:
#                     # 音频和视频直接移动张量
#                     inputs[mod] = batch[key].to(device)
#             labels = batch['label'].squeeze().to(device)  # Remove extra dimension [batch_size, 1] -> [batch_size]
#
#             if use_cam_loss:
#                 if cam_type == 'T_to_CAM' or cam_type == 'Tcam_to_CAM':
#                     outputs, cam_loss, text_loss = model(inputs, labels=None)  # Model expects a dictionary of inputs
#                 elif cam_type == 'AV_to_CAM' or cam_type == 'AVcam_to_CAM':
#                     outputs, cam_loss, audio_loss, video_loss = model(inputs,
#                                                                       labels=None)  # Model expects a dictionary of inputs
#             else:
#                 outputs = model(inputs, labels=None)
#             loss = torch.nn.functional.cross_entropy(outputs, labels)
#             total_loss += loss.item()
#             # audio_loss += audio_loss.item()
#             # video_loss += video_loss.item()
#
#             preds = torch.argmax(outputs, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#     avg_loss = total_loss / len(test_loader)
#     accuracy = correct / total
#     weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
#     conf_mat = confusion_matrix(all_labels, all_preds, labels=range(num_labels))
#     class_totals = conf_mat.sum(axis=1)
#     class_correct = conf_mat.diagonal()
#     class_accuracies = {i: (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0) for i in
#                         range(num_labels)}
#
#     if ema is not None:
#         ema.restore()
#
#     return avg_loss, accuracy, weighted_f1, class_accuracies

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix  # 引入 sklearn 的 f1_score
import random  # 引入 random 模块
import os  # 引入 os 模块以设置环境变量
import numpy as np


def convert_to_acc7_label(score):
    if np.isnan(score):
        return 3
    if -3.0 <= score <= -2.5:
        return 0  # 强烈消极 (-3)
    elif -2.5 < score <= -1.5:
        return 1  # 消极 (-2)
    elif -1.5 < score <= -0.5:
        return 2  # 弱消极 (-1)
    elif -0.5 < score <= 0.5:
        return 3  # 中性 (0)
    elif 0.5 < score <= 1.5:
        return 4  # 弱积极 (+1)
    elif 1.5 < score <= 2.5:
        return 5  # 积极 (+2)
    elif 2.5 < score <= 3.0:
        return 6  # 强烈积极 (+3)
    else:
        return 3


def convert_to_acc2_label2(score):
    if score < 0:
        return 0
    else:
        return 1


def convert_to_acc5_label5(score):
    if score <= -1.5:
        return 0
    elif -1.5 < score <= -0.5:
        return 1
    elif -0.5 < score <= 0.5:
        return 2
    elif 0.5 < score <= 1.5:
        return 3
    else:
        return 4



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


def evaluate(model, test_loader, ema, device):
    model.eval()
    if ema is not None:
        ema.apply_shadow()
    total_loss = 0
    all_preds = []
    all_labels = []
    modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = {}
            for mod in model.modalities:
                key = modality_map[mod]
                if key == 'text':
                    inputs[mod] = {
                        'input_ids': batch[key]['input_ids'].to(device),
                        'attention_mask': batch[key]['attention_mask'].to(device),
                        'target_start_pos': batch[key]['target_start_pos'].to(device),
                        'target_end_pos': batch[key]['target_end_pos'].to(device)
                    }
                else:
                    inputs[mod] = batch[key].to(device)
            labels = batch['label'].float().to(device)
            if isinstance(model, torch.nn.DataParallel):
                outputs, _ = model.module(inputs)
            else:
                outputs, _ = model(inputs)
            # 回归任务使用 MSE Loss
            loss = torch.nn.functional.mse_loss(outputs.squeeze(-1), labels.squeeze(-1))
            total_loss += loss.item()
            all_preds.extend(outputs.squeeze(-1).cpu().numpy())
            all_labels.extend(labels.squeeze(-1).cpu().numpy())
    avg_loss = total_loss / len(test_loader)
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    mae = np.mean(np.abs(all_preds_np - all_labels_np))
    if all_preds_np.std() > 1e-6 and all_labels_np.std() > 1e-6:
        corr_matrix = np.corrcoef(all_preds_np, all_labels_np)
        corr = corr_matrix[0, 1]
    else:
        corr = 0.0
    discrete_preds = np.array([convert_to_acc7_label(p) for p in all_preds_np])
    discrete_labels = np.array([convert_to_acc7_label(l) for l in all_labels_np])
    two_discrete_preds = np.array([convert_to_acc2_label2(l) for l in all_preds_np])
    two_discrete_labels = np.array([convert_to_acc2_label2(l) for l in all_labels_np])
    correct_two = (two_discrete_preds == two_discrete_labels).sum()
    correct = (discrete_preds == discrete_labels).sum()
    total_two = len(correct_two)
    total = len(discrete_labels)
    acc2 = correct_two / total_two if total_two > 0 else 0.0
    acc7 = correct / total if total > 0 else 0.0
    if ema is not None:
        ema.restore()
    return avg_loss, mae, corr, acc7, acc2


# --- 对 train 函数的修改（仅移除分类/CAM 依赖） ---
def train(model, train_loader, optimizer, scheduler, ema, device, use_raw_audio):
    model.train()
    total_loss = 0
    modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
    for batch in tqdm(train_loader, desc="Training"):
        inputs = {}
        for mod in model.modalities:
            key = modality_map[mod]
            if key == 'text':
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
                inputs[mod] = batch[key].to(device)
            labels = batch['label'].float().to(device)
        optimizer.zero_grad()
        if isinstance(model, torch.nn.DataParallel):
            outputs, _ = model.module(inputs)
        else:
            outputs, _ = model(inputs)
        # 回归任务使用 MSE Loss
        loss = torch.nn.functional.mse_loss(outputs.squeeze(-1), labels.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if ema is not None:
            ema.update()
    scheduler.step()
    return total_loss / len(train_loader)
