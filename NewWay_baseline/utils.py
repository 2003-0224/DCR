import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error, accuracy_score
from scipy.stats import pearsonr
import random
import os
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


def _default_prepare_inputs(batch, modalities, device, use_raw_audio):
    modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
    inputs = {}
    for mod in modalities:
        key = modality_map[mod]
        if key == 'text' and isinstance(batch[key], dict):
            inputs[mod] = {
                'input_ids': batch[key]['input_ids'].to(device),
                'attention_mask': batch[key]['attention_mask'].to(device),
                'target_start_pos': batch[key]['target_start_pos'].to(device),
                'target_end_pos': batch[key]['target_end_pos'].to(device)
            }
        elif key == 'audio' and use_raw_audio and isinstance(batch[key], dict):
            inputs[mod] = {
                'input_values': batch[key]['input_values'].to(device),
                'attention_mask': batch[key]['attention_mask'].to(device)
            }
        else:
            inputs[mod] = batch[key].to(device)
    labels = batch['label'].to(device)
    return inputs, labels


# Training function
def train(model, train_loader, optimizer, scheduler, ema, device,
          use_cam_loss, use_raw_audio, cam_type,
          task_type='classification', prepare_inputs_fn=None):
    model.train()
    total_loss = 0
    # Loss functions
    loss_fn_cls = nn.CrossEntropyLoss()
    loss_fn_reg = nn.MSELoss()
    for batch in tqdm(train_loader, desc="Training"):
        if prepare_inputs_fn:
            inputs, labels = prepare_inputs_fn(batch, model.modalities, device)
        else:
            inputs, labels = _default_prepare_inputs(batch, model.modalities, device, use_raw_audio)
        if task_type == 'regression':
            labels = labels.view(-1, 1).float()
        else:
            labels = labels.view(-1).long()
        optimizer.zero_grad()
        # Regression Logic (Standard MSE training)
        if task_type == 'regression':
            outputs = model(inputs, labels)
            if isinstance(outputs, tuple):
                loss = outputs[1]
            else:
                loss = loss_fn_reg(outputs, labels)
            loss.backward()
            total_loss += loss.item()
        else:
            if use_cam_loss:
                # T_to_CAM / Tcam_to_CAM
                if cam_type in ['T_to_CAM', 'Tcam_to_CAM']:
                    outputs, cam_loss, text_loss = model(inputs, labels)
                    loss = loss_fn_cls(outputs, labels)
                    loss_all = loss + 0.2 * cam_loss + 0.2 * text_loss
                # AV_to_CAM / AVcam_to_CAM
                elif cam_type in ['AV_to_CAM', 'AVcam_to_CAM']:
                    ret = model(inputs, labels)
                    if len(ret) == 4:
                        outputs, cam_loss, audio_loss, video_loss = ret
                        loss = loss_fn_cls(outputs, labels)
                        loss_all = loss + 0.05 * cam_loss + 0.05 * audio_loss + 0.05 * video_loss
                    else:
                        outputs, cam_loss, *_ = ret
                        loss = loss_fn_cls(outputs, labels)
                        loss_all = loss + 0.1 * cam_loss
                loss_all.backward()
                total_loss += loss_all.item()
            else:
                outputs = model(inputs, labels)
                if isinstance(outputs, tuple):
                    loss = outputs[1]
                else:
                    loss = loss_fn_cls(outputs, labels)
                loss.backward()
                total_loss += loss.item()
        optimizer.step()
        if ema is not None:
            ema.update()
    scheduler.step()
    return total_loss / len(train_loader)


def evaluate_regression(model, test_loader, ema, device,
                        use_cam_loss=False, cam_type="T_to_CAM",
                        use_raw_audio=False, prepare_inputs_fn=None):
    model.eval()
    if ema is not None:
        ema.apply_shadow()
    total_loss = 0
    all_preds = []
    all_labels = []
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Regression"):
            if prepare_inputs_fn:
                inputs, labels = prepare_inputs_fn(batch, model.modalities, device)
            else:
                inputs, labels = _default_prepare_inputs(batch, model.modalities, device, use_raw_audio)
            labels = labels.view(-1, 1).float()
            outputs = model(inputs, labels=None)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    avg_loss = total_loss / len(test_loader)
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    mae = mean_absolute_error(labels_arr, preds_arr)
    pearson_corr, _ = pearsonr(labels_arr, preds_arr)

    def map_to_7_class(scores):
        y_class = []
        for s in scores:
            if s <= -2.5:
                c = 0
            elif -2.5 < s <= -1.5:
                c = 1
            elif -1.5 < s <= -0.5:
                c = 2
            elif -0.5 < s <= 0.5:
                c = 3
            elif 0.5 < s <= 1.5:
                c = 4
            elif 1.5 < s <= 2.5:
                c = 5
            else:
                c = 6
            y_class.append(c)
        return np.array(y_class)  # 直接返回 Numpy 数组

    # 转换
    def map_to_5_class(scores):
        y_class = []
        for s in scores:
            if s <= -1.5:
                c = 0
            elif -1.5 < s <= -0.5:
                c = 1
            elif -0.5 < s <= 0.5:
                c = 2
            elif 0.5 < s <= 1.5:
                c = 3
            else:
                c = 4
            y_class.append(c)
        return np.array(y_class)

    preds_7 = map_to_7_class(preds_arr)
    labels_7 = map_to_7_class(labels_arr)
    preds_5 = map_to_5_class(preds_arr)
    labels_5 = map_to_5_class(labels_arr)
    for i in range(7):
        # 提取属于该类别的索引
        class_idx = (labels_7 == i)
        class_count = np.sum(class_idx)

        if class_count > 0:
            correct_count = np.sum(preds_7[class_idx] == i)
            acc = correct_count / class_count
            print(f"Class {i} Acc: {acc:.4f} | Correct: {correct_count}/{class_count}")
        else:
            print(f"Class {i}: No samples present in this batch/split.")
    acc7 = accuracy_score(labels_7, preds_7)
    acc5 = accuracy_score(labels_5, preds_5)
    if ema is not None:
        ema.restore()
    return avg_loss, mae, pearson_corr, acc7, acc5


def evaluate(model, test_loader, ema, device, num_labels=7,
             use_cam_loss=False, cam_type="T_to_CAM",
             use_raw_audio=False, prepare_inputs_fn=None):
    model.eval()
    if ema is not None:
        ema.apply_shadow()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Classification"):
            if prepare_inputs_fn:
                inputs, labels = prepare_inputs_fn(batch, model.modalities, device)
            else:
                inputs, labels = _default_prepare_inputs(batch, model.modalities, device, use_raw_audio)
            labels = labels.view(-1).long()
            if use_cam_loss:
                ret = model(inputs, labels=None)
                if isinstance(ret, tuple):
                    outputs = ret[0]
                else:
                    outputs = ret
            else:
                outputs = model(inputs, labels=None)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            total_loss += loss.item()
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
    class_accuracies = {i: (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0)
                        for i in range(num_labels)}
    if ema is not None:
        ema.restore()
    return avg_loss, accuracy, weighted_f1, class_accuracies
