# import torch
# from tqdm import tqdm
# from sklearn.metrics import f1_score, confusion_matrix
# import numpy as np
#
#
# def train_epoch(model, train_loader, optimizer, device, modality_key, ema=None):
#     """
#     Train one epoch for a unimodal model.
#     - model: nn.Module, forward returns (logits, pooled_feat) or logits
#     - train_loader: yields batch dict with keys 'text'/'audio'/'video' and 'label'
#     - modality_key: 'T' / 'A' / 'V'
#     - ema: optional EMA object with .update() method
#     """
#     model.train()
#     total_loss = 0.0
#     modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
#
#     for batch in tqdm(train_loader, desc="Training", leave=False):
#         key = modality_map[modality_key]
#         # prepare inputs
#         if modality_key == 'T':
#             # dataset returns batch['text'] either tensor or dict (we handle both)
#             text_data = batch[key]
#             if isinstance(text_data, dict):
#                 inputs = {k: v.to(device) for k, v in text_data.items()}
#             else:
#                 inputs = text_data.to(device)
#         else:
#             inputs = batch[key].to(device)
#
#         labels = batch['label'].to(device)
#         # ensure labels shape (B,)
#         labels = labels.view(-1)
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
#         loss = torch.nn.functional.cross_entropy(logits, labels)
#         loss.backward()
#         optimizer.step()
#
#         # update EMA if provided (assumes ema.update() updates internal shadow weights)
#         if ema is not None:
#             try:
#                 ema.update()
#             except TypeError:
#                 # some EMA implementations expect model param list or model itself
#                 try:
#                     ema.update(model)
#                 except Exception:
#                     pass
#
#         total_loss += loss.item()
#
#     avg_loss = total_loss / max(1, len(train_loader))
#     return avg_loss
#
#
# def evaluate_epoch(model, test_loader, device, modality_key, num_labels=7):
#     """
#     Evaluate model on test_loader.
#     Returns: avg_loss, accuracy, weighted_f1, class_accuracies(dict)
#     """
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
#
#     modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
#
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Evaluating", leave=False):
#             key = modality_map[modality_key]
#             if modality_key == 'T':
#                 text_data = batch[key]
#                 if isinstance(text_data, dict):
#                     inputs = {k: v.to(device) for k, v in text_data.items()}
#                 else:
#                     inputs = text_data.to(device)
#             else:
#                 inputs = batch[key].to(device)
#
#             labels = batch['label'].to(device)
#             labels = labels.view(-1)
#
#             outputs = model(inputs)
#             logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
#             loss = torch.nn.functional.cross_entropy(logits, labels)
#             total_loss += loss.item()
#
#             preds = torch.argmax(logits, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#             all_preds.extend(preds.cpu().numpy().tolist())
#             all_labels.extend(labels.cpu().numpy().tolist())
#
#     avg_loss = total_loss / max(1, len(test_loader))
#     accuracy = (correct / total) if total > 0 else 0.0
#     weighted_f1 = 0.0
#     try:
#         weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
#     except Exception:
#         weighted_f1 = 0.0
#
#     # confusion matrix and per-class accuracies
#     if len(all_labels) > 0:
#         conf_mat = confusion_matrix(all_labels, all_preds, labels=list(range(num_labels)))
#         class_totals = conf_mat.sum(axis=1)
#         class_correct = conf_mat.diagonal()
#         class_accuracies = {i: (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0) for i in
#                             range(num_labels)}
#     else:
#         class_accuracies = {i: 0.0 for i in range(num_labels)}
#
#     return avg_loss, accuracy, weighted_f1, class_accuracies, all_labels, all_preds

import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
import random
import os


def quantize_for_acc7(scores: np.ndarray) -> np.ndarray:
    quantized_labels = np.full(scores.shape, 3, dtype=int)
    quantized_labels[(-3.0 <= scores) & (scores <= -2.5)] = 0
    quantized_labels[(-2.5 < scores) & (scores <= -1.5)] = 1
    quantized_labels[(-1.5 < scores) & (scores <= -0.5)] = 2
    quantized_labels[(-0.5 < scores) & (scores <= 0.5)] = 3
    quantized_labels[(0.5 < scores) & (scores <= 1.5)] = 4
    quantized_labels[(1.5 < scores) & (scores <= 2.5)] = 5
    quantized_labels[(2.5 < scores) & (scores <= 3.0)] = 6
    return quantized_labels


def train_epoch(model, train_loader, optimizer, device, modality_key, ema=None):
    model.train()
    total_loss = 0.0
    modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
    criterion = torch.nn.MSELoss()
    for batch in tqdm(train_loader, desc="Training", leave=False):
        key = modality_map[modality_key]
        if modality_key == 'T':
            text_data = batch[key]
            if isinstance(text_data, dict):
                inputs = {k: v.to(device) for k, v in text_data.items()}
            else:
                inputs = text_data.to(device)
        else:
            inputs = batch[key].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        regression_output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        loss = criterion(regression_output, labels)
        loss.backward()
        optimizer.step()
        if ema is not None:
            try:
                ema.update()
            except TypeError:
                try:
                    ema.update(model)
                except Exception:
                    pass
        total_loss += loss.item()
    avg_loss = total_loss / max(1, len(train_loader))
    return avg_loss


def evaluate_epoch(model, test_loader, device, modality_key, num_labels=None, ema=None):
    model.eval()
    if ema is not None:
        ema.apply_shadow()
    total_loss = 0.0
    all_preds_cont = []
    all_labels_cont = []
    modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            key = modality_map[modality_key]

            if modality_key == 'T':
                text_data = batch[key]
                if isinstance(text_data, dict):
                    inputs = {k: v.to(device) for k, v in text_data.items()}
                else:
                    inputs = text_data.to(device)
            else:
                inputs = batch[key].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            regression_output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            loss = criterion(regression_output, labels)
            total_loss += loss.item()
            all_preds_cont.extend(regression_output.cpu().squeeze().numpy())
            all_labels_cont.extend(labels.cpu().squeeze().numpy())
    if ema is not None:
        ema.restore()
    avg_loss = total_loss / max(1, len(test_loader))
    all_labels_cont = np.array(all_labels_cont)
    all_preds_cont = np.array(all_preds_cont)
    mae = np.mean(np.abs(all_preds_cont - all_labels_cont))
    try:
        correlation, _ = pearsonr(all_labels_cont, all_preds_cont)
        if np.isnan(correlation): correlation = 0.0
    except ValueError:
        correlation = 0.0
    # 真实标签量化
    quantized_labels = quantize_for_acc7(all_labels_cont)
    # 预测分数量化
    quantized_preds = quantize_for_acc7(all_preds_cont)
    # 计算 ACC7
    acc7 = accuracy_score(quantized_labels, quantized_preds)
    corr_dict = {'Corr': correlation, 'MAE': mae}  # 包含 MAE 也方便后续打印
    return avg_loss, acc7, mae, corr_dict, all_labels_cont, all_preds_cont
