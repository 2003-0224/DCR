"""Train bimodal fusion models (T+A and T+V) using raw text with RoBERTa fine-tuning.

Each fusion model distills CAM-style temporal weights from the auxiliary modality (audio or
video) into the text stream, then performs cross-attention fusion before classification.
Training strategy mirrors the baseline setup: cosine LR schedule, EMA, patience-based early
stopping, and detailed metric reporting.
"""

import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import EMA, set_seed
from NewWay_baseline.T_raw_data_dataset import T_raw_MELDDataset


def normalize_sample_name(name):
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    name = str(name).replace("\\", "/")
    parts = name.split("/")
    if len(parts) >= 3 and parts[-3].isdigit() and parts[-2].isdigit():
        return f"dia{parts[-3]}_utt{parts[-2]}"
    base = parts[-1]
    return os.path.splitext(base)[0]


class CAMModule(nn.Module):
    """CAM module used to distill auxiliary modality importance over time."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None):
        pooled = features.mean(dim=1)
        logits = self.classifier(pooled)
        weights = self.classifier.weight
        cam = torch.einsum('blh,ch->blc', features, weights) + self.classifier.bias.view(1, 1, -1)

        time_weights = torch.ones(features.size(0), features.size(1), device=features.device) / features.size(1)
        if labels is not None and self.training:
            idx = labels.view(-1, 1, 1).expand(-1, features.size(1), 1)
            true_cam = cam.gather(dim=2, index=idx).squeeze(-1)
            time_weights = F.softmax(true_cam, dim=1)

        aux_loss = F.cross_entropy(logits, labels) if labels is not None else torch.zeros(1, device=features.device)
        return time_weights, logits, aux_loss


class FusionModel(nn.Module):
    """Text (RoBERTa) + auxiliary modality fusion with CAM-guided cross-attention."""

    def __init__(self, hidden_dim: int, num_classes: int, aux_dim: int, num_heads: int = 4, target_seq_len: int = 38):
        super().__init__()
        from transformers import RobertaModel

        self.target_seq_len = target_seq_len
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.text_proj = nn.Linear(1024, hidden_dim)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=1,
        )

        self.aux_proj = nn.Linear(aux_dim, hidden_dim)
        self.aux_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=1,
        )

        self.cam = CAMModule(hidden_dim, num_classes)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fusion_ln = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.aux_loss_weight = 0.2

    def _extract_text_tokens(self, text_inputs: Dict[str, torch.Tensor]):
        outputs = self.roberta(text_inputs['input_ids'], text_inputs['attention_mask'])
        hidden_states = outputs.last_hidden_state
        batch_size, _, hidden = hidden_states.shape
        tokens = torch.zeros(batch_size, self.target_seq_len, hidden, device=hidden_states.device)
        masks = torch.zeros(batch_size, self.target_seq_len, 1, device=hidden_states.device)
        start_pos = text_inputs['target_start_pos']
        end_pos = text_inputs['target_end_pos']
        for i in range(batch_size):
            start = start_pos[i].item()
            end = end_pos[i].item()
            segment = hidden_states[i, start:end]
            length = min(segment.size(0), self.target_seq_len)
            if length > 0:
                tokens[i, :length] = segment[:length]
                masks[i, :length] = 1.0
        return tokens, masks

    def forward(self, text_inputs, aux_seq, labels=None):
        text_tokens, text_mask = self._extract_text_tokens(text_inputs)
        text_h = self.text_proj(text_tokens)
        text_h = self.text_encoder(text_h)

        if aux_seq.dim() == 2:
            aux_seq = aux_seq.unsqueeze(1)
        aux_h = self.aux_proj(aux_seq.float())
        aux_h = self.aux_encoder(aux_h)

        time_weights, aux_logits, aux_ce = self.cam(aux_h, labels)
        weighted_aux = torch.einsum('bl,blh->bh', time_weights, aux_h)
        attn_out, _ = self.cross_attn(text_h, aux_h, aux_h)
        fused = self.fusion_ln(text_h + attn_out)
        pooled = fused.mean(dim=1) + weighted_aux
        logits = self.classifier(pooled)
        total_loss = torch.zeros(1, device=logits.device)
        if labels is not None:
            total_loss = F.cross_entropy(logits, labels) + self.aux_loss_weight * aux_ce
        return logits, aux_logits, total_loss


def collate_aux_features(batch: Dict[str, torch.Tensor], modality: str) -> torch.Tensor:
    feat = batch[modality]
    if feat.dim() == 2:
        feat = feat.unsqueeze(1)
    return feat.float()


def train_epoch(model, loader, optimizer, device, fusion_mode):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        text_inputs = {k: v.to(device) for k, v in batch['text'].items()}
        aux_key = 'audio' if fusion_mode == 'AT' else 'video'
        aux_seq = collate_aux_features(batch, aux_key).to(device)
        labels = batch['label'].squeeze(-1).to(device)

        logits, aux_logits, loss = model(text_inputs, aux_seq, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_epoch(model, loader, device, fusion_mode, num_labels):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            text_inputs = {k: v.to(device) for k, v in batch['text'].items()}
            aux_key = 'audio' if fusion_mode == 'AT' else 'video'
            aux_seq = collate_aux_features(batch, aux_key).to(device)
            labels = batch['label'].squeeze(-1).to(device)
            logits, aux_logits, loss = model(text_inputs, aux_seq, labels)
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    preds = torch.argmax(logits_tensor, dim=1).numpy()
    labels_np = labels_tensor.numpy()

    accuracy = float((preds == labels_np).mean())
    try:
        weighted_f1 = f1_score(labels_np, preds, average='weighted')
    except ValueError:
        weighted_f1 = 0.0
    conf_mat = confusion_matrix(labels_np, preds, labels=list(range(num_labels)))
    class_totals = conf_mat.sum(axis=1)
    class_correct = conf_mat.diagonal()
    class_acc = {
        f"class_{i}_acc": (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0)
        for i in range(num_labels)
    }
    return total_loss / len(loader), accuracy, weighted_f1, class_acc


def run_fusion_training(fusion_mode: str):
    assert fusion_mode in {'AT', 'VT'}
    seed = 42
    num_labels = 7
    num_epochs = 60
    batch_size = 32
    hidden_dim = 512
    lr = 1e-5
    eta_min = 1e-6
    patience = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    data_paths = {
        'train': {
            'text': '/data/yuyangchen/data/MELD/text_features/train_text_features.npz',
            'text_raw_path': '/data/yuyangchen/data/MELD/processed_train_T_emo.csv',
            'audio': '/data/yuyangchen/data/MELD/audio_features/train_audio_features_processed.npz',
            'video': '/data/yuyangchen/data/MELD/face_features/train_video_features_processed.npz',
        },
        'test': {
            'text': '/data/yuyangchen/data/MELD/text_features/test_text_features.npz',
            'text_raw_path': '/data/yuyangchen/data/MELD/processed_test_T_emo.csv',
            'audio': '/data/yuyangchen/data/MELD/audio_features/test_audio_features_processed.npz',
            'video': '/data/yuyangchen/data/MELD/face_features/test_video_features_processed.npz',
        }
    }

    modalities = ['T', 'A'] if fusion_mode == 'AT' else ['T', 'V']
    train_dataset = T_raw_MELDDataset(
        data_paths['train']['text'],
        data_paths['train']['audio'],
        data_paths['train']['video'],
        modalities=modalities,
        split='train',
        feature_type='sequence_features',
        text_path=data_paths['train']['text_raw_path']
    )
    test_dataset = T_raw_MELDDataset(
        data_paths['test']['text'],
        data_paths['test']['audio'],
        data_paths['test']['video'],
        modalities=modalities,
        split='test',
        feature_type='sequence_features',
        text_path=data_paths['test']['text_raw_path']
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    aux_key = 'audio' if fusion_mode == 'AT' else 'video'
    aux_dim = train_dataset.aligned_data[aux_key].shape[-1]

    model = FusionModel(hidden_dim, num_labels, aux_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=8, eta_min=eta_min)
    ema = EMA(model, decay=0.999)

    emotion_labels = {
        3: "anger",
        5: "disgust",
        6: "fear",
        0: "joy",
        2: "neutral",
        1: "sadness",
        4: "surprise"
    }

    best_f1 = 0.0
    best_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, fusion_mode)

        ema.apply_shadow()
        train_eval_loss, train_acc, train_f1, train_class_acc = evaluate_epoch(model, train_eval_loader, device, fusion_mode, num_labels)
        test_loss, test_acc, test_f1, test_class_acc = evaluate_epoch(model, test_loader, device, fusion_mode, num_labels)
        ema.restore()

        scheduler.step()

        print(f"[Fusion {fusion_mode}] Epoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss(opt): {train_loss:.4f} | Train Loss(eval): {train_eval_loss:.4f} | Train Acc: {train_acc:.4f} | Train W-F1: {train_f1:.4f}"
        )
        print(f"Test  Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test W-F1: {test_f1:.4f}")

        train_class_str = "  " + "  ".join([f"{emotion_labels.get(int(k.split('_')[1]), k)}: {v:.4f}" for k, v in train_class_acc.items()])
        test_class_str = "  " + "  ".join([f"{emotion_labels.get(int(k.split('_')[1]), k)}: {v:.4f}" for k, v in test_class_acc.items()])
        print("Train Class-wise:" + train_class_str)
        print("Test  Class-wise:" + test_class_str)

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_acc = test_acc
            patience_counter = 0
            ema.apply_shadow()
            best_state = model.state_dict()
            ema.restore()
            print(f"  >> New best W-F1: {best_f1:.4f} (Acc {best_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("  Early stopping triggered.")
                break

    if best_state is not None:
        save_root = f"/data/yuyangchen/checkpoints/fusion_{fusion_mode.lower()}"
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, f"fusion_{fusion_mode.lower()}_best_wf1_{best_f1:.4f}_seed_{seed}.pth")
        torch.save(best_state, save_path)
        print(f"Saved best fusion model to {save_path}")
        print(f"Final Best Metrics -> Accuracy: {best_acc:.4f}, Weighted-F1: {best_f1:.4f}")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    for mode in ['AT', 'VT']:
        run_fusion_training(mode)


if __name__ == "__main__":
    main()
