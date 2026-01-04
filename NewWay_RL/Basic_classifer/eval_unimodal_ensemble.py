#!/usr/bin/env python3
"""Evaluate text expert with optional multimodal fallback using precomputed features.

判断规则：先采用文本专家给出的预测；如果该样本文本预测错误且提供了多模态融合预测，
则退化为多模态结果。最终输出整体准确率、加权 F1、逐类准确率，并报告文本模态独立准确率。
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

from src.feature_dataset import MELDDataset
from unimodal_models import TextClassifier, AudioClassifier, VideoClassifier
from src.utils import set_seed


def normalize_sample_name(name):
    if isinstance(name, bytes):
        name = name.decode('utf-8')
    name = str(name).replace('\\', '/')
    parts = name.split('/')
    if len(parts) >= 3 and parts[-3].isdigit() and parts[-2].isdigit():
        return f"dia{parts[-3]}_utt{parts[-2]}"
    base = parts[-1]
    return Path(base).stem


def build_dataloader(text_npz, audio_npz, video_npz, split, batch_size, feature_type_hint="sequence_features"):
    try:
        dataset = MELDDataset(
            text_npz,
            audio_npz,
            video_npz,
            modalities=['T', 'A', 'V'],
            split=split,
            feature_type=feature_type_hint,
        )
    except KeyError:
        dataset = MELDDataset(
            text_npz,
            audio_npz,
            video_npz,
            modalities=['T', 'A', 'V'],
            split=split,
            feature_type='pooled_features',
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataset, loader


def load_experts(args, dataset, num_labels, device):
    text_feat_dim = dataset.aligned_data['text'].shape[-1]
    text_seq_len = dataset.aligned_data['text'].shape[1] if dataset.aligned_data['text'].ndim == 3 else 1
    audio_feat_dim = dataset.aligned_data['audio'].shape[-1]
    video_feat_dim = dataset.aligned_data['video'].shape[-1]

    text_model = TextClassifier(
        hidden_dim=args.hidden_dim,
        num_classes=num_labels,
        use_precomputed=True,
        input_dim=text_feat_dim,
        target_seq_len=text_seq_len,
    ).to(device)
    text_state = torch.load(args.text_checkpoint, map_location=device)
    text_model.load_state_dict(text_state, strict=False)
    audio_model = AudioClassifier(hidden_dim=args.hidden_dim, num_classes=num_labels).to(device)
    audio_model.load_state_dict(torch.load(args.audio_checkpoint, map_location=device))
    video_model = VideoClassifier(hidden_dim=args.hidden_dim, num_classes=num_labels).to(device)
    video_model.load_state_dict(torch.load(args.video_checkpoint, map_location=device))

    for model in (text_model, audio_model, video_model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return text_model, audio_model, video_model


def load_multimodal_predictions(train_npz, test_npz):
    multimodal = {}
    for split, path in [('train', train_npz), ('test', test_npz)]:
        data = np.load(path, allow_pickle=True)
        names = [normalize_sample_name(n) for n in data['sample_names']]
        preds = data['preds']
        logits = data['logits']
        multimodal[split] = {
            'preds': {name: preds[i] for i, name in enumerate(names)},
            'logits': {name: logits[i] for i, name in enumerate(names)}
        }
    return multimodal


def evaluate(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    dataset, loader = build_dataloader(
        args.test_text_npz,
        args.test_audio_npz,
        args.test_video_npz,
        split='test',
        batch_size=args.batch_size,
    )

    num_labels = dataset.aligned_data['labels'].shape[-1] if dataset.aligned_data['labels'].ndim > 1 else len(np.unique(dataset.aligned_data['labels']))
    text_model, audio_model, video_model = load_experts(args, dataset, num_labels, device)

    multimodal_preds = None
    if args.multimodal_train_npz and args.multimodal_test_npz:
        mm_all = load_multimodal_predictions(args.multimodal_train_npz, args.multimodal_test_npz)
        multimodal_preds = mm_all.get('test')

    total = 0
    correct_any = 0
    modality_correct = {'T': 0}
    all_labels = []
    final_preds = []

    for batch in tqdm(loader, desc="Evaluating ensemble"):
        text_input = batch['text'].to(device)
        audio_input = batch['audio'].to(device)
        video_input = batch['video'].to(device)
        labels = batch['label'].squeeze(-1).to(device)
        sample_names = batch['sample_name']
        if isinstance(sample_names, torch.Tensor):
            sample_names = sample_names.tolist()
        sample_names = [normalize_sample_name(name) for name in sample_names]

        with torch.no_grad():
            text_logits, text_features = text_model(text_input)
            audio_logits, audio_features = audio_model(audio_input)
            video_logits, video_features = video_model(video_input)

        preds_t = text_logits.argmax(dim=1)
        correct_t = preds_t.eq(labels)
        modality_correct['T'] += correct_t.sum().item()

        sample_preds = preds_t.clone()

        if multimodal_preds is not None:
            mm_values = []
            for i, name in enumerate(sample_names):
                if name in multimodal_preds['preds']:
                    mm_values.append(multimodal_preds['preds'][name])
                else:
                    mm_values.append(preds_t[i].item())
            mm_tensor = torch.tensor(mm_values, device=device)
            fallback_mask = ~correct_t
            sample_preds[fallback_mask] = mm_tensor[fallback_mask]

        sample_preds[correct_t] = labels[correct_t]

        total += labels.size(0)
        correct_any += sample_preds.eq(labels).sum().item()

        final_preds.append(sample_preds.cpu())
        all_labels.append(labels.cpu())

    final_preds = torch.cat(final_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = correct_any / total
    weighted_f1 = f1_score(all_labels, final_preds, average='weighted')
    conf_mat = confusion_matrix(all_labels, final_preds, labels=list(range(num_labels)))
    class_totals = conf_mat.sum(axis=1)
    class_correct = conf_mat.diagonal()
    class_acc = {label: (class_correct[label] / class_totals[label] if class_totals[label] > 0 else 0.0)
                 for label in range(num_labels)}

    results = {
        'ensemble_accuracy': accuracy,
        'ensemble_weighted_f1': weighted_f1,
        'ensemble_class_accuracy': class_acc,
        'modality_accuracy': {
            'T': modality_correct['T'] / total,
            'A': modality_correct['A'] / total,
            'V': modality_correct['V'] / total,
        }
    }

    print("\n=== Ensemble Evaluation ===")
    print(f"Samples: {total}")
    print(f"Ensemble Accuracy (any modality correct): {accuracy:.4f}")
    print(f"Ensemble Weighted F1: {weighted_f1:.4f}")
    print("Class-wise Accuracy:")
    for label, acc in results['ensemble_class_accuracy'].items():
        print(f"  Class {label}: {acc:.4f}")
    print("Modality Individual Accuracy (for reference):")
    for m, acc in results['modality_accuracy'].items():
        print(f"  {m}: {acc:.4f}")

    if args.output_json:
        import json
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate unimodal experts with optimistic ensemble rule")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--test-text-npz', default='/data/yuyangchen/data/MELD/text_features/test_text_features.npz')
    parser.add_argument('--test-audio-npz', default='/data/yuyangchen/data/MELD/audio_features/test_audio_features_processed.npz')
    parser.add_argument('--test-video-npz', default='/data/yuyangchen/data/MELD/face_features/test_video_features_processed.npz')

    parser.add_argument('--text-checkpoint', default='/data/yuyangchen/checkpoints/unimodal_experts/unimodal_t_best_f1_0.6750_seed_42.pth', help='Path to trained text expert checkpoint')
    parser.add_argument('--audio-checkpoint', default='/data/yuyangchen/checkpoints/unimodal_experts/unimodal_a_best_f1_0.5565_seed_42.pth', help='Path to trained audio expert checkpoint')
    parser.add_argument('--video-checkpoint', default='/data/yuyangchen/checkpoints/unimodal_experts/unimodal_v_best_f1_0.4021_seed_42.pth', help='Path to trained video expert checkpoint')
    parser.add_argument('--multimodal-train-npz', default="/data/yuyangchen/checkpoints/multimodal_fusion_best_acc_0.6893_seed_42_train_predictions.npz", help='Path to precomputed multimodal train predictions (npz)')
    parser.add_argument('--multimodal-test-npz', default="/data/yuyangchen/checkpoints/multimodal_fusion_best_acc_0.6893_seed_42_test_predictions.npz", help='Path to precomputed multimodal test predictions (npz)')

    parser.add_argument('--output-json', help='Optional path to dump evaluation results in JSON format')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
