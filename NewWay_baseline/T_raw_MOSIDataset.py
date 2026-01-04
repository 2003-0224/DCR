import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Set, Union
import pandas as pd
from transformers import RobertaTokenizer
from collections import defaultdict


def convert_to_acc7_label(score):
    """
    将连续分数转换为 7 分类标签 (0-6)。
    注意：此函数仅在 task_type='classification' 时被调用。
    """
    try:
        score = float(score)
    except (ValueError, TypeError):
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


def _standardize_ids(raw_ids: np.ndarray) -> Dict[str, int]:
    id_map = {}
    for i, item in enumerate(raw_ids):
        if isinstance(item, bytes):
            standardized_id = item.decode('utf-8').strip()
        else:
            standardized_id = str(item).strip()
        if standardized_id not in id_map:
            id_map[standardized_id] = i
    return id_map


class T_raw_MOSIDataset(Dataset):
    def __init__(
            self,
            text_npz: str,  # 未使用，但保留
            audio_npz: str,
            video_npz: str,
            modalities: List[str],
            split: str = 'train',
            feature_type: str = 'sequence_features',
            text_path: str = '/data/home/chenqian/CMU-MOSEI/label_utf8_clean.csv',
            max_seq_length: int = 128,
            # *** 修改 1: 新增 task_type 参数 ***
            task_type: str = 'regression'
    ):
        self.modalities = [m for m in modalities if m in 'TAV']  # 仅保留 T, A, V
        self.split = split
        self.feature_type = feature_type
        self.text_path = text_path
        self.max_seq_length = max_seq_length
        # *** 修改 2: 存储 task_type ***
        self.task_type = task_type

        self.tokenizer = RobertaTokenizer.from_pretrained("/data/home/chenqian/Roberta-large/Roberta-large")

        # *** 修改 3: 加载数据时传入 task_type 依赖 ***
        self.text_data = self.load_mosi_text_data()

        self.text_id_map = _standardize_ids(self.text_data['sample_names'])
        self.text_id_set = set(self.text_id_map.keys())

        self.audio_data = self.load_features(audio_npz, 'audio') if 'A' in self.modalities else None
        self.video_data = self.load_features(video_npz, 'video') if 'V' in self.modalities else None

        # 初始化 ID 映射和集合
        self.av_id_maps: Dict[str, Dict[str, int]] = {}
        self.av_id_sets: Dict[str, Set[str]] = {}
        if 'A' in self.modalities and self.audio_data is not None:
            self.av_id_maps['A'] = _standardize_ids(self.audio_data['sample_names'])
            self.av_id_sets['A'] = set(self.av_id_maps['A'].keys())
        if 'V' in self.modalities and self.video_data is not None:
            self.av_id_maps['V'] = _standardize_ids(self.video_data['sample_names'])
            self.av_id_sets['V'] = set(self.av_id_maps['V'].keys())

        # 对齐特征和样本名 (计算交集)
        all_id_sets = [self.text_id_set] + list(self.av_id_sets.values())
        if not all_id_sets:
            raise ValueError("No modality data loaded for alignment.")

        # 获取交集作为最终样本名列表
        self.sample_names = sorted(list(set.intersection(*all_id_sets)))
        print(f"{split} split: {len(self.sample_names)} samples after alignment")

        # 计算对齐后的原始索引
        text_indices = [self.text_id_map[name] for name in self.sample_names]

        # 文本和标签
        if 'T' in self.modalities:
            # 使用预计算的索引提取数据
            self.text_input_ids = self.text_data['input_ids'][text_indices]
            self.text_attention_mask = self.text_data['attention_mask'][text_indices]
            self.text_target_start_pos = torch.zeros(len(self.sample_names), dtype=torch.long)
            self.text_target_end_pos = torch.zeros(len(self.sample_names), dtype=torch.long)
            self.labels = self.text_data['labels'][text_indices]
            print(f"Aligned text input_ids shape: {self.text_input_ids.shape}")
            print(f"Aligned labels shape: {self.labels.shape} (Type: {self.labels.dtype})")

        # 音频特征
        if 'A' in self.modalities:
            audio_indices = [self.av_id_maps['A'][name] for name in self.sample_names]
            self.audio_features = self.audio_data[self.feature_type][audio_indices]
            print(f"Aligned audio shape: {self.audio_features.shape}")

        # 视频特征
        if 'V' in self.modalities:
            video_indices = [self.av_id_maps['V'][name] for name in self.sample_names]
            self.video_features = self.video_data[self.feature_type][video_indices]
            print(f"Aligned video shape: {self.video_features.shape}")

    def load_mosi_text_data(self):
        """
        加载 MOSEI 标签 CSV，根据 task_type 返回分类标签或连续回归分数，并对文本进行 RoBERTa 编码。
        """
        data = pd.read_csv(self.text_path)
        split_mode = 'train' if self.split == 'train' else 'test'

        if split_mode == 'train':
            data = data[data['mode'] == 'train'].reset_index(drop=True)
        elif split_mode == 'test':
            data = data[data['mode'] == 'test'].reset_index(drop=True)

        # 编码文本和收集结果
        data['full_sample_name'] = data['video_id'].astype(str) + '_' + data['clip_id'].astype(str)
        sample_names = data['full_sample_name'].tolist()
        texts = data['text'].tolist()

        # *** 修改 4: 标签处理逻辑切换 ***
        if self.task_type == 'regression':
            # 回归任务：使用原始连续分数
            labels_to_use = data['label'].tolist()
            label_dtype = torch.float
            print(f"Using continuous scores for regression.")
        else:
            # 分类任务：应用 ACC-7 分类
            data['label_7c'] = data['label'].apply(convert_to_acc7_label)
            labels_to_use = data['label_7c'].tolist()
            label_dtype = torch.long
            print(f"Using 7-class labels for classification.")
            print(f"ACC-7 label distribution (0-6):\n{data['label_7c'].value_counts().sort_index()}")

        input_ids_list = []
        attention_mask_list = []
        for text in texts:
            encoding = self.tokenizer(
                str(text),
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids_list.append(encoding["input_ids"].squeeze(0))
            attention_mask_list.append(encoding["attention_mask"].squeeze(0))

        print(f"Loaded MOSEI {self.split} set from {self.text_path}: {len(data)} samples.")

        return {
            "sample_names": sample_names,
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            # *** 修改 5: 使用 label_dtype ***
            "labels": torch.tensor(labels_to_use, dtype=label_dtype)
        }

    def load_features(self, npz_path: str, modality: str) -> Dict[str, Any]:
        try:
            data = np.load(npz_path, allow_pickle=True)
            features = {key: data[key] for key in data}
            if self.feature_type not in features:
                raise KeyError(
                    f"Feature type '{self.feature_type}' not found in {npz_path}. Available keys: {list(features.keys())}")
            if 'sample_names' not in features:
                raise KeyError(f"'sample_names' not found in {npz_path}")
            print(f"Loaded {modality} features from {npz_path}: {len(features['sample_names'])} samples, "
                  f"{self.feature_type} shape: {features[self.feature_type].shape}")
            return features
        except Exception as e:
            print(f"Error loading {modality} features from {npz_path}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, str]]:
        sample = {}

        if 'T' in self.modalities:
            sample['text'] = {
                'input_ids': self.text_input_ids[idx],  # (max_seq_length,)
                'attention_mask': self.text_attention_mask[idx],  # (max_seq_length,)
                'target_start_pos': self.text_target_start_pos[idx],  # scalar (0)
                'target_end_pos': self.text_target_end_pos[idx]  # scalar (0)
            }

        # A/V 特征是 NumPy 数组，转换为 Tensor
        if 'A' in self.modalities:
            # 这里的 self.audio_features 是 NumPy 数组，需要 tensor() 转换
            sample['audio'] = torch.tensor(self.audio_features[idx], dtype=torch.float32)
        if 'V' in self.modalities:
            sample['video'] = torch.tensor(self.video_features[idx], dtype=torch.float32)

        # *** 修改 6: 标签类型和形状切换 ***
        label_value = self.labels[idx]
        if self.task_type == 'regression':
            # 回归任务：浮点数，形状 [1]
            sample['label'] = torch.tensor([label_value.item()], dtype=torch.float32)
        else:
            # 分类任务：长整型，形状 [1]
            sample['label'] = torch.tensor([label_value.item()], dtype=torch.long)

        sample['sample_name'] = self.sample_names[idx]
        return sample