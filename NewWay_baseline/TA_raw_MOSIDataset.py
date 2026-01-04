# 文件：TA_raw_data_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import pandas as pd
from transformers import RobertaTokenizer
from tqdm import tqdm

# 假设您已将 MOSI 的音频数据集类导入
# 请确保 MOSIAudioDataset 在此文件可访问
from audio_dataset import MOSIAudioDataset  # <--- 引入 MOSIAudioDataset


# ... (以及 convert_to_acc7_label 函数 和 T_raw_MOSIDataset 中的 load_mosi_text_data 函数)


# --- ACC-7 七分类转换函数 (从您的 MOSIDataset 中引入) ---
def convert_to_acc7_label(score):
    """
    将 CMU-MOSI 的连续情感分数 [-3.0, 3.0] 转换为 0-6 的七分类标签。
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


# --- MOSI 文本加载函数 (从 T_raw_MOSIDataset 引入) ---
def load_mosi_text_data(text_path, split, max_seq_length=128):
    """
    加载 MOSI 标签 CSV，应用 ACC-7 分类，并对文本进行 RoBERTa 编码。
    """
    tokenizer = RobertaTokenizer.from_pretrained("/data/home/chenqian/Roberta-large/Roberta-large")

    data = pd.read_csv(text_path)
    split_mode = 'train_test' if split == 'train' else 'valid'

    if split_mode == 'train_test':
        data = data[data['mode'].isin(['train', 'test'])].reset_index(drop=True)
    elif split_mode == 'valid':
        data = data[data['mode'] == 'valid'].reset_index(drop=True)

    data['label_7c'] = data['label'].apply(convert_to_acc7_label)
    data['full_sample_name'] = data['video_id'].astype(str) + '_' + data['clip_id'].astype(str)
    sample_names = data['full_sample_name'].tolist()
    texts = data['text'].tolist()
    labels_7c = data['label_7c'].tolist()

    input_ids_list = []
    attention_mask_list = []

    for text in texts:
        encoding = tokenizer(
            str(text),
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids_list.append(encoding["input_ids"].squeeze(0))
        attention_mask_list.append(encoding["attention_mask"].squeeze(0))

    print(f"Loaded MOSI {split} raw text from {text_path}: {len(data)} samples.")

    return {
        "sample_names": sample_names,
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.tensor(labels_7c, dtype=torch.long)
    }


class TA_raw_MOSIDataset(Dataset):
    """
    PyTorch Dataset for CMU-MOSI multimodal emotion recognition (ACC-7).
    Loads raw text, raw audio, and aligned video features.
    """

    def __init__(
            self,
            text_npz: str,  # 占位符
            audio_npz: str,  # 占位符
            video_npz: str,
            modalities: List[str],
            split: str = 'train',
            feature_type: str = 'sequence_features',
            text_path: str = '/data/home/chenqian/CMU_MOSI/CMU-MOSI/label.csv',
            audio_csv_path: str = '/data/home/chenqian/CMU_MOSI/CMU-MOSI/all_train_data.csv',
            audio_data_path: str = '/data/home/chenqian/CMU_MOSI/CMU-MOSI/all_train_data.csv'
    ):

        self.modalities = modalities
        self.split = split
        self.feature_type = feature_type
        self.text_path = text_path
        self.audio_csv_path = audio_csv_path
        self.audio_data_path = audio_data_path

        # **使用统一的文本加载函数**
        self.text_data = load_mosi_text_data(self.text_path, self.split)

        # **【核心修改点：使用 MOSIAudioDataset 加载原始音频】**
        self.audio_data = self.load_audio_features(
            csv_path=audio_csv_path,
            audio_directory=audio_data_path,
            model_type='Whisper'  # 假设您使用的是 Whisper
        ) if 'A' in modalities else None

        self.video_data = self.load_features(video_npz, 'video') if 'V' in modalities else None

        # Get common sample names (intersection across selected modalities)
        sample_names_sets = []
        if 'T' in modalities:
            sample_names_sets.append(set(self.text_data['sample_names']))
        if 'A' in modalities:
            sample_names_sets.append(set(self.audio_data['sample_names']))
        if 'V' in modalities:
            # 这里的 video_npz 必须包含 MOSI clip_id 作为 sample_names
            sample_names_sets.append(set(self.video_data['sample_names']))

        self.sample_names = sorted(list(set.intersection(*sample_names_sets)))
        print(f"{split} split: {len(self.sample_names)} samples after alignment")

        # 提取对齐后的特征 (保持与 T_raw_MOSIDataset 相似的逻辑)
        if 'T' in modalities:
            indices = [self.text_data['sample_names'].index(name) for name in self.sample_names]
            self.text_input_ids = self.text_data['input_ids'][indices]
            self.text_attention_mask = self.text_data['attention_mask'][indices]
            self.text_target_start_pos = torch.zeros(len(self.sample_names), dtype=torch.long)
            self.text_target_end_pos = torch.zeros(len(self.sample_names), dtype=torch.long)
            self.labels = self.text_data['labels'][indices]  # ACC-7 标签
            print(f"Aligned text input_ids shape: {self.text_input_ids.shape}")

        if 'A' in modalities:
            indices = [self.audio_data['sample_names'].index(name) for name in self.sample_names]
            self.audio_input_values = self.audio_data['input_values'][indices]
            self.audio_attention_mask = self.audio_data['attention_mask'][indices]
            print(f"Aligned audio input_values shape: {self.audio_input_values.shape}")

        if 'V' in modalities:
            indices = [np.where(self.video_data['sample_names'] == name)[0][0] for name in self.sample_names]
            self.video_features = self.video_data[self.feature_type][indices]
            print(f"Aligned video shape: {self.video_features.shape}")

        print(f"Aligned labels shape: {self.labels.shape}")

    # load_features (用于加载 NPZ 格式的 V 特征，保持不变)
    def load_features(self, npz_path: str, modality: str) -> Dict[str, Any]:
        """Load features from .npz file and validate."""
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

    # **【核心修改点：利用 MOSIAudioDataset 封装的逻辑】**
    def load_audio_features(self, csv_path, audio_directory, model_type, max_length=12*16000, local_model_path=None):
        """
        使用 MOSIAudioDataset 加载原始音频特征，仅保留特征和样本名。
        """
        # MOSI 的训练集/测试集 CSV 路径通常是分开的，这里使用 train/test 的 CSV
        audio_dataset = MOSIAudioDataset(
            csv_path=csv_path,
            audio_directory=audio_directory,
            max_length=max_length,
            # 多模态训练时通常不使用缓存，以防止 GPU 内存不足或数据不一致。
            # 这里设置 cache_dir=None，除非您确保缓存目录划分清晰
            cache_dir=None,
            model_type=model_type,
            local_model_path=local_model_path
        )

        sample_names = []
        input_values_list = []
        attention_mask_list = []

        # 遍历 MOSIAudioDataset 并提取所需数据
        for idx in tqdm(range(len(audio_dataset)), desc=f"Loading MOSI raw audio for {self.split}"):
            sample = audio_dataset[idx]
            # 确保 MOSIAudioDataset 的 __getitem__ 返回正确的数据
            input_values_list.append(sample["input_values"])
            attention_mask_list.append(sample["attention_mask"])
            sample_names.append(sample["sample_name"])

        return {
            "sample_names": sample_names,
            "input_values": torch.stack(input_values_list),
            "attention_mask": torch.stack(attention_mask_list),
        }

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample = {}
        if 'T' in self.modalities:
            sample['text'] = {
                'input_ids': self.text_input_ids[idx],
                'attention_mask': self.text_attention_mask[idx],
                'target_start_pos': self.text_target_start_pos[idx],  # 0
                'target_end_pos': self.text_target_end_pos[idx]  # 0
            }
        if 'A' in self.modalities:
            sample['audio'] = {
                'input_values': self.audio_input_values[idx],
                'attention_mask': self.audio_attention_mask[idx]
            }
        if 'V' in self.modalities:
            sample['video'] = torch.tensor(self.video_features[idx], dtype=torch.float32)
        sample['label'] = self.labels[idx]  # ACC-7 Label
        sample['sample_name'] = self.sample_names[idx]  # MOSI clip_id
        return sample
