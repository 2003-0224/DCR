import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Set
from collections import defaultdict
from typing import Union 


def convert_to_acc7_label(score):
    """
    将连续分数转换为 7 分类标签 (0-6)。
    """
    try:
        score = float(score)
    except (ValueError, TypeError):
        return 3  # 默认中性
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
    # 将 numpy ID 数组转换为 标准化 ID (str) 到索引 (int) 的映射字典。
    id_map = {}
    for i, item in enumerate(raw_ids):
        if isinstance(item, bytes):
            standardized_id = item.decode('utf-8').strip()
        else:
            standardized_id = str(item).strip()
        if standardized_id not in id_map:
            id_map[standardized_id] = i
    return id_map


class MOSIDataset(Dataset):
    def __init__(
            self,
            text_npz: str,
            audio_npz: str,
            video_npz: str,
            modalities: List[str],
            split: str = 'train',
            feature_type: str = 'pooled_features',
            task_type: str = 'classification'
    ):
        self.modalities = [m for m in modalities if m in 'TAV']
        self.split = split
        self.feature_type = feature_type
        # 存储 task_type
        self.task_type = task_type
        # 加载特征
        self.text_data = self.load_features(text_npz, 'text')
        self.audio_data = self.load_features(audio_npz, 'audio') if 'A' in modalities else None
        self.video_data = self.load_features(video_npz, 'video') if 'V' in modalities else None

        # 构建 ID 映射字典
        self.id_maps: Dict[str, Dict[str, int]] = {}
        self.id_sets: Dict[str, Set[str]] = {}
        for mod, data in [('T', self.text_data), ('A', self.audio_data), ('V', self.video_data)]:
            if data is not None and mod in self.modalities:
                id_map = _standardize_ids(data['sample_names'])
                self.id_maps[mod] = id_map
                self.id_sets[mod] = set(id_map.keys())

        # 计算(交集)
        if not self.id_sets:
            raise ValueError("No valid modality data loaded for alignment.")
        common_ids_set = set.intersection(*self.id_sets.values())
        self.sample_names = sorted(list(common_ids_set))
        print(f"{split} split: {len(self.sample_names)} samples after alignment")

        # 对齐数据
        self.aligned_data = self.align_data()

    def load_features(self, npz_path: str, modality: str) -> Dict[str, Any]:
        try:
            data = np.load(npz_path, allow_pickle=True)
            features = {key: data[key] for key in data}

            # 验证特征键是否完整
            if 'sample_names' not in features:
                raise KeyError(f"'sample_names' not found in {npz_path}")
            if modality == 'text' and 'labels' not in features:
                # Text data 必须包含连续分数 (labels)
                raise KeyError(f"'labels' (continuous scores) not found in {npz_path}. Text data must contain scores.")

            # 兼容性检查：如果特征是序列，可能需要 sequence_features
            feature_key = self.feature_type
            if feature_key not in features:
                if 'sequence_features' in features:
                    feature_key = 'sequence_features'
                else:
                    if modality != 'text' or feature_key != 'pooled_features':
                        raise KeyError(f"Feature key '{self.feature_type}' not found in {npz_path}.")

            print(f"Loaded {modality} features from {npz_path}: {len(features['sample_names'])} samples, "
                  f"Feature shape: {features.get(feature_key, 'N/A').shape}")
            return features
        except Exception as e:
            print(f"Error loading {modality} features from {npz_path}: {e}")
            raise

    def align_data(self) -> Dict[str, np.ndarray]:
        """
        Align features across modalities using pre-computed ID maps.
        根据 self.task_type 处理标签：分类 (ACC-7) 或回归 (连续分数)。
        """
        aligned_data = defaultdict(list)
        # 使用 Text Modality 的 ID 映射来查找原始索引
        text_id_map = self.id_maps.get('T', {})
        for sample_id in self.sample_names:
            try:
                # 提取和转换标签 (以文本数据为准)
                idx_t = text_id_map[sample_id]
                continuous_score = self.text_data['labels'][idx_t]
                if self.task_type == 'classification':
                    # 分类任务：将连续分数转换为 ACC-7 标签
                    label = convert_to_acc7_label(continuous_score)
                elif self.task_type == 'regression':
                    # 回归任务：直接使用连续分数 (确保为浮点数)
                    try:
                        label = float(continuous_score)
                    except (ValueError, TypeError):
                        label = 0.0
                else:
                    raise ValueError(f"Unknown task_type: {self.task_type}")
                aligned_data['labels'].append(label)
                # 提取特征
                if 'T' in self.modalities:
                    aligned_data['text'].append(self.text_data[self.feature_type][idx_t])
                if 'A' in self.modalities and self.audio_data is not None:
                    idx_a = self.id_maps['A'][sample_id]
                    aligned_data['audio'].append(self.audio_data[self.feature_type][idx_a])
                if 'V' in self.modalities and self.video_data is not None:
                    idx_v = self.id_maps['V'][sample_id]
                    aligned_data['video'].append(self.video_data[self.feature_type][idx_v])
            except KeyError:
                print(f"Warning: Aligned ID '{sample_id}' not found in one of the ID maps during data extraction.")
                continue
            except IndexError:
                print(f"Warning: Index error for ID '{sample_id}' during data extraction.")
                continue
        # 转换为 numpy 数组
        final_aligned_data = {}
        for key, value_list in aligned_data.items():
            if value_list:
                try:
                    final_aligned_data[key] = np.array(value_list)
                    print(f"Aligned {key} shape: {final_aligned_data[key].shape}")
                except ValueError as e:
                    print(f"Error converting {key} to numpy array: {e}")
                    raise
        return final_aligned_data

    def __len__(self) -> int:
        # 返回数据集的样本数量。
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        获取数据集的一个样本。
        """
        sample = {}
        if 'T' in self.modalities:
            sample['text'] = torch.FloatTensor(self.aligned_data['text'][idx])
        if 'A' in self.modalities:
            sample['audio'] = torch.FloatTensor(self.aligned_data['audio'][idx])
        if 'V' in self.modalities:
            sample['video'] = torch.FloatTensor(self.aligned_data['video'][idx])
        # *** 修改 4: 标签类型切换 ***
        label_value = self.aligned_data['labels'][idx]
        if self.task_type == 'regression':
            sample['label'] = torch.FloatTensor([label_value])
        else:
            sample['label'] = torch.LongTensor([label_value.item()])
        sample['sample_name'] = self.sample_names[idx]
        return sample


def get_dataloader(
        text_npz: str,
        audio_npz: str,
        video_npz: str,
        modalities: List[str],
        split: str = 'train',
        batch_size: int = 32,
        shuffle: bool = True,
        feature_type: str = 'pooled_features',
        task_type: str = 'classification'
) -> DataLoader:
    # MOSEIDataset
    dataset = MOSIDataset(text_npz, audio_npz, video_npz, modalities, split, feature_type, task_type=task_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)