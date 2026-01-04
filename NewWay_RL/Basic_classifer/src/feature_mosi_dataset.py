# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from typing import List, Dict, Any, Set
# from torch.utils.data import DataLoader
# from collections import defaultdict
#
#
# def _standardize_ids(raw_ids: np.ndarray) -> Dict[str, int]:
#     # 将 numpy ID 数组转换为 标准化 ID (str) 到索引 (int) 的映射字典。
#     id_map = {}
#     for i, item in enumerate(raw_ids):
#         if isinstance(item, bytes):
#             standardized_id = item.decode('utf-8').strip()
#         else:
#             standardized_id = str(item).strip()
#         if standardized_id not in id_map:
#             id_map[standardized_id] = i
#     return id_map
#
#
# class MOSIDataset(Dataset):
#     def __init__(
#             self,
#             text_npz: str,
#             audio_npz: str,
#             video_npz: str,
#             modalities: List[str],
#             split: str = 'train',
#             feature_type: str = 'pooled_features'
#     ):
#         # 只保留合法模态标记
#         self.modalities = [m for m in modalities if m in 'TAV']
#         self.split = split
#         self.feature_type = feature_type
#
#         # 加载特征（若某模态不需要传入的 npz，可以传相同文件或空字符串，但 load_features 会报错）
#         self.text_data = self.load_features(text_npz, 'text')
#         self.audio_data = self.load_features(audio_npz, 'audio') if ('A' in self.modalities) else None
#         self.video_data = self.load_features(video_npz, 'video') if ('V' in self.modalities) else None
#
#         # 构建 ID 映射字典（sample_names 必须存在）
#         self.id_maps: Dict[str, Dict[str, int]] = {}
#         self.id_sets: Dict[str, Set[str]] = {}
#         # for mod, data in [('T', self.text_data), ('A', self.audio_data), ('V', self.video_data)]:
#         #     if data is not None and mod in self.modalities:
#         #         id_map = _standardize_ids(data['sample_names'])
#         #         self.id_maps[mod] = id_map
#         #         self.id_sets[mod] = set(id_map.keys())
#         # 永远处理 T
#         id_map_T = _standardize_ids(self.text_data['sample_names'])
#         self.id_maps['T'] = id_map_T
#         self.id_sets['T'] = set(id_map_T.keys())
#
#         # 额外处理 A / V（是否参与取决于 self.modalities）
#         for mod, data in [('A', self.audio_data), ('V', self.video_data)]:
#             if data is not None:  # A/V 文件存在
#                 id_map = _standardize_ids(data['sample_names'])
#                 self.id_maps[mod] = id_map
#                 self.id_sets[mod] = set(id_map.keys())
#
#
#         # 交集对齐
#         if not self.id_sets:
#             raise ValueError("No valid modality data loaded for alignment.")
#         common_ids_set = set.intersection(*self.id_sets.values())
#         self.sample_names = sorted(list(common_ids_set))
#         print(f"{split} split: {len(self.sample_names)} samples after alignment")
#
#         # 对齐并把所需的 features 写入 aligned_data
#         self.aligned_data = self.align_data()
#
#     def load_features(self, npz_path: str, modality: str) -> Dict[str, Any]:
#         if not npz_path or not os.path.exists(npz_path):
#             raise FileNotFoundError(f"Feature file not found: {npz_path}")
#         try:
#             data = np.load(npz_path, allow_pickle=True)
#             features = {key: data[key] for key in data}
#             # 必要键检查
#             if 'sample_names' not in features:
#                 raise KeyError(f"'sample_names' not found in {npz_path}")
#             if modality == 'text' and 'labels' not in features:
#                 raise KeyError(f"'labels' (continuous scores) not found in {npz_path}. Text data must contain scores.")
#             # 选择 feature key：优先使用传入的 feature_type，否则回退
#             feature_key = self.feature_type
#             if feature_key not in features:
#                 if 'sequence_features' in features:
#                     feature_key = 'sequence_features'
#                 elif 'pooled_features' in features:
#                     feature_key = 'pooled_features'
#                 else:
#                     raise KeyError(f"Feature key '{self.feature_type}' not found in {npz_path}. Available: {list(features.keys())}")
#             print(f"Loaded {modality} features from {npz_path}: {len(features['sample_names'])} samples, "
#                   f"Selected feature key: {feature_key}, "
#                   f"Example feature shape: {features.get(feature_key).shape if feature_key in features else 'N/A'}")
#             features['selected_feature_key'] = feature_key
#             return features
#         except Exception as e:
#             print(f"Error loading {modality} features from {npz_path}: {e}")
#             raise
#
#     def align_data(self) -> Dict[str, np.ndarray]:
#         """
#         Align features across modalities using pre-computed ID maps and convert continuous scores to ACC-7 labels.
#         Returns a dict with keys like 'text','audio','video','labels' each mapped to numpy arrays.
#         """
#         aligned_data = defaultdict(list)
#         text_id_map = self.id_maps.get('T', {})
#
#         for sample_id in self.sample_names:
#             try:
#                 # 标签以 text npz 为准
#                 idx_t = text_id_map[sample_id]
#                 continuous_score = self.text_data['labels'][idx_t]
#                 aligned_data['labels'].append(int(continuous_score))
#
#                 # 文本特征
#                 if 'T' in self.modalities:
#                     t_key = self.text_data['selected_feature_key']
#                     t_feat = self.text_data[t_key][idx_t]
#                     aligned_data['text'].append(t_feat)
#
#                 # 音频特征
#                 if 'A' in self.modalities and self.audio_data is not None:
#                     idx_a = self.id_maps['A'][sample_id]
#                     a_key = self.audio_data['selected_feature_key']
#                     a_feat = self.audio_data[a_key][idx_a]
#                     aligned_data['audio'].append(a_feat)
#
#                 # 视频特征
#                 if 'V' in self.modalities and self.video_data is not None:
#                     idx_v = self.id_maps['V'][sample_id]
#                     v_key = self.video_data['selected_feature_key']
#                     v_feat = self.video_data[v_key][idx_v]
#                     aligned_data['video'].append(v_feat)
#
#             except KeyError:
#                 # 某些 id 可能在某个 modality 中找不到，已在 sample_names 交集中过滤，这里为保险起见捕获
#                 print(f"Warning: Aligned ID '{sample_id}' not found in one of the ID maps during data extraction.")
#                 continue
#             except IndexError:
#                 print(f"Warning: Index error for ID '{sample_id}' during data extraction.")
#                 continue
#
#         # 转换为 numpy 数组；如果某些 modality 没有元素则不创建键
#         final_aligned_data = {}
#         for key, value_list in aligned_data.items():
#             if not value_list:
#                 continue
#             try:
#                 arr = np.array(value_list)
#                 final_aligned_data[key] = arr
#                 print(f"Aligned {key} shape: {arr.shape}")
#             except Exception as e:
#                 print(f"Error converting {key} to numpy array: {e}")
#                 raise
#         if 'labels' in final_aligned_data:
#             labels = final_aligned_data['labels']
#
#             # 确保 labels 是一个一维整数数组
#             if labels.ndim > 1:
#                 labels = labels.flatten()
#             labels = labels.astype(np.int32)
#
#             # 使用 numpy 统计唯一值及其计数
#             unique_labels, counts = np.unique(labels, return_counts=True)
#
#             # 创建一个字典，包含所有 0-6 类别，并初始化为 0 (确保未出现的类别也显示0)
#             label_counts = {i: 0 for i in range(7)}
#             for label, count in zip(unique_labels, counts):
#                 if 0 <= label <= 6:
#                     label_counts[label] = count
#
#             print("\n--- ACC-7 标签类别数量统计 ---")
#             print("| Label (情感强度) | 计数 |")
#             print("|:---:|:---:|")
#
#             # 打印 MOSEI 7类情感的友好名称（通常为 -3 到 +3）
#             emotion_map = {
#                 0: "强负面 (-3)", 1: "负面 (-2)", 2: "弱负面 (-1)", 3: "中性 (0)",
#                 4: "弱正面 (+1)", 5: "正面 (+2)", 6: "强正面 (+3)"
#             }
#
#             for label, count in label_counts.items():
#                 emotion_name = emotion_map.get(label, f"Unknown ({label})")
#                 print(f"| {emotion_name} | {count} |")
#             print("-----------------------------------")
#             print(f"总样本数: {len(labels)}")
#         return final_aligned_data
#
#     def __len__(self) -> int:
#         return len(self.sample_names)
#
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         sample = {}
#         # 文本可能是 sequence (N, L, D) 或 pooled (N, D)
#         if 'T' in self.modalities:
#             t = self.aligned_data['text'][idx]
#             t_arr = np.array(t)
#             # 如果是二维 (feat_dim,) 变为 (1, feat_dim) 以便 batch stack
#             if t_arr.ndim == 1:
#                 t_arr = t_arr[np.newaxis, :]
#             sample['text'] = torch.from_numpy(t_arr).float()
#
#         if 'A' in self.modalities:
#             a = self.aligned_data['audio'][idx]
#             a_arr = np.array(a)
#             if a_arr.ndim == 1:
#                 a_arr = a_arr[np.newaxis, :]
#             sample['audio'] = torch.from_numpy(a_arr).float()
#
#         if 'V' in self.modalities:
#             v = self.aligned_data['video'][idx]
#             v_arr = np.array(v)
#             if v_arr.ndim == 1:
#                 v_arr = v_arr[np.newaxis, :]
#             sample['video'] = torch.from_numpy(v_arr).float()
#
#         # 标签为单值 long tensor
#         label = int(self.aligned_data['labels'][idx])
#         sample['label'] = torch.tensor(label, dtype=torch.long)
#
#         # 原始样本名（字符串）
#         sample['sample_name'] = self.sample_names[idx]
#         return sample
#
# def get_dataloader(
#         text_npz: str,
#         audio_npz: str,
#         video_npz: str,
#         modalities: List[str],
#         split: str = 'train',
#         batch_size: int = 32,
#         shuffle: bool = True,
#         feature_type: str = 'pooled_features'
# ) -> DataLoader:
#     # MOSEIDataset
#     dataset = MOSIDataset(text_npz, audio_npz, video_npz, modalities, split, feature_type)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Set
from collections import defaultdict


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


class MOSIDataset(Dataset):
    def __init__(
            self,
            text_npz: str,
            audio_npz: str,
            video_npz: str,
            modalities: List[str],
            split: str = 'train',
            feature_type: str = 'pooled_features'
    ):
        self.modalities = [m for m in modalities if m in 'TAV']
        self.split = split
        self.feature_type = feature_type
        # 加载特征
        self.text_data = self.load_features(text_npz, 'text')
        self.audio_data = self.load_features(audio_npz, 'audio') if 'A' in modalities else None
        self.video_data = self.load_features(video_npz, 'video') if 'V' in modalities else None
        # 构建 ID 映射字典
        self.id_maps: Dict[str, Dict[str, int]] = {}
        self.id_sets: Dict[str, Set[str]] = {}
        # for mod, data in [('T', self.text_data), ('A', self.audio_data), ('V', self.video_data)]:
        #     if data is not None and mod in self.modalities:
        #         id_map = _standardize_ids(data['sample_names'])
        #         self.id_maps[mod] = id_map
        #         self.id_sets[mod] = set(id_map.keys())
        id_map_T = _standardize_ids(self.text_data['sample_names'])
        self.id_maps['T'] = id_map_T
        self.id_sets['T'] = set(id_map_T.keys())
        for mod, data in [('A', self.audio_data), ('V', self.video_data)]:
            if data is not None:  # A/V 文件存在
                id_map = _standardize_ids(data['sample_names'])
                self.id_maps[mod] = id_map
                self.id_sets[mod] = set(id_map.keys())
        # 计算交集
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
            if 'sample_names' not in features:
                raise KeyError(f"'sample_names' not found in {npz_path}")
            if modality == 'text' and 'labels' not in features:
                # 'labels' 必须包含连续分数
                raise KeyError(f"'labels' (continuous scores) not found in {npz_path}. Text data must contain scores.")
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
        aligned_data = defaultdict(list)
        text_id_map = self.id_maps.get('T', {})
        for sample_id in self.sample_names:
            try:
                # 提取连续分数标签 (以文本数据为准)
                idx_t = text_id_map[sample_id]
                continuous_score = self.text_data['labels'][idx_t]
                # 核心修改：直接使用原始连续分数作为回归标签
                aligned_data['labels'].append(float(continuous_score))
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
                    if key == 'labels':
                        # 确保标签是 float32 类型
                        final_aligned_data[key] = np.array(value_list, dtype=np.float32)
                    else:
                        final_aligned_data[key] = np.array(value_list)
                    print(f"Aligned {key} shape: {final_aligned_data[key].shape}")
                except ValueError as e:
                    print(f"Error converting {key} to numpy array: {e}")
                    raise
        return final_aligned_data

    def __len__(self) -> int:
        # 返回数据集的样本数量。
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

        # 核心修改：返回 FloatTensor 作为回归标签
        sample['label'] = torch.FloatTensor([self.aligned_data['labels'][idx].item()])
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
        feature_type: str = 'pooled_features'
) -> DataLoader:
    dataset = MOSIDataset(text_npz, audio_npz, video_npz, modalities, split, feature_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

