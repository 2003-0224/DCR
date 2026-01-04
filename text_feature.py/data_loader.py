# 这是分类任务代码


# import torch
# from torch import nn
# import transformers
# from transformers import RobertaTokenizer
# from torch.utils.data import DataLoader, Dataset
#
# import pandas as pd
# import numpy as np
# import string
#
#
# # --- ACC-7 七分类转换函数 ---
# def convert_to_acc7_label(score):
#     """
#     将 CMU-MOSI 的连续情感分数 [-3.0, 3.0] 转换为 0-6 的七分类标签。
#     这是研究中用于计算 ACC-7 的标准分箱方法。
#     """
#     score = float(score)
#     if -3.0 <= score <= -2.5:
#         return 0  # 强烈消极 (-3)
#     elif -2.5 < score <= -1.5:
#         return 1  # 消极 (-2)
#     elif -1.5 < score <= -0.5:
#         return 2  # 弱消极 (-1)
#     elif -0.5 < score <= 0.5:
#         return 3  # 中性 (0)
#     elif 0.5 < score <= 1.5:
#         return 4  # 弱积极 (+1)
#     elif 1.5 < score <= 2.5:
#         return 5  # 积极 (+2)
#     elif 2.5 < score <= 3.0:
#         return 6  # 强烈积极 (+3)
#     else:
#         # 确保返回一个有效的类别
#         return 3
#
#     # --- 针对 MOSI 独白/单片段的 Dataset 类 ---
#
#
# class MOSIDataset(Dataset):
#     def __init__(self, data_path, tokenizer, split_mode='train_test', max_seq_length=128):
#         """
#         初始化 MOSI 数据集类，用于单片段 ACC-7 分类任务。
#
#         参数：
#         - data_path (str): CSV 文件路径（例如 label.csv）。
#         - tokenizer: 文本 tokenizer。
#         - split_mode (str): 'train_valid' (合并 train 和 valid) 或 'test'。
#         - max_seq_length (int): 文本最大序列长度（默认 128）。
#         """
#         # 读取数据
#         self.data = pd.read_csv(data_path)
#         self.tokenizer = tokenizer
#         self.max_seq_length = max_seq_length
#
#         # 1. 划分数据集：合并 'train' 和 'test'
#         if split_mode in ['train', 'valid', 'test']:
#             self.data = self.data[self.data['mode'] == split_mode].reset_index(drop=True)
#         else:
#             raise ValueError("split_mode must be 'train', 'valid', or 'test'.")
#
#         # 2. 应用 ACC-7 分类：将连续分数 'label' 转换为 7 类别 (0-6)
#         self.data['label_7c'] = self.data['label'].apply(convert_to_acc7_label)
#
#         # 3. 提取文本和标签列表
#         self.texts = self.data['text'].tolist()
#         self.labels_7c = self.data['label_7c'].tolist()
#         self.sample_names = (self.data['video_id'].astype(str) + '_' + self.data['clip_id'].astype(str)).tolist()
#
#         # 打印数据集统计信息
#         print(f"Loaded MOSI {split_mode} set from {data_path}")
#         print(f"Total clips/utterances: {len(self.data)}")
#         print(f"ACC-7 label distribution (0-6):\n{self.data['label_7c'].value_counts().sort_index()}")
#
#     def __len__(self):
#         """
#         返回数据集的大小（总片段数）。
#         """
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         """
#         获取第 idx 个样本。
#
#         返回：
#         - dict: 包含编码后的文本特征和 7-Class 标签 (0-6)。
#         """
#         # 1. 获取目标片段信息
#         target_utterance = str(self.texts[idx])
#         label_7c = self.labels_7c[idx]
#         sample_name = self.sample_names[idx]
#
#         # 2. 编码输入 (只编码当前片段)
#         encoding = self.tokenizer(
#             target_utterance,
#             max_length=self.max_seq_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#
#         # 3. 准备输出
#         input_ids = encoding["input_ids"].squeeze(0)
#         attention_mask = encoding["attention_mask"].squeeze(0)
#         # 标签使用 Long 类型，适用于分类任务
#         label = torch.tensor(label_7c, dtype=torch.long)
#
#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "label": label,
#             "sample_name": sample_name
#             # 如果是多模态，可以在这里添加 'acoustic_features' 和 'visual_features'
#         }


import torch
from torch import nn
import transformers
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import string


class MOSIDataset(Dataset):
    def __init__(self, data_path, tokenizer, split_mode='train_test', max_seq_length=128):
        """
        初始化 MOSI 数据集类，用于单片段 V-Score 回归任务。

        参数：
        - data_path (str): CSV 文件路径（例如 label_utf8.csv）。
        - tokenizer: 文本 tokenizer。
        - split_mode (str): 'train', 'valid' 或 'test'。
        - max_seq_length (int): 文本最大序列长度（默认 128）。
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        if split_mode in ['train', 'valid', 'test']:
            self.data = self.data[self.data['mode'] == split_mode].reset_index(drop=True)
        else:
            raise ValueError("split_mode must be 'train', 'valid', or 'test'.")
        self.texts = self.data['text'].tolist()
        self.labels = self.data['label'].tolist()
        self.sample_names = (self.data['video_id'].astype(str) + '_' + self.data['clip_id'].astype(str)).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_utterance = str(self.texts[idx])
        original_label = self.labels[idx]
        sample_name = self.sample_names[idx]

        encoding = self.tokenizer(
            target_utterance,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor([original_label], dtype=torch.float)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "sample_name": sample_name
        }
