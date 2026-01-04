import torch
from torch.utils.data import Dataset, DataLoader
# *** 替换 torchaudio，使用 soundfile 和 scipy ***
import soundfile as sf  # 新增：用于音频加载
from scipy.signal import resample_poly  # 新增：用于重采样
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoProcessor

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# --- ACC-7 七分类转换函数 (保持不变) ---
def convert_to_acc7_label(score):
    """
    将 CMU-MOSI 的连续情感分数 [-3.0, 3.0] 转换为 0-6 的七分类标签。
    适用于情感分数所在的列。
    """
    try:
        score = float(score)
    except (ValueError, TypeError):
        # 处理 NaN 或无效值
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


# --- MOSI 音频数据集类 (适配 soundfile + scipy) ---
class MOSIAudioDataset(Dataset):
    def __init__(self, csv_path, audio_directory=None, max_length=96000, cache_dir=None, model_type=None,
                 local_model_path=None):
        df = pd.read_csv(csv_path)

        # === 核心修复 1: 查找正确的情感分数列名 (解决 KeyError) ===
        if 'label' in df.columns:
            label_col = 'label'
        elif 'MOSI情感分数' in df.columns:
            label_col = 'MOSI情感分数'
        elif 'score' in df.columns:
            label_col = 'score'
        else:
            raise KeyError(
                "CSV file must contain an emotional score column named 'label', 'score', or 'MOSI情感分数'. Please check your CSV file header.")

        # 使用动态确定的 label_col 来删除 NaN 行
        self.df = df.dropna(subset=[label_col]).reset_index(drop=True)

        self.audio_file_paths = []
        self.sample_names = []
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # 应用 ACC-7 分类到正确的列
        self.targets_7c = self.df[label_col].apply(convert_to_acc7_label).tolist()

        self.sampling_rate = 16000
        self.max_length = max_length

        # === 核心修复 2: 健壮的音频文件路径查找 ===
        if 'audio_path' in self.df.columns:
            # 模式 A: 使用 'audio_path' 列中的绝对路径
            for i in range(len(self.df)):
                file_path = self.df['audio_path'][i]
                file_name = os.path.basename(file_path)
                # 1. 移除扩展名 (如 .wav)
                path_no_ext = os.path.splitext(file_path)[0]

                # 2. 使用 os.path.sep 分割，获取最后两级路径：[...]/video_id/clip_id
                parts = path_no_ext.split(os.path.sep)

                if len(parts) >= 2:
                    # 拼接为 video_id_clip_id
                    video_id = parts[-2]
                    clip_id = parts[-1]
                    formatted_name = f"{video_id}_{clip_id}"
                else:
                    # 结构不符合预期，使用整个文件名并警告
                    formatted_name = os.path.basename(path_no_ext)
                    print(f"Warning: Path structure too short for sample name extraction: {file_path}")

                self.sample_names.append(formatted_name)
                if os.path.exists(file_path):
                    self.audio_file_paths.append(file_path)
                else:
                    print(f"Warning: Audio file not found at absolute path: {file_path}")
                    self.audio_file_paths.append(None)

        elif 'video_id' in self.df.columns and 'clip_id' in self.df.columns and audio_directory:
            # 模式 B: 使用 video_id/clip_id 格式（旧 MOSI/MELD 格式）
            for i in range(len(self.df)):
                video_id = self.df['video_id'][i]
                clip_id = self.df['clip_id'][i]
                file_name = f"{video_id}_{clip_id}.wav"
                file_path = os.path.join(audio_directory, file_name)

                if os.path.exists(file_path):
                    self.audio_file_paths.append(file_path)
                    self.sample_names.append(file_name)
                else:
                    self.audio_file_paths.append(None)
                    self.sample_names.append(file_name)
        else:
            raise ValueError(
                "CSV must contain either 'audio_path' or both 'video_id' and 'clip_id' columns, and 'audio_directory' must be provided for the latter.")

        # 初始化特征提取器 (Processor)
        if model_type == "WavLM":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
        elif model_type == "Data2Vec":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/data2vec-audio-large-960h")
        elif model_type == "Wav2Vec2":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")
        elif model_type == "Hubert":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        elif model_type == 'Whisper':
            # === 核心修复 3: 适配本地 Whisper 模型加载 (健壮性检查) ===
            if local_model_path and os.path.isdir(local_model_path):
                print(f"Loading Whisper Processor from local path: {local_model_path}")
                self.processor = AutoProcessor.from_pretrained(local_model_path)
            else:
                print(
                    f"Warning: local_model_path not found or provided. Loading Whisper Processor from HuggingFace Hub (openai/whisper-large-v3)")
                self.processor = AutoProcessor.from_pretrained("/data/home/chenqian/whisper_large_v3")
            # ============================================

        self.model_type = model_type

        # 加载数据
        self.cached_data = self._load_data()

        # 数据清理和填充 (保留原代码逻辑)
        for i in range(len(self.cached_data)):
            if self.model_type != 'Whisper' and self.cached_data[i]['input_values'].shape[-1] != self.max_length:
                if i > 0:
                    self.cached_data[i]['input_values'] = self.cached_data[i - 1]['input_values']
                    self.cached_data[i]['attention_mask'] = self.cached_data[i - 1]['attention_mask']

    def _load_data(self):
        """核心：使用 soundfile + scipy 进行加载和重采样"""
        data_list = []
        for file_path in tqdm(self.audio_file_paths, desc="Generating data"):
            if file_path is None:
                if self.model_type == 'Whisper':
                    num_mels = self.processor.feature_extractor.config.num_mel_bins
                    max_frames = (self.max_length // self.sampling_rate) * 100
                    input_values = torch.zeros(num_mels, max_frames)
                else:
                    input_values = torch.zeros(self.max_length)
                attention_mask = torch.zeros(input_values.shape[-1], dtype=torch.long)
            else:
                try:
                    # *** 1. soundfile 加载音频 ***
                    sound_data_multichannel, sr = sf.read(file_path, dtype='float32')

                    # 2. 转换为单声道 (取平均)
                    if sound_data_multichannel.ndim > 1:
                        sound_data_np = np.mean(sound_data_multichannel, axis=1)
                    else:
                        sound_data_np = sound_data_multichannel

                    # 3. 手动重采样 (如果需要)
                    if sr != self.sampling_rate:
                        # 使用 resample_poly 进行高效的重采样
                        num = self.sampling_rate
                        den = sr
                        sound_data_np = resample_poly(sound_data_np, num, den)

                except Exception as e:
                    # 如果 soundfile/scipy 加载或重采样失败
                    print(f"ERROR: soundfile/scipy failed to load/process {file_path}. Using zero data. Error: {e}")
                    sound_data_np = np.zeros(self.max_length, dtype=np.float32)

                # 提取特征
                processed = self.processor(
                    sound_data_np,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True
                )

                if self.model_type == "Whisper":
                    input_values = processed['input_features'].squeeze(0)
                    attention_mask = torch.ones(input_values.shape[-1], dtype=torch.long)
                else:
                    input_values = processed['input_values'].squeeze(0)
                    attention_mask = processed['attention_mask'].squeeze(
                        0) if 'attention_mask' in processed else torch.ones_like(input_values, dtype=torch.long)

            data = {"input_values": input_values, "attention_mask": attention_mask}
            data_list.append(data)
        return data_list

    def __getitem__(self, index):
        # 标签是 0-6 整数
        label = self.targets_7c[index]
        data = self.cached_data[index]

        return {
            "input_values": data["input_values"],
            "attention_mask": data["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long),
            "sample_name": self.sample_names[index]
        }

    def __len__(self):
        return len(self.df)


# --- MOSI 数据加载器函数 (保持不变) ---
def data_loader_mosi_audio(train_csv_path, test_csv_path, audio_directory=None, batch_size=32, max_seq_length=96000,
                           model_type=None, num_workers=4,
                           cache_dir="/data/home/chenqian/cache", local_model_path="/data/home/chenqian/whisper_large_v3/"):
    # 训练集
    train_data = MOSIAudioDataset(
        csv_path=train_csv_path,
        audio_directory=audio_directory,
        max_length=max_seq_length,
        cache_dir=os.path.join(cache_dir, "train_valid"),
        model_type=model_type,
        local_model_path=local_model_path
    )

    # 测试集
    test_data = MOSIAudioDataset(
        csv_path=test_csv_path,
        audio_directory=audio_directory,
        max_length=max_seq_length,
        cache_dir=os.path.join(cache_dir, "test"),
        model_type=model_type,
        local_model_path=local_model_path
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
