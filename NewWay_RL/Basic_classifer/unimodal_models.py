# import torch
# import torch.nn as nn
# from transformers import RobertaModel
#
# # 请确保下面模块路径对你的工程是正确的（你原来用的是 src.transformer_FacialMMT）
# from src.transformer_FacialMMT import MELDTransEncoder, AdditiveAttention
#
#
# class Config:
#     def __init__(self, hidden_size, num_attention_heads, intermediate_size,
#                  attention_probs_dropout_prob, hidden_dropout_prob, layer_norm_eps):
#         self.hidden_size = hidden_size
#         self.num_attention_heads = num_attention_heads
#         self.intermediate_size = intermediate_size
#         self.attention_probs_dropout_prob = attention_probs_dropout_prob
#         self.hidden_dropout_prob = hidden_dropout_prob
#         self.layer_norm_eps = layer_norm_eps
#
#
# class TextClassifier(nn.Module):
#     """
#     TextClassifier 支持两种输入模式：
#       - use_precomputed=True: inputs 为 tensor (B, L, D) 或 (B, 1, D)（预计算特征）
#       - use_precomputed=False: inputs 为 dict 包含 'input_ids','attention_mask','target_start_pos','target_end_pos'
#     forward 返回 (logits, pooled_feature)
#     """
#
#     def __init__(self,
#                  hidden_dim=512,
#                  num_classes=7,
#                  use_precomputed=False,
#                  input_dim=1024,
#                  target_seq_len=38):
#         super(TextClassifier, self).__init__()
#         self.use_precomputed = use_precomputed
#         self.target_seq_len = target_seq_len
#
#         if self.use_precomputed:
#             # 直接将预计算的 token/word-level embedding 投影到 hidden_dim
#             self.roberta_model = None
#             self.text_proj = nn.Linear(input_dim, hidden_dim)
#         else:
#             # 使用 Roberta 原始模型（如需在训练中使用 tokenizer）
#             self.roberta_model = RobertaModel.from_pretrained("/data/home/chenqian/Roberta-large")
#             # Roberta-large 的 hidden size 通常为1024
#             self.text_proj = nn.Linear(1024, hidden_dim)
#
#         # transformer encoder 用于对齐序列并聚合上下文
#         config = Config(hidden_size=hidden_dim,
#                         num_attention_heads=8,
#                         intermediate_size=4 * hidden_dim,
#                         attention_probs_dropout_prob=0.1,
#                         hidden_dropout_prob=0.1,
#                         layer_norm_eps=1e-6)
#
#         # MELDTransEncoder 是你工程中的 Transformer 实现
#         self.text_transformer = MELDTransEncoder(config=config, layer_num=1, get_max_lens=self.target_seq_len,
#                                                  hidden_size=hidden_dim)
#
#         self.attention = AdditiveAttention(hidden_dim, hidden_dim)
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(hidden_dim, num_classes)
#
#     def forward(self, inputs):
#         # inputs 支持两种形式（tensor 或 dict）
#         if self.use_precomputed:
#             # 预计算特征（tensor）：形状 (B, L, D) 或 (B, D) / (B,1,D)
#             text_seq = inputs
#             # 若用户意外传入形状 (B, D) -> expand为 (B,1,D)
#             if text_seq.dim() == 2:
#                 text_seq = text_seq.unsqueeze(1)
#             # 如果 seq_len 与 target_seq_len 不同，进行截断或 padding
#             if self.target_seq_len is not None:
#                 seq_len = text_seq.size(1)
#                 if seq_len < self.target_seq_len:
#                     pad = torch.zeros(
#                         text_seq.size(0),
#                         self.target_seq_len - seq_len,
#                         text_seq.size(2),
#                         dtype=text_seq.dtype,
#                         device=text_seq.device,
#                     )
#                     text_seq = torch.cat([text_seq, pad], dim=1)
#                 elif seq_len > self.target_seq_len:
#                     text_seq = text_seq[:, :self.target_seq_len, :]
#
#         else:
#             # 来自 tokenizer 的输入字典
#             text_input_ids = inputs['input_ids']
#             text_attention_mask = inputs['attention_mask']
#             text_output = self.roberta_model(text_input_ids, text_attention_mask)
#             text_feat = text_output.last_hidden_state  # (B, L, H)
#             # 如果有 target_start/end，裁剪到 target_seq_len
#             batch_size = text_input_ids.shape[0]
#             hidden_size = text_feat.shape[-1]
#             word_level_features = torch.zeros(batch_size, self.target_seq_len, hidden_size, device=text_feat.device)
#             target_start_pos = inputs.get('target_start_pos', None)
#             target_end_pos = inputs.get('target_end_pos', None)
#             if target_start_pos is None or target_end_pos is None:
#                 # 没有位置信息则直接截取前 target_seq_len
#                 word_level_features[:, :min(self.target_seq_len, text_feat.size(1))] = text_feat[:, :self.target_seq_len]
#             else:
#                 for i in range(batch_size):
#                     start = int(target_start_pos[i].item())
#                     end = int(target_end_pos[i].item())
#                     curr_utt_len = end - start
#                     if curr_utt_len > self.target_seq_len:
#                         curr_utt_len = self.target_seq_len
#                     if curr_utt_len > 0:
#                         word_level_features[i, :curr_utt_len] = text_feat[i, start:start + curr_utt_len]
#             text_seq = word_level_features
#
#         # 投影到 hidden_dim 并通过 transformer
#         text_seq = self.text_proj(text_seq)
#         text_feat_transformed = self.text_transformer(text_seq, attention_mask=None)
#
#         pooled_output, _ = self.attention(text_feat_transformed)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         return logits, pooled_output
#
#
# class AudioClassifier(nn.Module):
#     def __init__(self, audio_dim=1280, hidden_dim=512, num_classes=7):
#         super(AudioClassifier, self).__init__()
#         self.audio_proj = nn.Linear(audio_dim, hidden_dim)
#
#         config = Config(hidden_size=hidden_dim,
#                         num_attention_heads=8,
#                         intermediate_size=4 * hidden_dim,
#                         attention_probs_dropout_prob=0.1,
#                         hidden_dropout_prob=0.1,
#                         layer_norm_eps=1e-6)
#         self.audio_transformer = MELDTransEncoder(config=config, layer_num=1, get_max_lens=60, hidden_size=hidden_dim)
#
#         self.attention = AdditiveAttention(hidden_dim, hidden_dim)
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(hidden_dim, num_classes)
#
#     def forward(self, inputs):
#         # inputs: tensor (B, L, D) 或 (B, D)/(B,1,D)
#         audio_seq = inputs
#         if audio_seq.dim() == 2:
#             audio_seq = audio_seq.unsqueeze(1)
#         audio_seq = self.audio_proj(audio_seq)
#         audio_feat = self.audio_transformer(audio_seq, attention_mask=None)
#
#         pooled_output, _ = self.attention(audio_feat)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         return logits, pooled_output
#
#
# class VideoClassifier(nn.Module):
#     def __init__(self, video_dim=768, hidden_dim=512, num_classes=7):
#         super(VideoClassifier, self).__init__()
#         self.video_proj = nn.Linear(video_dim, hidden_dim)
#
#         config = Config(hidden_size=hidden_dim,
#                         num_attention_heads=8,
#                         intermediate_size=4 * hidden_dim,
#                         attention_probs_dropout_prob=0.1,
#                         hidden_dropout_prob=0.1,
#                         layer_norm_eps=1e-6)
#         self.video_transformer = MELDTransEncoder(config=config, layer_num=1, get_max_lens=40, hidden_size=hidden_dim)
#
#         self.attention = AdditiveAttention(hidden_dim, hidden_dim)
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(hidden_dim, num_classes)
#
#     def forward(self, inputs):
#         video_seq = inputs
#         if video_seq.dim() == 2:
#             video_seq = video_seq.unsqueeze(1)
#         video_seq = self.video_proj(video_seq)
#         video_feat = self.video_transformer(video_seq, attention_mask=None)
#
#         pooled_output, _ = self.attention(video_feat)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         return logits, pooled_output

import torch
import torch.nn as nn
from transformers import RobertaModel
from src.transformer_FacialMMT import MELDTransEncoder, AdditiveAttention


class Config:
    def __init__(self, hidden_size, num_attention_heads, intermediate_size,
                 attention_probs_dropout_prob, hidden_dropout_prob, layer_norm_eps):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps


class TextClassifier(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 num_classes=1,  # <--- MODIFIED: 从 7 改为 1 (回归输出)
                 use_precomputed=False,
                 input_dim=1024,
                 target_seq_len=38):
        super(TextClassifier, self).__init__()
        self.use_precomputed = use_precomputed
        self.target_seq_len = target_seq_len

        if self.use_precomputed:
            self.roberta_model = None
            self.text_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.roberta_model = RobertaModel.from_pretrained("/data/home/chenqian/Roberta-large")
            self.text_proj = nn.Linear(1024, hidden_dim)
        # transformer encoder 用于对齐序列并聚合上下文
        config = Config(hidden_size=hidden_dim,
                        num_attention_heads=8,
                        intermediate_size=4 * hidden_dim,
                        attention_probs_dropout_prob=0.1,
                        hidden_dropout_prob=0.1,
                        layer_norm_eps=1e-6)
        self.text_transformer = MELDTransEncoder(config=config, layer_num=1, get_max_lens=self.target_seq_len,
                                                 hidden_size=hidden_dim)
        self.attention = AdditiveAttention(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        if self.use_precomputed:
            text_seq = inputs
            if text_seq.dim() == 2:
                text_seq = text_seq.unsqueeze(1)
            if self.target_seq_len is not None:
                seq_len = text_seq.size(1)
                if seq_len < self.target_seq_len:
                    pad = torch.zeros(
                        text_seq.size(0),
                        self.target_seq_len - seq_len,
                        text_seq.size(2),
                        dtype=text_seq.dtype,
                        device=text_seq.device,
                    )
                    text_seq = torch.cat([text_seq, pad], dim=1)
                elif seq_len > self.target_seq_len:
                    text_seq = text_seq[:, :self.target_seq_len, :]
        else:
            text_input_ids = inputs['input_ids']
            text_attention_mask = inputs['attention_mask']
            text_output = self.roberta_model(text_input_ids, text_attention_mask)
            text_feat = text_output.last_hidden_state

            batch_size = text_input_ids.shape[0]
            hidden_size = text_feat.shape[-1]
            word_level_features = torch.zeros(batch_size, self.target_seq_len, hidden_size, device=text_feat.device)
            target_start_pos = inputs.get('target_start_pos', None)
            target_end_pos = inputs.get('target_end_pos', None)
            if target_start_pos is None or target_end_pos is None:
                word_level_features[:, :min(self.target_seq_len, text_feat.size(1))] = text_feat[:,
                                                                                       :self.target_seq_len]
            else:
                for i in range(batch_size):
                    start = int(target_start_pos[i].item())
                    end = int(target_end_pos[i].item())
                    curr_utt_len = end - start
                    if curr_utt_len > self.target_seq_len:
                        curr_utt_len = self.target_seq_len
                    if curr_utt_len > 0:
                        word_level_features[i, :curr_utt_len] = text_feat[i, start:start + curr_utt_len]
            text_seq = word_level_features
        text_seq = self.text_proj(text_seq)
        text_feat_transformed = self.text_transformer(text_seq, attention_mask=None)
        pooled_output, _ = self.attention(text_feat_transformed)
        pooled_output = self.dropout(pooled_output)
        regression_output = self.classifier(pooled_output)
        return regression_output, pooled_output


class AudioClassifier(nn.Module):

    def __init__(self, audio_dim=1280, hidden_dim=512, num_classes=1):  # <--- MODIFIED
        super(AudioClassifier, self).__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        config = Config(hidden_size=hidden_dim,
                        num_attention_heads=8,
                        intermediate_size=4 * hidden_dim,
                        attention_probs_dropout_prob=0.1,
                        hidden_dropout_prob=0.1,
                        layer_norm_eps=1e-6)
        self.audio_transformer = MELDTransEncoder(config=config, layer_num=1, get_max_lens=60, hidden_size=hidden_dim)
        self.attention = AdditiveAttention(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        audio_seq = inputs
        if audio_seq.dim() == 2:
            audio_seq = audio_seq.unsqueeze(1)
        audio_seq = self.audio_proj(audio_seq)
        audio_feat = self.audio_transformer(audio_seq, attention_mask=None)
        pooled_output, _ = self.attention(audio_feat)
        pooled_output = self.dropout(pooled_output)
        regression_output = self.classifier(pooled_output)

        return regression_output, pooled_output


class VideoClassifier(nn.Module):

    def __init__(self, video_dim=768, hidden_dim=512, num_classes=1):  # <--- MODIFIED
        super(VideoClassifier, self).__init__()
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        config = Config(hidden_size=hidden_dim,
                        num_attention_heads=8,
                        intermediate_size=4 * hidden_dim,
                        attention_probs_dropout_prob=0.1,
                        hidden_dropout_prob=0.1,
                        layer_norm_eps=1e-6)
        self.video_transformer = MELDTransEncoder(config=config, layer_num=1, get_max_lens=40, hidden_size=hidden_dim)

        self.attention = AdditiveAttention(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        video_seq = inputs
        if video_seq.dim() == 2:
            video_seq = video_seq.unsqueeze(1)
        video_seq = self.video_proj(video_seq)
        video_feat = self.video_transformer(video_seq, attention_mask=None)

        pooled_output, _ = self.attention(video_feat)
        pooled_output = self.dropout(pooled_output)
        regression_output = self.classifier(pooled_output)

        return regression_output, pooled_output
