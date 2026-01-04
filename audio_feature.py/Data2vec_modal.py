# 分类模型
# import torch
# import torch.nn as nn
# # import torchaudio
# import pandas as pd
# import os
# import numpy as np
# from transformers import Data2VecAudioModel, WavLMModel, Wav2Vec2Model, HubertModel, WhisperModel, WhisperConfig
# # from peft import LoraConfig, get_peft_model, IA3Config
# import torch.nn as nn
# from transformer_FacialMMT import MELDTransEncoder
# from chenyin_whisper import CustomWhisperEncoderLayer, BaseModel
# from transformers.models.whisper import modeling_whisper as whisper_model
#
# import logging
# import transformers
#
# # 设置日志级别为 ERROR，屏蔽 WARNING
# logging.getLogger("transformers").setLevel(logging.ERROR)
#
# whisper_model.WhisperEncoderLayer = CustomWhisperEncoderLayer
#
#
# # 配置类
# class Config:
#     def __init__(self, hidden_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob,
#                  hidden_dropout_prob, layer_norm_eps):
#         self.hidden_size = hidden_size
#         self.num_attention_heads = num_attention_heads
#         self.intermediate_size = intermediate_size
#         self.attention_probs_dropout_prob = attention_probs_dropout_prob
#         self.hidden_dropout_prob = hidden_dropout_prob
#         self.layer_norm_eps = layer_norm_eps
#
#
# # 情感分类模型
# class EmotionClassifier(nn.Module):
#     def __init__(self, num_classes=7, use_lora=False, use_transformer=False, weight_attn=False, model_type=None, max_length = 6, use_adapters=False):
#         super(EmotionClassifier, self).__init__()
#         self.model_type = model_type
#         if model_type == "WavLM":
#             self.model = WavLMModel.from_pretrained("microsoft/wavlm-large")
#             self.feature_dim = self.model.config.hidden_size
#         elif model_type == "Data2Vec":
#             self.model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-large-960h")
#             self.feature_dim = self.model.config.hidden_size
#         elif model_type == "Wav2Vec2":
#             self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
#             self.feature_dim = self.model.config.hidden_size
#         elif model_type == "Hubert":
#             self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
#             self.feature_dim = self.model.config.hidden_size
#         elif model_type == 'Whisper':
#             # 加载预训练模型
#             config = WhisperConfig.from_pretrained("/data/home/chenqian/whisper_large_v3/")  # WhisperPretrain没有使用掩码的注意力效果
#             config.chunk_length = max_length
#             config.adapter_dim = 196
#             config.adapter_scale = 0.1
#             self.model = WhisperModel.from_pretrained("/data/home/chenqian/whisper_large_v3/", config = config)
#             self.feature_dim = 1280
#             self.model = self.model.encoder
#
#             # 保存原始的位置编码权重
#             original_weights = self.model.embed_positions.weight.data
#
#             # 修改位置编码的最大长度，同时保持预训练权重
#             max_positions = int(max_length * 100 / 2)  # 略大于你的最大长度30
#             self.model.config.max_source_positions = max_positions
#             self.model.embed_positions = nn.Embedding(
#                 max_positions,
#                 self.model.config.d_model
#             )
#             # 将原始权重的前max_positions个复制到新的embedding中
#             self.model.embed_positions.weight.data[:] = original_weights[:max_positions]
#             self.model.embed_positions.requires_grad_(False)
#
#         self.classifier = nn.Linear(self.feature_dim, num_classes)  # hidden_size=1024
#         self.use_lora = use_lora
#         self.use_transformer = use_transformer
#         self.weight_attn = weight_attn
#         self.use_adapters = use_adapters
#
#         if self.use_transformer:
#             # 配置 MELDTransEncoder
#             config = Config(
#                 hidden_size=self.feature_dim,
#                 num_attention_heads=8,  # 1024 ÷ 8 = 128
#                 intermediate_size=4 * self.feature_dim,  # 4 × 1024 = 4096
#                 attention_probs_dropout_prob=0.1,
#                 hidden_dropout_prob=0.1,
#                 layer_norm_eps=1e-6
#             )
#             self.transformer = MELDTransEncoder(
#                 config=config,
#                 layer_num=1,  # 3 层 Transformer
#                 get_max_lens=max_length*5,  # 包含起始 token
#                 hidden_size=self.feature_dim
#             )
#
#         if use_adapters:
#             for param in self.model.parameters():
#                 param.requires_grad = False
#
#             # 仅训练 Adapter 参数
#             for layer in self.model.layers:
#                 if isinstance(layer, CustomWhisperEncoderLayer):
#                     for param in layer.S_Adapter.parameters():
#                         param.requires_grad = True
#                     for param in layer.MLP_Adapter.parameters():
#                         param.requires_grad = True
#             num_param = sum(p.numel()
#                             for p in self.model.parameters() if p.requires_grad)/1e6
#             num_total_param = sum(p.numel() for p in self.model.parameters())/1e6
#             print(f"Trainable parameters: {num_param:.2f}M")
#             print(f"Total parameters: {num_total_param:.2f}M")
#
#
#         self.dropout = nn.Dropout(0.2)  # 增加 Dropout
#
#         if self.weight_attn:
#             self.attention = nn.Linear(self.feature_dim, 1)
#
#         # 添加 Conv1d 层以减少序列长度
#         self.conv1d = nn.Conv1d(
#             in_channels=self.feature_dim,
#             out_channels=self.feature_dim,
#             kernel_size=10,
#             stride=10,
#             padding=0
#         )
#
#     def forward(self, input_values, attention_mask=None, output_features=False):
#         if self.use_adapters:
#             outputs = self.model(input_values, attention_mask=None)
#         else:
#             with torch.no_grad():
#                 outputs = self.model(input_values, attention_mask=None)
#         hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
#
#         # 使用 Conv1d 降采样
#         hidden_states = hidden_states.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
#         hidden_states = self.conv1d(hidden_states)  # [batch_size, hidden_size, new_seq_len]
#         hidden_states = hidden_states.transpose(1, 2)  # [batch_size, new_seq_len, hidden_size]
#         new_seq_len = hidden_states.shape[1]  # 例如 600 / 10 = 60 (约 64)
#
#
#
#         # print(hidden_states.shape)
#         if self.use_transformer:
#             transformer_output = self.transformer(hidden_states, None)  # (batch_size, max_seq_len + 1, input_dim)
#             if self.weight_attn:
#                 attn_weights = torch.softmax(self.attention(transformer_output), dim=1)
#                 pooled_output = torch.sum(transformer_output * attn_weights, dim=1)
#             else:
#                 if self.model_type == "Whisper":
#                     pooled_output = torch.mean(transformer_output, dim=1)
#                 else:
#                     mask = attention_mask.unsqueeze(-1).expand_as(transformer_output)
#                     masked_output = transformer_output * mask
#                     sum_output = masked_output.sum(dim=1)
#                     sum_mask = mask.sum(dim=1)
#                     pooled_output = sum_output / (sum_mask + 1e-10)
#         else:
#             if self.weight_attn:
#                 attn_weights = torch.softmax(self.attention(hidden_states), dim=1)  # [batch_size, seq_len, 1]
#                 pooled_output = torch.sum(hidden_states * attn_weights, dim=1)
#             else:
#                 pooled_output = torch.mean(hidden_states, dim=1)
#         pooled_output_end = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output_end)
#         if output_features:
#             return logits, pooled_output, hidden_states
#         else:
#             return logits
#
#
#     def save_adapters(self, save_path):
#         """保存 Adapter 参数和 Conv1d 参数"""
#         if not self.use_adapters:
#             print("Adapter mode is not enabled. Skipping save_adapters.")
#             return
#
#         adapter_state_dict = {}
#         # 保存 Adapter 参数
#         for idx, layer in enumerate(self.model.layers):
#             if isinstance(layer, CustomWhisperEncoderLayer):
#                 adapter_state_dict[f"layer_{idx}_S_Adapter"] = layer.S_Adapter.state_dict()
#                 adapter_state_dict[f"layer_{idx}_MLP_Adapter"] = layer.MLP_Adapter.state_dict()
#
#         # 保存 Conv1d 参数
#         adapter_state_dict["conv1d"] = self.conv1d.state_dict()
#
#         # 保存到指定路径
#         torch.save(adapter_state_dict, save_path)
#         print(f"Adapter and Conv1d parameters saved to {save_path}")
#
#     def load_adapters(self, load_path):
#         """加载 Adapter 参数和 Conv1d 参数"""
#         if not self.use_adapters:
#             print("Adapter mode is not enabled. Skipping load_adapters.")
#             return
#
#         adapter_state_dict = torch.load(load_path, map_location=torch.device('cpu'))
#         # 加载 Adapter 参数
#         for idx, layer in enumerate(self.model.layers):
#             if isinstance(layer, CustomWhisperEncoderLayer):
#                 layer.S_Adapter.load_state_dict(adapter_state_dict[f"layer_{idx}_S_Adapter"])
#                 layer.MLP_Adapter.load_state_dict(adapter_state_dict[f"layer_{idx}_MLP_Adapter"])
#
#         # 加载 Conv1d 参数
#         self.conv1d.load_state_dict(adapter_state_dict["conv1d"])
#
#         print(f"Adapter and Conv1d parameters loaded from {load_path}")


# 回归模型
import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from transformers import Data2VecAudioModel, WavLMModel, Wav2Vec2Model, HubertModel, WhisperModel, WhisperConfig
from transformer_FacialMMT import MELDTransEncoder
from chenyin_whisper import CustomWhisperEncoderLayer, BaseModel
from transformers.models.whisper import modeling_whisper as whisper_model
import logging
import transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
whisper_model.WhisperEncoderLayer = CustomWhisperEncoderLayer


# 配置类
class Config:
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob,
                 hidden_dropout_prob, layer_norm_eps):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps


# 情感回归模型
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=1, use_lora=False, use_transformer=False, weight_attn=False, model_type=None,
                 max_length=6, use_adapters=False):
        super(EmotionClassifier, self).__init__()
        self.model_type = model_type
        if model_type == "WavLM":
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-large")
            self.feature_dim = self.model.config.hidden_size
        elif model_type == "Data2Vec":
            self.model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-large-960h")
            self.feature_dim = self.model.config.hidden_size
        elif model_type == "Wav2Vec2":
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
            self.feature_dim = self.model.config.hidden_size
        elif model_type == "Hubert":
            self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
            self.feature_dim = self.model.config.hidden_size
        elif model_type == 'Whisper':
            # 加载预训练模型
            config = WhisperConfig.from_pretrained(
                "/data/home/chenqian/whisper_large_v3/")  # WhisperPretrain没有使用掩码的注意力效果
            config.chunk_length = max_length
            config.adapter_dim = 196
            config.adapter_scale = 0.1
            self.model = WhisperModel.from_pretrained("/data/home/chenqian/whisper_large_v3/", config=config)
            self.feature_dim = 1280
            self.model = self.model.encoder
            # 保存原始的位置编码权重
            original_weights = self.model.embed_positions.weight.data
            # 修改位置编码的最大长度，同时保持预训练权重
            max_positions = int(max_length * 100 / 2)  # 略大于你的最大长度30
            self.model.config.max_source_positions = max_positions
            self.model.embed_positions = nn.Embedding(
                max_positions,
                self.model.config.d_model
            )
            self.model.embed_positions.weight.data[:] = original_weights[:max_positions]
            self.model.embed_positions.requires_grad_(False)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.use_lora = use_lora
        self.use_transformer = use_transformer
        self.weight_attn = weight_attn
        self.use_adapters = use_adapters
        if self.use_transformer:
            config = Config(
                hidden_size=self.feature_dim,
                num_attention_heads=8,
                intermediate_size=4 * self.feature_dim,
                attention_probs_dropout_prob=0.1,
                hidden_dropout_prob=0.1,
                layer_norm_eps=1e-6
            )
            self.transformer = MELDTransEncoder(
                config=config,
                layer_num=1,  # 3 层 Transformer
                get_max_lens=max_length * 5,
                hidden_size=self.feature_dim
            )
        if use_adapters:
            for param in self.model.parameters():
                param.requires_grad = False
            # 仅训练 Adapter 参数
            for layer in self.model.layers:
                if isinstance(layer, CustomWhisperEncoderLayer):
                    for param in layer.S_Adapter.parameters():
                        param.requires_grad = True
                    for param in layer.MLP_Adapter.parameters():
                        param.requires_grad = True
            num_param = sum(p.numel()
                            for p in self.model.parameters() if p.requires_grad) / 1e6
            num_total_param = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"Trainable parameters: {num_param:.2f}M")
            print(f"Total parameters: {num_total_param:.2f}M")
        self.dropout = nn.Dropout(0.2)  # 增加 Dropout
        if self.weight_attn:
            self.attention = nn.Linear(self.feature_dim, 1)
        # 添加 Conv1d 层以减少序列长度
        self.conv1d = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=10,
            stride=10,
            padding=0
        )

    def forward(self, input_values, attention_mask=None, output_features=False):
        if self.use_adapters:
            outputs = self.model(input_values, attention_mask=None)
        else:
            with torch.no_grad():
                outputs = self.model(input_values, attention_mask=None)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        hidden_states = hidden_states.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        hidden_states = self.conv1d(hidden_states)  # [batch_size, hidden_size, new_seq_len]
        hidden_states = hidden_states.transpose(1, 2)  # [batch_size, new_seq_len, hidden_size]
        if self.use_transformer:
            transformer_output = self.transformer(hidden_states, None)  # (batch_size, max_seq_len + 1, input_dim)
            if self.weight_attn:
                attn_weights = torch.softmax(self.attention(transformer_output), dim=1)
                pooled_output = torch.sum(transformer_output * attn_weights, dim=1)
            else:
                if self.model_type == "Whisper":
                    pooled_output = torch.mean(transformer_output, dim=1)
                else:
                    pooled_output = torch.mean(transformer_output, dim=1)
        else:
            if self.weight_attn:
                attn_weights = torch.softmax(self.attention(hidden_states), dim=1)  # [batch_size, seq_len, 1]
                pooled_output = torch.sum(hidden_states * attn_weights, dim=1)
            else:
                pooled_output = torch.mean(hidden_states, dim=1)

        pooled_output_end = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_end)
        if output_features:
            return logits, pooled_output, hidden_states
        else:
            return logits

    def save_adapters(self, save_path):
        if not self.use_adapters:
            print("Adapter mode is not enabled. Skipping save_adapters.")
            return
        adapter_state_dict = {}
        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, CustomWhisperEncoderLayer):
                adapter_state_dict[f"layer_{idx}_S_Adapter"] = layer.S_Adapter.state_dict()
                adapter_state_dict[f"layer_{idx}_MLP_Adapter"] = layer.MLP_Adapter.state_dict()
        adapter_state_dict["conv1d"] = self.conv1d.state_dict()
        # 保存到指定路径
        torch.save(adapter_state_dict, save_path)
        print(f"Adapter and Conv1d parameters saved to {save_path}")

    def load_adapters(self, load_path):
        if not self.use_adapters:
            print("Adapter mode is not enabled. Skipping load_adapters.")
            return
        adapter_state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, CustomWhisperEncoderLayer):
                layer.S_Adapter.load_state_dict(adapter_state_dict[f"layer_{idx}_S_Adapter"])
                layer.MLP_Adapter.load_state_dict(adapter_state_dict[f"layer_{idx}_MLP_Adapter"])
        self.conv1d.load_state_dict(adapter_state_dict["conv1d"])
        print(f"Adapter and Conv1d parameters loaded from {load_path}")