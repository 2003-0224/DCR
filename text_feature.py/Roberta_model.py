# 分类模型
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import RobertaTokenizer, RobertaModel
# from tqdm import tqdm
# import torch.nn as nn
# import warnings
#
# # from peft import LoraConfig, get_peft_model, IA3Config
#
# warnings.filterwarnings("ignore", category=FutureWarning)
#
#
# # 定义 Adapter 模块
# class Adapter(nn.Module):
#     def __init__(self, hidden_size, adapter_size=64):
#         super(Adapter, self).__init__()
#         self.down_project = nn.Linear(hidden_size, adapter_size)
#         self.activation = nn.ReLU()
#         self.up_project = nn.Linear(adapter_size, hidden_size)
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         # x: [batch_size, seq_len, hidden_size]
#         down = self.down_project(x)  # [batch_size, seq_len, adapter_size]
#         activated = self.activation(down)
#         up = self.up_project(activated)  # [batch_size, seq_len, hidden_size]
#         return self.dropout(up) + x  # 残差连接
#
#
# # 情感分类模型, 序列级特征：
# class EmotionClassifier(nn.Module):
#     def __init__(self, roberta_model, num_labels, use_lora, use_adapters, adapter_size=128):
#         super(EmotionClassifier, self).__init__()
#         self.roberta = roberta_model
#         self.classifier = nn.Linear(1024, num_labels)  # RoBERTa 的隐藏层维度为 768
#         self.hidden_size = self.roberta.config.hidden_size
#         self.use_lora = use_lora
#         self.use_adapters = use_adapters
#
#         # 应用 LoRA 到 self.roberta
#         if self.use_lora:
#             lora_config = LoraConfig(
#                 r=8,  # 低秩维度，可调整（4、8、16 等）
#                 lora_alpha=16,  # 缩放因子
#                 target_modules=["query", "value"],  # 对注意力层的 q 和 v 矩阵应用 LoRA
#                 lora_dropout=0.1,  # Dropout
#                 bias="none",  # 不调整偏置
#                 # task_type="SEQ_CLS"  # 序列分类任务
#             )
#             self.roberta = get_peft_model(self.roberta, lora_config)
#             print("Applied LoRA to self.roberta")
#             self.roberta.print_trainable_parameters()  # 查看可训练参数数量
#
#         if self.use_adapters:
#
#             # 冻结 RoBERTa 的所有参数
#             for param in self.roberta.parameters():
#                 param.requires_grad = False
#             print("Froze all RoBERTa parameters for manual Adapters")
#
#             self.adapters = nn.ModuleList()
#             # 为 RoBERTa 的每一层添加 Adapter（roberta-large 有 24 层）
#             for _ in range(self.roberta.config.num_hidden_layers):
#                 self.adapters.append(Adapter(self.hidden_size, adapter_size))
#             print(f"Added manual Adapters with size {adapter_size} to {self.roberta.config.num_hidden_layers} layers")
#
#     def forward(self, input_ids, attention_mask):
#         # 提取 RoBERTa 特征
#         if self.use_adapters:
#             with torch.no_grad():
#                 outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
#                 # 获取所有隐藏层状态
#                 all_hidden_states = outputs.hidden_states  # Tuple of [batch_size, seq_len, hidden_size] for each layer
#         else:
#             outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#
#         # 手动应用 Adapters
#         if self.use_adapters:
#             adapted_hidden_states = []
#             for i, hidden_state in enumerate(all_hidden_states):
#                 if i == 0:  # 第一层（嵌入层输出）不加 Adapter
#                     adapted_hidden_states.append(hidden_state)
#                 else:
#                     # 应用 Adapter
#                     adapted_state = self.adapters[i - 1](hidden_state)
#                     adapted_hidden_states.append(adapted_state)
#             sequence_output = adapted_hidden_states[-1]  # 最后一层输出
#         else:
#             sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
#
#         # Mean Pooling（忽略填充 token）
#         mask = attention_mask.unsqueeze(-1).expand_as(sequence_output)  # [batch_size, seq_len, hidden_size]
#         masked_output = sequence_output * mask
#         sum_output = masked_output.sum(dim=1)  # [batch_size, hidden_size]
#         sum_mask = mask.sum(dim=1)  # [batch_size, hidden_size]
#         pooled_output = sum_output / (sum_mask + 1e-10)  # [batch_size, hidden_size]
#
#         # 分类
#         logits = self.classifier(pooled_output)
#         return logits
#
#
# # # 情感分类模型, target utterance的序列特征：
# class WordEmotionClassifier(nn.Module):
#     def __init__(self, roberta_model, num_labels, max_utt_seq_len=38, use_transformer=False, add_history_token=False,
#                  Roberta_w=None, Roberta_w_plus=None, no_grad_Roberta=False, no_grad_Roberta_plus=False, use_lora=False,
#                  use_adapters=False, adapter_size=128):
#         super(WordEmotionClassifier, self).__init__()
#         self.roberta = roberta_model
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
#         self.hidden_size = self.roberta.config.hidden_size  # 1024 for roberta-large, 768 for roberta-base
#         self.max_utt_seq_len = max_utt_seq_len
#         self.use_transformer = use_transformer
#         self.add_history_token = add_history_token
#         self.no_grad_Roberta = no_grad_Roberta
#         self.Roberta_w_plus = Roberta_w_plus
#         self.no_grad_Roberta_plus = no_grad_Roberta_plus
#         self.roberta_plus = None
#         self.use_lora = use_lora
#         self.use_adapters = use_adapters
#
#         # 如果提供了 Roberta_w，加载第一阶段微调后的参数
#         if Roberta_w is not None:
#             self.Roberta_w = Roberta_w
#             self.roberta.load_state_dict(Roberta_w)
#             print("Loaded first-stage fine-tuned RoBERTa parameters from Roberta_w.")
#
#         # 应用 LoRA 到 self.roberta
#         if self.use_lora:
#             lora_config = LoraConfig(
#                 r=8,  # 低秩维度，可调整（4、8、16 等）
#                 lora_alpha=16,  # 缩放因子
#                 target_modules=["query", "value"],  # 对注意力层的 q 和 v 矩阵应用 LoRA
#                 lora_dropout=0.1,  # Dropout
#                 bias="none",  # 不调整偏置
#                 # task_type="SEQ_CLS"  # 序列分类任务
#             )
#             self.roberta = get_peft_model(self.roberta, lora_config)
#             print("Applied LoRA to self.roberta")
#             self.roberta.print_trainable_parameters()  # 查看可训练参数数量
#
#         # 手动添加 Adapters 并冻结 RoBERTa 参数
#         if self.use_adapters:
#             # 冻结 RoBERTa 的所有参数
#             for param in self.roberta.parameters():
#                 param.requires_grad = False
#             print("Froze all RoBERTa parameters for manual Adapters")
#
#             # 添加 Adapters 到每一层
#             self.adapters = nn.ModuleList()
#             for _ in range(self.roberta.config.num_hidden_layers):
#                 self.adapters.append(Adapter(self.hidden_size, adapter_size))
#             print(f"Added manual Adapters with size {adapter_size} to {self.roberta.config.num_hidden_layers} layers")
#
#         # 如果提供了 Roberta_w_plus，创建独立副本并应用 LoRA
#         if self.Roberta_w_plus is not None:
#             from copy import deepcopy
#             self.roberta_plus = deepcopy(roberta_model)
#             self.roberta_plus.load_state_dict(Roberta_w_plus)
#             print("Loaded Roberta_w_plus into self.roberta_plus")
#         else:
#             self.roberta_plus = None
#
#         # 冻结参数（LoRA 默认冻结原始权重，可选显式冻结）
#         if self.no_grad_Roberta:
#             for param in self.roberta.parameters():
#                 param.requires_grad = False
#         if self.no_grad_Roberta_plus and self.roberta_plus is not None:
#             for param in self.roberta_plus.parameters():
#                 param.requires_grad = False
#
#         # 对word level序列级特征使用transformer来进行预测：
#         # Transformer 块（官方实现）
#         if use_transformer:
#             self.transformer_block = nn.TransformerEncoder(
#                 nn.TransformerEncoderLayer(
#                     d_model=self.hidden_size,  # 输入和输出的维度（匹配 RoBERTa 的 hidden_size）
#                     nhead=8,  # 注意力头数
#                     dim_feedforward=2048,  # 前馈网络的中间维度
#                     dropout=0.2,  # Dropout 比率
#                     activation='relu',  # 前馈网络的激活函数
#                     batch_first=True  # 输入格式为 (batch_size, seq_len, hidden_size)
#                 ),
#                 num_layers=1  # Transformer 层数
#             )
#             self.dropout = nn.Dropout(0.1)
#             self.classifier = nn.Linear(self.hidden_size, num_labels)  # 基于 Transformer 输出进行分类
#         else:
#             # 不使用 Transformer 时，直接池化后分类
#             self.dropout = nn.Dropout(0.1)
#             self.classifier = nn.Linear(self.hidden_size, num_labels)
#
#     def forward(self, input_ids, attention_mask, target_start_pos=None, target_end_pos=None, output_features=False,
#                 **kwargs):
#         # if kwargs:
#         #     print("Forward received extra kwargs:", kwargs)
#
#         # 获取 RoBERTa 输出（冻结参数）
#         if self.no_grad_Roberta:
#             with torch.no_grad():  # 确保 RoBERTa 不计算梯度
#                 if self.use_adapters:
#                     outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask,
#                                            output_hidden_states=True)
#                     # 获取所有隐藏层状态
#                     all_hidden_states = outputs.hidden_states  # Tuple of [batch_size, seq_len, hidden_size] for each layer
#                 else:
#                     outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         else:
#             outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#
#         # 手动应用 Adapters
#         if self.use_adapters:
#             adapted_hidden_states = []
#             for i, hidden_state in enumerate(all_hidden_states):
#                 if i == 0:  # 第一层（嵌入层输出）不加 Adapter
#                     adapted_hidden_states.append(hidden_state)
#                 else:
#                     # 应用 Adapter
#                     adapted_state = self.adapters[i - 1](hidden_state)
#                     adapted_hidden_states.append(adapted_state)
#             hidden_states = adapted_hidden_states[-1]  # 最后一层输出
#         else:
#             hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
#
#         # 提取目标话语的词级特征
#         batch_size = input_ids.shape[0]
#         hidden_size = hidden_states.shape[-1]
#         word_level_features = torch.zeros(batch_size, self.max_utt_seq_len, hidden_size).to(input_ids.device)
#         word_level_masks = torch.zeros(batch_size, self.max_utt_seq_len).to(input_ids.device)
#
#         if target_start_pos is not None and target_end_pos is not None:
#             for i in range(batch_size):
#                 start = target_start_pos[i].item()
#                 end = target_end_pos[i].item()
#                 curr_utt_len = end - start
#                 if curr_utt_len > self.max_utt_seq_len:
#                     curr_utt_len = self.max_utt_seq_len
#                 if curr_utt_len > 0:
#                     word_level_features[i, :curr_utt_len] = hidden_states[i, start:start + curr_utt_len]
#                     word_level_masks[i, :curr_utt_len] = 1
#         else:
#             raise ValueError("target_start_pos and target_end_pos must be provided for word-level feature extraction")
#
#         # 如果 add_history_token 为 True，添加历史信息 token
#         if self.add_history_token:
#             # 计算整个序列（context + target）的均值池化，作为历史信息 token
#
#             full_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)  # (batch_size, seq_len, hidden_size)
#
#             if self.Roberta_w_plus is not None:  # plus作为历史信息的模型：
#                 if self.no_grad_Roberta_plus:
#                     with torch.no_grad():  # 确保 RoBERTa 不计算梯度
#                         outputs = self.roberta_plus(input_ids=input_ids, attention_mask=attention_mask)
#                 else:
#                     outputs = self.roberta_plus(input_ids=input_ids, attention_mask=attention_mask)
#                 hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
#
#             # 在当前对话前，加入历史信息：
#             masked_hidden_states = hidden_states * full_mask  # 屏蔽 padding
#             sum_hidden_states = masked_hidden_states.sum(dim=1)  # (batch_size, hidden_size)
#             sum_full_mask = full_mask.sum(dim=1)  # (batch_size, hidden_size)
#             history_token = sum_hidden_states / (sum_full_mask + 1e-10)  # (batch_size, hidden_size)
#
#             # 将历史信息 token 添加到目标序列的最前面
#             new_max_seq_len = self.max_utt_seq_len + 1
#             enhanced_features = torch.zeros(batch_size, new_max_seq_len, hidden_size).to(input_ids.device)
#             enhanced_masks = torch.zeros(batch_size, new_max_seq_len).to(input_ids.device)
#             enhanced_features[:, 0, :] = history_token  # 第一位是历史 token
#             enhanced_features[:, 1:self.max_utt_seq_len + 1, :] = word_level_features  # 后面接目标序列
#             enhanced_masks[:, 0] = 1  # 历史 token 的 mask 为 1
#             enhanced_masks[:, 1:self.max_utt_seq_len + 1] = word_level_masks  # 目标序列的 mask
#         else:
#             # 不添加历史 token，使用原始的目标序列
#             enhanced_features = word_level_features
#             enhanced_masks = word_level_masks
#
#         # 根据 use_transformer 参数选择处理方式
#         if self.use_transformer:
#             # 使用 Transformer 处理词级特征
#             # 输入格式: (batch_size, max_utt_seq_len, hidden_size)
#             # padding_mask: (batch_size, max_utt_seq_len), True 表示忽略
#             # 使用 Transformer 处理词级特征（可能是增强后的）
#             padding_mask = (enhanced_masks == 0).to(input_ids.device)  # (batch_size, max_utt_seq_len 或 new_max_seq_len)
#             transformer_output = self.transformer_block(enhanced_features, src_key_padding_mask=padding_mask)
#             # transformer_output: (batch_size, max_utt_seq_len, hidden_size)
#
#             # 池化 Transformer 输出（忽略 padding）
#             mask = enhanced_masks.unsqueeze(-1).expand_as(transformer_output)  # (batch_size, seq_len, hidden_size)
#             masked_output = transformer_output * mask
#             sum_output = masked_output.sum(dim=1)  # (batch_size, hidden_size)
#             sum_mask = mask.sum(dim=1)  # (batch_size, hidden_size)
#             pooled_output = sum_output / (sum_mask + 1e-10)  # (batch_size, hidden_size)
#
#             # 分类
#             pooled_output = self.dropout(pooled_output)
#             logits = self.classifier(pooled_output)  # (batch_size, num_labels)
#         else:
#             # 不使用 Transformer，直接池化后分类
#             # Mean Pooling（忽略填充 token）, 直接对word level的数据进行池化，后续可以进行用transformer进行分类：
#             mask = enhanced_masks.unsqueeze(-1).expand_as(enhanced_features)  # (batch_size, seq_len, hidden_size)
#             masked_output = enhanced_features * mask
#             sum_output = masked_output.sum(dim=1)  # (batch_size, hidden_size)
#             sum_mask = mask.sum(dim=1)  # (batch_size, hidden_size)
#             pooled_output = sum_output / (sum_mask + 1e-10)  # (batch_size, hidden_size)
#
#             # 分类
#             pooled_output_end = self.dropout(pooled_output)
#             logits = self.classifier(pooled_output_end)  # (batch_size, num_labels)
#
#         if output_features:
#             return logits, pooled_output, enhanced_features, enhanced_masks
#
#         else:
#             return logits


# 回归模型
import torch
from torch import nn
import transformers
from transformers import RobertaTokenizer
from peft import LoraConfig, get_peft_model
import warnings


# 定义 Adapter 模块
class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        down = self.down_project(x)
        activated = self.activation(down)
        up = self.up_project(activated)
        return self.dropout(up) + x


# 情感回归模型 (单片段/序列级特征):
class EmotionClassifier(nn.Module):
    def __init__(self, roberta_model, num_labels, use_lora, use_adapters, adapter_size=128):
        super(EmotionClassifier, self).__init__()
        self.roberta = roberta_model
        self.classifier = nn.Linear(1024, 1)
        self.hidden_size = self.roberta.config.hidden_size
        self.use_lora = use_lora
        self.use_adapters = use_adapters
        if self.use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
            )
            self.roberta = get_peft_model(self.roberta, lora_config)

        if self.use_adapters:
            for param in self.roberta.parameters():
                param.requires_grad = False

            self.adapters = nn.ModuleList()
            for _ in range(self.roberta.config.num_hidden_layers):
                self.adapters.append(Adapter(self.hidden_size, adapter_size))

    def forward(self, input_ids, attention_mask):
        if self.use_adapters:
            with torch.no_grad():
                outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                all_hidden_states = outputs.hidden_states
        else:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_adapters:
            adapted_hidden_states = []
            for i, hidden_state in enumerate(all_hidden_states):
                if i == 0:
                    adapted_hidden_states.append(hidden_state)
                else:
                    adapted_state = self.adapters[i - 1](hidden_state)
                    adapted_hidden_states.append(adapted_state)
            sequence_output = adapted_hidden_states[-1]
        else:
            sequence_output = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand_as(sequence_output)
        sum_output = (sequence_output * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1)
        pooled_output = sum_output / (sum_mask + 1e-10)
        logits = self.classifier(pooled_output)  # (batch_size, 1)
        return logits


# 情感回归模型 (Word-level/复杂):
class WordEmotionClassifier(nn.Module):
    def __init__(self, roberta_model, num_labels, max_utt_seq_len=38, use_transformer=False, add_history_token=False,
                 Roberta_w=None, Roberta_w_plus=None, no_grad_Roberta=False, no_grad_Roberta_plus=False, use_lora=False,
                 use_adapters=False, adapter_size=128):
        super(WordEmotionClassifier, self).__init__()
        self.roberta = roberta_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 1)
        self.hidden_size = self.roberta.config.hidden_size
        self.max_utt_seq_len = max_utt_seq_len
        self.use_transformer = use_transformer
        self.add_history_token = add_history_token
        self.no_grad_Roberta = no_grad_Roberta
        self.Roberta_w_plus = Roberta_w_plus
        self.no_grad_Roberta_plus = no_grad_Roberta_plus
        self.roberta_plus = None
        self.use_lora = use_lora
        self.use_adapters = use_adapters
        if Roberta_w is not None:
            self.Roberta_w = Roberta_w
            self.roberta.load_state_dict(Roberta_w)

        if self.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.1,
                                     bias="none")
            self.roberta = get_peft_model(self.roberta, lora_config)

        if self.use_adapters:
            for param in self.roberta.parameters():
                param.requires_grad = False
            self.adapters = nn.ModuleList(
                [Adapter(self.hidden_size, adapter_size) for _ in range(self.roberta.config.num_hidden_layers)])

        if self.Roberta_w_plus is not None:
            from copy import deepcopy
            self.roberta_plus = deepcopy(roberta_model)
            self.roberta_plus.load_state_dict(Roberta_w_plus)
        else:
            self.roberta_plus = None
        if self.no_grad_Roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False
        if self.no_grad_Roberta_plus and self.roberta_plus is not None:
            for param in self.roberta_plus.parameters():
                param.requires_grad = False
        if use_transformer:
            self.transformer_block = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, dim_feedforward=2048, dropout=0.2,
                                           activation='relu', batch_first=True),
                num_layers=1
            )
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.hidden_size, 1)
        else:
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask, target_start_pos=None, target_end_pos=None, output_features=False,
                **kwargs):
        if self.no_grad_Roberta:
            with torch.no_grad():
                if self.use_adapters:
                    outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask,
                                           output_hidden_states=True)
                    all_hidden_states = outputs.hidden_states
                else:
                    outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_adapters:
            adapted_hidden_states = []
            for i, hidden_state in enumerate(all_hidden_states):
                if i == 0:
                    adapted_hidden_states.append(hidden_state)
                else:
                    adapted_state = self.adapters[i - 1](hidden_state)
                    adapted_hidden_states.append(adapted_state)
            hidden_states = adapted_hidden_states[-1]
        else:
            hidden_states = outputs.last_hidden_state
        batch_size = input_ids.shape[0]
        hidden_size = hidden_states.shape[-1]
        word_level_features = torch.zeros(batch_size, self.max_utt_seq_len, hidden_size).to(input_ids.device)
        word_level_masks = torch.zeros(batch_size, self.max_utt_seq_len).to(input_ids.device)
        if target_start_pos is not None and target_end_pos is not None:
            for i in range(batch_size):
                start = target_start_pos[i].item()
                end = target_end_pos[i].item()
                curr_utt_len = end - start
                if curr_utt_len > self.max_utt_seq_len:
                    curr_utt_len = self.max_utt_seq_len
                if curr_utt_len > 0:
                    word_level_features[i, :curr_utt_len] = hidden_states[i, start:start + curr_utt_len]
                    word_level_masks[i, :curr_utt_len] = 1
        else:
            raise ValueError("target_start_pos and target_end_pos must be provided for word-level feature extraction")
        if self.add_history_token:
            full_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            if self.Roberta_w_plus is not None:
                if self.no_grad_Roberta_plus:
                    with torch.no_grad():
                        outputs = self.roberta_plus(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.roberta_plus(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
            masked_hidden_states = hidden_states * full_mask
            sum_hidden_states = masked_hidden_states.sum(dim=1)
            sum_full_mask = full_mask.sum(dim=1)
            history_token = sum_hidden_states / (sum_full_mask + 1e-10)
            new_max_seq_len = self.max_utt_seq_len + 1
            enhanced_features = torch.zeros(batch_size, new_max_seq_len, hidden_size).to(input_ids.device)
            enhanced_masks = torch.zeros(batch_size, new_max_seq_len).to(input_ids.device)
            enhanced_features[:, 0, :] = history_token
            enhanced_features[:, 1:self.max_utt_seq_len + 1, :] = word_level_features
            enhanced_masks[:, 0] = 1
            enhanced_masks[:, 1:self.max_utt_seq_len + 1] = word_level_masks
        else:
            enhanced_features = word_level_features
            enhanced_masks = word_level_masks
        if self.use_transformer:
            padding_mask = (enhanced_masks == 0).to(input_ids.device)
            transformer_output = self.transformer_block(enhanced_features, src_key_padding_mask=padding_mask)
            mask = enhanced_masks.unsqueeze(-1).expand_as(transformer_output)
            sum_output = (transformer_output * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1)
            pooled_output = sum_output / (sum_mask + 1e-10)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            mask = enhanced_masks.unsqueeze(-1).expand_as(enhanced_features)
            sum_output = (enhanced_features * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1)
            pooled_output = sum_output / (sum_mask + 1e-10)
            pooled_output_end = self.dropout(pooled_output)
            logits = self.classifier(pooled_output_end)  # (batch_size, 1)
        if output_features:
            return logits, pooled_output, enhanced_features, enhanced_masks
        else:
            return logits
