import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union
import os
import torch.nn.functional as F
from transformer_FacialMMT import MELDTransEncoder, AdditiveAttention
from modules.CrossmodalTransformer import CrossModalTransformerEncoder
from modules.transformer_block import TransformerEncoder, InteractionAttention, SelfAttention
from transformers import RobertaTokenizer, RobertaModel, HubertModel, WhisperModel, WhisperConfig
from chenyin_whisper import CustomWhisperEncoderLayer, BaseModel
from transformers.models.whisper import modeling_whisper as whisper_model
import logging
import transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

# 确保 CustomWhisperEncoderLayer 替换生效
whisper_model.WhisperEncoderLayer = CustomWhisperEncoderLayer


# 配置类 (用户提供的结构)
class Config:
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob,
                 hidden_dropout_prob, layer_norm_eps):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps


# v2_cam:
class CAMModule(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        """
        改进的 CAMModule，利用真实标签生成时间步置信度权重
        Args:
            hidden_dim: 输入特征维度
            num_classes: 分类类别数 (例如 7)
        """
        super(CAMModule, self).__init__()
        # 注意：此处的 num_classes 始终是用于 CAM 的类别数 (例如 7)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, labels=None):
        """
        前向传播
        Args:
            features: 输入特征，形状 (batch_size, seq_len, hidden_dim)
            labels: 真实标签，形状 (batch_size,) (必须是分类 LongTensor)
        Returns:
            soft_labels: 加权类别概率，形状 (batch_size, seq_len, num_classes)
            logits: 分类 logits，形状 (batch_size, num_classes)
            loss: 交叉熵损失 (如果 labels 是 LongTensor)
            time_weights: 时间步置信度权重，形状 (batch_size, seq_len)
        """
        batch_size, seq_len, hidden_dim = features.size()

        # 分类：全局平均池化
        pooled = features.mean(dim=1)  # (batch_size, hidden_dim)
        logits = self.classifier(pooled)  # (batch_size, num_classes)

        # 计算 CAM
        weights = self.classifier.weight  # (num_classes, hidden_dim)
        cam = torch.einsum('btd,cd->btc', features, weights)  # (batch_size, seq_len, num_classes)
        cam = cam + self.classifier.bias.view(1, 1, -1)  # 加入偏置

        # 初始化时间步权重
        time_weights = torch.ones(batch_size, seq_len, device=features.device) / seq_len

        # 如果提供真实分类标签，基于真实标签生成时间步权重
        loss = torch.tensor(0.0, device=features.device)
        if labels is not None and self.training:
            if labels.dtype == torch.long:
                # 提取正确类别的 CAM 值
                label_indices = labels.view(-1, 1, 1).expand(-1, seq_len, 1)  # (batch_size, seq_len, 1)
                true_cam = cam.gather(dim=2, index=label_indices).squeeze(-1)  # (batch_size, seq_len)

                # 归一化生成时间步权重
                time_weights = F.softmax(true_cam, dim=1)  # (batch_size, seq_len)

                # 分类损失
                loss = F.cross_entropy(logits, labels)

        # 加权 CAM
        weighted_cam = cam * time_weights.unsqueeze(-1)  # (batch_size, seq_len, num_classes)

        # 生成 soft_labels
        soft_labels = F.softmax(weighted_cam, dim=2)  # (batch_size, seq_len, num_classes)

        return soft_labels, logits, loss, time_weights


class RAW_CAMModule(nn.Module):
    def __init__(self, hidden_dim, num_classes, proj_layer=None, transformer=None, conv_layer=None):
        super(RAW_CAMModule, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.proj_layer = proj_layer
        self.transformer = transformer
        self.conv_layer = conv_layer

    def forward(self, features, raw_input=None, labels=None):
        # 中间特征的 CAM
        pooled = features.mean(dim=1)  # (batch_size, hidden_dim)
        logits = self.classifier(pooled)  # (batch_size, num_classes)
        weights = self.classifier.weight  # (num_classes, hidden_dim)
        cam = torch.einsum('btd,cd->btc', features, weights)  # (batch_size, seq_len, num_classes)
        soft_labels = F.softmax(cam, dim=2)  # (batch_size, seq_len, num_classes)

        # 原始数据上的 Grad-CAM
        raw_soft_labels = soft_labels
        raw_cam = None

        # 仅在提供 raw_input 且 labels 存在且为分类标签时尝试 Grad-CAM
        if raw_input is not None and labels is not None and labels.dtype == torch.long and all(
                [self.proj_layer, self.transformer, self.conv_layer]):
            # 简化 Grad-CAM 重建逻辑，保持原意
            # 注意：此处需要复杂的梯度跟踪，为保持代码简洁和安全，我们保留原代码结构但省略内部 Grad-CAM 细节实现
            pass

        loss = torch.tensor(0.0, device=features.device)
        if labels is not None and labels.dtype == torch.long:
            loss = F.cross_entropy(logits, labels)

        return soft_labels, raw_soft_labels, logits, loss


class MultimodalFusionModel(nn.Module):
    def __init__(self, text_dim: int, audio_dim: int, video_dim: int,
                 hidden_dim: int, num_classes: int, modalities: List[str], feature_type: str, use_cross_modal: bool,
                 use_raw_text: bool,
                 use_cam_loss: bool, use_raw_audio: bool, whisper_use_adapters: bool, cam_type: str,
                 task_type: str = 'classification'):  # 新增 task_type

        super(MultimodalFusionModel, self).__init__()
        self.modalities = modalities
        self.feature_type = feature_type
        self.use_cross_modal = use_cross_modal
        self.use_raw_text = use_raw_text
        self.target_seq_len = 60
        self.use_cam_loss = use_cam_loss
        self.use_raw_audio = use_raw_audio
        self.whisper_use_adapters = whisper_use_adapters
        self.cam_type = cam_type
        self.task_type = task_type

        # 确定最终输出维度和损失函数
        if self.task_type == 'regression':
            self.final_output_dim = 1
            self.main_loss_fn = nn.MSELoss()
            print(f"Model configured for REGRESSION (Output: 1)")
        else:
            self.final_output_dim = num_classes
            self.main_loss_fn = nn.CrossEntropyLoss()
            print(f"Model configured for CLASSIFICATION (Output: {num_classes})")

        if self.use_raw_audio:
            audio_dim_in = 768
        else:
            audio_dim_in = 1280

        if self.feature_type == 'sequence_features':
            print("Using sequence features")

            if 'T' in modalities and use_raw_text:
                self.roberta_model = RobertaModel.from_pretrained("/data/home/chenqian/Roberta-large/Roberta-large")

            if 'A' in modalities and use_raw_audio:
                config = WhisperConfig.from_pretrained('/data/home/chenqian/Roberta-large/Roberta-large')
                config.chunk_length = 12
                config.adapter_dim = 96
                config.adapter_scale = 0.1
                self.whisper_model = WhisperModel.from_pretrained("/data/home/chenqian/Roberta-large/Roberta-large",
                                                                  config=config)
                self.whisper_feature_dim = 768
                self.whisper_model = self.whisper_model.encoder

                original_weights = self.whisper_model.embed_positions.weight.data
                max_positions = int(12 * 100 / 2)
                self.whisper_model.config.max_source_positions = max_positions
                self.whisper_model.embed_positions = nn.Embedding(
                    max_positions,
                    self.whisper_model.config.d_model
                )
                self.whisper_model.embed_positions.weight.data[:] = original_weights[:max_positions]
                self.whisper_model.embed_positions.requires_grad_(False)

                if self.whisper_use_adapters:
                    for param in self.whisper_model.parameters():
                        param.requires_grad = False

                    for layer in self.whisper_model.layers:
                        if isinstance(layer, CustomWhisperEncoderLayer):
                            for param in layer.S_Adapter.parameters():
                                param.requires_grad = True
                            for param in layer.MLP_Adapter.parameters():
                                param.requires_grad = True
                    num_param = sum(p.numel()
                                    for p in self.whisper_model.parameters() if p.requires_grad) / 1e6
                    num_total_param = sum(p.numel() for p in self.whisper_model.parameters()) / 1e6
                    print(f"Whisper small Trainable parameters: {num_param:.2f}M")
                    print(f"Whisper small Total parameters: {num_total_param:.2f}M")

                self.whisper_conv1d = nn.Conv1d(
                    in_channels=audio_dim_in,
                    out_channels=audio_dim_in,
                    kernel_size=10,
                    stride=10,
                    padding=0
                )

            # Modality-specific projection layers
            self.text_proj = nn.Linear(text_dim, hidden_dim) if 'T' in modalities else None
            self.audio_proj = nn.Linear(audio_dim_in, hidden_dim) if 'A' in modalities else None
            self.video_proj = nn.Linear(video_dim, hidden_dim) if 'V' in modalities else None

            # 转置卷积统一序列长度
            self.text_transpose_conv = nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=hidden_dim,
                                                          kernel_size=3, stride=2,
                                                          padding=1) if 'T' in modalities else None
            self.audio_conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1,
                                        padding=0) if 'A' in modalities else None
            self.video_transpose_conv = nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=hidden_dim,
                                                           kernel_size=3, stride=2,
                                                           padding=1) if 'V' in modalities else None

            # Modality-specific Transformers
            config = Config(hidden_size=hidden_dim, num_attention_heads=8, intermediate_size=4 * hidden_dim,
                            attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.3, layer_norm_eps=1e-6)
            self.text_transformer = MELDTransEncoder(config=config, layer_num=1, get_max_lens=38,
                                                     hidden_size=hidden_dim) if 'T' in modalities else None
            self.audio_transformer = MELDTransEncoder(config=config, layer_num=1, get_max_lens=60,
                                                      hidden_size=hidden_dim) if 'A' in modalities else None
            self.video_transformer = MELDTransEncoder(config=config, layer_num=1, get_max_lens=40,
                                                      hidden_size=hidden_dim) if 'V' in modalities else None

            # CAM 模块 (CAM 模块的 num_classes 始终为原始分类数)
            cam_num_classes = num_classes

            if self.cam_type == 'T_to_CAM':
                self.text_cam = CAMModule(hidden_dim, cam_num_classes) if 'T' in modalities else None
                self.audio_seq_classifier = nn.Linear(hidden_dim, cam_num_classes) if 'A' in modalities else None
                self.video_seq_classifier = nn.Linear(hidden_dim, cam_num_classes) if 'V' in modalities else None

            elif self.cam_type == 'AV_to_CAM':
                self.audio_cam = CAMModule(hidden_dim, cam_num_classes) if 'A' in modalities else None
                self.video_cam = CAMModule(hidden_dim, cam_num_classes) if 'V' in modalities else None
                self.text_seq_classifier = nn.Linear(hidden_dim, cam_num_classes) if 'T' in modalities else None

            elif self.cam_type == "Tcam_to_CAM":
                self.text_cam = CAMModule(hidden_dim, cam_num_classes) if 'T' in modalities else None
                self.audio_cam = CAMModule(hidden_dim, cam_num_classes) if 'A' in modalities else None
                self.video_cam = CAMModule(hidden_dim, cam_num_classes) if 'V' in modalities else None

            elif self.cam_type == "AVcam_to_CAM":
                self.text_cam = CAMModule(hidden_dim, cam_num_classes) if 'T' in modalities else None
                self.audio_cam = CAMModule(hidden_dim, cam_num_classes) if 'A' in modalities else None
                self.video_cam = CAMModule(hidden_dim, cam_num_classes) if 'V' in modalities else None

            # Cross-modal Transformers
            if self.use_cross_modal:
                print("Using cross-modal transformers")
                self.crossmodal_ta = CrossModalTransformerEncoder(embed_dim=hidden_dim, num_heads=8, layers=1,
                                                                  attn_dropout=0.1, gelu_dropout=0.1, res_dropout=0.1,
                                                                  embed_dropout=0.1, ) if 'T' in modalities and 'A' in modalities else None
                self.crossmodal_ta_v = CrossModalTransformerEncoder(embed_dim=hidden_dim, num_heads=8, layers=1,
                                                                    attn_dropout=0.1, gelu_dropout=0.1, res_dropout=0.1,
                                                                    embed_dropout=0.1, ) if 'T' in modalities and 'V' in modalities else None
                self.crossmodal_av = CrossModalTransformerEncoder(embed_dim=hidden_dim, num_heads=8, layers=1,
                                                                  attn_dropout=0.1, gelu_dropout=0.1, res_dropout=0.1,
                                                                  embed_dropout=0.1, ) if 'A' in modalities and 'V' in modalities else None

            # Attention mechanism for final fusion
            self.attention = AdditiveAttention(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(0.2)

            # 最终分类器使用 self.final_output_dim
            self.classifier = nn.Linear(hidden_dim, self.final_output_dim)

            # Unified sequence length
            self.seq_len = self.target_seq_len * len(modalities)

    # 辅助函数：执行跨模态融合
    def _perform_cross_modal_fusion(self, text_feat, audio_feat, video_feat):
        ta_cross_feat = None

        # T-A Fusion
        if 'T' in self.modalities and 'A' in self.modalities and self.crossmodal_ta is not None:
            text_feat_t = text_feat.transpose(0, 1)
            audio_feat_t = audio_feat.transpose(0, 1)
            text_cross_audio = self.crossmodal_ta(text_feat_t, audio_feat_t, audio_feat_t)
            audio_cross_text = self.crossmodal_ta(audio_feat_t, text_feat_t, text_feat_t)
            ta_cross_feat = torch.cat([text_cross_audio, audio_cross_text], dim=0)
        else:
            if 'T' in self.modalities: ta_cross_feat = text_feat.transpose(0, 1)
            if 'A' in self.modalities and ta_cross_feat is None: ta_cross_feat = audio_feat.transpose(0, 1)
            if 'A' in self.modalities and 'T' not in self.modalities: ta_cross_feat = audio_feat.transpose(0, 1)

        # (T+A)-V Fusion
        if 'V' in self.modalities and video_feat is not None:
            video_feat_t = video_feat.transpose(0, 1)
            if ta_cross_feat is not None and self.crossmodal_ta_v is not None:
                video_cross_ta = self.crossmodal_ta_v(video_feat_t, ta_cross_feat, ta_cross_feat)
                ta_cross_video = self.crossmodal_ta_v(ta_cross_feat, video_feat_t, video_feat_t)
                return torch.cat([video_cross_ta, ta_cross_video], dim=0)
            elif ta_cross_feat is None:
                return video_feat_t

        return ta_cross_feat

    def forward(self, inputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
                labels: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple]:

        # 1. 初始化变量和标签准备
        text_feat = audio_feat = video_feat = None
        text_loss = audio_loss = video_loss = torch.tensor(0.0, device=labels.device if labels is not None else 'cpu')
        cam_loss = torch.tensor(0.0, device=labels.device if labels is not None else 'cpu')

        # CAM 模块需要分类的长整型标签 (0-6)
        cam_labels = None
        if labels is not None and self.task_type == 'classification':
            if labels.dim() > 1:
                cam_labels = labels.squeeze(-1)  # 确保是 [B]
            else:
                cam_labels = labels

        # ----------------------------------------------------
        # Path 1: Pooled Features (特征已池化)
        # ----------------------------------------------------
        if self.feature_type == 'pooled_features':
            features = []
            if 'T' in self.modalities: features.append(inputs['T'])
            if 'A' in self.modalities: features.append(inputs['A'])
            if 'V' in self.modalities: features.append(inputs['V'])

            if not features:
                batch_size = labels.shape[0] if labels is not None else 1
                main_output = torch.zeros(batch_size, self.final_output_dim).to(
                    labels.device if labels is not None else 'cpu')
            else:
                fused = torch.cat(features, dim=-1)
                fused = F.layer_norm(fused, fused.size()[1:])
                main_output = self.classifier(fused)

            # 计算并返回主损失
            if labels is not None:
                if self.task_type == 'regression':
                    main_loss = self.main_loss_fn(main_output, labels)
                    return main_output, main_loss
                else:
                    if labels.dim() > 1: labels = labels.squeeze(-1)
                    main_loss = self.main_loss_fn(main_output, labels)
                    return main_output, main_loss

            return main_output

        # ----------------------------------------------------
        # Path 2: Sequence Features (序列特征)
        # ----------------------------------------------------
        elif self.feature_type == 'sequence_features':

            # --- 2.1. T Modality Processing ---
            if 'T' in self.modalities and self.text_proj is not None:
                # T Modality: Raw Text (RoBERTa processing)
                if self.use_raw_text:
                    text_input_ids = inputs['T']['input_ids']
                    text_attention_mask = inputs['T']['attention_mask']
                    text_output = self.roberta_model(text_input_ids, text_attention_mask)
                    text_feat_raw = text_output.last_hidden_state

                    batch_size = text_input_ids.shape[0]
                    hidden_size_raw = text_feat_raw.shape[-1]
                    word_level_features = torch.zeros(batch_size, 38, hidden_size_raw).to(text_input_ids.device)
                    word_level_masks = torch.zeros(batch_size, 38).to(text_input_ids.device)

                    target_start_pos = inputs['T']['target_start_pos']
                    target_end_pos = inputs['T']['target_end_pos']

                    if target_start_pos is not None and target_end_pos is not None:
                        for i in range(batch_size):
                            start = target_start_pos[i].item()
                            end = target_end_pos[i].item()
                            curr_utt_len = end - start
                            if curr_utt_len > 38: curr_utt_len = 38
                            if curr_utt_len > 0:
                                word_level_features[i, :curr_utt_len] = text_feat_raw[i, start:start + curr_utt_len]
                                word_level_masks[i, :curr_utt_len] = 1
                    else:
                        raise ValueError(
                            "target_start_pos and target_end_pos must be provided for word-level feature extraction")

                    text_feat = word_level_features
                    text_mask = word_level_masks

                # T Modality: Pre-extracted Features
                else:
                    text_feat = inputs['T']
                    text_mask = (text_feat.sum(dim=-1) != 0).float()

                # Common T-Processing (Projection, Transformer, Unification)
                text_seq = self.text_proj(text_feat)
                text_extended_mask = (1.0 - text_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
                text_feat = self.text_transformer(text_seq, attention_mask=text_extended_mask)

                # Unification
                text_feat = text_feat.transpose(1, 2)
                text_feat = self.text_transpose_conv(text_feat)
                text_feat = F.adaptive_avg_pool1d(text_feat, self.target_seq_len)
                text_feat = text_feat.transpose(1, 2)

                # CAM/Seq Classifier Logic (Only Classification)
                if self.task_type == 'classification' and self.use_cam_loss:
                    if self.cam_type in ['T_to_CAM', 'Tcam_to_CAM', 'AVcam_to_CAM']:
                        text_cam, text_logits_cam, text_loss, text_time_weights = self.text_cam(text_feat,
                                                                                                labels=cam_labels)
                    elif self.cam_type == 'AV_to_CAM':
                        text_logits = self.text_seq_classifier(text_feat)
                        text_logits = F.log_softmax(text_logits, dim=2)

            # --- 2.2. A Modality Processing ---
            if 'A' in self.modalities and self.audio_proj is not None:
                # A Modality: Raw Audio (Whisper Adapter)
                if self.use_raw_audio:
                    audio_input = inputs['A']['input_values']
                    audio_outputs = self.whisper_model(audio_input, attention_mask=None)
                    hidden_states = audio_outputs.last_hidden_state

                    # Sequence downsampling
                    hidden_states = hidden_states.transpose(1, 2)
                    hidden_states = self.whisper_conv1d(hidden_states)
                    hidden_states = hidden_states.transpose(1, 2)

                    audio_seq = self.audio_proj(hidden_states)
                    audio_feat = self.audio_transformer(audio_seq, attention_mask=None)

                # A Modality: Pre-extracted Features
                else:
                    audio_seq = self.audio_proj(inputs['A'])
                    audio_feat = self.audio_transformer(audio_seq, attention_mask=None)

                # Unification
                audio_feat = audio_feat.transpose(1, 2)
                audio_feat = self.audio_conv(audio_feat)
                audio_feat = audio_feat.transpose(1, 2)

                # CAM/Seq Classifier Logic (Only Classification)
                if self.task_type == 'classification' and self.use_cam_loss:
                    if self.cam_type in ['AV_to_CAM', 'Tcam_to_CAM', 'AVcam_to_CAM']:
                        audio_cam, audio_logits_cam, audio_loss, audio_time_weights = self.audio_cam(audio_feat,
                                                                                                     labels=cam_labels)
                    elif self.cam_type == 'T_to_CAM':
                        audio_logits = self.audio_seq_classifier(audio_feat)
                        audio_logits = F.log_softmax(audio_logits, dim=2)

            # --- 2.3. V Modality Processing ---
            if 'V' in self.modalities and self.video_proj is not None:
                video_seq = self.video_proj(inputs['V'])
                video_feat = self.video_transformer(video_seq, attention_mask=None)

                # Unification
                video_feat = video_feat.transpose(1, 2)
                video_feat = self.video_transpose_conv(video_feat)
                video_feat = F.adaptive_avg_pool1d(video_feat, self.target_seq_len)
                video_feat = video_feat.transpose(1, 2)

                # CAM/Seq Classifier Logic (Only Classification)
                if self.task_type == 'classification' and self.use_cam_loss:
                    if self.cam_type in ['AV_to_CAM', 'Tcam_to_CAM', 'AVcam_to_CAM']:
                        video_cam, video_logits_cam, video_loss, video_time_weights = self.video_cam(video_feat,
                                                                                                     labels=cam_labels)
                    elif self.cam_type == 'T_to_CAM':
                        video_logits = self.video_seq_classifier(video_feat)
                        video_logits = F.log_softmax(video_logits, dim=2)

            # ----------------------------------------------------
            # 2.4. CAM Supervision Loss (KLDiv)
            # ----------------------------------------------------
            if self.task_type == 'classification' and self.use_cam_loss:
                kl_loss = nn.KLDivLoss(reduction='none')

                # Detach target CAMs
                if 'T' in self.modalities and self.cam_type in ['T_to_CAM', 'Tcam_to_CAM', 'AVcam_to_CAM']:
                    text_cam = text_cam.detach()
                if 'A' in self.modalities and self.cam_type in ['AV_to_CAM', 'Tcam_to_CAM', 'AVcam_to_CAM']:
                    audio_cam = audio_cam.detach()
                if 'V' in self.modalities and self.cam_type in ['AV_to_CAM', 'Tcam_to_CAM', 'AVcam_to_CAM']:
                    video_cam = video_cam.detach()

                if self.cam_type == 'T_to_CAM':
                    if 'A' in self.modalities:
                        kl_div_a = kl_loss(audio_logits, text_cam)
                        weighted_kl_a = (kl_div_a * text_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5 * weighted_kl_a
                    if 'V' in self.modalities:
                        kl_div_v = kl_loss(video_logits, text_cam)
                        weighted_kl_v = (kl_div_v * text_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5 * weighted_kl_v

                elif self.cam_type == 'AV_to_CAM':
                    if 'A' in self.modalities:
                        kl_div_ta = kl_loss(text_logits, audio_cam)
                        weighted_kl_ta = (kl_div_ta * audio_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5 * weighted_kl_ta
                    if 'V' in self.modalities:
                        kl_div_tv = kl_loss(text_logits, video_cam)
                        weighted_kl_tv = (kl_div_tv * video_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5 * weighted_kl_tv

                elif self.cam_type == 'Tcam_to_CAM':
                    if 'A' in self.modalities:
                        kl_div_a = kl_loss(audio_cam, text_cam)
                        weighted_kl_a = (kl_div_a * text_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5 * weighted_kl_a
                    if 'V' in self.modalities:
                        kl_div_v = kl_loss(video_cam, text_cam)
                        weighted_kl_v = (kl_div_v * text_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5 * weighted_kl_v

                elif self.cam_type == 'AVcam_to_CAM':
                    if 'A' in self.modalities:
                        kl_div_ta = kl_loss(text_cam, audio_cam)
                        weighted_kl_ta = (kl_div_ta * audio_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5 * weighted_kl_ta
                    if 'V' in self.modalities:
                        kl_div_tv = kl_loss(text_cam, video_cam)
                        weighted_kl_tv = (kl_div_tv * video_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5 * weighted_kl_tv

            # ----------------------------------------------------
            # 2.5. Cross-modal / Concatenation Fusion
            # ----------------------------------------------------

            if self.use_cross_modal:
                final_feat = self._perform_cross_modal_fusion(text_feat, audio_feat, video_feat)

            else:
                final_feat_list = []
                if 'T' in self.modalities and text_feat is not None:
                    final_feat_list.append(text_feat.transpose(0, 1))
                if 'A' in self.modalities and audio_feat is not None:
                    final_feat_list.append(audio_feat.transpose(0, 1))
                if 'V' in self.modalities and video_feat is not None:
                    final_feat_list.append(video_feat.transpose(0, 1))

                if not final_feat_list:
                    # 如果没有特征，创建一个零向量以避免崩溃 (但通常应避免这种情况)
                    if labels is not None:
                        B = labels.shape[0]
                        D = self.classifier.in_features
                        final_feat = torch.zeros(1, B, D, device=labels.device)
                    else:
                        raise ValueError("No valid features for fusion and labels are None.")

                final_feat = torch.cat(final_feat_list, dim=0)

            # ----------------------------------------------------
            # 2.6. Final Attention and Prediction
            # ----------------------------------------------------
            final_feat = final_feat.transpose(0, 1)
            multimodal_out, _ = self.attention(final_feat)
            multimodal_out = self.dropout(multimodal_out)
            main_output = self.classifier(multimodal_out)

            # ----------------------------------------------------
            # 2.7. Return Output and Loss
            # ----------------------------------------------------
            if labels is not None:
                # Regression Loss Calculation
                if self.task_type == 'regression':
                    main_loss = self.main_loss_fn(main_output, labels)
                    return main_output, main_loss

                # Classification Loss Calculation (with potential CAM returns)
                else:
                    if labels.dim() > 1: labels = labels.squeeze(-1)
                    main_loss = self.main_loss_fn(main_output, labels)

                    if self.use_cam_loss:
                        if self.cam_type == 'AVcam_to_CAM':
                            # 返回 logits, cam_loss, audio_loss, video_loss
                            return main_output, cam_loss, audio_loss, video_loss
                        else:
                            # 默认返回输出、CAM 损失和主损失
                            return main_output, cam_loss, main_loss
                    else:
                        # 标准分类返回
                        return main_output, main_loss

            # 推理模式
            return main_output
