import torch
import torch.nn as nn
from transformers import RobertaModel


class MultimodalFusionModel(nn.Module):
    def __init__(
        self,
        text_dim=1024,
        audio_dim=768,
        video_dim=768,
        hidden_dim=256,
        modalities=('T', 'A', 'V'),
        feature_type="sequence_features",
        use_raw_text=True,
        task_type='regression',
        roberta_path="/data/home/chenqian/Roberta-large/Roberta-large"
    ):
        super().__init__()

        self.modalities = modalities
        self.feature_type = feature_type
        self.task_type = task_type
        self.use_cam_loss = False        # 强制关闭
        if 'T' in modalities and use_raw_text:
            self.text_encoder = RobertaModel.from_pretrained(roberta_path)
            text_dim = self.text_encoder.config.hidden_size
        else:
            self.text_encoder = None
        self.text_proj = nn.Linear(text_dim, hidden_dim) if 'T' in modalities else None
        self.audio_proj = nn.Linear(audio_dim, hidden_dim) if 'A' in modalities else None
        self.video_proj = nn.Linear(video_dim, hidden_dim) if 'V' in modalities else None
        fusion_dim = hidden_dim * len(modalities)
        self.classifier = nn.Linear(fusion_dim, 1)
        self.main_loss_fn = nn.MSELoss()

    def forward(self, inputs, labels=None):
        """
        inputs:
            T: dict(input_ids, attention_mask) or Tensor[B, L, D]
            A: Tensor[B, L, D] or Tensor[B, D]
            V: Tensor[B, L, D] or Tensor[B, D]
        """
        pooled_feats = []
        if 'T' in self.modalities:
            if isinstance(inputs['T'], dict):
                out = self.text_encoder(
                    input_ids=inputs['T']['input_ids'],
                    attention_mask=inputs['T']['attention_mask']
                ).last_hidden_state            # (B, L, D)
                text_feat = out
            else:
                text_feat = inputs['T']

            if text_feat.dim() == 3:
                text_feat = text_feat.mean(dim=1)

            pooled_feats.append(self.text_proj(text_feat))
        if 'A' in self.modalities:
            audio_feat = inputs['A']
            if audio_feat.dim() == 3:
                audio_feat = audio_feat.mean(dim=1)
            pooled_feats.append(self.audio_proj(audio_feat))
        if 'V' in self.modalities:
            video_feat = inputs['V']
            if video_feat.dim() == 3:
                video_feat = video_feat.mean(dim=1)
            pooled_feats.append(self.video_proj(video_feat))

        fused_feat = torch.cat(pooled_feats, dim=-1)   # (B, hidden_dim * M)
        output = self.classifier(fused_feat)           # (B, 1)
        if labels is not None:
            labels = labels.view(-1, 1)
            loss = self.main_loss_fn(output, labels)
            return output, loss

        return output
