import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from data_loader import data_loader_mosi_audio
from Data2vec_modal import EmotionClassifier
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 特征提取函数
def extract_features(model, dataloader, device, pool_chunk=10):
    model.eval()
    features = []
    labels = []
    sample_names = []
    sequence_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            input_ids = batch['input_values'].to(device)
            labels_batch = batch['label'].to(device)
            sample_names_batch = batch['sample_name']
            base_encoder = getattr(model, 'model', model)
            # Whisper (及类似模型) 通常不需要 attention_mask
            encoder_outputs = base_encoder(input_ids, attention_mask=None)
            seq_features = encoder_outputs.last_hidden_state  # [Batch, SeqLen, HiddenSize]
            pooled_features = seq_features.mean(dim=1)
            # 序列分块池化 (您的原始逻辑)
            if seq_features.size(1) % pool_chunk != 0:
                pad_len = pool_chunk - (seq_features.size(1) % pool_chunk)
                pad = seq_features[:, -1:, :].expand(-1, pad_len, -1)
                seq_features = torch.cat([seq_features, pad], dim=1)
            new_len = seq_features.size(1) // pool_chunk
            seq_features_chunked = seq_features.view(seq_features.size(0), new_len, pool_chunk, -1).mean(dim=2)
            features.append(pooled_features.cpu().numpy())
            labels.append(labels_batch.cpu().numpy())
            sample_names.extend(sample_names_batch)
            sequence_features.append(seq_features_chunked.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    sequence_features = np.concatenate(sequence_features, axis=0)
    return features, labels, sample_names, sequence_features


# 主函数
def main():
    # 配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_type = "Whisper"
    batch_size = 16  # 提取特征时可以使用稍大的 batch_size
    max_length = 12  # 12 秒
    num_labels = 1
    use_transformer = False
    use_adapters = True  # 确保与您训练的模型一致
    use_lora = False
    weight_attn = False
    max_seq_length = max_length * 16000
    train_csv_file = "/data/home/chenqian/CMU-MOSEI/train_acc7.csv"
    test_csv_file = "/data/home/chenqian/CMU-MOSEI/test_acc7.csv"
    valid_csv_file = "/data/home/chenqian/CMU-MOSEI/valid_acc7.csv"
    local_whisper_path = "/data/home/chenqian/whisper_large_v3/"  # 您的本地 Whisper 路径
    output_dir = '/data/home/chenqian/regression_models/audio_model'  # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    train_loader, valid_loader, test_loader = data_loader_mosi_audio(
        train_csv_path=train_csv_file,
        valid_csv_path=valid_csv_file,
        test_csv_path=test_csv_file,
        audio_directory=None,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        model_type=model_type,
        num_workers=0,
        local_model_path=local_whisper_path
    )
    # 初始化模型
    model = EmotionClassifier(
        num_classes=num_labels,
        use_lora=use_lora,
        use_transformer=use_transformer,
        weight_attn=weight_attn,
        model_type=model_type,
        max_length=max_length,
        use_adapters=use_adapters
    ).to(device)
    MODEL_CHECKPOINT_PATH = "/data/home/chenqian/regression_models/audio_model/best_Whisper_valid_acc7_0.5389_seed_41_adapter.pth"
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(f"模型检查点未找到: {MODEL_CHECKPOINT_PATH}。请先运行 train.py 并更新此路径。")
    print(f"Loading trained weights from: {MODEL_CHECKPOINT_PATH}")
    if use_adapters:
        model.load_adapters(MODEL_CHECKPOINT_PATH)
    else:
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()
    with torch.no_grad():
        # 提取并保存特征
        for split, loader in [('train', train_loader), ('test', test_loader)]:
            print(f"Processing {split} set...")
            features, labels, sample_names, sequence_features = extract_features(model, loader, device)
            # 保存特征
            output_path = os.path.join(output_dir, f'{split}_audio_features.npz')
            np.savez(
                output_path,
                pooled_features=features,
                labels=labels,
                sample_names=sample_names,
                sequence_features=sequence_features
            )
            print(f"Saved features to {output_path}")
            print(f"Pooled feature shape: {features.shape}, Sequence feature shape: {sequence_features.shape}, "
                  f"Label shape: {labels.shape}, Sample names: {len(sample_names)}")


if __name__ == '__main__':
    main()
