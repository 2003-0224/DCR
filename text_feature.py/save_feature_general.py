import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import numpy as np
import os
from Roberta_model import EmotionClassifier 
from data_loader import MOSIDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    pooled_features = []
    sequence_features = []
    sequence_masks = []
    labels = []
    indices = []
    sample_names = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting General features")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_batch = batch["label"].to(device)
        sample_name_batch = batch.get("sample_name")
        outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        pooled = last_hidden[:, 0, :]
        # 累加数据
        pooled_features.append(pooled.cpu().numpy())
        sequence_features.append(last_hidden.cpu().numpy())
        sequence_masks.append(attention_mask.cpu().numpy())
        labels.append(label_batch.cpu().numpy())
        if isinstance(sample_name_batch, torch.Tensor):
            sample_name_batch = sample_name_batch.tolist()
        sample_names.extend(sample_name_batch)
        indices.extend(
            np.arange(batch_idx * input_ids.size(0), (batch_idx + 1) * input_ids.size(0)))
    pooled_features = np.concatenate(pooled_features, axis=0)
    sequence_features = np.concatenate(sequence_features, axis=0)
    sequence_masks = np.concatenate(sequence_masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    return pooled_features, sequence_features, sequence_masks, labels, indices, np.array(sample_names)


def process_split(model, tokenizer, data_path, split_mode, device, batch_size, max_seq_length, output_dir):
    print(f"\n===== Processing {split_mode} set (General Base Model) =====")
    dataset = MOSIDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        split_mode=split_mode,
        max_seq_length=max_seq_length
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pooled_features, sequence_features, sequence_masks, labels, indices, sample_names = extract_features(model, dataloader, device)
    if split_mode == "train":
        save_path = os.path.join(output_dir, "train_text_features_general.npz")
    elif split_mode == "test":
        save_path = os.path.join(output_dir, "test_text_features_general.npz")
    np.savez(
        save_path,
        pooled_features=pooled_features,
        sequence_features=sequence_features,
        sequence_masks=sequence_masks,
        labels=labels,
        indices=indices,
        sample_names=sample_names
    )
    print(f"Saved {split_mode} features to: {save_path}")
    print(f"Pooled feature shape: {pooled_features.shape}")
    print(f"Sequence feature shape: {sequence_features.shape}")
    print(f"Sequence mask shape: {sequence_masks.shape}")
    print(f"Label shape: {labels.shape}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_labels = 1
    batch_size = 32
    max_seq_length = 160
    # === 路径配置 ===
    data_path = "/data/home/chenqian/CMU-MOSEI/label_utf8_clean.csv"
    local_roberta_path = "/data/home/chenqian/Roberta-large/Roberta-large"
    output_dir = "/data/home/chenqian/regression_models/text_model"
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = RobertaTokenizer.from_pretrained(local_roberta_path)
    roberta_model_general = RobertaModel.from_pretrained(local_roberta_path)
    model_general = EmotionClassifier(roberta_model_general, num_labels=num_labels, use_lora=False, use_adapters=False)
    model_general.to(device)
    print("--- Starting General Feature Extraction (Base RoBERTa) ---")
    for split_mode in ["train", "test"]:
        process_split(
            model=model_general,
            tokenizer=tokenizer,
            data_path=data_path,
            split_mode=split_mode,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            output_dir=output_dir
        )
    del model_general, roberta_model_general
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()