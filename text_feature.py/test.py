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
def collect_predictions(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    all_sample_names = []

    for batch in tqdm(dataloader, desc="Collecting predictions"):
        # 准备输入
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 提取真实标签 (并转换为 [B] 形状的 float Tensor)
        labels = batch["label"].to(device).float().squeeze(-1)

        sample_name_batch = batch.get("sample_name")

        # 模型前向传播，只获取 logits (预期形状通常是 [B, 1])
        # 假设 EmotionClassifier 的 forward 方法返回 logits Tensor
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # 提取 logits (回归预测)
        # .detach() 停止梯度跟踪; .squeeze(-1) 将形状从 [B, 1] -> [B]
        # 如果 outputs 是 tuple/list, 取第一个元素作为 logits:
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0].detach().squeeze(-1)
        else:
            logits = outputs.detach().squeeze(-1)

        # 累加数据
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        # 处理样本名称
        if isinstance(sample_name_batch, torch.Tensor):
            sample_name_batch = sample_name_batch.tolist()
        all_sample_names.extend(sample_name_batch)

    # 拼接所有批次结果
    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.array([])
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    names_np = np.array(all_sample_names)

    # 确保 logits_np 和 labels_np 形状为 [N, 1] 以便存储 (N: 样本总数)
    if logits_np.ndim == 1:
        logits_np = logits_np[:, np.newaxis]
    if labels_np.ndim == 1:
        labels_np = labels_np[:, np.newaxis]

    return {
        'logits': logits_np,
        'labels': labels_np,
        'sample_names': names_np,
    }


def process_split_and_save_predictions(model, tokenizer, data_path, split_mode, device, batch_size, max_seq_length,
                                       output_dir):
    print(f"\n===== Processing {split_mode} set =====")

    # 使用您的 MOSIDataset 类
    dataset = MOSIDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        split_mode=split_mode,
        max_seq_length=max_seq_length
    )
    # shuffle=False 确保预测结果的顺序与样本名称对应
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 收集预测结果
    results = collect_predictions(model, dataloader, device)

    # 定义预测结果保存路径
    pred_save_path = os.path.join(output_dir, f"{split_mode}_predictions.npz")

    # 保存预测结果 (logits, labels, sample_names)
    np.savez(
        pred_save_path,
        logits=results['logits'],
        labels=results['labels'],
        sample_names=results['sample_names']
    )
    print(f"Saved {split_mode} predictions to: {pred_save_path}")

    # 打印形状
    print(f"Logits shape (Predictions): {results['logits'].shape}")
    print(f"Label shape (Ground Truth): {results['labels'].shape}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_labels = 1
    batch_size = 32
    max_seq_length = 160

    # 请根据您的实际环境修改以下路径
    data_path = "/data/home/chenqian/CMU-MOSEI/label_utf8_clean.csv"
    local_roberta_path = "/data/home/chenqian/Roberta-large/Roberta-large"
    output_dir = "/data/home/chenqian/regression_models/text_model"
    os.makedirs(output_dir, exist_ok=True)
    stage1_model_path = (
        "/data/home/chenqian/regression_models/text_model/seed_45_Acc7_0.5422.pth"
    )

    # 初始化 Tokenizer 和 RoBERTa 主体
    tokenizer = RobertaTokenizer.from_pretrained(local_roberta_path)
    roberta_model = RobertaModel.from_pretrained(local_roberta_path)

    # 使用您的 EmotionClassifier 类
    model = EmotionClassifier(roberta_model, num_labels=num_labels, use_lora=False, use_adapters=False)
    model.to(device)

    # 加载微调权重
    if os.path.exists(stage1_model_path):
        state_dict = torch.load(stage1_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded full model weights from {stage1_model_path}")
    else:
        print(f"Warning: trained model not found at {stage1_model_path}, using base weights.")

    # 对训练集和测试集进行预测和保存
    for split_mode in ["train", "test"]:
        process_split_and_save_predictions(
            model=model,
            tokenizer=tokenizer,
            data_path=data_path,
            split_mode=split_mode,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main() # 运行 main 函数
