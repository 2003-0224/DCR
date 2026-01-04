import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Union, Optional
import os
from tqdm import tqdm
from feature_mosi_dataset import MOSIDataset
from T_raw_MOSIDataset import T_raw_MOSIDataset
from baseline_model import MultimodalFusionModel
from utils import EMA, train, evaluate, set_seed, evaluate_regression
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
import gc
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def append_results_to_csv(filepath: str, seed: int, mae: float, corr: float, acc7: float, acc5: float):
    """将单轮种子的最佳结果（MAE, Corr, Acc7, Acc5）追加写入 CSV 文件。"""
    # 检查文件是否存在以确定是否写入头部
    file_exists = os.path.exists(filepath)
    with open(filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 如果文件不存在，写入 CSV 头部
        if not file_exists:
            writer.writerow(['Seed', 'Best Test MAE', 'Best Test Corr', 'Best Test Acc7', 'Best Test Acc5'])
        # 写入本轮种子结果
        writer.writerow([
            seed,
            f"{mae:.4f}",
            f"{corr:.4f}",
            f"{acc7:.4f}",
            f"{acc5:.4f}"
        ])
    print(f"Results for Seed {seed} appended to {filepath}")


def prepare_inputs(batch: Dict[str, Any], modalities: List[str], device: torch.device) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    根据 batch 内容和模态列表准备模型输入。
    处理嵌套字典（如 raw text/audio inputs）和普通 Tensor。
    """
    modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
    inputs = {}
    for mod in modalities:
        key = modality_map[mod]
        if key not in batch:
            continue
        data = batch[key]
        if isinstance(data, dict):
            inputs[mod] = {k: v.to(device) for k, v in data.items() if isinstance(v, torch.Tensor)}
        elif isinstance(data, torch.Tensor):
            inputs[mod] = data.to(device)
        else:
            inputs[mod] = data

    return inputs


def collect_predictions(model: nn.Module, data_loader: DataLoader, device: torch.device, modalities: List[str],
                        use_cam_loss: bool, cam_type: str) -> Dict[str, np.ndarray]:
    """
    收集数据加载器中的所有样本的预测结果、logits、真实标签和样本名称。
    """
    model.eval()
    all_logits = []
    all_labels = []
    all_names = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting predictions"):
            inputs = prepare_inputs(batch, modalities, device)
            labels = batch['label']
            # 标签准备：回归任务需要 FloatTensor [B, 1]
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device).float()  # 转换为 FloatTensor
            else:
                labels = torch.tensor(labels, device=device).float()
            labels = labels.view(-1, 1)  # 确保是 [B, 1] 形状的 FloatTensor
            # 模型推理
            outputs = model(inputs, labels=None)
            if use_cam_loss and isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits.detach()
            # 收集样本名称 (处理多种可能的返回格式)
            names = batch.get('sample_name')
            if names is not None:
                if isinstance(names, (torch.Tensor, np.ndarray)):
                    names = names.tolist()
                elif isinstance(names, (str, bytes)):
                    names = [str(names)]
                all_names.extend([str(n) for n in names])

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits_np = torch.cat(all_logits).numpy() if all_logits else np.array([])
    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    names_np = np.array(all_names)
    return {
        'sample_names': names_np,
        'predictions': logits_np,
        'labels': labels_np,
    }


def main():
    global_best_model_path = ("/data/home/chenqian/regression_models/multi_model/multimodal_best_mse_0.7017_corr_0"
                              ".7216_seed_43.pth")

    num_labels = 7
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task_type = 'regression'
    use_cam_loss = False
    modalities = ['T', "A", "V"]
    hidden_dim = 128
    feature_tpye = "sequence_features"
    use_cross_modal = True
    use_raw_text = True
    cam_type = "AVcam_to_CAM"
    use_raw_audio = False
    whisper_use_adapters = True

    if use_raw_audio:
        batch_size = 16
    if feature_tpye == "pooled_features":
        use_cross_modal = True

    data_paths = {
        'train': {
            'text': '/data/home/chenqian/regression_models/text_model/train_text_features.npz',
            'text_raw_path': '/data/home/chenqian/CMU-MOSEI/label_utf8_clean.csv',
            'audio': '/data/home/chenqian/regression_models/audio_model/train_audio_features.npz',
            'video': '/data/home/chenqian/regression_models/video_model/train_video_features.npz',
            'audio_csv_path': '/data/home/chenqian/CMU-MOSEI/train_sentiment.csv',
            'audio_data_path': '/data/home/chenqian/CMU-MOSEI/train_sentiment.csv'
        },
        'test': {
            'text': '/data/home/chenqian/regression_models/text_model/test_text_features.npz',
            'text_raw_path': '/data/home/chenqian/CMU-MOSEI/label_utf8_clean.csv',
            'audio': '/data/home/chenqian/regression_models/audio_model/test_audio_features.npz',
            'video': '/data/home/chenqian/regression_models/video_model/test_video_features.npz',
            'audio_csv_path': '/data/home/chenqian/CMU-MOSEI/test_sentiment.csv',
            'audio_data_path': '/data/home/chenqian/CMU-MOSEI/test_sentiment.csv'
        }
    }

    def build_dataset(split: str):
        cfg = data_paths[split]
        if use_raw_text:
            return T_raw_MOSIDataset(
                cfg['text'], cfg['audio'], cfg['video'], modalities, split=split, feature_type=feature_tpye,
                text_path=cfg['text_raw_path']
            )
        else:
            return MOSIDataset(
                cfg['text'], cfg['audio'], cfg['video'], modalities, split=split, feature_type=feature_tpye
            )

    def create_model():
        return MultimodalFusionModel(
            text_dim=1024, audio_dim=768, video_dim=768, hidden_dim=hidden_dim, num_classes=num_labels,
            modalities=modalities, feature_type=feature_tpye, use_cross_modal=use_cross_modal,
            use_raw_text=use_raw_text, use_cam_loss=use_cam_loss, use_raw_audio=use_raw_audio,
            whisper_use_adapters=whisper_use_adapters, cam_type=cam_type, task_type=task_type
        ).to(device)

    print(f"\n=== Starting Prediction Mode ===\n")
    original_train_dataset = build_dataset('train')
    original_test_dataset = build_dataset('test')

    # 创建 DataLoader
    train_loader_best = DataLoader(original_train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                   pin_memory=True)
    test_loader_best = DataLoader(original_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                  pin_memory=True)

    # 2. 实例化模型并加载权重
    best_model = create_model()
    try:
        print(f"Loading model state from: {global_best_model_path}")
        best_model.load_state_dict(torch.load(global_best_model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {global_best_model_path}")
        return
    except RuntimeError as e:
        print(f"Error loading state dict: {e}. Check if the model architecture matches the saved weights.")
        return

    # 3. 收集预测结果
    print("\n--- Collecting Train Set Predictions ---")
    train_preds = collect_predictions(best_model, train_loader_best, device, modalities, use_cam_loss, cam_type)

    print("\n--- Collecting Test Set Predictions ---")
    test_preds = collect_predictions(best_model, test_loader_best, device, modalities, use_cam_loss, cam_type)

    # 4. 保存预测结果

    # 从模型路径生成前缀，确保预测文件和权重文件关联
    train_pred_path = f"/data/home/chenqian/regression_models/multi_model/train_predictions_RECALC.npz"
    test_pred_path = f"/data/home/chenqian/regression_models/multi_model/test_predictions_RECALC.npz"

    # 回归任务，保存 predictions 和 labels
    np.savez(train_pred_path, **train_preds)
    np.savez(test_pred_path, **test_preds)

    print("\n=== Prediction Complete ===")
    print(f"Saved train predictions to: {train_pred_path}")
    print(f"Saved test predictions to: {test_pred_path}")

    # 5. 内存清理
    del best_model
    del train_loader_best
    del test_loader_best
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
