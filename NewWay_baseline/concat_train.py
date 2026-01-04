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

# 1. 启用双卡训练
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def append_results_to_csv(filepath: str, seed: int, mae: float, mse: float, corr: float, acc7: float, acc5: float):
    """将单轮种子的最佳结果追加写入 CSV 文件。"""
    file_exists = os.path.exists(filepath)
    with open(filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ['Seed', 'Best Test MAE', 'Best Test MSE', 'Best Test Corr', 'Best Test Acc7', 'Best Test Acc5'])
        writer.writerow([
            seed,
            f"{mae:.4f}",
            f"{mse:.4f}",
            f"{corr:.4f}",
            f"{acc7:.4f}",
            f"{acc5:.4f}"
        ])
    print(f"Results for Seed {seed} appended to {filepath}")


def prepare_inputs(batch: Dict[str, Any], modalities: List[str], device: torch.device) -> Dict[
    str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
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
    model.eval()
    all_logits = []
    all_labels = []
    all_names = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting predictions"):
            inputs = prepare_inputs(batch, modalities, device)
            labels = batch['label']
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device).float()
            else:
                labels = torch.tensor(labels, device=device).float()
            labels = labels.view(-1, 1)

            outputs = model(inputs, labels=None)
            if use_cam_loss and isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits.detach()

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
    seeds = [41, 42, 43]
    num_labels = 7
    num_epochs = 32
    # 双卡训练，建议增大 batch_size 以提升效率
    batch_size = 64
    lr = 1e-5
    eta_min = 2e-7
    patience = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    output_dir = "/data/home/chenqian/regression_models/concat_multi_model"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, "regression_metrics_by_seed.csv")

    if os.path.exists(csv_filename):
        os.remove(csv_filename)
        print(f"Removed existing results file: {csv_filename}")

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
            return T_raw_MOSIDataset(cfg['text'], cfg['audio'], cfg['video'], modalities, split=split,
                                     feature_type=feature_tpye, text_path=cfg['text_raw_path'])
        return MOSIDataset(cfg['text'], cfg['audio'], cfg['video'], modalities, split=split, feature_type=feature_tpye)

    def create_model():
        model = MultimodalFusionModel(
            text_dim=1024, audio_dim=768, video_dim=768, hidden_dim=hidden_dim, num_classes=num_labels,
            modalities=modalities, feature_type=feature_tpye, use_cross_modal=use_cross_modal,
            use_raw_text=use_raw_text, use_cam_loss=use_cam_loss, use_raw_audio=use_raw_audio,
            whisper_use_adapters=whisper_use_adapters, cam_type=cam_type, task_type=task_type
        ).to(device)
        # 启用 DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training.")
            model = nn.DataParallel(model)
        return model

    # 核心指标追踪：回归任务以 MAE 越低越好为准
    best_mses = []
    best_maes = []
    global_best_mae = float('inf')
    global_best_mse = float('inf')
    global_best_seed = None
    global_best_model_state = None
    global_best_corr = -1.0

    for seed in seeds:
        print(f"\n=== Training with Seed {seed} (Target: MAE) ===\n")
        set_seed(seed)
        train_dataset = build_dataset('train')
        test_dataset = build_dataset('test')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=eta_min)
        ema = EMA(model, decay=0.999)

        best_mae = float('inf')  # 当前 seed 的最佳 MAE
        best_mse_at_best_mae = float('inf')
        best_corr = 0.0
        best_acc7 = 0.0
        best_acc5 = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            # 训练阶段
            train_loss = train(model, train_loader, optimizer, scheduler, ema, device, use_cam_loss=use_cam_loss,
                               cam_type=cam_type, use_raw_audio=use_raw_audio, task_type=task_type)

            # 测试阶段评估
            test_loss, test_mae, test_corr, test_acc7, test_acc5 = evaluate_regression(
                model, test_loader, ema, device, use_cam_loss, cam_type=cam_type,
                use_raw_audio=use_raw_audio, prepare_inputs_fn=None
            )

            test_mse = test_loss
            print(
                f"Epoch {epoch + 1}/{num_epochs} | Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}, Test Corr: {test_corr:.4f}")

            # --- 全局/当前种子保存逻辑：以 MAE 为准 ---
            if test_mae < global_best_mae:
                global_best_mae = test_mae
                global_best_mse = test_mse
                global_best_seed = seed
                global_best_corr = test_corr

                # 应用 EMA 并提取状态字典
                ema.apply_shadow()
                # 如果使用了 DataParallel，保存 .module.state_dict()
                if isinstance(model, nn.DataParallel):
                    global_best_model_state = model.module.state_dict()
                else:
                    global_best_model_state = model.state_dict()
                ema.restore()
                print(f"--> New Global Best MAE: {global_best_mae:.4f}")

            if test_mae < best_mae:
                best_mae = test_mae
                best_mse_at_best_mae = test_mse
                best_corr = test_corr
                best_acc7 = test_acc7
                best_acc5 = test_acc5
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            scheduler.step()

        best_mses.append(best_mse_at_best_mae)
        best_maes.append(best_mae)
        append_results_to_csv(csv_filename, seed, best_mae, best_mse_at_best_mae, best_corr, best_acc7, best_acc5)

        # 清理内存
        del model, optimizer, scheduler, ema, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    # --- 保存全局最佳模型 ---
    if global_best_model_state is not None:
        save_path = os.path.join(output_dir, f"best_mae_{global_best_mae:.4f}_seed_{global_best_seed}.pth")
        torch.save(global_best_model_state, save_path)
        print(f"\nSaved Global Best Model: {save_path}")

        # 重新加载最佳种子进行预测保存
        set_seed(global_best_seed)
        best_model = create_model()
        # 加载时注意 DataParallel 的兼容性
        if isinstance(best_model, nn.DataParallel):
            best_model.module.load_state_dict(global_best_model_state)
        else:
            best_model.load_state_dict(global_best_model_state)

        train_loader_best = DataLoader(build_dataset('train'), batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader_best = DataLoader(build_dataset('test'), batch_size=batch_size, shuffle=False, num_workers=4)

        train_preds = collect_predictions(best_model, train_loader_best, device, modalities, use_cam_loss, cam_type)
        test_preds = collect_predictions(best_model, test_loader_best, device, modalities, use_cam_loss, cam_type)

        np.savez(os.path.join(output_dir, "best_train_preds.npz"), **train_preds)
        np.savez(os.path.join(output_dir, "best_test_preds.npz"), **test_preds)

    print(f"\nFinal MAE Average: {np.mean(best_maes):.4f} ± {np.std(best_maes):.4f}")


if __name__ == "__main__":
    main()