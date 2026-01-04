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
    seeds = [41, 42]
    num_labels = 7  # 在回归任务中不使用，但保留以避免 MultimodalFusionModel 初始化报错
    num_epochs = 64
    batch_size = 32
    lr = 1e-4
    eta_min = 2e-7
    patience = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task_type = 'regression'
    use_cam_loss = False  # 回归任务中禁用 CAM 损失
    modalities = ['T', "A", "V"]
    hidden_dim = 128
    feature_tpye = "sequence_features"
    use_cross_modal = True
    use_raw_text = True
    cam_type = "AVcam_to_CAM"  # 禁用后不重要，但保留
    use_raw_audio = False
    whisper_use_adapters = True
    output_dir = "/data/home/chenqian/regression_models/multi_model"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, "regression_metrics_by_seed.csv")
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
        print(f"Removed existing results file: {csv_filename}")
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
                cfg['text'],
                cfg['audio'],
                cfg['video'],
                modalities,
                split=split,
                feature_type=feature_tpye,
                text_path=cfg['text_raw_path']
            )
        else:
            return MOSIDataset(
                cfg['text'],
                cfg['audio'],
                cfg['video'],
                modalities,
                split=split,
                feature_type=feature_tpye
            )

    def create_model():
        # 传入 task_type='regression'
        return MultimodalFusionModel(
            text_dim=1024,
            audio_dim=768,
            video_dim=768,
            hidden_dim=hidden_dim,
            num_classes=num_labels,
            modalities=modalities,
            feature_type=feature_tpye,
            use_cross_modal=use_cross_modal,
            use_raw_text=use_raw_text,
            use_cam_loss=use_cam_loss,  # False
            use_raw_audio=use_raw_audio,
            whisper_use_adapters=whisper_use_adapters,
            cam_type=cam_type,
            task_type=task_type  # 'regression'
        ).to(device)
    # 回归任务指标追踪
    best_mses = []  # MSE 越低越好
    best_maes = []  # MAE 越低越好
    seed_best_metrics: Dict[int, Dict[str, float]] = {}
    global_best_mse = float('inf')
    global_best_mae = float('inf')
    global_best_seed = None
    global_best_model_state = None
    global_best_corr = -1.0
    for seed in seeds:
        print(f"\n=== Training with Seed {seed} (Regression) ===\n")
        set_seed(seed)
        train_dataset = build_dataset('test')
        test_dataset = build_dataset('train')
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=eta_min)
        ema = EMA(model, decay=0.999)
        best_mse = float('inf')
        best_mae = float('inf')
        best_corr = 0.0
        best_acc7 = 0.0
        best_acc5 = 0.0
        patience_counter = 0
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, scheduler, ema, device, use_cam_loss=use_cam_loss,
                               cam_type=cam_type, use_raw_audio=use_raw_audio, task_type=task_type)
            # 评估训练集
            train_loss_eval, train_mae, train_corr, train_acc7, train_acc5 = evaluate_regression(
                model, train_loader, ema, device, use_cam_loss, cam_type=cam_type,
                use_raw_audio=use_raw_audio, prepare_inputs_fn=None
            )
            test_loss, test_mae, test_corr, test_acc7, test_acc5 = evaluate_regression(
                model, test_loader, ema, device, use_cam_loss, cam_type=cam_type,  # use_cam_loss=False
                use_raw_audio=use_raw_audio, prepare_inputs_fn=None  # 传入 None 使用默认 prepare_inputs
            )
            test_mse = test_loss
            train_mse = train_loss_eval
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"Train Loss: {train_loss_eval:.4f}, Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}, Train "
                f"Corr: {train_corr:.4f}, Train ACC5:{train_acc5:.4f}, Train ACC7: {train_acc7:.4f}")
            print(
                f"Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}, Test Corr: {test_corr:.4f}, Test ACC5: {test_acc5:.4f}, Test ACC7: {test_acc7:.4f}")
            print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

            # --- 模型保存与早停逻辑 (基于 MSE，越低越好) ---
            if test_mse < global_best_mse or test_corr > global_best_corr:
                global_best_mse = test_mse
                global_best_mae = test_mae
                global_best_seed = seed
                global_best_corr = test_corr
                ema.apply_shadow()
                global_best_model_state = model.state_dict()
                ema.restore()
                print(
                    f"New global best found: Test MSE={global_best_mse:.4f}, Test MAE={global_best_mae:.4f}, Seed={seed}, Epoch={epoch + 1}")
            if test_mse < best_mse or test_corr > best_corr:
                best_mse = test_mse
                best_mae = test_mae
                best_corr = test_corr
                best_acc7 = test_acc7
                best_acc5 = test_acc5
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Patience Counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
        # 记录当前种子的最佳结果
        best_mses.append(best_mse)
        best_maes.append(best_mae)
        print(f"\nSeed {seed} - Best Test MSE: {best_mse:.4f}, Corresponding MAE: {best_mae:.4f}")
        append_results_to_csv(csv_filename, seed, best_mae, best_corr, best_acc7, best_acc5)
        # 内存清理
        print(f"\nClearing GPU memory for Seed {seed}...")
        del model
        del optimizer
        del scheduler
        del ema
        del train_loader
        del test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # --- 最终结果处理和保存 ---
    if global_best_mse != float('inf') and global_best_model_state is not None:
        save_path = os.path.join(output_dir, f"multimodal_best_mse_{global_best_mse:.4f}_corr_{global_best_corr:.4f}_seed_{global_best_seed}.pth")
        torch.save(global_best_model_state, save_path)
        print(f"\nSaved global best model parameters to: {save_path}")
        print(
            f"Global Best Test MSE: {global_best_mse:.4f}, MAE: {global_best_mae:.4f}, Seed: {global_best_seed}")
        # 收集最佳模型的训练集和测试集预测结果
        set_seed(global_best_seed)
        best_train_dataset = build_dataset('train')
        best_test_dataset = build_dataset('test')
        train_loader_best = DataLoader(best_train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                       pin_memory=True)
        test_loader_best = DataLoader(best_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                      pin_memory=True)
        best_model = create_model()
        best_model.load_state_dict(global_best_model_state)
        train_preds = collect_predictions(best_model, train_loader_best, device, modalities, use_cam_loss, cam_type)
        test_preds = collect_predictions(best_model, test_loader_best, device, modalities, use_cam_loss, cam_type)
        # 保存预测结果
        prefix = os.path.splitext(save_path)[0]
        train_pred_path = f"{prefix}_train_predictions2.npz"
        test_pred_path = f"{prefix}_test_predictions2.npz"
        # 回归任务，保存 predictions 和 labels
        np.savez(train_pred_path, **train_preds)
        np.savez(test_pred_path, **test_preds)
        print(f"Saved train predictions to: {train_pred_path}")
        print(f"Saved test predictions to: {test_pred_path}")
    # 计算所有种子的平均结果
    mse_mean = np.mean(best_mses)
    mse_std = np.std(best_mses)
    mae_mean = np.mean(best_maes)
    mae_std = np.std(best_maes)
    print("\n=== Final Results Across Seeds ===")
    print(f"Best Test MSEs: {best_mses}")
    print(f"Mean Test MSE: {mse_mean:.4f}, Std Dev: {mse_std:.4f}")
    print(f"Corresponding MAEs: {best_maes}")
    print(f"Mean Test MAE: {mae_mean:.4f}, Std Dev: {mae_std:.4f}")


if __name__ == "__main__":
    main()
