# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import os
# import gc
# # 复用您原来的代码
# from src.feature_mosi_dataset import MOSIDataset
# from src.utils import set_seed, EMA
# # 导入新的模型和工具函数
# from unimodal_models import TextClassifier, AudioClassifier, VideoClassifier
# from unimodal_utils import train_epoch, evaluate_epoch
# from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
#
# def main():
#     # --- 主配置区 ---
#     MODALITY_TO_TRAIN = 'V'  # 修改这里来选择模态: 'T', 'A', 或 'V'
#     # Parameters
#     seed = 42
#     num_labels = 7
#     num_epochs = 32
#     batch_size = 32
#     lr = 1e-4
#     eta_min = 2e-7
#     patience = 8
#     hidden_dim = 128
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     print(f"=== Training Unimodal Classifier for Modality: {MODALITY_TO_TRAIN} with Seed {seed} ===")
#     set_seed(seed)
#
#     # Data paths (请根据实际路径修改)
#     data_paths = {
#         'train': {
#             'text': '/data/home/chenqian/regression_models/text_model/train_text_features_general.npz',
#             'audio': '/data/home/chenqian/regression_models/audio_model/train_audio_features_general.npz',
#             'video': '/data/home/chenqian/regression_models/video_model/train_video_features_general.npz',
#         },
#         'test': {
#             'text': '/data/home/chenqian/regression_models/text_model/test_text_features_general.npz',
#             'audio': '/data/home/chenqian/regression_models/audio_model/test_audio_features_general.npz',
#             'video': '/data/home/chenqian/regression_models/video_model/test_video_features_general.npz',
#         }
#     }
#     emotion_labels = {
#         0: "Strong Negative (-3)", 1: "Negative (-2)", 2: "Weak Negative (-1)", 3: "Neutral (0)",
#         4: "Weak Positive (+1)", 5: "Positive (+2)", 6: "Strong Positive (+3)"
#     }
#
#     # 优先使用 sequence_features（如果 npz 中存在），否则会回退到 pooled_features（由 MOSIDataset 处理）
#     feature_type_choice = 'sequence_features'
#
#     # 根据要训练的模态，创建 dataset 与 model
#     if MODALITY_TO_TRAIN == 'T':
#         train_dataset = MOSIDataset(
#             data_paths['train']['text'],
#             data_paths['train']['audio'],
#             data_paths['train']['video'],
#             modalities=['T'],
#             split='train',
#             feature_type=feature_type_choice
#         )
#         test_dataset = MOSIDataset(
#             data_paths['test']['text'],
#             data_paths['test']['audio'],
#             data_paths['test']['video'],
#             modalities=['T'],
#             split='test',
#             feature_type=feature_type_choice
#         )
#         text_seq_len = train_dataset.aligned_data['text'].shape[1] if train_dataset.aligned_data['text'].ndim == 3 else 1
#         model = TextClassifier(
#             hidden_dim=hidden_dim,
#             num_classes=num_labels,
#             use_precomputed=True,
#             input_dim=1024,
#             target_seq_len=text_seq_len,
#         ).to(device)
#
#     elif MODALITY_TO_TRAIN == 'A':
#         train_dataset = MOSIDataset(
#             data_paths['train']['text'],
#             data_paths['train']['audio'],
#             data_paths['train']['video'],
#             split='train', modalities=['A'],
#             feature_type='sequence_features'
#         )
#         test_dataset = MOSIDataset(
#             data_paths['test']['text'],
#             data_paths['test']['audio'],
#             data_paths['test']['video'],
#             modalities=['A'],
#             split='test',
#             feature_type='sequence_features'
#         )
#         audio_feat = train_dataset.aligned_data['audio']
#         if audio_feat.ndim >= 2:
#             audio_feat_dim = audio_feat.shape[-1]
#         else:
#             raise ValueError("Unexpected audio feature shape: {}".format(audio_feat.shape))
#         model = AudioClassifier(audio_dim=audio_feat_dim, hidden_dim=hidden_dim, num_classes=num_labels).to(device)
#
#     elif MODALITY_TO_TRAIN == 'V':
#         train_dataset = MOSIDataset(
#             data_paths['train']['text'],
#             data_paths['train']['audio'],
#             data_paths['train']['video'],
#             split='train',
#             modalities=['V'],
#             feature_type='sequence_features'
#         )
#         test_dataset = MOSIDataset(
#             data_paths['test']['text'],
#             data_paths['test']['audio'],
#             data_paths['test']['video'],
#             modalities=['V'],
#             split='test',
#             feature_type='sequence_features'
#         )
#         video_feat = train_dataset.aligned_data['video']
#         if video_feat.ndim >= 2:
#             video_feat_dim = video_feat.shape[-1]
#         else:
#             raise ValueError("Unexpected video feature shape: {}".format(video_feat.shape))
#         model = VideoClassifier(video_dim=video_feat_dim, hidden_dim=hidden_dim, num_classes=num_labels).to(device)
#     else:
#         raise ValueError("MODALITY_TO_TRAIN must be one of 'T', 'A', or 'V'")
#     # DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = CosineAnnealingLR(optimizer, T_max=8, eta_min=eta_min)
#     ema = EMA(model, decay=0.999)
#     best_f1 = 0.0
#     patience_counter = 0
#     best_model_state = None
#     for epoch in range(num_epochs):
#         # 使用新的训练评估函数（已支持预计算文本）
#         train_loss = train_epoch(model, train_loader, optimizer, device, MODALITY_TO_TRAIN, ema=ema)
#         # 评估时应用 EMA 的影子权重
#         ema.apply_shadow()
#         train_eval_loss, train_accuracy, train_weighted_f1, train_class_accuracies, train_all_labels, train_all_preds = evaluate_epoch(
#             model, train_eval_loader, device, MODALITY_TO_TRAIN, num_labels
#         )
#         test_loss, test_accuracy, test_weighted_f1, class_accuracies, all_labels, all_preds = evaluate_epoch(
#             model, test_loader, device, MODALITY_TO_TRAIN, num_labels
#         )
#         ema.restore()
#         scheduler.step()
#         print(f"Epoch {epoch + 1}/{num_epochs}")
#         print(
#             f"Train Loss (opt): {train_loss:.4f} | Train Loss (eval): {train_eval_loss:.4f}"
#             f", Train Accuracy: {train_accuracy:.4f}, Train Weighted-F1: {train_weighted_f1:.4f}"
#         )
#         print(
#             f"Test  Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Weighted-F1: {test_weighted_f1:.4f}")
#
#         train_class_acc_str = "  "
#         for label_id, acc in train_class_accuracies.items():
#             emotion_name = emotion_labels.get(label_id, f"Label {label_id}")
#             train_class_acc_str += f"{emotion_name}: {acc:.4f}  "
#         print("Train Class-wise Accuracies:" + train_class_acc_str)
#         class_acc_str = "  "
#         for label_id, acc in class_accuracies.items():
#             emotion_name = emotion_labels.get(label_id, f"Label {label_id}")
#             class_acc_str += f"{emotion_name}: {acc:.4f}  "
#         print("Test  Class-wise Accuracies:" + class_acc_str)
#         if test_weighted_f1 > best_f1:
#             best_f1 = test_weighted_f1
#             best_accuracy = test_accuracy
#             patience_counter = 0
#             # 保存最佳模型状态（使用 EMA 的权重）
#             ema.apply_shadow()
#             best_model_state = model.state_dict()
#             ema.restore()
#             print(f"New best found! Weighted-F1: {best_f1:.4f}, Accuracy: {best_accuracy:.4f}")
#         else:
#             patience_counter += 1
#             print(f"Patience Counter: {patience_counter}/{patience}")
#             if patience_counter >= patience:
#                 print(f"Early stopping triggered after {epoch + 1} epochs.")
#                 break
#         if epoch == num_epochs - 1:
#             cm = confusion_matrix(all_labels, all_preds)
#             print(cm)
#     # 保存最终的最佳模型权重
#     if best_model_state:
#         save_dir = "/data/home/chenqian/models/unimodel_experts"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir,
#                                  f"unimodal_{MODALITY_TO_TRAIN.lower()}_best_acc_{best_accuracy:.4f}_best_f1_{best_f1:.4f}.pth")
#         torch.save(best_model_state, save_path)
#         print(f"\nSaved best unimodal model for '{MODALITY_TO_TRAIN}' to: {save_path}")
#         print(f"Final Best Metrics -> Accuracy: {best_accuracy:.4f}, Weighted-F1: {best_f1:.4f}")
#
#
# if __name__ == "__main__":
#     main()

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import gc
# 复用您原来的代码
from src.feature_mosi_dataset import MOSIDataset
from src.utils import set_seed, EMA
# 导入新的模型和工具函数 (已修改为回归)
from unimodal_models import TextClassifier, AudioClassifier, VideoClassifier
from unimodal_utils import train_epoch, evaluate_epoch
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # --- 主配置区 ---
    MODALITY_TO_TRAIN = 'V'
    seed = 42
    num_labels = 7
    num_epochs = 32
    batch_size = 32
    lr = 1e-4
    eta_min = 2e-7
    patience = 8
    hidden_dim = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Unimodal REGRESSION Classifier for Modality: {MODALITY_TO_TRAIN} with Seed {seed} ===")
    set_seed(seed)
    data_paths = {
        'train': {
            'text': '/data/home/chenqian/regression_models/text_model/train_text_features_general.npz',
            'audio': '/data/home/chenqian/regression_models/audio_model/train_audio_features_general.npz',
            'video': '/data/home/chenqian/regression_models/video_model/train_video_features_general.npz',
        },
        'test': {
            'text': '/data/home/chenqian/regression_models/text_model/test_text_features_general.npz',
            'audio': '/data/home/chenqian/regression_models/audio_model/test_audio_features_general.npz',
            'video': '/data/home/chenqian/regression_models/video_model/test_video_features_general.npz',
        }
    }
    # 情感标签现在仅用于辅助理解 ACC7 的类别，但实际模型输出是连续分数
    emotion_labels = {
        0: "Strong Negative (-3)", 1: "Negative (-2)", 2: "Weak Negative (-1)", 3: "Neutral (0)",
        4: "Weak Positive (+1)", 5: "Positive (+2)", 6: "Strong Positive (+3)"
    }
    feature_type_choice = 'sequence_features'
    if MODALITY_TO_TRAIN == 'T':
        train_dataset = MOSIDataset(
            data_paths['train']['text'],
            data_paths['train']['audio'],
            data_paths['train']['video'],
            modalities=['T'], split='train', feature_type=feature_type_choice
        )
        test_dataset = MOSIDataset(
            data_paths['test']['text'],
            data_paths['test']['audio'],
            data_paths['test']['video'],
            modalities=['T'], split='test', feature_type=feature_type_choice
        )
        text_seq_len = train_dataset.aligned_data['text'].shape[1] if train_dataset.aligned_data[
                                                                          'text'].ndim == 3 else 1
        model = TextClassifier(
            hidden_dim=hidden_dim, num_classes=1,
            use_precomputed=True, input_dim=1024, target_seq_len=text_seq_len,
        ).to(device)

    elif MODALITY_TO_TRAIN == 'A':
        train_dataset = MOSIDataset(
            data_paths['train']['text'], data_paths['train']['audio'], data_paths['train']['video'],
            split='train', modalities=['A'], feature_type='sequence_features'
        )
        test_dataset = MOSIDataset(
            data_paths['test']['text'], data_paths['test']['audio'], data_paths['test']['video'],
            modalities=['A'], split='test', feature_type='sequence_features'
        )
        audio_feat = train_dataset.aligned_data['audio']
        if audio_feat.ndim >= 2:
            audio_feat_dim = audio_feat.shape[-1]
        else:
            raise ValueError("Unexpected audio feature shape: {}".format(audio_feat.shape))
        # num_classes 在 unimodal_models 中已默认为 1
        model = AudioClassifier(audio_dim=audio_feat_dim, hidden_dim=hidden_dim, num_classes=1).to(device)

    elif MODALITY_TO_TRAIN == 'V':
        train_dataset = MOSIDataset(
            data_paths['train']['text'], data_paths['train']['audio'], data_paths['train']['video'],
            split='train', modalities=['V'], feature_type='sequence_features'
        )
        test_dataset = MOSIDataset(
            data_paths['test']['text'], data_paths['test']['audio'], data_paths['test']['video'],
            modalities=['V'], split='test', feature_type='sequence_features'
        )
        video_feat = train_dataset.aligned_data['video']
        if video_feat.ndim >= 2:
            video_feat_dim = video_feat.shape[-1]
        else:
            raise ValueError("Unexpected video feature shape: {}".format(video_feat.shape))
        model = VideoClassifier(video_dim=video_feat_dim, hidden_dim=hidden_dim, num_classes=1).to(device)
    else:
        raise ValueError("MODALITY_TO_TRAIN must be one of 'T', 'A', or 'V'")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=8, eta_min=eta_min)
    ema = EMA(model, decay=0.999)
    # 最佳指标记录
    best_corr = -1.0  # 使用相关系数作为主要选择标准
    best_acc7 = 0.0
    best_mae = float('inf')
    best_mse = float('inf')
    patience_counter = 0
    best_model_state = None
    for epoch in range(num_epochs):
        # 使用新的训练评估函数 (train_epoch 返回 MSE Loss)
        train_loss = train_epoch(model, train_loader, optimizer, device, MODALITY_TO_TRAIN, ema=ema)
        # 评估时应用 EMA 的影子权重
        ema.apply_shadow()
        train_eval_loss, train_acc7, train_neg_mae, train_corr_dict, _, _ = evaluate_epoch(
            model, train_eval_loader, device, MODALITY_TO_TRAIN, num_labels
        )
        test_loss, test_acc7, test_neg_mae, test_corr_dict, all_labels_cont, all_preds_cont = evaluate_epoch(
            model, test_loader, device, MODALITY_TO_TRAIN, num_labels
        )
        ema.restore()
        scheduler.step()
        # 提取实际的 MAE 和 Corr
        train_corr = train_corr_dict.get('Corr', 0.0)
        train_mae = train_corr_dict.get('MAE', 0.0)
        test_corr = test_corr_dict.get('Corr', 0.0)
        test_mae = test_corr_dict.get('MAE', 0.0)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(
            f"Train MSE Loss (opt): {train_loss:.4f} | Train MSE Loss (eval): {train_eval_loss:.4f}"
        )
        print(
            f"Train Metrics -> ACC7: {train_acc7:.4f}, Corr: {train_corr:.4f}, MAE: {train_mae:.4f}"
        )
        print(
            f"Test  Metrics -> MSE Loss: {test_loss:.4f}, ACC7: {test_acc7:.4f}, Corr: {test_corr:.4f}, MAE: {test_mae:.4f}"
        )
        # --- 最佳模型选择逻辑 (最大化 Corr) ---
        if test_corr > best_corr:
            best_corr = test_corr
            best_acc7 = test_acc7
            best_mae = test_mae
            best_mse = test_loss
            patience_counter = 0
            # 保存最佳模型状态（使用 EMA 的权重）
            ema.apply_shadow()
            best_model_state = model.state_dict()
            ema.restore()
            print(f"New best found! Corr: {best_corr:.4f}, ACC7: {best_acc7:.4f}, MAE: {best_mae:.4f}")
        else:
            patience_counter += 1
            print(f"Patience Counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # 保存最终的最佳模型权重
    if best_model_state:
        save_dir = "/data/home/chenqian/regression_models/unimodel_experts"
        os.makedirs(save_dir, exist_ok=True)
        # 将 Corr 作为保存名称的一部分
        save_path = os.path.join(save_dir,
                                 f"unimodal_{MODALITY_TO_TRAIN.lower()}_best_corr_{best_corr:.4f}_acc7_{best_acc7:.4f}_mae_{best_mae: .4f}.pth")
        torch.save(best_model_state, save_path)
        print(f"\nSaved best unimodal REGRESSION model for '{MODALITY_TO_TRAIN}' to: {save_path}")
        print(f"Final Best Metrics -> Corr: {best_corr:.4f}, ACC7: {best_acc7:.4f}, MAE: {best_mae:.4f}")


if __name__ == "__main__":
    main()
