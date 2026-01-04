# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# import torch.nn as nn
# from Data2vec_modal import EmotionClassifier
# from data_loader import data_loader_mosi_audio  # 导入保持不变
# from sklearn.metrics import f1_score, confusion_matrix
# import random
# import os
# import numpy as np
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from utils import EMA, train, evaluate, set_seed
# import gc
# import copy
# from collections import defaultdict
# import shutil
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
#
# # 测试集评估函数（使用 utils.evaluate 对测试集进行评估）
# def test_on_final_model(model_state, model_config, test_loader, device, num_labels, emotion_labels):
#     """加载最佳模型权重，并在测试集上运行评估。"""
#     # 重新初始化模型
#     temp_model = EmotionClassifier(
#         num_classes=num_labels,
#         use_lora=model_config['use_lora'],
#         use_transformer=model_config['use_transformer'],
#         weight_attn=model_config['weight_attn'],
#         model_type=model_config['model_type'],
#         max_length=model_config['max_length'],
#         use_adapters=model_config['use_adapters']
#     ).to(device)
#
#     # 加载权重
#     temp_model.load_state_dict(model_state)
#     print("\n--- Running Test Set Evaluation ---")
#
#     # 传入 ema=None，因为模型已经加载了最佳权重，无需再次应用影子参数。
#     test_loss, test_accuracy, test_weighted_f1, test_class_accuracies = evaluate(
#         temp_model, test_loader, ema=None, device=device, num_labels=num_labels)
#
#     # 格式化输出
#     print(f"TEST Loss: {test_loss:.4f}, TEST Accuracy: {test_accuracy:.4f}, TEST Weighted-F1: {test_weighted_f1:.4f}")
#     print("Class-wise Accuracies:")
#     class_acc_str = "  "
#     for label_id, acc in test_class_accuracies.items():
#         emotion_name = emotion_labels.get(label_id, f"Label {label_id}")
#         class_acc_str += f"{emotion_name}: {acc:.4f}  "
#     print(class_acc_str)
#
#     # 清理
#     del temp_model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     return test_loss, test_accuracy, test_weighted_f1, test_class_accuracies
#
#
# def main():
#     # ===============================================
#     # 路径配置
#     # ===============================================
#     train_csv_file = "/data/home/chenqian/CMU-MOSEI/train_acc7.csv"
#     valid_csv_file = "/data/home/chenqian/CMU-MOSEI/valid_acc7.csv"
#     test_csv_file = "/data/home/chenqian/CMU-MOSEI/test_acc7.csv"
#     audio_directory = "/data/home/chenqian/CMU-MOSEI/audio_16k"
#     model_save_dir = "/data/home/chenqian/models/audio_model"
#     results_csv_path = "/data/home/chenqian/models/audio_model/test_results.csv"
#     os.makedirs(model_save_dir, exist_ok=True)
#     # ===============================================
#
#     # 参数设置
#     seeds = [41]
#     num_labels = 7
#     num_epochs = 64
#     batch_size = 32
#     max_length_sec = 12
#     patience = 5
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     lr = 1e-4
#     eta_min = 2e-5
#     TEST_EVAL_INTERVAL = 8  # 每 8 轮评估一次测试集
#
#     # 更改的参数：
#     use_lora = False
#     use_transformer = False
#     weight_attn = False
#     use_adapters = True
#     model_type = "Whisper"
#     max_seq_length = max_length_sec * 16000  # 音频最大长度
#     if use_adapters:
#         batch_size = 8
#
#     # 存储测试集评估结果
#     all_test_results_detailed = []
#
#     # === 全局最佳模型跟踪变量 (基于验证集) ===
#     global_best_valid_acc = -1.0
#     global_best_model_state = None
#     global_best_seed = None
#
#     # 统一模型配置
#     model_config = {
#         'use_lora': use_lora, 'use_transformer': use_transformer, 'weight_attn': weight_attn,
#         'model_type': model_type, 'max_length': max_length_sec, 'use_adapters': use_adapters
#     }
#     # ==========================================
#
#     # MOSI ACC-7 数据集的情感标签映射：
#     emotion_labels = {
#         0: "Strong Negative (-3)", 1: "Negative (-2)", 2: "Weak Negative (-1)", 3: "Neutral (0)",
#         4: "Weak Positive (+1)", 5: "Positive (+2)", 6: "Strong Positive (+3)"
#     }
#
#     # 遍历随机种子
#     for seed in seeds:
#         print(f"\n=== Training with Seed {seed} ===\n")
#         set_seed(seed)
#
#         # 初始化模型
#         model = EmotionClassifier(num_classes=num_labels, **model_config).to(device)
#
#         # 加载数据
#         train_loader, valid_loader, test_loader = data_loader_mosi_audio(
#             train_csv_path=train_csv_file,
#             test_csv_path=test_csv_file,
#             valid_csv_path=valid_csv_file,
#             audio_directory=audio_directory,
#             batch_size=batch_size,
#             max_seq_length=max_seq_length,
#             model_type=model_type,
#             local_model_path="/data/home/chenqian/whisper_large_v3/"
#         )
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#         scheduler = CosineAnnealingLR(optimizer, T_max=8, eta_min=eta_min)
#         ema = EMA(model, decay=0.999)
#
#         # 当前种子下的最佳结果和早停变量
#         best_valid_acc_for_seed = 0.0
#         best_f1_for_acc_for_seed = 0.0
#         best_model_state_for_seed = None
#         patience_counter = 0
#
#         for epoch in range(num_epochs):
#             # 训练
#             train_loss = train(model, train_loader, optimizer, scheduler, ema, device)
#
#             # 验证集评估 (用于早停和最佳模型选择)
#             valid_loss, valid_accuracy, valid_weighted_f1, class_accuracies = evaluate(
#                 model, valid_loader, ema, device, num_labels)
#
#             print(f"Epoch {epoch + 1}/{num_epochs}")
#             print(f"Train Loss: {train_loss:.4f}, Current Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
#             # 验证集结果
#             print(
#                 f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Valid Weighted-F1: {valid_weighted_f1:.4f}")
#             print("Class-wise Accuracies:")
#             class_acc_str = "  "
#             for label_id, acc in class_accuracies.items():
#                 emotion_name = emotion_labels.get(label_id, f"Label {label_id}")
#                 class_acc_str += f"{emotion_name}: {acc:.4f}  "
#             print(class_acc_str)
#
#             # 最佳模型和早停逻辑 (基于验证集)
#             if valid_accuracy > best_valid_acc_for_seed:
#                 best_valid_acc_for_seed = valid_accuracy
#                 best_f1_for_acc_for_seed = valid_weighted_f1
#                 patience_counter = 0
#
#                 # 保存当前种子下的最佳权重 (EMA 平滑后的)
#                 ema.apply_shadow()
#                 best_model_state_for_seed = model.state_dict()
#                 ema.restore()
#
#                 # 检查是否是全局最佳
#                 if valid_accuracy > global_best_valid_acc:
#                     global_best_valid_acc = valid_accuracy
#                     global_best_seed = seed
#                     global_best_model_state = copy.deepcopy(best_model_state_for_seed)
#                     print(
#                         f"New global best found: Valid Accuracy={global_best_valid_acc:.4f}, Seed={seed}, Epoch={epoch + 1}")
#             else:
#                 patience_counter += 1
#                 print(f"Patience Counter: {patience_counter}/{patience}")
#
#             # 测试集评估逻辑：每 TEST_EVAL_INTERVAL 轮评估一次
#             if (epoch + 1) % TEST_EVAL_INTERVAL == 0 and best_model_state_for_seed is not None:
#                 # 测试评估
#                 test_loss, test_accuracy, test_weighted_f1, test_class_accuracies = test_on_final_model(
#                     best_model_state_for_seed, model_config, test_loader, device, num_labels, emotion_labels)
#                 result_entry = {
#                     "Seed": seed,
#                     "Epoch": epoch + 1,
#                     "Best_Valid_Acc_at_Epoch": best_valid_acc_for_seed,  # 记录当前模型在验证集上的最佳Acc
#                     "Best_Valid_F1_at_Epoch": best_f1_for_acc_for_seed,
#                     "Test_Acc": test_accuracy,
#                     "Test_Weighted_F1": test_weighted_f1,
#                 }
#                 # 添加类别准确率
#                 for label_id, acc in test_class_accuracies.items():
#                     emotion_name = emotion_labels.get(label_id, f"Label {label_id}")
#                     col_name = f"Test_Class_Acc_{emotion_name.replace(' ', '_').replace('(', '').replace(')', '')}"
#                     result_entry[col_name] = acc
#
#                 all_test_results_detailed.append(result_entry)
#
#             # 早停
#             if patience_counter >= patience:
#                 print(f"Early stopping triggered after {epoch + 1} epochs.")
#                 break
#
#         # 训练结束后的最终测试集评估
#         final_epoch_number = epoch + 1
#         if best_model_state_for_seed is not None:
#             print(f"\n--- Final Evaluation (Seed {seed}) on TEST Set ---")
#             final_test_loss, final_test_accuracy, final_test_weighted_f1, final_test_class_accuracies = test_on_final_model(
#                 best_model_state_for_seed, model_config, test_loader, device, num_labels, emotion_labels)
#
#             # 将最终测试结果添加到统一的详细列表
#             result_entry = {
#                 "Seed": seed,
#                 "Epoch": final_epoch_number,  # 使用最终的 epoch 编号标记
#                 "Best_Valid_Acc_at_Epoch": best_valid_acc_for_seed,
#                 "Best_Valid_F1_at_Epoch": best_f1_for_acc_for_seed,
#                 "Test_Acc": final_test_accuracy,
#                 "Test_Weighted_F1": final_test_weighted_f1,
#             }
#             # 添加类别准确率
#             for label_id, acc in final_test_class_accuracies.items():
#                 emotion_name = emotion_labels.get(label_id, f"Label {label_id}")
#                 col_name = f"Test_Class_Acc_{emotion_name.replace(' ', '_').replace('(', '').replace(')', '')}"
#                 result_entry[col_name] = acc
#
#             all_test_results_detailed.append(result_entry)
#         else:
#             print(f"Warning: No best model state saved for seed {seed}. Skipping final test evaluation.")
#
#         # 清理 GPU 显存
#         print(f"\nClearing GPU memory for Seed {seed}...")
#         del model
#         del optimizer
#         del scheduler
#         del ema
#         del train_loader
#         del valid_loader
#         del test_loader
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#     # 保存全局最佳模型参数
#     if global_best_model_state is not None:
#         final_path_base = os.path.join(model_save_dir,
#                                        f"best_{model_type}_valid_acc_{global_best_valid_acc:.4f}_seed_{global_best_seed}")
#
#         if use_adapters or use_lora:
#             # 对于 LoRA/Adapter，需要重新加载模型来使用 save_adapters
#             temp_model_for_save = EmotionClassifier(num_classes=num_labels, **model_config).to('cpu')
#             temp_model_for_save.load_state_dict(global_best_model_state)
#             adapter_save_path = f"{final_path_base}_adapter.pth"
#             if hasattr(temp_model_for_save, 'save_adapters'):
#                 temp_model_for_save.save_adapters(adapter_save_path)
#             else:
#                 torch.save(global_best_model_state, adapter_save_path)
#             del temp_model_for_save
#             print(f"\nSaved global best adapters parameters to: {adapter_save_path}")
#         else:
#             full_model_save_path = f"{final_path_base}.pth"
#             torch.save(global_best_model_state, full_model_save_path)
#             print(f"\nSaved global best full model parameters to: {full_model_save_path}")
#
#         print(f"Global Best Model (from Validation Set): Acc={global_best_valid_acc:.4f}, Seed={global_best_seed}")
#
#     if all_test_results_detailed:
#         results_df = pd.DataFrame(all_test_results_detailed)
#
#         # 筛选出每个种子的最终结果
#         final_results_df = results_df.loc[results_df.groupby('Seed')['Epoch'].idxmax()]
#
#         # 计算均值和方差
#         acc_mean = final_results_df['Test_Acc'].mean()
#         acc_std = final_results_df['Test_Acc'].std()
#         f1_mean = final_results_df['Test_Weighted_F1'].mean()
#         f1_std = final_results_df['Test_Weighted_F1'].std()
#
#         # 创建 summary rows
#         # 使用一个特殊的 Epoch 编号来区分总结行
#         summary_row = {"Seed": "Mean", "Test_Acc": acc_mean, "Test_Weighted_F1": f1_mean, "Epoch": -99}
#         std_row = {"Seed": "Std_Dev", "Test_Acc": acc_std, "Test_Weighted_F1": f1_std, "Epoch": -99}
#
#         # 添加类别准确率的均值/方差
#         for col in results_df.columns:
#             if col.startswith("Test_Class_Acc_") or col.startswith("Best_Valid_"):
#                 summary_row[col] = final_results_df[col].mean()
#                 std_row[col] = final_results_df[col].std()
#
#         # 将均值/方差添加到总 DataFrame 的末尾
#         results_df = pd.concat([results_df, pd.DataFrame([summary_row, std_row])], ignore_index=True)
#
#         # 最终结果和总结的 DataFrame
#         results_df.to_csv(results_csv_path, index=False, float_format='%.4f')
#         print(f"\nAll periodic and final test results saved to: {results_csv_path}")
#
#         print("\n=== Final Results Across Seeds (Test Set) ===")
#         print(f"Mean Test Accuracy (Final Epoch): {acc_mean:.4f} \u00B1 {acc_std:.4f}")
#         print(f"Mean Weighted-F1 (Final Epoch): {f1_mean:.4f} \u00B1 {f1_std:.4f}")
#     else:
#         print("\nNo test results to save.")
#
#
# if __name__ == "__main__":
#     main()

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from Data2vec_modal import EmotionClassifier
from data_loader import data_loader_mosi_audio
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from scipy.stats import pearsonr
import random
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import EMA, train, evaluate, set_seed
import gc
import copy
from collections import defaultdict
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def convert_to_acc7_label(score):
    score = float(score)
    if -3.0 <= score <= -2.5:
        return 0  # 强烈消极 (-3)
    elif -2.5 < score <= -1.5:
        return 1  # 消极 (-2)
    elif -1.5 < score <= -0.5:
        return 2  # 弱消极 (-1)
    elif -0.5 < score <= 0.5:
        return 3  # 中性 (0)
    elif 0.5 < score <= 1.5:
        return 4  # 弱积极 (+1)
    elif 1.5 < score <= 2.5:
        return 5  # 积极 (+2)
    elif 2.5 < score <= 3.0:
        return 6  # 强烈积极 (+3)
    else:
        return 3  # 默认中性


def evaluate_and_get_acc7(model, data_loader, ema, device):
    model.eval()
    if ema is not None:
        ema.apply_shadow()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in data_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)  # [B, 1]

            logits = model(input_values, attention_mask=attention_mask)  # [B, 1]
            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_preds.extend(logits.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    if ema is not None:
        ema.restore()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / len(data_loader)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    # 计算 Corr
    if len(all_labels) > 1 and all_labels.std() != 0 and all_preds.std() != 0:
        corr, _ = pearsonr(all_labels, all_preds)
    else:
        corr = 0.0
    true_acc7_labels = np.array([convert_to_acc7_label(l) for l in all_labels])
    pred_acc7_labels = np.array([convert_to_acc7_label(p) for p in all_preds])
    acc_7 = accuracy_score(true_acc7_labels, pred_acc7_labels)
    return avg_loss, mse, mae, corr, acc_7


# 测试集评估函数 (修改后以处理 ACC-7)
def test_on_final_model(model_state, model_config, test_loader, device):
    num_labels = 1
    temp_model = EmotionClassifier(
        num_classes=num_labels,
        use_lora=model_config['use_lora'],
        use_transformer=model_config['use_transformer'],
        weight_attn=model_config['weight_attn'],
        model_type=model_config['model_type'],
        max_length=model_config['max_length'],
        use_adapters=model_config['use_adapters']
    ).to(device)
    # 加载权重
    temp_model.load_state_dict(model_state)
    # 使用新的评估函数
    test_loss, test_mse, test_mae, test_corr, test_acc_7 = evaluate_and_get_acc7(
        temp_model, test_loader, ema=None, device=device)
    print(
        f"TEST Loss (MSE): {test_loss:.4f}, TEST MSE: {test_mse:.4f}, TEST MAE: {test_mae:.4f}, TEST Corr: {test_corr:.4f}, TEST ACC-7: {test_acc_7:.4f}")
    # 清理
    del temp_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 返回回归指标 + ACC-7
    return test_loss, test_mse, test_mae, test_corr, test_acc_7


def main():
    # ===============================================
    # 路径配置
    # ===============================================
    train_csv_file = "/data/home/chenqian/CMU-MOSEI/train_acc7.csv"
    valid_csv_file = "/data/home/chenqian/CMU-MOSEI/valid_acc7.csv"
    test_csv_file = "/data/home/chenqian/CMU-MOSEI/test_acc7.csv"
    audio_directory = "/data/home/chenqian/CMU-MOSEI/audio_16k"
    model_save_dir = "/data/home/chenqian/regression_models/audio_model"  # 更改保存目录
    results_csv_path = "/data/home/chenqian/regression_models/audio_model/test_results.csv"  # 更改结果文件名
    os.makedirs(model_save_dir, exist_ok=True)
    # 参数设置
    seeds = [41]
    num_labels = 1
    num_epochs = 64
    batch_size = 32
    max_length_sec = 12
    patience = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr = 1e-4
    eta_min = 2e-5
    TEST_EVAL_INTERVAL = 8
    # 更改的参数：
    use_lora = False
    use_transformer = False
    weight_attn = False
    use_adapters = True
    model_type = "Whisper"
    max_seq_length = max_length_sec * 16000
    if use_adapters:
        batch_size = 8
    # 存储测试集评估结果
    all_test_results_detailed = []
    # === 全局最佳模型跟踪变量 (基于验证集 ACC-7) ===
    global_best_valid_acc_7 = -1.0
    global_best_model_state = None
    global_best_seed = None
    # 统一模型配置
    model_config = {
        'use_lora': use_lora, 'use_transformer': use_transformer, 'weight_attn': weight_attn,
        'model_type': model_type, 'max_length': max_length_sec, 'use_adapters': use_adapters
    }
    # 遍历随机种子
    for seed in seeds:
        print(f"\n=== Training with Seed {seed} (Regression + Acc-7 as Metric) ===\n")
        set_seed(seed)
        # 初始化模型 (num_classes=1)
        model = EmotionClassifier(num_classes=num_labels, **model_config).to(device)
        # 加载数据
        train_loader, valid_loader, test_loader = data_loader_mosi_audio(
            train_csv_path=train_csv_file,
            test_csv_path=test_csv_file,
            valid_csv_path=valid_csv_file,
            audio_directory=audio_directory,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            model_type=model_type,
            local_model_path="/data/home/chenqian/whisper_large_v3/"
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=8, eta_min=eta_min)
        ema = EMA(model, decay=0.999)
        # 当前种子下的最佳结果和早停变量
        best_valid_acc_7_for_seed = -1.0
        best_valid_corr_for_seed = -1.0
        best_valid_mse_for_seed = float('inf')
        best_model_state_for_seed = None
        patience_counter = 0
        for epoch in range(num_epochs):
            # 训练 (使用 utils.train - MSE Loss)
            train_loss = train(model, train_loader, optimizer, scheduler, ema, device)
            # 验证集评估 (使用扩展函数 evaluate_and_get_acc7)
            valid_loss, valid_mse, valid_mae, valid_corr, valid_acc_7 = evaluate_and_get_acc7(
                model, valid_loader, ema, device)
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss (MSE): {train_loss:.4f}, Current Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            # 验证集结果 (回归指标 + ACC-7)
            print(
                f"Valid Loss (MSE): {valid_loss:.4f}, Valid MSE: {valid_mse:.4f}, Valid MAE: {valid_mae:.4f}, Valid Corr: {valid_corr:.4f}, Valid Acc-7: {valid_acc_7:.4f}")
            # 最佳模型和早停逻辑 (基于验证集 ACC-7)
            if valid_acc_7 > best_valid_acc_7_for_seed:
                best_valid_acc_7_for_seed = valid_acc_7
                best_valid_corr_for_seed = valid_corr
                best_valid_mse_for_seed = valid_mse
                patience_counter = 0
                # 保存当前种子下的最佳权重 (EMA 平滑后的)
                ema.apply_shadow()
                best_model_state_for_seed = model.state_dict()
                ema.restore()
                # 检查是否是全局最佳 (基于 ACC-7)
                if valid_acc_7 > global_best_valid_acc_7:
                    global_best_valid_acc_7 = valid_acc_7
                    global_best_seed = seed
                    global_best_model_state = copy.deepcopy(best_model_state_for_seed)
                    print(
                        f"New global best found: Valid Acc-7={global_best_valid_acc_7:.4f}, Seed={seed}, Epoch={epoch + 1}")
            else:
                patience_counter += 1
                print(f"Patience Counter: {patience_counter}/{patience}")
            # 测试集评估逻辑：每 TEST_EVAL_INTERVAL 轮评估一次
            if (epoch + 1) % TEST_EVAL_INTERVAL == 0 and best_model_state_for_seed is not None:
                # 测试评估 (返回 loss, mse, mae, corr, acc_7)
                test_loss, test_mse, test_mae, test_corr, test_acc_7 = test_on_final_model(
                    best_model_state_for_seed, model_config, test_loader, device)
                result_entry = {
                    "Seed": seed,
                    "Epoch": epoch + 1,
                    "Best_Valid_Acc7_at_Epoch": best_valid_acc_7_for_seed,  # 记录当前模型在验证集上的最佳Acc7
                    "Best_Valid_Corr_at_Epoch": best_valid_corr_for_seed,
                    "Test_MSE": test_mse,
                    "Test_MAE": test_mae,
                    "Test_Corr": test_corr,
                    "Test_Acc7": test_acc_7,  # 新增
                }
                all_test_results_detailed.append(result_entry)

            # 早停
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
        # 训练结束后的最终测试集评估
        final_epoch_number = epoch + 1
        if best_model_state_for_seed is not None:
            print(f"\n--- Final Evaluation (Seed {seed}) on TEST Set ---")
            final_test_loss, final_test_mse, final_test_mae, final_test_corr, final_test_acc_7 = test_on_final_model(
                best_model_state_for_seed, model_config, test_loader, device)
            # 将最终测试结果添加到统一的详细列表
            result_entry = {
                "Seed": seed,
                "Epoch": final_epoch_number,
                "Best_Valid_Acc7_at_Epoch": best_valid_acc_7_for_seed,
                "Best_Valid_Corr_at_Epoch": best_valid_corr_for_seed,
                "Test_MSE": final_test_mse,
                "Test_MAE": final_test_mae,
                "Test_Corr": final_test_corr,
                "Test_Acc7": final_test_acc_7,  # 新增
            }
            all_test_results_detailed.append(result_entry)
        else:
            print(f"Warning: No best model state saved for seed {seed}. Skipping final test evaluation.")
        # 清理 GPU 显存
        print(f"\nClearing GPU memory for Seed {seed}...")
        del model
        del optimizer
        del scheduler
        del ema
        del train_loader
        del valid_loader
        del test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # 保存全局最佳模型参数
    if global_best_model_state is not None:
        final_path_base = os.path.join(model_save_dir,
                                       f"best_{model_type}_valid_acc7_{global_best_valid_acc_7:.4f}_seed_{global_best_seed}")
        if use_adapters or use_lora:
            temp_model_for_save = EmotionClassifier(num_classes=num_labels, **model_config).to('cpu')
            temp_model_for_save.load_state_dict(global_best_model_state)
            adapter_save_path = f"{final_path_base}_adapter.pth"
            if hasattr(temp_model_for_save, 'save_adapters'):
                temp_model_for_save.save_adapters(adapter_save_path)
            else:
                torch.save(global_best_model_state, adapter_save_path)
            del temp_model_for_save
        else:
            full_model_save_path = f"{final_path_base}.pth"
            torch.save(global_best_model_state, full_model_save_path)
        print(f"Global Best Model (from Validation Set): Acc-7={global_best_valid_acc_7:.4f}, Seed={global_best_seed}")
    if all_test_results_detailed:
        results_df = pd.DataFrame(all_test_results_detailed)
        # 筛选出每个种子的最终结果
        final_results_df = results_df.loc[results_df.groupby('Seed')['Epoch'].idxmax()]
        # 计算均值和方差
        acc7_mean = final_results_df['Test_Acc7'].mean()
        acc7_std = final_results_df['Test_Acc7'].std()
        corr_mean = final_results_df['Test_Corr'].mean()
        corr_std = final_results_df['Test_Corr'].std()
        mse_mean = final_results_df['Test_MSE'].mean()
        mse_std = final_results_df['Test_MSE'].std()
        mae_mean = final_results_df['Test_MAE'].mean()
        mae_std = final_results_df['Test_MAE'].std()
        # 创建 summary rows
        summary_row = {"Seed": "Mean", "Test_Acc7": acc7_mean, "Test_Corr": corr_mean, "Test_MSE": mse_mean,
                       "Test_MAE": mae_mean, "Epoch": -99}
        std_row = {"Seed": "Std_Dev", "Test_Acc7": acc7_std, "Test_Corr": corr_std, "Test_MSE": mse_std,
                   "Test_MAE": mae_std, "Epoch": -99}
        # 添加验证集最佳指标的均值/方差
        for col in final_results_df.columns:
            if col.startswith("Best_Valid_"):
                summary_row[col] = final_results_df[col].mean()
                std_row[col] = final_results_df[col].std()
        # 将均值/方差添加到总 DataFrame 的末尾
        results_df = pd.concat([results_df, pd.DataFrame([summary_row, std_row])], ignore_index=True)
        # 最终结果和总结的 DataFrame
        results_df.to_csv(results_csv_path, index=False, float_format='%.4f')
        print(f"\nAll periodic and final test results saved to: {results_csv_path}")
        print("\n=== Final Results Across Seeds (Test Set - Regression + Acc-7) ===")
        print(f"Mean Test Acc-7 (Final Epoch): {acc7_mean:.4f} \u00B1 {acc7_std:.4f}")
        print(f"Mean Test Correlation (r): {corr_mean:.4f} \u00B1 {corr_std:.4f}")
        print(f"Mean Test MSE: {mse_mean:.4f} \u00B1 {mse_std:.4f}")
        print(f"Mean Test MAE: {mae_mean:.4f} \u00B1 {mae_std:.4f}")
    else:
        print("\nNo test results to save.")


if __name__ == "__main__":
    main()
