# 回归任务
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch.nn as nn
from Roberta_model import EmotionClassifier
from data_loader import MOSIDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import random
import os
import numpy as np
from collections import defaultdict
import shutil
from scipy.stats import pearsonr


# --- V-Score 到 ACC-7 分箱函数 ---
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


# 改进的设置随机种子函数 (保持不变)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 训练函数 (保持不变，使用 MSE Loss)
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        # 使用 MSE Loss 进行回归
        loss = torch.nn.functional.mse_loss(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 评估函数 - 新增 ACC-7 指标
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            # 使用 MSE Loss
            loss = torch.nn.functional.mse_loss(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / len(data_loader)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r, _ = pearsonr(all_labels, all_preds)
    true_acc7_labels = np.array([convert_to_acc7_label(l) for l in all_labels])
    pred_acc7_labels = np.array([convert_to_acc7_label(p) for p in all_preds])
    acc_7 = accuracy_score(true_acc7_labels, pred_acc7_labels)
    # 返回 MSE, MAE, Corr, ACC-7
    return avg_loss, mse, mae, r, acc_7


# 主函数
def main():
    # 参数设置
    seeds = [41, 42, 43, 44, 45]
    num_labels = 1
    num_epochs = 10
    batch_size = 16
    max_seq_length = 160
    context_len = 0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    version = 1
    use_lora = False
    use_adapters = False
    # 文件路径设置
    mosi_data_path = "/data/home/chenqian/CMU-MOSEI/label_utf8.csv"
    local_roberta_path = "/data/home/chenqian/Roberta-large/Roberta-large"
    results_csv_path = "/data/home/chenqian/regression_models/text_model/regression_acc7_results.csv"
    # === 模型保存路径 ===
    model_save_dir = "/data/home/chenqian/regression_models/text_model"
    os.makedirs(model_save_dir, exist_ok=True)
    if version == 5_1:
        use_lora = True
    if version == 6_1:
        use_adapters = True
    print(
        f"\nparam set: context_len (Monologue): {context_len}  max_seq_length: {max_seq_length}   device: {device}  version:{version}   use_lora:{use_lora}    use_adapters:{use_adapters}\n")
    # 存储每个种子的测试集结果
    all_test_results = []
    # === 全局最佳模型跟踪变量 ===
    global_best_valid_acc_7 = -1.0
    global_best_model_state_dict = None
    global_best_seed = None
    print(f"Loading RoBERTa Tokenizer from local path: {local_roberta_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_roberta_path)
    for seed in seeds:
        print(f"\n=== Training with Seed {seed} (Regression + Acc-7) ===\n")
        set_seed(seed)
        # 加载模型
        roberta_model = AutoModel.from_pretrained(local_roberta_path)
        best_model_state_dict_for_seed = None
        best_valid_corr_for_seed = -1.0
        best_valid_acc_7_for_seed = 0.0
        best_valid_mse_for_seed = float('inf')
        model = EmotionClassifier(roberta_model, num_labels, use_lora, use_adapters)
        model.to(device)
        # 2. 数据集加载
        print("Loading datasets...")
        train_dataset = MOSIDataset(mosi_data_path, tokenizer, split_mode='train', max_seq_length=max_seq_length)
        valid_dataset = MOSIDataset(mosi_data_path, tokenizer, split_mode='valid', max_seq_length=max_seq_length)
        test_dataset = MOSIDataset(mosi_data_path, tokenizer, split_mode='test', max_seq_length=max_seq_length)
        train_loader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, num_workers=0, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, num_workers=0, batch_size=batch_size, shuffle=False)
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
        # 3. 训练和验证
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, device)
            valid_loss, valid_mse, valid_mae, valid_corr, valid_acc_7 = evaluate(model, valid_loader, device)
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss (MSE): {train_loss:.4f}")
            print(
                f"Valid Loss (MSE): {valid_loss:.4f}, Valid MSE: {valid_mse:.4f}, Valid MAE: {valid_mae:.4f}, Valid Corr: {valid_corr:.4f}, Valid Acc-7: {valid_acc_7:.4f}")
            # 检查当前种子下的最佳模型 (基于相关系数)
            if valid_acc_7 > best_valid_acc_7_for_seed:
                best_valid_corr_for_seed = valid_corr
                best_valid_mse_for_seed = valid_mse
                best_valid_acc_7_for_seed = valid_acc_7
                best_model_state_dict_for_seed = model.state_dict()
                print(f"--> Seed {seed} new best valid ACC found: {best_valid_acc_7_for_seed:.4f} at epoch {epoch + 1}")
                # 检查是否为全局最佳模型
                if valid_acc_7 > global_best_valid_acc_7:
                    global_best_valid_acc_7 = valid_acc_7
                    global_best_seed = seed
                    global_best_model_state_dict = best_model_state_dict_for_seed
                    print(f"--> GLOBAL BEST UPDATED: Acc={global_best_valid_acc_7:.4f}, Seed={seed}, Epoch={epoch + 1}")
        # 测试集评估 (使用当前种子下的最佳模型)
        print(f"\n--- Evaluating Best Model (Seed {seed}) on TEST Set (Regression + Acc-7) ---")

        if best_model_state_dict_for_seed is not None:
            roberta_model_test = AutoModel.from_pretrained(local_roberta_path)
            best_model = EmotionClassifier(roberta_model_test, num_labels, use_lora, use_adapters)
            best_model.load_state_dict(best_model_state_dict_for_seed)
            best_model.to(device)
            test_loss, test_mse, test_mae, test_corr, test_acc_7 = evaluate(
                best_model, test_loader, device)
            print(
                f"Seed {seed} - TEST Loss (MSE): {test_loss:.4f}, TEST MSE: {test_mse:.4f}, TEST MAE: {test_mae:.4f}, TEST Corr: {test_corr:.4f}, TEST Acc-7: {test_acc_7:.4f}")
            model_path_this_seed = os.path.join(
                model_save_dir,
                f"seed_{seed}_testAcc7_{test_acc_7:.4f}.pth"
            )
            torch.save(best_model_state_dict_for_seed, model_path_this_seed)
            # 存储结果
            result_entry = {
                "Seed": seed,
                "Best_Valid_Corr": best_valid_corr_for_seed,
                "Best_Valid_MSE": best_valid_mse_for_seed,
                "Best_Valid_Acc_7": best_valid_acc_7_for_seed,
                "Test_MSE": test_mse,
                "Test_MAE": test_mae,
                "Test_Corr": test_corr,
                "Test_Acc_7": test_acc_7,
            }
            all_test_results.append(result_entry)
        else:
            print(f"Warning: No best model state saved for seed {seed}. Skipping test evaluation.")

    # 保存全局最佳模型参数
    if global_best_model_state_dict is not None:
        final_path = os.path.join(model_save_dir,
                                  f"stage1_best_on_valid_acc7_{global_best_valid_acc_7:.4f}_seed_{global_best_seed}.pth")
        torch.save(global_best_model_state_dict, final_path)
        print(f"\nSaved Global Best (Regression) parameters to: {final_path}")
        print(f"Global Best Model (from Validation Set): Acc={global_best_valid_acc_7:.4f}, Seed={global_best_seed}")

    # 结果保存到 CSV
    if all_test_results:
        results_df = pd.DataFrame(all_test_results)
        # 计算均值和方差
        corr_mean = results_df['Test_Corr'].mean()
        corr_std = results_df['Test_Corr'].std()
        mse_mean = results_df['Test_MSE'].mean()
        mse_std = results_df['Test_MSE'].std()
        mae_mean = results_df['Test_MAE'].mean()
        mae_std = results_df['Test_MAE'].std()
        acc_7_mean = results_df['Test_Acc_7'].mean()
        acc_7_std = results_df['Test_Acc_7'].std()
        summary_row = {"Seed": "Mean", "Test_Corr": corr_mean, "Test_MSE": mse_mean, "Test_MAE": mae_mean,
                       "Test_Acc_7": acc_7_mean}
        std_row = {"Seed": "Std_Dev", "Test_Corr": corr_std, "Test_MSE": mse_std, "Test_MAE": mae_std,
                   "Test_Acc_7": acc_7_std}
        for col in results_df.columns:
            if col.startswith("Best_Valid_"):
                summary_row[col] = results_df[col].mean()
                std_row[col] = results_df[col].std()
        results_df = pd.concat([results_df, pd.DataFrame([summary_row, std_row])], ignore_index=True)
        results_df.to_csv(results_csv_path, index=False, float_format='%.4f')
        print(f"\n All seed regression test results saved to: {results_csv_path}")
        print("\n=== Final Results Across Seeds (Test Set - Regression + Acc-7) ===")
        print(f"Mean Test Correlation (r): {corr_mean:.4f} \u00B1 {corr_std:.4f}")
        print(f"Mean Test MSE: {mse_mean:.4f} \u00B1 {mse_std:.4f}")
        print(f"Mean Test MAE: {mae_mean:.4f} \u00B1 {mae_std:.4f}")
        print(f"Mean Test Acc-7: {acc_7_mean:.4f} \u00B1 {acc_7_std:.4f}")
    else:
        print("\n No test results to save.")


if __name__ == "__main__":
    main()
