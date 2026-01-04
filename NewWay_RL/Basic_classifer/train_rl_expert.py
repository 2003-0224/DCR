# 增量训练版强化学习脚本

import os
import math
import random
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

from src.feature_dataset import MELDDataset
from src.utils import set_seed
from unimodal_models import TextClassifier, AudioClassifier, VideoClassifier
from rl_agent import RLAgent, ValueNet


def evaluate_agent(agent, experts, data_loader, device, num_labels, num_actions):
    was_training = agent.training
    agent.eval()

    all_preds = []
    all_labels = []
    action_counts = torch.zeros(num_actions, dtype=torch.long)

    with torch.no_grad():
        for batch in data_loader:
            text_input = batch['text'].to(device)
            audio_input = batch['audio'].to(device)
            video_input = batch['video'].to(device)
            labels = batch['label'].squeeze(-1).to(device)

            text_logits, text_features = experts['T'](text_input)
            audio_logits, audio_features = experts['A'](audio_input)
            video_logits, video_features = experts['V'](video_input)

            agent_inputs = {
                'T': text_input,
                'A': audio_input,
                'V': video_input,
            }

            action_logits = agent(agent_inputs)
            actions = torch.argmax(action_logits, dim=1)
            action_counts += torch.bincount(actions.cpu(), minlength=num_actions)

            expert_preds = torch.stack([
                torch.argmax(text_logits, dim=1),
                torch.argmax(audio_logits, dim=1),
                torch.argmax(video_logits, dim=1)
            ], dim=1)
            batch_indices = torch.arange(actions.size(0), device=actions.device)
            chosen_preds = expert_preds[batch_indices, actions]

            all_preds.append(chosen_preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if was_training:
        agent.train()

    if all_preds:
        preds_tensor = torch.cat(all_preds)
        labels_tensor = torch.cat(all_labels)
        preds_np = preds_tensor.numpy()
        labels_np = labels_tensor.numpy()
        accuracy = float((preds_np == labels_np).mean())
        try:
            weighted_f1 = f1_score(labels_np, preds_np, average='weighted')
        except ValueError:
            weighted_f1 = 0.0
        conf_mat = confusion_matrix(labels_np, preds_np, labels=list(range(num_labels)))
        class_totals = conf_mat.sum(axis=1)
        class_correct = conf_mat.diagonal()
        class_acc_dict = {
            f"class_{i}_acc": (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0)
            for i in range(num_labels)
        }
    else:
        accuracy = 0.0
        weighted_f1 = 0.0
        class_acc_dict = {f"class_{i}_acc": 0.0 for i in range(num_labels)}

    total_actions = max(action_counts.sum().item(), 1)
    action_distribution = ", ".join([
        f"{idx}:{(count.item() / total_actions) * 100:.1f}%" for idx, count in enumerate(action_counts)
    ])

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "class_acc": class_acc_dict,
        "action_dist": action_distribution
    }


def concat_inputs(modality, inputs_a, inputs_b):
    return torch.cat([inputs_a, inputs_b], dim=0)


def sample_from_buffer(buffer, modality, sample_size, device):
    if sample_size <= 0 or not buffer:
        return None, None
    sample_size = min(sample_size, len(buffer))
    samples = random.sample(buffer, sample_size)
    inputs = torch.stack([item['features'] for item in samples]).to(device)
    labels = torch.stack([item['label'] for item in samples]).to(device)
    return inputs, labels


def append_modal_samples(buffer, modality, inputs, labels):
    if inputs is None:
        return
    batch = labels.size(0)
    for i in range(batch):
        buffer.append({
            'features': inputs[i].detach().cpu(),
            'label': labels[i].detach().cpu().long(),
        })


def update_single_expert(
        modality,
        expert,
        optimizer,
        indices,
        text_input,
        audio_input,
        video_input,
        labels,
        grad_clip,
        replay_buffers,
        replay_ratio,
        device,
):
    if indices.numel() == 0:
        return None, 0, 0

    expert.train()
    optimizer.zero_grad()

    if modality == 'T':
        new_inputs = text_input.index_select(0, indices)
    elif modality == 'A':
        new_inputs = audio_input.index_select(0, indices)
    else:
        new_inputs = video_input.index_select(0, indices)

    new_labels = labels.index_select(0, indices)

    combined_inputs = new_inputs
    combined_labels = new_labels

    replay_inputs = None
    replay_labels = None
    if replay_ratio > 0 and replay_buffers[modality]:
        replay_count = max(1, int(new_labels.size(0) * replay_ratio))
        replay_inputs, replay_labels = sample_from_buffer(replay_buffers[modality], modality, replay_count, device)
        if replay_inputs is not None:
            combined_inputs = concat_inputs(modality, new_inputs, replay_inputs)
            combined_labels = torch.cat([new_labels, replay_labels], dim=0)

    logits, _ = expert(combined_inputs)

    loss = F.cross_entropy(logits, combined_labels)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(expert.parameters(), grad_clip)
    optimizer.step()
    expert.eval()

    append_modal_samples(replay_buffers[modality], modality, new_inputs, new_labels)

    replay_used = replay_labels.size(0) if replay_labels is not None else 0
    return loss.item(), new_labels.size(0), replay_used


def main():
    seed = 42
    num_labels = 7
    num_epochs = 50
    batch_size = 32
    agent_lr = 1e-4
    value_lr = 1e-4
    expert_lr = 5e-5
    hidden_dim = 512
    agent_embed_dim = 256
    value_hidden_dim = 256
    value_coef = 0.5
    entropy_coef = 0.01
    expert_grad_clip = 1.0
    replay_buffer_size = 2048
    replay_ratio = 1.0
    num_actions = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    expert_model_paths = {
        'T': '/data/yuyangchen/checkpoints/unimodal_experts/unimodal_t_best_f1_0.6674_seed_42.pth',
        'A': '/data/yuyangchen/checkpoints/unimodal_experts/unimodal_a_best_f1_0.5565_seed_42.pth',
        'V': '/data/yuyangchen/checkpoints/unimodal_experts/unimodal_v_best_f1_0.4021_seed_42.pth'
    }

    set_seed(seed)

    print("Loading expert models...")

    replay_buffers = {
        'T': deque(maxlen=replay_buffer_size),
        'A': deque(maxlen=replay_buffer_size),
        'V': deque(maxlen=replay_buffer_size)
    }

    data_paths = {
        'train': {
            'text': '/data/yuyangchen/data/MELD/text_features/train_text_features.npz',
            'audio': '/data/yuyangchen/data/MELD/audio_features/train_audio_features_processed.npz',
            'video': '/data/yuyangchen/data/MELD/face_features/train_video_features_processed.npz',
        },
        'test': {
            'text': '/data/yuyangchen/data/MELD/text_features/test_text_features.npz',
            'audio': '/data/yuyangchen/data/MELD/audio_features/test_audio_features_processed.npz',
            'video': '/data/yuyangchen/data/MELD/face_features/test_video_features_processed.npz',
        }
    }

    try:
        train_dataset = MELDDataset(
            data_paths['train']['text'],
            data_paths['train']['audio'],
            data_paths['train']['video'],
            modalities=['T', 'A', 'V'],
            feature_type='sequence_features'
        )
    except KeyError:
        train_dataset = MELDDataset(
            data_paths['train']['text'],
            data_paths['train']['audio'],
            data_paths['train']['video'],
            modalities=['T', 'A', 'V'],
            feature_type='pooled_features'
        )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    try:
        test_dataset = MELDDataset(
            data_paths['test']['text'],
            data_paths['test']['audio'],
            data_paths['test']['video'],
            modalities=['T', 'A', 'V'],
            split='test',
            feature_type='sequence_features'
        )
    except KeyError:
        test_dataset = MELDDataset(
            data_paths['test']['text'],
            data_paths['test']['audio'],
            data_paths['test']['video'],
            modalities=['T', 'A', 'V'],
            split='test',
            feature_type='pooled_features'
        )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    text_feat_dim = train_dataset.aligned_data['text'].shape[-1]
    text_seq_len = train_dataset.aligned_data['text'].shape[1] if train_dataset.aligned_data['text'].ndim == 3 else 1
    audio_feat_dim = train_dataset.aligned_data['audio'].shape[-1]
    video_feat_dim = train_dataset.aligned_data['video'].shape[-1]
    modality_dims = {'T': text_feat_dim, 'A': audio_feat_dim, 'V': video_feat_dim}

    text_expert = TextClassifier(
        hidden_dim=hidden_dim,
        num_classes=num_labels,
        use_precomputed=True,
        input_dim=text_feat_dim,
        target_seq_len=text_seq_len,
    ).to(device)
    text_state = torch.load(expert_model_paths['T'])
    text_expert.load_state_dict(text_state, strict=False)
    audio_expert = AudioClassifier(hidden_dim=hidden_dim, num_classes=num_labels).to(device)
    audio_expert.load_state_dict(torch.load(expert_model_paths['A']))
    video_expert = VideoClassifier(hidden_dim=hidden_dim, num_classes=num_labels).to(device)
    video_expert.load_state_dict(torch.load(expert_model_paths['V']))

    experts = {'T': text_expert, 'A': audio_expert, 'V': video_expert}
    for expert in experts.values():
        expert.eval()

    expert_optimizers = {
        'T': torch.optim.Adam(text_expert.parameters(), lr=expert_lr),
        'A': torch.optim.Adam(audio_expert.parameters(), lr=expert_lr),
        'V': torch.optim.Adam(video_expert.parameters(), lr=expert_lr)
    }

    agent = RLAgent(modality_dims=modality_dims, num_actions=num_actions, num_classes=num_labels,
                    embed_dim=agent_embed_dim).to(device)
    value_net = ValueNet(input_dim=agent_embed_dim, hidden_dim=value_hidden_dim).to(device)
    agent_optimizer = torch.optim.Adam(agent.parameters(), lr=agent_lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr)

    save_dir = "/data/yuyangchen/checkpoints/rl_agent"
    os.makedirs(save_dir, exist_ok=True)
    metrics_dir = os.path.join(save_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_log = []

    print("Starting RL with expert fine-tuning...")
    for epoch in range(num_epochs):
        total_rewards = 0
        total_loss = 0
        total_value_loss = 0
        total_entropy = 0
        epoch_preds = []
        epoch_labels = []
        epoch_actions = []
        expert_update_losses = defaultdict(list)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            text_input = batch['text'].to(device)
            audio_input = batch['audio'].to(device)
            video_input = batch['video'].to(device)
            labels = batch['label'].squeeze(-1).to(device)

            with torch.no_grad():
                text_logits, text_features = experts['T'](text_input)
                audio_logits, audio_features = experts['A'](audio_input)
                video_logits, video_features = experts['V'](video_input)

                text_probs = F.softmax(text_logits, dim=1)
                audio_probs = F.softmax(audio_logits, dim=1)
                video_probs = F.softmax(video_logits, dim=1)

                agent_inputs = {
                    'T': text_input,
                    'A': audio_input,
                    'V': video_input,
                }

            action_logits, agent_repr = agent(agent_inputs, return_repr=True)
            action_dist = Categorical(logits=action_logits)
            actions = action_dist.sample()

            with torch.no_grad():
                preds_t = torch.argmax(text_logits, dim=1)
                preds_a = torch.argmax(audio_logits, dim=1)
                preds_v = torch.argmax(video_logits, dim=1)

                all_preds = torch.stack([preds_t, preds_a, preds_v], dim=1)
                batch_indices = torch.arange(actions.size(0), device=actions.device)
                chosen_preds = all_preds[batch_indices, actions]
                rewards = (chosen_preds == labels).float() * 2 - 1

            values = value_net(agent_repr)
            log_probs = action_dist.log_prob(actions)

            with torch.no_grad():
                advantages = rewards - values.detach()
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                adv_std_val = adv_std.item()
                if math.isnan(adv_std_val) or adv_std_val < 1e-6:
                    advantages = advantages - adv_mean
                else:
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, rewards)
            entropy = action_dist.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            agent_optimizer.zero_grad()
            value_optimizer.zero_grad()
            loss.backward()
            agent_optimizer.step()
            value_optimizer.step()

            total_rewards += rewards.mean().item()
            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

            epoch_preds.append(chosen_preds.detach().cpu())
            epoch_labels.append(labels.detach().cpu())
            epoch_actions.append(actions.detach().cpu())

            for idx, modality in enumerate(['T', 'A', 'V']):
                modality_indices = (actions == idx).nonzero(as_tuple=True)[0]
                if modality_indices.numel() == 0:
                    continue
                update_stats = update_single_expert(
                    modality,
                    experts[modality],
                    expert_optimizers[modality],
                    modality_indices,
                    text_input,
                    audio_input,
                    video_input,
                    labels,
                    expert_grad_clip,
                    replay_buffers,
                    replay_ratio,
                    device,
                )
                if update_stats[0] is not None:
                    expert_update_losses[modality].append(update_stats)

        avg_reward = total_rewards / len(train_loader)
        avg_loss = total_loss / len(train_loader)
        avg_value_loss = total_value_loss / len(train_loader)
        avg_entropy = total_entropy / len(train_loader)

        epoch_preds_tensor = torch.cat(epoch_preds)
        epoch_labels_tensor = torch.cat(epoch_labels)
        epoch_actions_tensor = torch.cat(epoch_actions)
        preds_np = epoch_preds_tensor.numpy()
        labels_np = epoch_labels_tensor.numpy()
        epoch_accuracy = float((preds_np == labels_np).mean()) if preds_np.size > 0 else 0.0
        try:
            epoch_weighted_f1 = f1_score(labels_np, preds_np, average='weighted') if preds_np.size > 0 else 0.0
        except ValueError:
            epoch_weighted_f1 = 0.0

        if preds_np.size > 0:
            conf_mat = confusion_matrix(labels_np, preds_np, labels=list(range(num_labels)))
            class_totals = conf_mat.sum(axis=1)
            class_correct = conf_mat.diagonal()
            class_acc_dict = {
                f"class_{i}_acc": (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0)
                for i in range(num_labels)
            }
        else:
            class_acc_dict = {f"class_{i}_acc": 0.0 for i in range(num_labels)}

        action_counts = torch.bincount(epoch_actions_tensor, minlength=num_actions)
        total_actions_num = max(epoch_actions_tensor.numel(), 1)
        action_distribution = ", ".join(
            [f"{idx}:{(count.item() / total_actions_num) * 100:.1f}%" for idx, count in enumerate(action_counts)]
        )

        test_metrics = evaluate_agent(agent, experts, test_loader, device, num_labels, num_actions)

        expert_update_summary = {}
        for modality, stats in expert_update_losses.items():
            if not stats:
                continue
            losses = [item[0] for item in stats if item[0] is not None]
            new_counts = sum(item[1] for item in stats)
            replay_counts = sum(item[2] for item in stats)
            mean_loss = float(np.mean(losses)) if losses else 0.0
            avg_replay = replay_counts / max(len(stats), 1)
            expert_update_summary[modality] = {
                'loss': mean_loss,
                'avg_replay': avg_replay,
            }

        print(
            f"Epoch {epoch + 1} | Train Reward: {avg_reward:.4f} | Train Loss: {avg_loss:.4f} "
            f"| Train ValLoss: {avg_value_loss:.4f} | Train Entropy: {avg_entropy:.4f} "
            f"| Train Acc: {epoch_accuracy * 100:.2f}% | Train W-F1: {epoch_weighted_f1:.4f} "
            f"| Train Action Dist: [{action_distribution}]"
        )
        print("  Train Class Accuracies: " + ", ".join([f"{k}:{v * 100:.2f}%" for k, v in class_acc_dict.items()]))
        print(
            f"  Test  | Acc: {test_metrics['accuracy'] * 100:.2f}% | W-F1: {test_metrics['weighted_f1']:.4f} "
            f"| Action Dist: [{test_metrics['action_dist']}]"
        )
        print("  Test Class Accuracies: " + ", ".join(
            [f"{k}:{v * 100:.2f}%" for k, v in test_metrics['class_acc'].items()]))
        if expert_update_summary:
            print(
                "  Expert Update Summary: "
                + ", ".join(
                    [
                        f"{m}-loss:{info['loss']:.4f}|replay:{info['avg_replay']:.2f}"
                        for m, info in expert_update_summary.items()
                    ]
                )
            )

        metrics_entry = {
            "epoch": epoch + 1,
            "train_avg_reward": avg_reward,
            "train_avg_loss": avg_loss,
            "train_avg_value_loss": avg_value_loss,
            "train_avg_entropy": avg_entropy,
            "train_accuracy": epoch_accuracy,
            "train_weighted_f1": epoch_weighted_f1,
            "train_action_dist": action_distribution,
            "test_accuracy": test_metrics['accuracy'],
            "test_weighted_f1": test_metrics['weighted_f1'],
            "test_action_dist": test_metrics['action_dist']
        }
        metrics_entry.update({f"train_{k}": v for k, v in class_acc_dict.items()})
        metrics_entry.update({f"test_{k}": v for k, v in test_metrics['class_acc'].items()})
        for m, info in expert_update_summary.items():
            metrics_entry[f"expert_update_{m}_loss"] = info['loss']
            metrics_entry[f"expert_update_{m}_avg_replay"] = info['avg_replay']
        metrics_log.append(metrics_entry)

    agent_path = os.path.join(save_dir, "agent_expert_final.pth")
    torch.save(agent.state_dict(), agent_path)
    value_path = os.path.join(save_dir, "value_net_expert_final.pth")
    torch.save(value_net.state_dict(), value_path)
    torch.save(text_expert.state_dict(), os.path.join(save_dir, "text_expert_incremental.pth"))
    torch.save(audio_expert.state_dict(), os.path.join(save_dir, "audio_expert_incremental.pth"))
    torch.save(video_expert.state_dict(), os.path.join(save_dir, "video_expert_incremental.pth"))
    print(f"\nSaved agent to: {agent_path}")
    print(f"Saved value network to: {value_path}")

    metrics_df = pd.DataFrame(metrics_log)
    metrics_csv_path = os.path.join(metrics_dir, "train_metrics_expert.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"保存训练指标到: {metrics_csv_path}")

    optimization_plan = [
        "1. 为专家增加回放缓冲/混合旧样本，缓解灾难性遗忘",
        "2. 引入动作置信度阈值或奖励调整，过滤噪声样本",
        "3. 采用多步优势或 PPO 等更稳健算法，以适应专家参数持续变化",
        "4. 周期性在验证集评估专家乐性能量，触发回滚或再训练",
        "5. 探索层级策略：先决策模态，再选择是否触发专家更新"
    ]
    print("后续优化计划:")
    for item in optimization_plan:
        print(" - " + item)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    main()
