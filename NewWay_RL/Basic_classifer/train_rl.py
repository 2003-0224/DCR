# import math
# import os
# import random
#
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from sklearn.metrics import f1_score, confusion_matrix
# from torch.distributions import Categorical
# from torch.utils.data import DataLoader
# from tqdm import tqdm
#
# from src.feature_dataset import MELDDataset
# from src.utils import set_seed
# from unimodal_models import TextClassifier, AudioClassifier, VideoClassifier
# from rl_agent import RLAgent, ValueNet
#
#
# def normalize_sample_name(name):
#     if isinstance(name, bytes):
#         name = name.decode("utf-8")
#     name = str(name).replace("\\", "/")
#     parts = name.split("/")
#     if len(parts) >= 3 and parts[-3].isdigit() and parts[-2].isdigit():
#         return f"dia{parts[-3]}_utt{parts[-2]}"
#     base = parts[-1]
#     return os.path.splitext(base)[0]
#
#
# def load_general_feature_dict(npz_path, prefer_key='sequence_features'):
#     data = np.load(npz_path, allow_pickle=True)
#     available_keys = data.files
#     if prefer_key in available_keys:
#         key = prefer_key
#     elif 'sequence_features' in available_keys:
#         key = 'sequence_features'
#     elif 'pooled_features' in available_keys:
#         key = 'pooled_features'
#     else:
#         raise KeyError(f"No suitable feature key found in {npz_path}. Available keys: {available_keys}")
#
#     names = [normalize_sample_name(n) for n in data['sample_names']]
#     feats = data[key]
#     mapping = {name: feats[i].astype(np.float32) for i, name in enumerate(names)}
#     feature_shape = feats.shape[1:]
#     return mapping, feature_shape
#
#
# def build_agent_inputs(
#     sample_names,
#     feature_store,
#     default_features,
#     device,
#     expert_features,
#     text_logits=None,
#     audio_logits=None,
#     video_logits=None,
#     multimodal_store=None,
# ):
#     required_modalities = ['T', 'A', 'V']
#     for key in required_modalities:
#         if key not in expert_features:
#             raise KeyError(f"Missing expert feature for modality '{key}' in expert_features")
#
#     feature_inputs = {}
#     logits_inputs = {}
#
#     for modality in required_modalities:
#         feats = []
#         masks = []
#         mapping = feature_store[modality]
#         default = default_features[modality]
#         for raw_name in sample_names:
#             name = normalize_sample_name(raw_name)
#             feature = mapping.get(name)
#             if feature is None:
#                 feats.append(default)
#                 masks.append(0.0)
#             else:
#                 feats.append(feature)
#                 masks.append(1.0)
#         arr = np.stack(feats, axis=0).astype(np.float32)
#         mask_arr = np.array(masks, dtype=np.float32).reshape(-1, 1)
#         feature_inputs[modality] = {
#             'memory': torch.from_numpy(arr).to(device),
#             'mask': torch.from_numpy(mask_arr).to(device),
#             'query': expert_features[modality].detach().to(device).float(),
#         }
#
#     if video_logits is not None:
#         logits_inputs['V'] = video_logits.detach().to(device)
#     if text_logits is not None:
#         logits_inputs['T'] = text_logits.detach().to(device)
#     if audio_logits is not None:
#         logits_inputs['A'] = audio_logits.detach().to(device)
#
#     if multimodal_store is not None and text_logits is not None:
#         mm_vals = []
#         for i, raw_name in enumerate(sample_names):
#             key = normalize_sample_name(raw_name)
#             mm_logit = multimodal_store['logits'].get(key)
#             if mm_logit is None:
#                 mm_logit = text_logits[i].detach().cpu().numpy()
#             mm_vals.append(mm_logit)
#         mm_tensor = torch.tensor(mm_vals, device=device, dtype=text_logits.dtype)
#         logits_inputs['M'] = mm_tensor
#     elif text_logits is not None:
#         logits_inputs['M'] = text_logits.detach().to(device)
#
#     return feature_inputs, logits_inputs
#
#
# def augment_agent_inputs(
#     agent_inputs,
#     logit_inputs,
#     drop_prob_single=0.0,
#     drop_prob_double=0.0,
#     noise_std=0.0,
# ):
#     if logit_inputs is None:
#         logit_inputs = {}
#
#     modalities = list(agent_inputs.keys())
#     if not modalities:
#         return agent_inputs, logit_inputs
#
#     batch_size = agent_inputs[modalities[0]]['memory'].size(0)
#     device = agent_inputs[modalities[0]]['memory'].device
#
#     if drop_prob_single > 0.0 or drop_prob_double > 0.0:
#         rand_vals = torch.rand(batch_size, device=device)
#         drop_mask = torch.zeros(batch_size, len(modalities), dtype=torch.bool, device=device)
#         cumulative_double = drop_prob_double
#         cumulative_single = drop_prob_double + drop_prob_single
#         for i in range(batch_size):
#             drop_count = 0
#             if rand_vals[i] < cumulative_double:
#                 drop_count = min(2, len(modalities))
#             elif rand_vals[i] < cumulative_single:
#                 drop_count = min(1, len(modalities))
#             if drop_count > 0:
#                 selected = random.sample(modalities, k=drop_count)
#                 for mod in selected:
#                     drop_mask[i, modalities.index(mod)] = True
#     else:
#         drop_mask = None
#
#     for mod_idx, modality in enumerate(modalities):
#         entry = agent_inputs[modality]
#         mem = entry['memory']
#         mask = entry['mask']
#         query = entry['query']
#
#         if noise_std > 0.0:
#             mask_expand = mask
#             for _ in range(mem.dim() - mask.dim()):
#                 mask_expand = mask_expand.unsqueeze(-1)
#             mem += torch.randn_like(mem) * noise_std * mask_expand
#             query += torch.randn_like(query) * noise_std * mask
#
#         if drop_mask is not None:
#             to_drop = drop_mask[:, mod_idx]
#             if to_drop.any():
#                 mem[to_drop] = 0.0
#                 mask[to_drop] = 0.0
#                 query[to_drop] = 0.0
#                 if modality in logit_inputs:
#                     logit_inputs[modality][to_drop] = 0.0
#
#     return agent_inputs, logit_inputs
#
#
# def load_multimodal_predictions(train_npz, test_npz):
#     multimodal = {}
#     for split, path in [('train', train_npz), ('test', test_npz)]:
#         data = np.load(path, allow_pickle=True)
#         names = [normalize_sample_name(n) for n in data['sample_names']]
#         preds = data['preds']
#         logits = data['logits']
#         multimodal[split] = {
#             'preds': {names[i]: preds[i] for i in range(len(names))},
#             'logits': {names[i]: logits[i] for i in range(len(names))},
#         }
#     return multimodal
#
#
# def evaluate_agent(agent, experts, data_loader, device, num_labels, num_actions, agent_feature_store, default_features, multimodal_store=None):
#     was_training = agent.training
#     agent.eval()
#
#     all_preds = []
#     all_labels = []
#     action_counts = torch.zeros(num_actions, dtype=torch.long)
#
#     with torch.no_grad():
#         for batch in data_loader:
#             text_input = batch['text'].to(device)
#             audio_input = batch['audio'].to(device)
#             video_input = batch['video'].to(device)
#             labels = batch['label'].squeeze(-1).to(device)
#             sample_names = batch['sample_name']
#             if isinstance(sample_names, torch.Tensor):
#                 sample_names = sample_names.tolist()
#             if isinstance(sample_names, tuple):
#                 sample_names = list(sample_names)
#             if isinstance(sample_names, (str, bytes)):
#                 sample_names = [sample_names]
#
#             text_logits, text_features = experts['T'](text_input)
#             audio_logits, audio_features = experts['A'](audio_input)
#             video_logits, video_features = experts['V'](video_input)
#
#             expert_queries = {
#                 'T': text_features,
#                 'A': audio_features,
#                 'V': video_features,
#             }
#             agent_inputs, logit_inputs = build_agent_inputs(
#                 sample_names,
#                 agent_feature_store,
#                 default_features,
#                 device,
#                 expert_queries,
#                 text_logits=text_logits,
#                 audio_logits=audio_logits,
#                 video_logits=video_logits,
#                 multimodal_store=multimodal_store,
#             )
#
#             logit_payload = logit_inputs if logit_inputs else None
#             action_logits = agent(agent_inputs, logits=logit_payload)
#             actions = torch.argmax(action_logits, dim=1)
#
#             action_counts += torch.bincount(actions.cpu(), minlength=num_actions)
#
#             preds_t = torch.argmax(text_logits, dim=1)
#
#             if multimodal_store is not None:
#                 mm_preds = []
#                 for i, name in enumerate(sample_names):
#                     key = normalize_sample_name(name)
#                     mm_pred = multimodal_store['preds'].get(key)
#                     if mm_pred is None:
#                         mm_pred = preds_t[i].item()
#                     mm_preds.append(mm_pred)
#                 preds_m = torch.tensor(mm_preds, device=device, dtype=torch.long)
#             else:
#                 preds_m = preds_t.clone()
#
#             expert_preds = torch.stack([
#                 preds_t,
#                 preds_m,
#             ], dim=1)
#             batch_indices = torch.arange(actions.size(0), device=actions.device)
#             chosen_preds = expert_preds[batch_indices, actions]
#
#             all_preds.append(chosen_preds.detach().cpu())
#             all_labels.append(labels.detach().cpu())
#
#     if was_training:
#         agent.train()
#
#     if all_preds:
#         preds_tensor = torch.cat(all_preds)
#         labels_tensor = torch.cat(all_labels)
#         preds_np = preds_tensor.numpy()
#         labels_np = labels_tensor.numpy()
#         accuracy = float((preds_np == labels_np).mean())
#         try:
#             weighted_f1 = f1_score(labels_np, preds_np, average='weighted')
#         except ValueError:
#             weighted_f1 = 0.0
#         conf_mat = confusion_matrix(labels_np, preds_np, labels=list(range(num_labels)))
#         class_totals = conf_mat.sum(axis=1)
#         class_correct = conf_mat.diagonal()
#         class_acc_dict = {
#             f"class_{i}_acc": (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0)
#             for i in range(num_labels)
#         }
#     else:
#         accuracy = 0.0
#         weighted_f1 = 0.0
#         class_acc_dict = {f"class_{i}_acc": 0.0 for i in range(num_labels)}
#
#     total_actions = max(action_counts.sum().item(), 1)
#     action_distribution = ", ".join([
#         f"{idx}:{(count.item() / total_actions) * 100:.1f}%" for idx, count in enumerate(action_counts)
#     ])
#
#     return {
#         "accuracy": accuracy,
#         "weighted_f1": weighted_f1,
#         "class_acc": class_acc_dict,
#         "action_dist": action_distribution
#     }
#
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
#
# def main():
#     seeds = [41, 42, 43, 44, 45]
#     num_labels = 7
#     num_epochs = 300
#     batch_size = 64
#     agent_lr = 5e-6
#     value_lr = 1e-5
#     hidden_dim = 512
#     agent_embed_dim = 768
#     value_hidden_dim = 256
#     value_coef = 0.5
#     # entropy_coef = 0.01
#     entropy_coef = 0.03
#     num_actions = 2
#     general_pooling_mode = 'mean'
#     general_conv_kernel = 3
#     include_memory_tokens = False
#     modality_drop_prob_single = 0.2
#     modality_drop_prob_double = 0.05
#     agent_noise_std = 0.01
#     patience = 10
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     expert_model_paths = {
#         'T': '/data/home/chenqian/regression_models/unimodel_experts/unimodal_t_best_corr_0.7802_acc7_0.5192_mae_0.5333.pth',
#         'A': '/data/home/chenqian/regression_models/unimodel_experts/unimodal_a_best_corr_0.7105_acc7_0.4838_mae_ 0.6058.pth',
#         # 这里可以换成过拟合的模型，再次进行预测
#         'V': '/data/home/chenqian/regression_models/unimodel_experts/unimodal_v_best_corr_0.3914_acc7_0.4047_mae_ 0.8022.pth'
#     }
#
#     data_paths = {
#         'train': {
#             'text': '/data/home/chenqian/regression_models/text_model/train_text_features.npz',
#             'audio': '/data/home/chenqian/regression_models/audio_model/train_audio_features.npz',
#             'video': '/data/home/chenqian/regression_models/video_model/train_video_features.npz',
#         },
#         'test': {
#             'text': '/data/home/chenqian/regression_models/text_model/test_text_features.npz',
#             'audio': '/data/home/chenqian/regression_models/audio_model/test_audio_features.npz',
#             'video': '/data/home/chenqian/regression_models/video_model/test_video_features.npz',
#         }
#     }
#
#     print("Loading datasets and general features...")
#     train_dataset = MELDDataset(
#         data_paths['train']['text'],
#         data_paths['train']['audio'],
#         data_paths['train']['video'],
#         modalities=['T', 'A', 'V'],
#         feature_type='sequence_features'
#     )
#     test_dataset = MELDDataset(
#         data_paths['test']['text'],
#         data_paths['test']['audio'],
#         data_paths['test']['video'],
#         modalities=['T', 'A', 'V'],
#         split='test',
#         feature_type='sequence_features'
#     )
#     text_feat_dim = train_dataset.aligned_data['text'].shape[-1]
#     text_seq_len = train_dataset.aligned_data['text'].shape[1] if train_dataset.aligned_data['text'].ndim == 3 else 1
#     agent_feature_paths = {
#         'train': {
#             'T': '/data/home/chenqian/regression_models/text_model/train_text_features_general.npz',
#             'A': '/data/home/chenqian/regression_models/audio_model/train_audio_features_general.npz',
#             'V': '/data/home/chenqian/regression_models/video_model/train_video_features_general.npz',
#         },
#         'test': {
#             'T': '/data/home/chenqian/regression_models/text_model/test_text_features_general.npz',
#             'A': '/data/home/chenqian/regression_models/audio_model/test_audio_features_general.npz',
#             'V': '/data/home/chenqian/regression_models/video_model/test_video_features_general.npz',
#         }
#     }
#     agent_feature_store = {split: {} for split in ['train', 'test']}
#     agent_feature_defaults = {}
#     for split in ['train', 'test']:
#         for modality in ['T', 'A', 'V']:
#             mapping, feat_shape = load_general_feature_dict(agent_feature_paths[split][modality])
#             agent_feature_store[split][modality] = mapping
#             agent_feature_defaults.setdefault(modality, np.zeros(feat_shape, dtype=np.float32))
#     for modality in ['T', 'A', 'V']:
#         missing_train = [name for name in train_dataset.sample_names if normalize_sample_name(name) not in agent_feature_store['train'][modality]]
#         if missing_train:
#             print(f"Warning: {len(missing_train)} missing agent features for modality {modality} in training split. Example: {missing_train[:5]}")
#         missing_test = [name for name in test_dataset.sample_names if normalize_sample_name(name) not in agent_feature_store['test'][modality]]
#         if missing_test:
#             print(f"Warning: {len(missing_test)} missing agent features for modality {modality} in test split. Example: {missing_test[:5]}")
#     modality_dims = {}
#     sample_name_example = normalize_sample_name(train_dataset.sample_names[0])
#     for modality in ['T', 'A', 'V']:
#         sample_feat = agent_feature_store['train'][modality].get(sample_name_example)
#         if sample_feat is None:
#             sample_feat = agent_feature_defaults[modality]
#         modality_dims[modality] = sample_feat.shape[-1] if sample_feat.ndim >= 1 else 1
#
#     multimodal_train_npz = ('/data/home/chenqian/models/multi_model/multimodal_best_acc_0.5496_f1_0'
#                             '.5431_train_predictions.npz')
#     multimodal_test_npz = ('/data/home/chenqian/models/multi_model/multimodal_best_acc_0.5496_f1_0'
#                            '.5431_test_predictions.npz')
#     multimodal_available = os.path.exists(multimodal_train_npz) and os.path.exists(multimodal_test_npz)
#     multimodal_predictions = load_multimodal_predictions(multimodal_train_npz, multimodal_test_npz) if multimodal_available else None
#     # base_save_dir = "/data/yuyangchen/checkpoints/rl_agent"
#     # 修改entropy_coef
#     base_save_dir = "/data/home/chenqian/checkpoints/"
#     # 修改value_coef
#     # base_save_dir = "/data2/home/chenqian/checkpoints/valueCoef"
#     os.makedirs(base_save_dir, exist_ok=True)
#     best_records = []
#     for seed in seeds:
#         print(f"\n===== Training with seed {seed} =====")
#         set_seed(seed)
#
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#
#         text_expert = TextClassifier(
#             hidden_dim=hidden_dim,
#             num_classes=num_labels,
#             use_precomputed=True,
#             input_dim=text_feat_dim,
#             target_seq_len=text_seq_len,
#         ).to(device)
#         text_expert.load_state_dict(torch.load(expert_model_paths['T'], map_location=device), strict=False)
#
#         audio_expert = AudioClassifier(hidden_dim=hidden_dim, num_classes=num_labels).to(device)
#         audio_expert.load_state_dict(torch.load(expert_model_paths['A'], map_location=device))
#
#         video_expert = VideoClassifier(hidden_dim=hidden_dim, num_classes=num_labels).to(device)
#         video_expert.load_state_dict(torch.load(expert_model_paths['V'], map_location=device))
#
#         experts = {'T': text_expert, 'A': audio_expert, 'V': video_expert}
#         for expert in experts.values():
#             expert.eval()
#             for param in expert.parameters():
#                 param.requires_grad = False
#
#         sample_batch = next(iter(train_loader))
#         with torch.no_grad():
#             sample_text_feat = experts['T'](sample_batch['text'].to(device))[1]
#             sample_audio_feat = experts['A'](sample_batch['audio'].to(device))[1]
#             sample_video_feat = experts['V'](sample_batch['video'].to(device))[1]
#         query_dims = {
#             'T': sample_text_feat.shape[-1],
#             'A': sample_audio_feat.shape[-1],
#             'V': sample_video_feat.shape[-1],
#         }
#         del sample_batch
#         multimodal_train = multimodal_predictions['train'] if multimodal_predictions else None
#         multimodal_test = multimodal_predictions['test'] if multimodal_predictions else None
#
#         agent = RLAgent(
#             modality_dims=modality_dims,
#             num_actions=num_actions,
#             query_dims=query_dims,
#             embed_dim=agent_embed_dim,
#             general_pooling=general_pooling_mode,
#             conv_kernel_size=general_conv_kernel,
#             include_memory_tokens=include_memory_tokens,
#             num_classes=num_labels,
#         ).to(device)
#         value_net = ValueNet(input_dim=agent_embed_dim, hidden_dim=value_hidden_dim).to(device)
#         agent_optimizer = torch.optim.Adam(agent.parameters(), lr=agent_lr)
#         value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr)
#
#         save_dir_seed = os.path.join(base_save_dir, f"seed_{seed}")
#         os.makedirs(save_dir_seed, exist_ok=True)
#         metrics_dir = os.path.join(save_dir_seed, "metrics")
#         os.makedirs(metrics_dir, exist_ok=True)
#
#         metrics_log = []
#         best_wf1 = -1.0
#         best_record = {
#             'seed': seed,
#             'best_wf1': -1.0,
#             'best_acc': 0.0,
#             'best_action_dist': '',
#             'best_epoch': 0,
#         }
#
#         for epoch in range(num_epochs):
#             total_rewards = 0
#             total_loss = 0
#             total_value_loss = 0
#             total_entropy = 0
#             epoch_preds = []
#             epoch_labels = []
#             epoch_actions = []
#
#             for batch in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch+1}/{num_epochs}"):
#                 text_input = batch['text'].to(device)
#                 audio_input = batch['audio'].to(device)
#                 video_input = batch['video'].to(device)
#                 labels = batch['label'].squeeze(-1).to(device)
#                 sample_names = batch['sample_name']
#                 if isinstance(sample_names, torch.Tensor):
#                     sample_names = sample_names.tolist()
#                 if isinstance(sample_names, tuple):
#                     sample_names = list(sample_names)
#                 if isinstance(sample_names, (str, bytes)):
#                     sample_names = [sample_names]
#
#                 with torch.no_grad():
#                     text_logits, text_feat = experts['T'](text_input)
#                     audio_logits, audio_feat = experts['A'](audio_input)
#                     video_logits, video_feat = experts['V'](video_input)
#
#                     expert_queries = {
#                         'T': text_feat,
#                         'A': audio_feat,
#                         'V': video_feat,
#                     }
#                     agent_inputs, logit_inputs = build_agent_inputs(
#                         sample_names,
#                         agent_feature_store['train'],
#                         agent_feature_defaults,
#                         device,
#                         expert_queries,
#                         text_logits=text_logits,
#                         audio_logits=audio_logits,
#                         video_logits=video_logits,
#                         multimodal_store=multimodal_train,
#                     )
#                     agent_inputs, logit_inputs = augment_agent_inputs(
#                         agent_inputs,
#                         logit_inputs,
#                         drop_prob_single=modality_drop_prob_single,
#                         drop_prob_double=modality_drop_prob_double,
#                         noise_std=agent_noise_std,
#                     )
#
#                 logit_payload = logit_inputs if logit_inputs else None
#                 action_logits, agent_repr = agent(agent_inputs, logits=logit_payload, return_repr=True)
#                 action_dist = Categorical(logits=action_logits)
#                 actions = action_dist.sample()
#
#                 with torch.no_grad():
#                     preds_t = torch.argmax(text_logits, dim=1)
#
#                     if multimodal_train is not None:
#                         mm_vals = []
#                         for i, raw_name in enumerate(sample_names):
#                             key = normalize_sample_name(raw_name)
#                             mm_pred = multimodal_train['preds'].get(key)
#                             if mm_pred is None:
#                                 mm_pred = preds_t[i].item()
#                             mm_vals.append(mm_pred)
#                         preds_m = torch.tensor(mm_vals, device=device, dtype=torch.long)
#                     else:
#                         preds_m = preds_t.clone()
#
#                     probs_t = F.softmax(text_logits, dim=1)
#                     modality_probs = [probs_t]
#
#                     if multimodal_train is not None and 'logits' in multimodal_train:
#                         mm_logits = []
#                         for i, raw_name in enumerate(sample_names):
#                             key = normalize_sample_name(raw_name)
#                             mm_logit = multimodal_train['logits'].get(key)
#                             if mm_logit is None:
#                                 mm_logit = text_logits[i].detach().cpu().numpy()
#                             mm_logits.append(mm_logit)
#                         mm_tensor = torch.tensor(mm_logits, device=device, dtype=text_logits.dtype)
#                         probs_m = F.softmax(mm_tensor, dim=1)
#                     else:
#                         probs_m = F.one_hot(preds_m, num_classes=num_labels).float()
#
#                     modality_probs.append(probs_m)
#
#                     all_preds = torch.stack([preds_t, preds_m], dim=1)
#                     batch_indices = torch.arange(actions.size(0), device=actions.device)
#                     chosen_preds = all_preds[batch_indices, actions]
#
#                     stacked_probs = torch.stack([
#                         modality_probs[0][batch_indices, actions],
#                         modality_probs[1][batch_indices, actions],
#                     ], dim=1)
#                     chosen_probs = stacked_probs[torch.arange(actions.size(0), device=actions.device), actions]
#
#                     correct_mask = chosen_preds.eq(labels)
#                     rewards = torch.where(correct_mask, chosen_probs, -chosen_probs)
#
#                     epoch_preds.append(chosen_preds.detach().cpu())
#                     epoch_labels.append(labels.detach().cpu())
#                     epoch_actions.append(actions.detach().cpu())
#
#                 values = value_net(agent_repr)
#                 log_probs = action_dist.log_prob(actions)
#                 with torch.no_grad():
#                     advantages = rewards - values.detach()
#                     adv_mean = advantages.mean()
#                     adv_std = advantages.std()
#                     adv_std_val = adv_std.item()
#                     if math.isnan(adv_std_val) or adv_std_val < 1e-6:
#                         advantages = advantages - adv_mean
#                     else:
#                         advantages = (advantages - adv_mean) / (adv_std + 1e-8)
#                 policy_loss = -(log_probs * advantages).mean()
#                 value_loss = F.mse_loss(values, rewards)
#                 entropy = action_dist.entropy().mean()
#                 loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
#
#                 agent_optimizer.zero_grad()
#                 value_optimizer.zero_grad()
#                 loss.backward()
#                 agent_optimizer.step()
#                 value_optimizer.step()
#
#                 total_rewards += rewards.mean().item()
#                 total_loss += loss.item()
#                 total_value_loss += value_loss.item()
#                 total_entropy += entropy.item()
#
#             avg_reward = total_rewards / len(train_loader)
#             avg_loss = total_loss / len(train_loader)
#             avg_value_loss = total_value_loss / len(train_loader)
#             avg_entropy = total_entropy / len(train_loader)
#
#             epoch_preds_tensor = torch.cat(epoch_preds)
#             epoch_labels_tensor = torch.cat(epoch_labels)
#             epoch_actions_tensor = torch.cat(epoch_actions)
#             preds_np = epoch_preds_tensor.numpy()
#             labels_np = epoch_labels_tensor.numpy()
#             epoch_accuracy = float((preds_np == labels_np).mean()) if preds_np.size > 0 else 0.0
#             try:
#                 epoch_weighted_f1 = f1_score(labels_np, preds_np, average='weighted') if preds_np.size > 0 else 0.0
#             except ValueError:
#                 epoch_weighted_f1 = 0.0
#             if preds_np.size > 0:
#                 conf_mat = confusion_matrix(labels_np, preds_np, labels=list(range(num_labels)))
#                 class_totals = conf_mat.sum(axis=1)
#                 class_correct = conf_mat.diagonal()
#                 class_acc_dict = {
#                     f"class_{i}_acc": (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0)
#                     for i in range(num_labels)
#                 }
#             else:
#                 class_acc_dict = {f"class_{i}_acc": 0.0 for i in range(num_labels)}
#             action_counts = torch.bincount(epoch_actions_tensor, minlength=num_actions)
#             total_actions = max(epoch_actions_tensor.numel(), 1)
#             action_distribution = ", ".join(
#                 [f"{idx}:{(count.item() / total_actions) * 100:.1f}%" for idx, count in enumerate(action_counts)]
#             )
#
#             test_metrics = evaluate_agent(
#                 agent,
#                 experts,
#                 test_loader,
#                 device,
#                 num_labels,
#                 num_actions,
#                 agent_feature_store['test'],
#                 agent_feature_defaults,
#                 multimodal_test,
#             )
#
#             metrics_entry = {
#                 "seed": seed,
#                 "epoch": epoch + 1,
#                 "train_avg_reward": avg_reward,
#                 "train_avg_loss": avg_loss,
#                 "train_avg_value_loss": avg_value_loss,
#                 "train_avg_entropy": avg_entropy,
#                 "train_accuracy": epoch_accuracy,
#                 "train_weighted_f1": epoch_weighted_f1,
#                 "train_action_dist": action_distribution,
#                 "test_accuracy": test_metrics['accuracy'],
#                 "test_weighted_f1": test_metrics['weighted_f1'],
#                 "test_action_dist": test_metrics['action_dist']
#             }
#             metrics_entry.update({f"train_{k}": v for k, v in class_acc_dict.items()})
#             metrics_entry.update({f"test_{k}": v for k, v in test_metrics['class_acc'].items()})
#             metrics_log.append(metrics_entry)
#
#             print(
#                 f"Seed {seed} Epoch {epoch+1}/{num_epochs} | Train Reward: {avg_reward:.4f} | Train Loss: {avg_loss:.4f} "
#                 f"| Train ValLoss: {avg_value_loss:.4f} | Train Entropy: {avg_entropy:.4f} "
#                 f"| Train Acc: {epoch_accuracy * 100:.2f}% | Train W-F1: {epoch_weighted_f1:.4f} "
#                 f"| Train Action Dist: [{action_distribution}]"
#             )
#             print("  Train Class Accuracies: " + ", ".join([f"{k}:{v*100:.2f}%" for k, v in class_acc_dict.items()]))
#             print(
#                 f"  Test  | Acc: {test_metrics['accuracy'] * 100:.2f}% | W-F1: {test_metrics['weighted_f1']:.4f} "
#                 f"| Action Dist: [{test_metrics['action_dist']}]"
#             )
#             print("  Test Class Accuracies: " + ", ".join([f"{k}:{v*100:.2f}%" for k, v in test_metrics['class_acc'].items()]))
#
#             if test_metrics['weighted_f1'] > best_wf1:
#                 best_wf1 = test_metrics['weighted_f1']
#                 best_record.update({
#                     'best_wf1': test_metrics['weighted_f1'],
#                     'best_acc': test_metrics['accuracy'],
#                     'best_action_dist': test_metrics['action_dist'],
#                     'best_epoch': epoch + 1,
#                 })
#
#         save_dir_seed = os.path.join(base_save_dir, f"seed_{seed}")
#         save_path = os.path.join(save_dir_seed, "agent_final.pth")
#         torch.save(agent.state_dict(), save_path)
#         print(f"Saved trained RL agent to: {save_path}")
#
#         metrics_dir = os.path.join(save_dir_seed, "metrics")
#         os.makedirs(metrics_dir, exist_ok=True)
#         metrics_df = pd.DataFrame(metrics_log)
#         metrics_csv_path = os.path.join(metrics_dir, "train_metrics.csv")
#         metrics_df.to_csv(metrics_csv_path, index=False)
#         print(f"保存训练指标到: {metrics_csv_path}")
#
#         best_records.append(best_record)
#         print(
#             f"Seed {seed} best W-F1: {best_record['best_wf1']:.4f} | Best Acc: {best_record['best_acc'] * 100:.2f}% "
#             f"| Action Dist: [{best_record['best_action_dist']}] at epoch {best_record['best_epoch']}"
#         )
#
#         del agent, value_net, text_expert, audio_expert, video_expert
#         torch.cuda.empty_cache()
#
#     summary_df = pd.DataFrame(best_records)
#     summary_path = os.path.join(base_save_dir, "best_seed_summary.csv")
#     summary_df.to_csv(summary_path, index=False)
#     print("\n===== Best results across seeds =====")
#     print(summary_df)
#     print(f"保存最优指标到: {summary_path}")
#
#
# if __name__ == "__main__":
#     main()

import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from src.utils import set_seed, convert_to_acc7_label, convert_to_acc2_label2, convert_to_acc5_label5
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.feature_dataset import MELDDataset
from unimodal_models import TextClassifier, AudioClassifier, VideoClassifier
from rl_agent import RLAgent, ValueNet


def normalize_sample_name(name):
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    name = str(name).replace("\\", "/")
    parts = name.split("/")
    if len(parts) >= 3 and parts[-3].isdigit() and parts[-2].isdigit():
        return f"dia{parts[-3]}_utt{parts[-2]}"
    base = parts[-1]
    return os.path.splitext(base)[0]


def load_general_feature_dict(npz_path, prefer_key='sequence_features'):
    data = np.load(npz_path, allow_pickle=True)
    available_keys = data.files
    if prefer_key in available_keys:
        key = prefer_key
    elif 'sequence_features' in available_keys:
        key = 'sequence_features'
    elif 'pooled_features' in available_keys:
        key = 'pooled_features'
    else:
        raise KeyError(f"No suitable feature key found in {npz_path}. Available keys: {available_keys}")
    names = [normalize_sample_name(n) for i, n in enumerate(data['sample_names'])]
    feats = data[key]
    mapping = {name: feats[i].astype(np.float32) for i, name in enumerate(names)}
    feature_shape = feats.shape[1:]
    return mapping, feature_shape


def build_agent_inputs(
        sample_names,
        feature_store,
        default_features,
        device,
        expert_features,
        text_predictions=None,  # predictions/logits 现在是 1维 回归值
        audio_predictions=None,
        video_predictions=None,
        multimodal_store=None,
):
    required_modalities = ['T', 'A', 'V']
    for key in required_modalities:
        if key not in expert_features:
            raise KeyError(f"Missing expert feature for modality '{key}' in expert_features")
    feature_inputs = {}
    logits_inputs = {}  # 在回归任务中，这些是 'predictions'
    for modality in required_modalities:
        feats = []
        masks = []
        mapping = feature_store[modality]
        default = default_features[modality]
        for raw_name in sample_names:
            name = normalize_sample_name(raw_name)
            feature = mapping.get(name)
            if feature is None:
                feats.append(default)
                masks.append(0.0)
            else:
                feats.append(feature)
                masks.append(1.0)
        arr = np.stack(feats, axis=0).astype(np.float32)
        mask_arr = np.array(masks, dtype=np.float32).reshape(-1, 1)
        feature_inputs[modality] = {
            'memory': torch.from_numpy(arr).to(device),
            'mask': torch.from_numpy(mask_arr).to(device),
            'query': expert_features[modality].detach().to(device).float(),
        }
    # 使用回归预测值作为 Logit/Prediction 输入
    if video_predictions is not None:
        logits_inputs['V'] = video_predictions.detach().to(device)
    if text_predictions is not None:
        logits_inputs['T'] = text_predictions.detach().to(device)
    if audio_predictions is not None:
        logits_inputs['A'] = audio_predictions.detach().to(device)
    # 多模态预测
    if multimodal_store is not None and text_predictions is not None:
        mm_vals = []
        for i, raw_name in enumerate(sample_names):
            key = normalize_sample_name(raw_name)
            mm_pred = multimodal_store['preds'].get(key)
            if mm_pred is None:
                mm_pred = text_predictions[i].detach().cpu().numpy()
            mm_vals.append(mm_pred)
        mm_tensor = torch.tensor(mm_vals, device=device, dtype=text_predictions.dtype).unsqueeze(1)
        logits_inputs['M'] = mm_tensor
    elif text_predictions is not None:
        logits_inputs['M'] = text_predictions.detach().to(device)
    return feature_inputs, logits_inputs


def augment_agent_inputs(
        agent_inputs,
        logit_inputs,
        drop_prob_single=0.0,
        drop_prob_double=0.0,
        noise_std=0.0,
):
    if logit_inputs is None:
        logit_inputs = {}
    modalities = list(agent_inputs.keys())
    if not modalities:
        return agent_inputs, logit_inputs
    batch_size = agent_inputs[modalities[0]]['memory'].size(0)
    device = agent_inputs[modalities[0]]['memory'].device
    if drop_prob_single > 0.0 or drop_prob_double > 0.0:
        rand_vals = torch.rand(batch_size, device=device)
        drop_mask = torch.zeros(batch_size, len(modalities), dtype=torch.bool, device=device)
        cumulative_double = drop_prob_double
        cumulative_single = drop_prob_double + drop_prob_single
        for i in range(batch_size):
            drop_count = 0
            if rand_vals[i] < cumulative_double:
                drop_count = min(2, len(modalities))
            elif rand_vals[i] < cumulative_single:
                drop_count = min(1, len(modalities))
            if drop_count > 0:
                selected = random.sample(modalities, k=drop_count)
                for mod in selected:
                    drop_mask[i, modalities.index(mod)] = True
    else:
        drop_mask = None
    for mod_idx, modality in enumerate(modalities):
        entry = agent_inputs[modality]
        mem = entry['memory']
        mask = entry['mask']
        query = entry['query']
        if noise_std > 0.0:
            mask_expand = mask
            for _ in range(mem.dim() - mask.dim()):
                mask_expand = mask_expand.unsqueeze(-1)
            mem += torch.randn_like(mem) * noise_std * mask_expand
            query += torch.randn_like(query) * noise_std * mask
        if drop_mask is not None:
            to_drop = drop_mask[:, mod_idx]
            if to_drop.any():
                mem[to_drop] = 0.0
                mask[to_drop] = 0.0
                query[to_drop] = 0.0
                if modality in logit_inputs:
                    logit_inputs[modality][to_drop] = 0.0
    return agent_inputs, logit_inputs


def load_multimodal_predictions(train_npz, test_npz):
    multimodal = {}
    ACTUAL_PRED_KEY = 'predictions'
    for split, path in [('train', train_npz), ('test', test_npz)]:
        data = np.load(path, allow_pickle=True)
        # 增加检查，确保文件包含必要的键
        if 'sample_names' not in data.files or ACTUAL_PRED_KEY not in data.files:
            raise KeyError(
                f"Required keys ('sample_names', '{ACTUAL_PRED_KEY}') not found in {path}. "
                f"Available keys: {data.files}"
            )
        names = [normalize_sample_name(n) for i, n in enumerate(data['sample_names'])]
        multimodal[split] = {
            'preds': {names[i]: data[ACTUAL_PRED_KEY][i].item() for i in range(len(names))},
            'logits': {names[i]: data[ACTUAL_PRED_KEY][i] for i in range(len(names))},
        }
    return multimodal


def evaluate_agent(agent, experts, data_loader, device, num_actions, agent_feature_store, default_features,
                   multimodal_store=None):
    was_training = agent.training
    agent.eval()
    all_preds_reg = []
    all_labels_reg = []
    action_counts = torch.zeros(num_actions, dtype=torch.long)
    num_classes_acc7 = 7  # 用于 ACC7 计算
    with torch.no_grad():
        for batch in data_loader:
            # 专家输入是特征序列
            text_input = batch['text'].to(device)
            audio_input = batch['audio'].to(device)
            video_input = batch['video'].to(device)
            # 标签是连续回归值 (B,)
            labels = batch['label'].squeeze(-1).float().to(device)
            sample_names = batch['sample_name']
            if isinstance(sample_names, torch.Tensor):
                sample_names = sample_names.tolist()
            if isinstance(sample_names, tuple):
                sample_names = list(sample_names)
            if isinstance(sample_names, (str, bytes)):
                sample_names = [sample_names]
            # 专家输出: (regression_value, pooled_feature)
            text_reg, text_features = experts['T'](text_input)
            audio_reg, audio_features = experts['A'](audio_input)
            video_reg, video_features = experts['V'](video_input)
            # 确保回归输出是 (B, 1)
            text_reg = text_reg.squeeze(-1).unsqueeze(1)
            audio_reg = audio_reg.squeeze(-1).unsqueeze(1)
            video_reg = video_reg.squeeze(-1).unsqueeze(1)
            expert_queries = {
                'T': text_features,
                'A': audio_features,
                'V': video_features,
            }
            agent_inputs, prediction_inputs = build_agent_inputs(
                sample_names,
                agent_feature_store,
                default_features,
                device,
                expert_queries,
                text_predictions=text_reg,
                audio_predictions=audio_reg,
                video_predictions=video_reg,
                multimodal_store=multimodal_store,
            )
            logit_payload = prediction_inputs if prediction_inputs else None
            action_logits = agent(agent_inputs, logits=logit_payload)
            actions = torch.argmax(action_logits, dim=1)
            action_counts += torch.bincount(actions.cpu(), minlength=num_actions)
            preds_t = text_reg.squeeze(-1)
            if multimodal_store is not None:
                mm_preds = []
                for i, name in enumerate(sample_names):
                    key = normalize_sample_name(name)
                    mm_pred = multimodal_store['preds'].get(key)
                    if mm_pred is None:
                        mm_pred = preds_t[i].item()
                    mm_preds.append(mm_pred)
                preds_m = torch.tensor(mm_preds, device=device, dtype=torch.float)
            else:
                preds_m = preds_t.clone()
            # 专家预测堆叠 (回归值)
            expert_preds_reg = torch.stack([
                preds_t,
                preds_m,
            ], dim=1)
            # 根据 Agent 动作选择预测
            batch_indices = torch.arange(actions.size(0), device=actions.device)
            chosen_preds_reg = expert_preds_reg[batch_indices, actions]
            all_preds_reg.append(chosen_preds_reg.detach().cpu())
            all_labels_reg.append(labels.detach().cpu())
    if was_training:
        agent.train()
    if all_preds_reg:
        preds_tensor = torch.cat(all_preds_reg)
        labels_tensor = torch.cat(all_labels_reg)
        preds_np = preds_tensor.numpy()
        labels_np = labels_tensor.numpy()
        if preds_np.ndim > 1:
            preds_np = preds_np.flatten()
        if labels_np.ndim > 1:
            labels_np = labels_np.flatten()
        mae = np.mean(np.abs(preds_np - labels_np))
        if preds_np.std() > 1e-6 and labels_np.std() > 1e-6:
            print(f"evaluate_agent SHAPE: preds_np shape: {preds_np.shape}, labels_np shape: {labels_np.shape}")
            corr_matrix = np.corrcoef(preds_np, labels_np)
            corr = corr_matrix[0, 1]
        else:
            corr = 0.0
        # ACC7 (使用 CMU-MOSI 离散化规则)
        discrete_preds = np.array([convert_to_acc7_label(p) for p in preds_np])
        discrete_labels = np.array([convert_to_acc7_label(l) for l in labels_np])
        two_discrete_preds = np.array([convert_to_acc2_label2(l) for l in preds_np])
        two_discrete_labels = np.array([convert_to_acc2_label2(l) for l in labels_np])
        has_not_zero = labels_np != 0.0
        if has_not_zero.sum() > 0:
            acc2_new = float((two_discrete_preds[has_not_zero] == two_discrete_labels[has_not_zero]).mean())
            try:
                f1_new = f1_score(
                    two_discrete_labels[has_not_zero],
                    two_discrete_preds[has_not_zero],
                    average='weighted'
                )
            except ValueError:
                f1_new = 0.0
        else:
            acc2_new = 0.0
            f1_new = 0.0
        five_discrete_preds = np.array([convert_to_acc5_label5(l) for l in preds_np])
        five_discrete_labels = np.array([convert_to_acc5_label5(l) for l in labels_np])
        try:
            weighted_f1 = f1_score(two_discrete_labels, two_discrete_preds, average='weighted')
        except ValueError:
            weighted_f1 = 0.0
        acc2 = float((two_discrete_preds == two_discrete_labels).mean())
        acc5 = float((five_discrete_preds == five_discrete_labels).mean())
        acc7 = float((discrete_preds == discrete_labels).mean())
        # 计算 ACC7 混淆矩阵 (用于类准确率)
        conf_mat = confusion_matrix(discrete_labels, discrete_preds, labels=list(range(num_classes_acc7)))
        class_totals = conf_mat.sum(axis=1)
        class_correct = conf_mat.diagonal()
        class_acc_dict = {
            f"class_{i}_acc": (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0)
            for i in range(num_classes_acc7)
        }
    else:
        mae = 0.0
        corr = 0.0
        acc7 = 0.0
        acc2 = 0.0
        acc2_new = 0.0
        acc5 = 0.0
        weighted_f1 = 0.0
        f1_new = 0.0
        class_acc_dict = {f"class_{i}_acc": 0.0 for i in range(num_classes_acc7)}
    total_actions = max(action_counts.sum().item(), 1)
    action_distribution = ", ".join([
        f"{idx}:{(count.item() / total_actions) * 100:.1f}%" for idx, count in enumerate(action_counts)
    ])
    return {
        "mae": mae,
        "corr": corr,
        "acc7": acc7,
        "acc2": acc2,
        "acc2_new": acc2_new,
        "acc5": acc5,
        "weighted_f1": weighted_f1,
        "f1_new": f1_new,
        "class_acc": class_acc_dict,
        "action_dist": action_distribution
    }


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # 多尝试几个种子；由于奖励的高方差和batch的随机抽取问题，最佳结果复现难度较高，可多次尝试
    # seeds = [41, 42, 43, 44]
    seeds = [41, 42, 43]
    num_labels_regression = 1  # 最终输出维度为 1 (回归)
    num_epochs = 100
    batch_size = 64
    agent_lr = 5e-6
    value_lr = 1e-5
    hidden_dim = 128
    agent_embed_dim = 768
    value_hidden_dim = 256
    value_coef = 0.5
    entropy_coef = 0.03
    num_actions = 2
    general_pooling_mode = 'mean'
    general_conv_kernel = 3
    include_memory_tokens = False
    modality_drop_prob_single = 0.2
    modality_drop_prob_double = 0.05
    agent_noise_std = 0.01
    patience = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 专家模型路径（注意：这些模型现在应该有回归头 num_classes=1）
    expert_model_paths = {
        'T': '/data/home/chenqian/regression_models/unimodel_experts/unimodal_t_best_corr_0.7912_acc7_0.5222_mae_ 0.5273.pth',
        'A': '/data/home/chenqian/regression_models/unimodel_experts/unimodal_a_best_corr_0.7310_acc7_0.5095_mae_ 0.5822.pth',
        'V': '/data/home/chenqian/regression_models/unimodel_experts/unimodal_v_best_corr_0.4503_acc7_0.4034_mae_ 0.7789.pth'
    }

    data_paths = {
        'train': {
            'text': '/data/home/chenqian/regression_models/text_model/train_text_features.npz',
            'audio': '/data/home/chenqian/regression_models/audio_model/train_audio_features.npz',
            'video': '/data/home/chenqian/regression_models/video_model/train_video_features.npz',
        },
        'test': {
            'text': '/data/home/chenqian/regression_models/text_model/test_text_features.npz',
            'audio': '/data/home/chenqian/regression_models/audio_model/test_audio_features.npz',
            'video': '/data/home/chenqian/regression_models/video_model/test_video_features.npz',
        }
    }
    print("Loading datasets and general features...")
    train_dataset = MELDDataset(
        data_paths['train']['text'],
        data_paths['train']['audio'],
        data_paths['train']['video'],
        modalities=['T', 'A', 'V'],
        feature_type='sequence_features'
    )
    test_dataset = MELDDataset(
        data_paths['test']['text'],
        data_paths['test']['audio'],
        data_paths['test']['video'],
        modalities=['T', 'A', 'V'],
        split='test',
        feature_type='sequence_features'
    )
    text_feat_dim = train_dataset.aligned_data['text'].shape[-1]
    text_seq_len = train_dataset.aligned_data['text'].shape[1] if train_dataset.aligned_data['text'].ndim == 3 else 1
    agent_feature_paths = {
        'train': {
            'T': '/data/home/chenqian/regression_models/text_model/train_text_features_general.npz',
            'A': '/data/home/chenqian/regression_models/audio_model/train_audio_features_general.npz',
            'V': '/data/home/chenqian/regression_models/video_model/train_video_features_general.npz',
        },
        'test': {
            'T': '/data/home/chenqian/regression_models/text_model/test_text_features_general.npz',
            'A': '/data/home/chenqian/regression_models/audio_model/test_audio_features_general.npz',
            'V': '/data/home/chenqian/regression_models/video_model/test_video_features_general.npz',
        }
    }
    agent_feature_store = {split: {} for split in ['train', 'test']}
    agent_feature_defaults = {}
    for split in ['train', 'test']:
        for modality in ['T', 'A', 'V']:
            mapping, feat_shape = load_general_feature_dict(agent_feature_paths[split][modality])
            agent_feature_store[split][modality] = mapping
            agent_feature_defaults.setdefault(modality, np.zeros(feat_shape, dtype=np.float32))
    for modality in ['T', 'A', 'V']:
        missing_train = [name for name in train_dataset.sample_names if
                         normalize_sample_name(name) not in agent_feature_store['train'][modality]]
        if missing_train:
            print(
                f"Warning: {len(missing_train)} missing agent features for modality {modality} in training split. Example: {missing_train[:5]}")
        missing_test = [name for name in test_dataset.sample_names if
                        normalize_sample_name(name) not in agent_feature_store['test'][modality]]
        if missing_test:
            print(
                f"Warning: {len(missing_test)} missing agent features for modality {modality} in test split. Example: {missing_test[:5]}")
    modality_dims = {}
    sample_name_example = normalize_sample_name(train_dataset.sample_names[0])
    for modality in ['T', 'A', 'V']:
        sample_feat = agent_feature_store['train'][modality].get(sample_name_example)
        if sample_feat is None:
            sample_feat = agent_feature_defaults[modality]
        modality_dims[modality] = sample_feat.shape[-1] if sample_feat.ndim >= 1 else 1
    multimodal_train_npz = '/data/home/chenqian/regression_models/without_afd_model/multimodal_best_mse_0.5381_corr_0.7633_seed_42_train_predictions.npz'
    multimodal_test_npz = '/data/home/chenqian/regression_models/without_afd_model/multimodal_best_mse_0.5381_corr_0.7633_seed_42_test_predictions.npz'
    multimodal_available = os.path.exists(multimodal_train_npz) and os.path.exists(multimodal_test_npz)
    multimodal_predictions = load_multimodal_predictions(multimodal_train_npz,
                                                         multimodal_test_npz) if multimodal_available else None
    base_save_dir = "/data/home/chenqian/without_afd_checkpoints"  # 更改保存目录以区分任务
    os.makedirs(base_save_dir, exist_ok=True)
    best_records = []
    for seed in seeds:
        print(f"\n===== Training with seed {seed} =====")
        set_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        text_expert = TextClassifier(
            hidden_dim=hidden_dim,
            num_classes=num_labels_regression,  # 1
            use_precomputed=True,
            input_dim=text_feat_dim,
            target_seq_len=text_seq_len,
        ).to(device)
        text_expert.load_state_dict(torch.load(expert_model_paths['T'], map_location=device), strict=False)
        audio_expert = AudioClassifier(hidden_dim=hidden_dim, num_classes=num_labels_regression).to(device)
        audio_expert.load_state_dict(torch.load(expert_model_paths['A'], map_location=device), strict=False)
        video_expert = VideoClassifier(hidden_dim=hidden_dim, num_classes=num_labels_regression).to(device)
        video_expert.load_state_dict(torch.load(expert_model_paths['V'], map_location=device), strict=False)
        experts = {'T': text_expert, 'A': audio_expert, 'V': video_expert}
        for expert in experts.values():
            expert.eval()
            for param in expert.parameters():
                param.requires_grad = False
        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            sample_text_feat = experts['T'](sample_batch['text'].to(device))[1]
            sample_audio_feat = experts['A'](sample_batch['audio'].to(device))[1]
            sample_video_feat = experts['V'](sample_batch['video'].to(device))[1]
        query_dims = {
            'T': sample_text_feat.shape[-1],
            'A': sample_audio_feat.shape[-1],
            'V': sample_video_feat.shape[-1],
        }
        del sample_batch
        multimodal_train = multimodal_predictions['train'] if multimodal_predictions else None
        multimodal_test = multimodal_predictions['test'] if multimodal_predictions else None
        agent = RLAgent(
            modality_dims=modality_dims,
            num_actions=num_actions,
            query_dims=query_dims,
            embed_dim=agent_embed_dim,
            general_pooling=general_pooling_mode,
            conv_kernel_size=general_conv_kernel,
            include_memory_tokens=include_memory_tokens,
            num_classes=num_labels_regression,  # 1
        ).to(device)
        value_net = ValueNet(input_dim=agent_embed_dim, hidden_dim=value_hidden_dim).to(device)
        agent_optimizer = torch.optim.Adam(agent.parameters(), lr=agent_lr)
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr)
        save_dir_seed = os.path.join(base_save_dir, f"seed_{seed}")
        os.makedirs(save_dir_seed, exist_ok=True)
        metrics_dir = os.path.join(save_dir_seed, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_csv_path = os.path.join(metrics_dir, "log.csv")
        metrics_log = []
        metrics_header_keys = [
            "seed",
            "epoch",
            "train_avg_reward",
            "train_avg_loss",
            "train_avg_value_loss",
            "train_avg_entropy",
            "train_mae",
            "train_corr",
            "train_acc7",
            "train_acc2",
            "train_acc5",
            "train_weighted_f1",
            "train_action_dist",
            "test_mae",
            "test_corr",
            "test_acc7",
            "test_acc2",
            "test_acc2_new",
            "test_acc5",
            "test_weighted_f1",
            "test_f1_new",
            "test_action_dist"
        ]
        metrics_header_keys.extend([f"train_class_{i}_acc" for i in range(7)])
        metrics_header_keys.extend([f"test_class_{i}_acc" for i in range(7)])
        metrics_header = ",".join(metrics_header_keys) + "\n"
        with open(metrics_csv_path, 'w') as f:
            f.write(metrics_header)
        best_corr = -1.0
        best_mae = 100
        best_record = {
            'seed': seed,
            'best_corr': -1.0,
            'best_mae': 999.0,
            'best_acc7': 0.0,
            'best_action_dist': '',
            'best_epoch': 0,
        }
        patience_count = 0
        for epoch in range(num_epochs):
            agent.train()
            value_net.train()
            total_rewards = 0
            total_loss = 0
            total_value_loss = 0
            total_entropy = 0
            epoch_preds_reg = []
            epoch_labels_reg = []
            epoch_actions = []
            for batch in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch + 1}/{num_epochs}"):
                text_input = batch['text'].to(device)
                audio_input = batch['audio'].to(device)
                video_input = batch['video'].to(device)
                labels = batch['label'].squeeze(-1).float().to(device)
                sample_names = batch['sample_name']
                if isinstance(sample_names, torch.Tensor):
                    sample_names = sample_names.tolist()
                if isinstance(sample_names, tuple):
                    sample_names = list(sample_names)
                if isinstance(sample_names, (str, bytes)):
                    sample_names = [sample_names]
                with torch.no_grad():
                    # 专家输出:
                    text_reg, text_feat = experts['T'](text_input)
                    audio_reg, audio_feat = experts['A'](audio_input)
                    video_reg, video_feat = experts['V'](video_input)
                    # 确保回归输出是 (B, 1)
                    text_reg = text_reg.squeeze(-1).unsqueeze(1)
                    audio_reg = audio_reg.squeeze(-1).unsqueeze(1)
                    video_reg = video_reg.squeeze(-1).unsqueeze(1)
                    expert_queries = {
                        'T': text_feat,
                        'A': audio_feat,
                        'V': video_feat,
                    }
                    agent_inputs, prediction_inputs = build_agent_inputs(
                        sample_names,
                        agent_feature_store['train'],
                        agent_feature_defaults,
                        device,
                        expert_queries,
                        text_predictions=text_reg,
                        audio_predictions=audio_reg,
                        video_predictions=video_reg,
                        multimodal_store=multimodal_train,
                    )
                    agent_inputs, prediction_inputs = augment_agent_inputs(
                        agent_inputs,
                        prediction_inputs,
                        drop_prob_single=modality_drop_prob_single,
                        drop_prob_double=modality_drop_prob_double,
                        noise_std=agent_noise_std,
                    )
                logit_payload = prediction_inputs if prediction_inputs else None
                action_logits, agent_repr = agent(agent_inputs, logits=logit_payload, return_repr=True)
                action_dist = Categorical(logits=action_logits)
                actions = action_dist.sample()
                with torch.no_grad():
                    preds_t = text_reg.squeeze(-1)
                    if multimodal_train is not None:
                        mm_preds = []
                        for i, raw_name in enumerate(sample_names):
                            key = normalize_sample_name(raw_name)
                            mm_pred = multimodal_train['preds'].get(key)
                            if mm_pred is None:
                                mm_pred = preds_t[i].item()
                            mm_preds.append(mm_pred)
                        preds_m = torch.tensor(mm_preds, device=device, dtype=torch.float)
                    else:
                        preds_m = preds_t.clone()
                    expert_preds_reg = torch.stack([preds_t, preds_m], dim=1)
                    batch_indices = torch.arange(actions.size(0), device=actions.device)
                    chosen_preds_reg = expert_preds_reg[batch_indices, actions]
                    preds_np = chosen_preds_reg.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    if preds_np.ndim > 1:
                        preds_np = preds_np.flatten()
                    if labels_np.ndim > 1:
                        labels_np = labels_np.flatten()
                    if preds_np.std() > 1e-6 and labels_np.std() > 1e-6:
                        corr_matrix = np.corrcoef(preds_np, labels_np)
                        corr = corr_matrix[0, 1]
                    else:
                        corr = 0.0
                    # 2. 奖励为 Corr (批次级奖励)
                    rewards = torch.tensor((corr + 1.0) / 2.0, dtype=torch.float32, device=device)
                    rewards = rewards.expand(actions.size(0))
                    epoch_preds_reg.append(chosen_preds_reg.detach().cpu())
                    epoch_labels_reg.append(labels.detach().cpu())
                    epoch_actions.append(actions.detach().cpu())
                values = value_net(agent_repr).squeeze(-1)  # ValueNet 输出 (B,)
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
            avg_reward = total_rewards / len(train_loader)
            avg_loss = total_loss / len(train_loader)
            avg_value_loss = total_value_loss / len(train_loader)
            avg_entropy = total_entropy / len(train_loader)
            epoch_preds_tensor = torch.cat(epoch_preds_reg)
            epoch_labels_tensor = torch.cat(epoch_labels_reg)
            preds_np = epoch_preds_tensor.numpy()
            labels_np = epoch_labels_tensor.numpy()
            if preds_np.ndim > 1:
                preds_np = preds_np.flatten()
            if labels_np.ndim > 1:
                labels_np = labels_np.flatten()
            epoch_mae = np.mean(np.abs(preds_np - labels_np))
            if preds_np.std() > 1e-6 and labels_np.std() > 1e-6:
                epoch_corr = np.corrcoef(preds_np, labels_np)[0, 1]
            else:
                epoch_corr = 0.0
            # ACC7
            discrete_preds = np.array([convert_to_acc7_label(p) for p in preds_np])
            discrete_labels = np.array([convert_to_acc7_label(l) for l in labels_np])
            # ACC2
            two_discrete_preds = np.array([convert_to_acc2_label2(p) for p in preds_np])
            two_discrete_labels = np.array([convert_to_acc2_label2(l) for l in labels_np])
            # ACC3
            five_discrete_preds = np.array([convert_to_acc5_label5(p) for p in preds_np])
            five_discrete_labels = np.array([convert_to_acc5_label5(l) for l in labels_np])
            try:
                epoch_weighted_f1 = f1_score(two_discrete_labels, two_discrete_preds, average='weighted')
            except ValueError:
                epoch_weighted_f1 = 0.0
            epoch_acc5 = float((five_discrete_preds == five_discrete_labels).mean())
            epoch_acc2 = float((two_discrete_preds == two_discrete_labels).mean())
            epoch_acc7 = float((discrete_preds == discrete_labels).mean())
            # ACC7 混淆矩阵 (用于类准确率)
            conf_mat = confusion_matrix(discrete_labels, discrete_preds, labels=list(range(7)))
            class_totals = conf_mat.sum(axis=1)
            class_correct = conf_mat.diagonal()
            class_acc_dict = {
                f"class_{i}_acc": (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0)
                for i in range(7)
            }
            epoch_actions_tensor = torch.cat(epoch_actions)
            action_counts = torch.bincount(epoch_actions_tensor, minlength=num_actions)
            total_actions = max(epoch_actions_tensor.numel(), 1)
            action_distribution = ", ".join(
                [f"{idx}:{(count.item() / total_actions) * 100:.1f}%" for idx, count in enumerate(action_counts)]
            )
            # --- 测试集评估---
            test_metrics = evaluate_agent(
                agent,
                experts,
                test_loader,
                device,
                num_actions,
                agent_feature_store['test'],
                agent_feature_defaults,
                multimodal_test,
            )
            # 记录 metrics
            metrics_entry = {
                "seed": seed,
                "epoch": epoch + 1,
                "train_avg_reward": avg_reward,
                "train_avg_loss": avg_loss,
                "train_avg_value_loss": avg_value_loss,
                "train_avg_entropy": avg_entropy,
                "train_mae": epoch_mae,
                "train_corr": epoch_corr,
                "train_acc7": epoch_acc7,
                "train_acc2": epoch_acc2,
                "train_acc5": epoch_acc5,
                "train_weighted_f1": epoch_weighted_f1,
                "train_action_dist": action_distribution,
                "test_mae": test_metrics['mae'],
                "test_corr": test_metrics['corr'],
                "test_acc7": test_metrics['acc7'],
                "test_acc2": test_metrics['acc2'],
                "test_acc2_new": test_metrics['acc2_new'],
                "test_acc5": test_metrics['acc5'],
                "test_weighted_f1": test_metrics['weighted_f1'],
                "test_f1_new": test_metrics['f1_new'],
                "test_action_dist": test_metrics['action_dist']
            }
            metrics_entry.update({f"train_{k}": v for k, v in class_acc_dict.items()})
            metrics_entry.update({f"test_{k}": v for k, v in test_metrics['class_acc'].items()})
            data_row = ",".join([str(metrics_entry[key]) for key in metrics_header_keys]) + "\n"
            with open(metrics_csv_path, 'a') as f:
                f.write(data_row)
            metrics_log.append(metrics_entry)
            print(
                f"Seed {seed} Epoch {epoch + 1}/{num_epochs} | Train Reward: {avg_reward:.4f} | Train Loss: {avg_loss:.4f} "
                f"| Train ValLoss: {avg_value_loss:.4f} | Train Entropy: {avg_entropy:.4f} "
                f"| Train MAE: {epoch_mae:.4f} | Train Corr: {epoch_corr:.4f} | Train ACC7: {epoch_acc7 * 100:.2f}% | Train ACC2 {epoch_acc2 * 100:.2f}% | Train ACC5: {epoch_acc5 * 100: .2f}% | Train Weighted F1: {epoch_weighted_f1 * 100:.2f}%"
                f"| Train Action Dist: [{action_distribution}]"
            )
            print("  Train Class Accuracies: " + ", ".join([f"{k}:{v * 100:.2f}%" for k, v in class_acc_dict.items()]))
            print(
                f" Test  | MAE: {test_metrics['mae']:.4f} | Corr: {test_metrics['corr']:.4f} | "
                f" ACC7: {test_metrics['acc7'] * 100:.2f}% | ACC2: {test_metrics['acc2'] * 100:.2f}% |"
                f" ACC5: {test_metrics['acc5'] * 100: .2f}% | WF1: {test_metrics['weighted_f1'] * 100:.2f}% |"
                f" ACC2_new: {test_metrics['acc2_new'] * 100:.2f}% | f1_new: {test_metrics['f1_new'] * 100:.2f}%"
                f"| Action Dist: [{test_metrics['action_dist']}]"
            )
            print("  Test Class Accuracies: " + ", ".join(
                [f"{k}:{v * 100:.2f}%" for k, v in test_metrics['class_acc'].items()]))
            # --- 保存最佳模型 (基于 Corr和MAE) ---
            if test_metrics['corr'] > best_corr or test_metrics['mae'] < best_mae:
                if test_metrics['corr'] > best_corr:
                    best_corr = test_metrics['corr']
                elif test_metrics['mae'] < best_mae:
                    best_mae = test_metrics['mae']
                best_record.update({
                    'best_corr': test_metrics['corr'],
                    'best_mae': test_metrics['mae'],
                    'best_acc7': test_metrics['acc7'],
                    'best_action_dist': test_metrics['action_dist'],
                    'best_epoch': epoch + 1,
                })
                save_path_agent = os.path.join(save_dir_seed, f"agent_best_corr_{best_corr:.4f}.pth")
                save_path_value = os.path.join(save_dir_seed, f"value_best_corr_{best_corr:.4f}.pth")
                torch.save(agent.state_dict(), save_path_agent)
                torch.save(value_net.state_dict(), save_path_value)
                print(f"Saved best RL models based on Corr: {test_metrics['corr']:.4f},MAE：{test_metrics['mae']:.4f}")
                patience_count = 0
            else:
                patience_count += 1
                print(f"Early Stopping : {patience_count} | {patience}")
            if patience_count >= patience:
                break
        save_dir_seed = os.path.join(base_save_dir, f"seed_{seed}")
        save_path_agent_final = os.path.join(save_dir_seed, "agent_final.pth")
        save_path_value_final = os.path.join(save_dir_seed, "value_final.pth")
        torch.save(agent.state_dict(), save_path_agent_final)
        torch.save(value_net.state_dict(), save_path_value_final)
        print(f"Saved final RL agent to: {save_path_agent_final}")
        best_records.append(best_record)
        print(
            f"Seed {seed} best Corr: {best_record['best_corr']:.4f} | Best MAE: {best_record['best_mae']:.4f} | Best ACC7: {best_record['best_acc7'] * 100:.2f}% "
            f"| Action Dist: [{best_record['best_action_dist']}] at epoch {best_record['best_epoch']}"
        )
        del agent, value_net, text_expert, audio_expert, video_expert
        torch.cuda.empty_cache()
    summary_df = pd.DataFrame(best_records)
    summary_path = os.path.join(base_save_dir, "best_seed_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n===== Best results across seeds =====")
    print(summary_df)
    print(f"保存最优指标到: {summary_path}")


if __name__ == "__main__":
    main()
