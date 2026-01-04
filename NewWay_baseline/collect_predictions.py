import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from feature_dataset import MELDDataset
from T_raw_data_dataset import T_raw_MELDDataset
from TA_raw_data_dataset import TA_raw_MELDDataset
from baseline_model import MultimodalFusionModel
from utils import set_seed


def normalize_sample_names(names):
    if isinstance(names, torch.Tensor):
        names = names.tolist()
    if isinstance(names, np.ndarray):
        names = names.tolist()
    if isinstance(names, (str, bytes)):
        names = [names]
    return [str(n) for n in names]


def prepare_inputs(batch, modalities, device):
    modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
    inputs = {}
    for mod in modalities:
        key = modality_map[mod]
        data = batch[key]
        if isinstance(data, dict):
            inputs[mod] = {k: v.to(device) for k, v in data.items()}
        else:
            inputs[mod] = data.to(device)
    return inputs


def collect_predictions(model, data_loader, device, modalities, use_cam_loss, cam_type):
    model.eval()
    all_logits, all_preds, all_labels, all_names = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting predictions"):
            inputs = prepare_inputs(batch, modalities, device)
            labels = batch['label']
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            else:
                labels = torch.tensor(labels, device=device)
            labels = labels.view(-1)

            outputs = model(inputs, labels=None)
            if use_cam_loss:
                outputs = outputs[0]

            logits = outputs.detach()
            preds = torch.argmax(logits, dim=1)

            names = normalize_sample_names(batch.get('sample_name'))

            all_logits.append(logits.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_names.extend(names)

    logits_np = torch.cat(all_logits).numpy() if all_logits else np.array([])
    preds_np = torch.cat(all_preds).numpy() if all_preds else np.array([])
    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    names_np = np.array(all_names)

    return {
        'sample_names': names_np,
        'logits': logits_np,
        'preds': preds_np,
        'labels': labels_np,
    }


def summarize_metrics(split_name, labels, preds):
    acc = accuracy_score(labels, preds) if labels.size and preds.size else 0.0
    try:
        weighted_f1 = f1_score(labels, preds, average='weighted') if labels.size else 0.0
    except ValueError:
        weighted_f1 = 0.0
    unique_labels = sorted(set(labels.tolist())) if labels.size else []
    cm = confusion_matrix(labels, preds, labels=unique_labels) if unique_labels else None
    print(f"{split_name} Accuracy: {acc:.4f}, Weighted F1: {weighted_f1:.4f}")
    if cm is not None:
        print(f"{split_name} Confusion Matrix:\n{cm}")


def build_dataset(cfg, modalities, feature_type, use_raw_text, use_raw_audio, split):
    if use_raw_text and use_raw_audio:
        return TA_raw_MELDDataset(
            cfg['text'], cfg['audio'], cfg['video'], modalities,
            split=split,
            feature_type=feature_type,
            text_path=cfg['text_raw_path'],
            audio_csv_path=cfg['audio_csv_path'],
            audio_data_path=cfg['audio_data_path']
        )
    elif use_raw_text:
        return T_raw_MELDDataset(
            cfg['text'], cfg['audio'], cfg['video'], modalities,
            split=split,
            feature_type=feature_type,
            text_path=cfg['text_raw_path']
        )
    else:
        return MELDDataset(
            cfg['text'], cfg['audio'], cfg['video'], modalities,
            split=split,
            feature_type=feature_type
        )


def create_model(modalities, feature_type, hidden_dim, num_labels,
                 use_cross_modal, use_raw_text, use_cam_loss,
                 use_raw_audio, whisper_use_adapters, cam_type, device):
    model = MultimodalFusionModel(
        text_dim=1024,
        audio_dim=768,
        video_dim=768,
        hidden_dim=hidden_dim,
        num_classes=num_labels,
        modalities=modalities,
        feature_type=feature_type,
        use_cross_modal=use_cross_modal,
        use_raw_text=use_raw_text,
        use_cam_loss=use_cam_loss,
        use_raw_audio=use_raw_audio,
        whisper_use_adapters=whisper_use_adapters,
        cam_type=cam_type
    ).to(device)
    return model


def main():
    checkpoint_path = "/data/yuyangchen/checkpoints/multimodal_fusion_best_acc_0.6893_seed_42.pth"
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    # Configuration (must match training setup)
    seed = 42
    num_labels = 7
    batch_size = 32
    modalities = ['T', 'A', 'V']
    hidden_dim = 512
    feature_type = 'sequence_features'
    use_cross_modal = True
    use_raw_text = True
    use_cam_loss = True
    use_raw_audio = False
    whisper_use_adapters = True
    cam_type = 'AVcam_to_CAM'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    data_paths = {
        'train': {
            'text': '/data/yuyangchen/data/MELD/text_features/train_text_features.npz',
            'text_raw_path': '/data/yuyangchen/data/MELD/processed_train_T_emo.csv',
            'audio': '/data/yuyangchen/data/MELD/audio_features/train_audio_features_processed.npz',
            'video': '/data/yuyangchen/data/MELD/face_features/train_video_features_processed.npz',
            'audio_csv_path': '/data/yuyangchen/data/MELD/train_sent_emo.csv',
            'audio_data_path': '/data/yuyangchen/data/MELD/train_A/train'
        },
        'test': {
            'text': '/data/yuyangchen/data/MELD/text_features/test_text_features.npz',
            'text_raw_path': '/data/yuyangchen/data/MELD/processed_test_T_emo.csv',
            'audio': '/data/yuyangchen/data/MELD/audio_features/test_audio_features_processed.npz',
            'video': '/data/yuyangchen/data/MELD/face_features/test_video_features_processed.npz',
            'audio_csv_path': '/data/yuyangchen/data/MELD/test_sent_emo.csv',
            'audio_data_path': '/data/yuyangchen/data/MELD/test_A/test'
        }
    }

    train_dataset = build_dataset(data_paths['train'], modalities, feature_type, use_raw_text, use_raw_audio, 'train')
    test_dataset = build_dataset(data_paths['test'], modalities, feature_type, use_raw_text, use_raw_audio, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = create_model(modalities, feature_type, hidden_dim, num_labels,
                         use_cross_modal, use_raw_text, use_cam_loss,
                         use_raw_audio, whisper_use_adapters, cam_type, device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)

    train_preds = collect_predictions(model, train_loader, device, modalities, use_cam_loss, cam_type)
    test_preds = collect_predictions(model, test_loader, device, modalities, use_cam_loss, cam_type)

    prefix = os.path.splitext(checkpoint_path)[0]
    train_pred_path = f"{prefix}_train_predictions.npz"
    test_pred_path = f"{prefix}_test_predictions.npz"

    np.savez(train_pred_path, **train_preds)
    np.savez(test_pred_path, **test_preds)

    print(f"Saved train predictions to: {train_pred_path}")
    print(f"Saved test predictions to: {test_pred_path}")

    summarize_metrics("Train", train_preds['labels'], train_preds['preds'])
    summarize_metrics("Test", test_preds['labels'], test_preds['preds'])


if __name__ == "__main__":
    main()
