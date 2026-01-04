import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any


class MELDDataset(Dataset):
    def __init__(
            self,
            text_npz: str,
            audio_npz: str,
            video_npz: str,
            modalities: List[str],
            split: str = 'train',
            feature_type: str = 'pooled_features'
    ):
        """
        Initialize the dataset.
        
        Args:
            text_npz (str): Path to text features .npz file.
            audio_npz (str): Path to audio features .npz file.
            video_npz (str): Path to video features .npz file.
            modalities (List[str]): List of modalities to include (e.g., ['T', 'A', 'V']).
            split (str): Dataset split ('train' or 'test').
            feature_type (str): Type of feature to load (e.g., 'pooled_features', 'sequence_features').
        """
        self.labels = None
        self.modalities = modalities
        self.split = split
        self.feature_type = feature_type
        self.text_data = self.load_features(text_npz, 'text')
        self.audio_data = self.load_features(audio_npz, 'audio') if 'A' in modalities else None
        self.video_data = self.load_features(video_npz, 'video') if 'V' in modalities else None
        sample_names_sets = []
        if 'T' in modalities:
            sample_names_sets.append(set(self.text_data['sample_names']))
        if 'A' in modalities:
            sample_names_sets.append(set(self.audio_data['sample_names']))
        if 'V' in modalities:
            sample_names_sets.append(set(self.video_data['sample_names']))
        self.sample_names = sorted(list(set.intersection(*sample_names_sets)))
        print(f"{split} split: {len(self.sample_names)} samples after alignment")
        self.aligned_data = self.align_data()

    def _normalize_sample_name(self, name: Any) -> str:
        """Normalize sample identifiers across modalities to canonical 'diaX_uttY' format."""
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        name = str(name)
        name = name.replace('\\', '/')
        parts = name.split('/')
        if len(parts) >= 3 and parts[-3].isdigit() and parts[-2].isdigit():
            return f"dia{parts[-3]}_utt{parts[-2]}"
        base = parts[-1]
        base = os.path.splitext(base)[0]
        return base

    def load_features(self, npz_path: str, modality: str) -> Dict[str, Any]:
        try:
            data = np.load(npz_path, allow_pickle=True)
            features = {key: data[key] for key in data}
            if self.feature_type not in features:
                raise KeyError(
                    f"Feature type '{self.feature_type}' not found in {npz_path}. Available keys: {list(features.keys())}")
            if 'sample_names' not in features:
                raise KeyError(f"'sample_names' not found in {npz_path}")
            normalized_names = np.array([self._normalize_sample_name(n) for n in features['sample_names']])
            features['sample_names'] = normalized_names
            print(f"Loaded {modality} features from {npz_path}: {len(features['sample_names'])} samples, "
                  f"{self.feature_type} shape: {features[self.feature_type].shape}")
            return features
        except Exception as e:
            print(f"Error loading {modality} features from {npz_path}: {e}")
            raise

    def align_data(self) -> Dict[str, np.ndarray]:
        aligned_data = {
            'text': [],
            'audio': [],
            'video': [],
            'labels': []
        }
        skipped_samples = []
        for sample in self.sample_names:
            idx = np.where(self.text_data['sample_names'] == sample)[0][0]
            aligned_data['labels'].append(self.text_data['labels'][idx])
            try:
                if 'T' in self.modalities:
                    idx = np.where(self.text_data['sample_names'] == sample)[0][0]
                    aligned_data['text'].append(self.text_data[self.feature_type][idx])
                    # aligned_data['labels'].append(self.text_data['labels'][idx])  # Assume labels are consistent
                if 'A' in self.modalities:
                    idx = np.where(self.audio_data['sample_names'] == sample)[0][0]
                    aligned_data['audio'].append(self.audio_data[self.feature_type][idx])
                if 'V' in self.modalities:
                    idx = np.where(self.video_data['sample_names'] == sample)[0][0]
                    aligned_data['video'].append(self.video_data[self.feature_type][idx])
            except IndexError:
                skipped_samples.append(sample)
                continue
        if skipped_samples:
            print(f"Skipped {len(skipped_samples)} samples due to missing data: {skipped_samples[:5]}...")
        for key in aligned_data:
            if aligned_data[key]:
                try:
                    aligned_data[key] = np.array(aligned_data[key])
                    print(f"Aligned {key} shape: {aligned_data[key].shape}")
                except ValueError as e:
                    print(f"Error converting {key} to numpy array: {e}")
                    raise
        self.labels = aligned_data['labels']
        return aligned_data

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {}
        if 'T' in self.modalities:
            sample['text'] = torch.FloatTensor(self.aligned_data['text'][idx])
        if 'A' in self.modalities:
            sample['audio'] = torch.FloatTensor(self.aligned_data['audio'][idx])
        if 'V' in self.modalities:
            sample['video'] = torch.FloatTensor(self.aligned_data['video'][idx])
        # sample['label'] = torch.FloatTensor([self.aligned_data['labels'][idx]])
        label_value = self.aligned_data['labels'][idx]
        sample['label'] = torch.from_numpy(np.array([label_value])).float()
        sample['sample_name'] = self.sample_names[idx]
        return sample


def get_dataloader(
        text_npz: str,
        audio_npz: str,
        video_npz: str,
        modalities: List[str],
        split: str = 'train',
        batch_size: int = 32,
        shuffle: bool = True,
        feature_type: str = 'pooled_features'
) -> DataLoader:
    dataset = MELDDataset(text_npz, audio_npz, video_npz, modalities, split, feature_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
