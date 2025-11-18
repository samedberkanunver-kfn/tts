"""
Turkish TTS Dataset Module for StyleTTS2 Fine-Tuning

This module provides dataset loading, preprocessing, and collation for Turkish TTS training.
Uses zeynepgulhan/mediaspeech-with-cv-tr dataset from Hugging Face.

Features:
- Automatic dataset loading and caching
- Audio resampling to 24kHz
- Text filtering and normalization
- Mel-spectrogram conversion
- Train/validation splitting
- Multi-speaker support

Author: Claude Code
"""

import re
import warnings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

try:
    from datasets import load_dataset, Dataset as HFDataset, DatasetDict
except ImportError:
    raise ImportError("datasets package not found. Install: pip install datasets")

from .phonemizer import TurkishPhonemizer


class TurkishTTSDataset(Dataset):
    """
    Turkish Text-to-Speech dataset for StyleTTS2 fine-tuning.

    Loads zeynepgulhan/mediaspeech-with-cv-tr dataset, processes audio to mel-spectrograms,
    and phonemizes text for training.

    Attributes:
        dataset: Hugging Face dataset
        phonemizer: Turkish phonemizer
        config: Dataset configuration dictionary
        target_sample_rate: Target audio sample rate (24kHz for StyleTTS2)
        mel_transform: Mel-spectrogram transform
    """

    def __init__(
        self,
        config: Dict[str, Any],
        phonemizer: TurkishPhonemizer,
        split: str = "train",
        limit_samples: Optional[int] = None
    ):
        """
        Initialize Turkish TTS dataset.

        Args:
            config: Configuration dictionary with 'data' section
            phonemizer: TurkishPhonemizer instance
            split: Dataset split ("train" or "test")
            limit_samples: Limit number of samples (for debugging)
        """
        self.config = config['data']
        self.phonemizer = phonemizer
        self.split = split
        self.target_sample_rate = self.config.get('sample_rate', 24000)

        # Load dataset from Hugging Face
        print(f"Loading {self.config['dataset_name']} ({split} split)...")
        self.dataset = self._load_and_filter_dataset(limit_samples)

        # Build mel-spectrogram transform
        self.mel_transform = self._build_mel_transform()

        # Build speaker vocabulary (for multi-speaker support)
        self.speaker_vocab = self._build_speaker_vocab()

        print(f"Dataset loaded: {len(self.dataset)} samples")

    def _load_and_filter_dataset(self, limit_samples: Optional[int] = None) -> HFDataset:
        """
        Load and filter dataset based on configuration.

        Args:
            limit_samples: Maximum number of samples to load

        Returns:
            Filtered Hugging Face dataset
        """
        # Load dataset
        dataset = load_dataset(
            self.config['dataset_name'],
            split=self.split,
            cache_dir=self.config.get('cache_dir')
        )

        # Limit samples if specified (for debugging)
        if limit_samples is not None:
            dataset = dataset.select(range(min(limit_samples, len(dataset))))

        # Filter by text length (word count)
        min_words = self.config.get('filter_min_words', 3)
        max_words = self.config.get('filter_max_words', 20)

        def filter_by_words(example):
            words = example['sentence'].split()
            return min_words <= len(words) <= max_words

        dataset = dataset.filter(filter_by_words)

        # Filter by audio duration
        min_duration = self.config.get('filter_min_duration', 0.5)
        max_duration = self.config.get('filter_max_duration', 15.0)

        def filter_by_duration(example):
            audio = example['audio']
            duration = len(audio['array']) / audio['sampling_rate']
            return min_duration <= duration <= max_duration

        dataset = dataset.filter(filter_by_duration)

        # Filter Turkish-only text (remove English/code-switching)
        turkish_pattern = r'^[a-zçğıöşüA-ZÇĞİÖŞÜ\s\'\-,.!?0-9]+$'

        def filter_turkish_only(example):
            return bool(re.match(turkish_pattern, example['sentence']))

        dataset = dataset.filter(filter_turkish_only)

        return dataset

    def _build_mel_transform(self) -> torchaudio.transforms.MelSpectrogram:
        """
        Build mel-spectrogram transform based on configuration.

        Returns:
            MelSpectrogram transform
        """
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.config.get('n_fft', 2048),
            hop_length=self.config.get('hop_length', 300),
            win_length=self.config.get('win_length', 1200),
            n_mels=self.config.get('n_mels', 80),
            f_min=self.config.get('mel_fmin', 0),
            f_max=self.config.get('mel_fmax', 8000),
            power=2.0,
        )

    def _build_speaker_vocab(self) -> Dict[str, int]:
        """
        Build speaker vocabulary for multi-speaker training.

        Note: mediaspeech-with-cv-tr might not have explicit speaker IDs.
        This is a placeholder for future multi-speaker support.

        Returns:
            Speaker ID to index mapping
        """
        # Check if dataset has speaker information
        if 'speaker_id' in self.dataset.column_names:
            unique_speakers = set(self.dataset['speaker_id'])
            return {spk: idx for idx, spk in enumerate(sorted(unique_speakers))}
        else:
            # Single speaker (or no speaker info)
            return {"default": 0}

    def _resample_audio(self, audio_array: np.ndarray, orig_sample_rate: int) -> torch.Tensor:
        """
        Resample audio to target sample rate.

        Args:
            audio_array: Audio waveform as numpy array
            orig_sample_rate: Original sample rate

        Returns:
            Resampled audio as torch tensor
        """
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_array).float()

        # Ensure mono (average channels if stereo)
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)

        # Add channel dimension
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Resample if needed
        if orig_sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=self.target_sample_rate
            )
            audio_tensor = resampler(audio_tensor)

        return audio_tensor

    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio amplitude to [-1, 1].

        Args:
            audio: Audio tensor

        Returns:
            Normalized audio tensor
        """
        # Remove DC offset
        audio = audio - audio.mean()

        # Normalize to [-1, 1]
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val

        return audio

    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio waveform to mel-spectrogram.

        Args:
            audio: Audio tensor (1, T)

        Returns:
            Mel-spectrogram tensor (n_mels, T_mel)
        """
        # Compute mel-spectrogram
        mel = self.mel_transform(audio)

        # Convert to log scale (add small epsilon to avoid log(0))
        mel = torch.log(mel + 1e-5)

        # Remove channel dimension
        mel = mel.squeeze(0)

        return mel

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - 'text': Original text (str)
                - 'phonemes': Phoneme string (str)
                - 'phoneme_ids': Phoneme indices (torch.LongTensor)
                - 'mel': Mel-spectrogram (torch.FloatTensor, n_mels × T)
                - 'audio': Normalized audio waveform (torch.FloatTensor, 1 × T)
                - 'speaker_id': Speaker index (torch.LongTensor)
        """
        # Get raw data
        example = self.dataset[idx]

        # Extract text
        text = example['sentence']

        # Phonemize text
        phonemes = self.phonemizer.phonemize(text, normalize=True)
        phoneme_ids = torch.LongTensor(self.phonemizer.encode(phonemes))

        # Extract and process audio
        audio_array = example['audio']['array']
        orig_sample_rate = example['audio']['sampling_rate']

        # Resample to target sample rate
        audio = self._resample_audio(audio_array, orig_sample_rate)

        # Normalize audio
        audio = self._normalize_audio(audio)

        # Convert to mel-spectrogram
        mel = self._audio_to_mel(audio)

        # Get speaker ID
        if 'speaker_id' in example:
            speaker_id = self.speaker_vocab[example['speaker_id']]
        else:
            speaker_id = 0  # Default speaker

        speaker_id = torch.LongTensor([speaker_id])

        return {
            'text': text,
            'phonemes': phonemes,
            'phoneme_ids': phoneme_ids,
            'mel': mel,
            'audio': audio,
            'speaker_id': speaker_id,
        }


class TTSCollator:
    """
    Collator for batching TTS samples with padding.

    Handles variable-length sequences by padding phoneme IDs and mel-spectrograms
    to the maximum length in each batch.
    """

    def __init__(self, pad_token_id: int = 0):
        """
        Initialize collator.

        Args:
            pad_token_id: Phoneme ID used for padding
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Args:
            batch: List of dataset items

        Returns:
            Batched tensors with padding:
                - 'text': List of original texts
                - 'phonemes': List of phoneme strings
                - 'phoneme_ids': Padded phoneme IDs (B, max_phoneme_len)
                - 'phoneme_lengths': Original phoneme lengths (B,)
                - 'mel': Padded mel-spectrograms (B, n_mels, max_mel_len)
                - 'mel_lengths': Original mel lengths (B,)
                - 'audio': List of audio tensors (variable length)
                - 'speaker_ids': Speaker IDs (B,)
        """
        # Extract fields
        texts = [item['text'] for item in batch]
        phonemes = [item['phonemes'] for item in batch]
        phoneme_ids = [item['phoneme_ids'] for item in batch]
        mels = [item['mel'] for item in batch]
        audios = [item['audio'] for item in batch]
        speaker_ids = torch.cat([item['speaker_id'] for item in batch])

        # Get lengths before padding
        phoneme_lengths = torch.LongTensor([len(p) for p in phoneme_ids])
        mel_lengths = torch.LongTensor([mel.size(1) for mel in mels])

        # Pad phoneme IDs
        phoneme_ids_padded = pad_sequence(
            phoneme_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )

        # Pad mel-spectrograms
        # Mels have shape (n_mels, T), need to pad along T dimension
        max_mel_len = max(mel.size(1) for mel in mels)
        n_mels = mels[0].size(0)

        mel_padded = torch.zeros(len(mels), n_mels, max_mel_len)
        for i, mel in enumerate(mels):
            mel_padded[i, :, :mel.size(1)] = mel

        return {
            'text': texts,
            'phonemes': phonemes,
            'phoneme_ids': phoneme_ids_padded,
            'phoneme_lengths': phoneme_lengths,
            'mel': mel_padded,
            'mel_lengths': mel_lengths,
            'audio': audios,
            'speaker_ids': speaker_ids,
        }


# ============================================================================
# Utility functions
# ============================================================================

def create_dataloader(
    dataset: TurkishTTSDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create DataLoader for TTS dataset.

    Args:
        dataset: TurkishTTSDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (not useful for MPS)

    Returns:
        DataLoader instance
    """
    collator = TTSCollator(pad_token_id=dataset.phonemizer.phoneme_vocab['<PAD>'])

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
    )


def split_dataset(
    dataset: HFDataset,
    train_ratio: float = 0.8
) -> Tuple[HFDataset, HFDataset]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Hugging Face dataset
        train_ratio: Ratio of training data

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    split_dict = dataset.train_test_split(test_size=1.0 - train_ratio, seed=42)
    return split_dict['train'], split_dict['test']


def build_phoneme_vocab_from_dataset(
    dataset: HFDataset,
    phonemizer: TurkishPhonemizer,
    save_path: Optional[str] = None
) -> Dict[str, int]:
    """
    Build phoneme vocabulary from entire dataset.

    Args:
        dataset: Hugging Face dataset
        phonemizer: TurkishPhonemizer instance
        save_path: Optional path to save vocabulary

    Returns:
        Phoneme vocabulary
    """
    print("Building phoneme vocabulary from dataset...")

    # Extract all texts
    texts = dataset['sentence']

    # Build vocabulary
    vocab = phonemizer.build_vocab_from_texts(texts, min_frequency=1)

    print(f"Vocabulary size: {len(vocab)}")

    # Save if path provided
    if save_path:
        phonemizer.save_vocab(save_path)
        print(f"Vocabulary saved to {save_path}")

    return vocab


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    import yaml
    from phonemizer import TurkishPhonemizer

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create phonemizer
    phonemizer = TurkishPhonemizer()

    # Load dataset (limited to 100 samples for testing)
    print("\n" + "="*50)
    print("Testing TurkishTTSDataset")
    print("="*50 + "\n")

    dataset = TurkishTTSDataset(
        config=config,
        phonemizer=phonemizer,
        split="train",
        limit_samples=100  # Limit for quick testing
    )

    # Build vocabulary
    full_dataset = load_dataset(
        config['data']['dataset_name'],
        split="train[:100]"  # Use small subset
    )
    vocab = build_phoneme_vocab_from_dataset(full_dataset, phonemizer)

    # Test dataset
    print(f"\nDataset size: {len(dataset)}")
    print(f"Speaker vocabulary: {dataset.speaker_vocab}")

    # Get a sample
    print("\n" + "-"*50)
    print("Sample item:")
    print("-"*50)
    sample = dataset[0]
    print(f"Text: {sample['text']}")
    print(f"Phonemes: {sample['phonemes']}")
    print(f"Phoneme IDs shape: {sample['phoneme_ids'].shape}")
    print(f"Mel shape: {sample['mel'].shape}")
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Speaker ID: {sample['speaker_id'].item()}")

    # Test dataloader
    print("\n" + "-"*50)
    print("Testing DataLoader:")
    print("-"*50)
    dataloader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )

    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Phoneme IDs shape: {batch['phoneme_ids'].shape}")
    print(f"Mel shape: {batch['mel'].shape}")
    print(f"Phoneme lengths: {batch['phoneme_lengths']}")
    print(f"Mel lengths: {batch['mel_lengths']}")
    print(f"Speaker IDs: {batch['speaker_ids']}")
