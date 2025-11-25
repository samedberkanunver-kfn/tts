"""
Kokoro-82M Training Script for Turkish Fine-Tuning with LoRA

Fine-tunes the pretrained Kokoro-82M model on Turkish dataset using
LoRA (Low-Rank Adaptation) for parameter-efficient training.

Features:
- LoRA fine-tuning (only ~109K trainable params)
- Turkish dataset (zeynepgulhan/mediaspeech-with-cv-tr)
- Mel-spectrogram reconstruction loss
- Apple Silicon (MPS) support
- Checkpoint saving
- TensorBoard logging
- A100 Optimizations: AMP, TF32, Dynamic Workers

Usage:
    # Debug mode (100 samples)
    python -m src.kokoro_train --limit-samples 100

    # Full training (A100 recommended: batch-size 32-64)
    python -m src.kokoro_train --batch-size 64

Author: Claude Code
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional, Any
import time
import os

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import load_dataset
import torchaudio

from .kokoro_model import KokoroWithLoRA
from .phonemizer import TurkishPhonemizer


class KokoroTurkishDataset(Dataset):
    """
    Turkish TTS dataset for Kokoro fine-tuning.

    Loads zeynepgulhan/mediaspeech-with-cv-tr and prepares:
    - Turkish text → IPA phonemes → Kokoro token IDs
    - Audio → 24kHz waveform
    """

    def __init__(
        self,
        split: str = "train",
        limit_samples: Optional[int] = None,
        vocab: Dict[str, int] = None,
        sample_rate: int = 24000
    ):
        """
        Initialize dataset.

        Args:
            split: "train" or "test"
            limit_samples: Limit number of samples (for debugging)
            vocab: Kokoro phoneme vocabulary
            sample_rate: Target sample rate (24kHz for Kokoro)
        """
        self.split = split
        self.vocab = vocab
        self.sample_rate = sample_rate

        # Create phonemizer
        self.phonemizer = TurkishPhonemizer()

        # Load vocabulary if not provided
        vocab_path = "data/phoneme_vocab.json"
        if Path(vocab_path).exists():
            self.phonemizer.load_vocab(vocab_path)

        # Load dataset
        print(f"Loading dataset ({split})...")
        dataset = load_dataset(
            "zeynepgulhan/mediaspeech-with-cv-tr",
            split=split
        )

        # Limit samples
        if limit_samples:
            dataset = dataset.select(range(min(limit_samples, len(dataset))))

        # Filter by text length and audio duration (avoid very short/long)
        def filter_fn(example):
            words = example['sentence'].split()
            # Text: 2-15 words (shorter to reduce predicted audio length)
            if not (2 <= len(words) <= 15):
                return False
            # Audio: 0.5s - 8s (shorter samples to prevent OOM)
            audio_len = len(example['audio']['array'])
            sr = example['audio']['sampling_rate']
            duration = audio_len / sr
            if not (0.5 <= duration <= 8.0):
                return False
            return True

        dataset = dataset.filter(filter_fn)

        self.dataset = dataset
        print(f"✅ Loaded {len(self.dataset)} samples")

    def text_to_tokens(self, text: str) -> torch.LongTensor:
        """
        Convert Turkish text to Kokoro token IDs.

        Args:
            text: Turkish text

        Returns:
            Token IDs (T,)
        """
        # Phonemize (espeak-ng → IPA)
        phonemes = self.phonemizer.phonemize(text, normalize=True)

        # Map to Kokoro vocab
        token_ids = []
        for char in phonemes:
            if char in self.vocab:
                token_ids.append(self.vocab[char])
            # Skip unknown chars

        # Add BOS/EOS
        token_ids = [0] + token_ids + [0]

        return torch.LongTensor(token_ids)

    def process_audio(self, audio_array, orig_sr):
        """Process audio to 24kHz mono."""
        # To tensor
        audio = torch.from_numpy(audio_array).float()

        # Ensure mono
        if audio.dim() > 1:
            audio = audio.mean(dim=0)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            audio = resampler(audio)

        # Normalize
        audio = audio - audio.mean()
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val

        return audio.squeeze(0)  # (T,)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Text → tokens
        text = example['sentence']
        tokens = self.text_to_tokens(text)

        # Audio
        audio_array = example['audio']['array']
        orig_sr = example['audio']['sampling_rate']
        audio = self.process_audio(audio_array, orig_sr)

        return {
            'text': text,
            'tokens': tokens,
            'audio': audio
        }


def collate_fn(batch):
    """Collate batch with padding."""
    texts = [item['text'] for item in batch]
    tokens = [item['tokens'] for item in batch]
    audios = [item['audio'] for item in batch]

    # Pad tokens to fixed length (reduced to 128 for memory efficiency)
    FIXED_TOKEN_LEN = 128
    tokens_padded = torch.zeros(len(tokens), FIXED_TOKEN_LEN, dtype=torch.long)
    token_lengths = torch.LongTensor([min(len(t), FIXED_TOKEN_LEN) for t in tokens])

    for i, t in enumerate(tokens):
        length = min(len(t), FIXED_TOKEN_LEN)
        tokens_padded[i, :length] = t[:length]

    # Pad audios
    max_audio_len = max(len(a) for a in audios)
    audios_padded = torch.zeros(len(audios), max_audio_len)
    audio_lengths = torch.LongTensor([len(a) for a in audios])

    for i, a in enumerate(audios):
        audios_padded[i, :len(a)] = a

    return {
        'text': texts,
        'tokens': tokens_padded,
        'token_lengths': token_lengths,
        'audio': audios_padded,
        'audio_lengths': audio_lengths
    }


class KokoroTrainer:
    """
    Trainer for Kokoro-82M fine-tuning on Turkish.
    """

    def __init__(
        self,
        model: KokoroWithLoRA,
        voice_embedding: torch.Tensor,
        device: torch.device,
        lr: float = 1e-4,
        log_dir: str = "runs/kokoro"
    ):
        """
        Initialize trainer.

        Args:
            model: KokoroWithLoRA instance
            voice_embedding: Reference voice (256-dim)
            device: Training device
            lr: Learning rate
            log_dir: TensorBoard log directory
        """
        self.model = model
        self.voice_embedding = voice_embedding.to(device)
        self.device = device

        # Optimizer (only LoRA params)
        self.optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        # AMP Scaler for Mixed Precision Training
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        # Mel transform for loss
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=2048,
            hop_length=300,
            win_length=1200,
            n_mels=80,
            f_min=0,
            f_max=8000,
            power=2.0
        ).to(device)

        # TensorBoard
        self.writer = SummaryWriter(log_dir)

        # State
        self.global_step = 0
        self.best_loss = float('inf')

    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        successful_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            tokens = batch['tokens'].to(self.device)
            token_lengths = batch['token_lengths'].to(self.device)
            target_audio = batch['audio'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)

            batch_size = tokens.size(0)

            # Check for inf/nan in raw target audio
            if not torch.isfinite(target_audio).all():
                print(f"Skipping batch {batch_idx}: raw target_audio contains inf/nan")
                continue

            # Normalize target audio early (before any processing)
            target_audio = target_audio / (target_audio.abs().max() + 1e-8)

            # Reference style (same for all in batch)
            ref_s = self.voice_embedding.unsqueeze(0).expand(batch_size, -1)

            # Forward with Mixed Precision
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                try:
                    pred_audio, pred_dur = self.model.model(tokens, ref_s=ref_s, speed=1.0)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print(f"OOM error, clearing cache and skipping batch")
                    else:
                        print(f"Forward error: {e}")
                    continue

                # Ensure pred_audio is 2D (batch, time)
                if pred_audio.dim() == 1:
                    pred_audio = pred_audio.unsqueeze(0)

                # Limit max audio length to prevent OOM (max 10 seconds at 24kHz)
                MAX_AUDIO_LEN = 240000  # 10 seconds
                if pred_audio.size(1) > MAX_AUDIO_LEN:
                    pred_audio = pred_audio[:, :MAX_AUDIO_LEN]
                if target_audio.size(1) > MAX_AUDIO_LEN:
                    target_audio = target_audio[:, :MAX_AUDIO_LEN]

                # Check for inf/nan in model output early
                if not torch.isfinite(pred_audio).all():
                    print(f"Warning batch {batch_idx}: model output contains inf/nan, skipping")
                    self.optimizer.zero_grad()
                    continue

                # Debug: print shapes
                if batch_idx == 0:
                    print(f"\\n[DEBUG] Batch {batch_idx}:")
                    print(f"  pred_audio shape: {pred_audio.shape}")
                    print(f"  target_audio shape: {target_audio.shape}")

                # Align lengths (pad to match target)
                max_len = target_audio.size(1)

                if pred_audio.size(1) < max_len:
                    # Pad predicted
                    pad_len = max_len - pred_audio.size(1)
                    pred_audio = F.pad(pred_audio, (0, pad_len))
                elif pred_audio.size(1) > max_len:
                    # Truncate predicted
                    pred_audio = pred_audio[:, :max_len]

                # Normalize pred_audio
                pred_audio_norm = pred_audio / (pred_audio.abs().max() + 1e-8)

                # Convert to mel in float32 (outside autocast to prevent overflow)
                with torch.amp.autocast('cuda', enabled=False):
                    pred_mel = self.mel_transform(pred_audio_norm.float().unsqueeze(1))
                    target_mel = self.mel_transform(target_audio.float().unsqueeze(1))

                    # Convert to log-mel with clamping
                    pred_mel = torch.log(pred_mel.clamp(min=1e-5, max=1e8))
                    target_mel = torch.log(target_mel.clamp(min=1e-5, max=1e8))

                # Match mel lengths
                min_mel_len = min(pred_mel.size(2), target_mel.size(2))
                pred_mel = pred_mel[:, :, :min_mel_len]
                target_mel = target_mel[:, :, :min_mel_len]

                # Mel reconstruction loss
                loss = F.l1_loss(pred_mel, target_mel)

            # Skip if loss is inf/nan - debug why
            if not torch.isfinite(loss):
                # Debug: find the source of inf
                has_inf_pred = not torch.isfinite(pred_mel).all()
                has_inf_target = not torch.isfinite(target_mel).all()
                target_min = target_mel.min().item()
                target_max = target_mel.max().item()
                print(f"Skipping batch {batch_idx}: loss={loss.item():.2f}, "
                      f"inf_pred={has_inf_pred}, inf_target={has_inf_target}, "
                      f"target_range=[{target_min:.2f}, {target_max:.2f}]")
                # Clear any bad gradients
                self.optimizer.zero_grad()
                continue

            # Debug: print loss
            if batch_idx == 0:
                print(f"  Loss value: {loss.item():.6f}")

            # Backward with Scaler
            self.optimizer.zero_grad()
            
            if self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                self.optimizer.step()

            # Logging
            total_loss += loss.item()
            successful_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)

            self.global_step += 1

        if successful_batches == 0:
            return float('inf')
        return total_loss / successful_batches

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate."""
        self.model.eval()
        total_loss = 0.0
        successful_batches = 0

        for batch in tqdm(val_loader, desc="Validating"):
            tokens = batch['tokens'].to(self.device)
            target_audio = batch['audio'].to(self.device)

            batch_size = tokens.size(0)
            ref_s = self.voice_embedding.unsqueeze(0).expand(batch_size, -1)

            try:
                pred_audio, _ = self.model.model(tokens, ref_s=ref_s, speed=1.0)
            except Exception as e:
                print(f"Forward error: {e}")
                continue

            if pred_audio.dim() == 1:
                pred_audio = pred_audio.unsqueeze(0)

            # Align
            max_len = target_audio.size(1)
            if pred_audio.size(1) < max_len:
                pred_audio = F.pad(pred_audio, (0, max_len - pred_audio.size(1)))
            elif pred_audio.size(1) > max_len:
                pred_audio = pred_audio[:, :max_len]

            # Normalize audio to [-1, 1] to prevent mel overflow
            pred_audio_norm = pred_audio / (pred_audio.abs().max() + 1e-8)
            target_audio_norm = target_audio / (target_audio.abs().max() + 1e-8)

            # Mel loss in float32 (prevent overflow)
            pred_mel = self.mel_transform(pred_audio_norm.float().unsqueeze(1))
            target_mel = self.mel_transform(target_audio_norm.float().unsqueeze(1))

            # Convert to log-mel with clamping
            pred_mel = torch.log(pred_mel.clamp(min=1e-5, max=1e8))
            target_mel = torch.log(target_mel.clamp(min=1e-5, max=1e8))

            min_mel_len = min(pred_mel.size(2), target_mel.size(2))
            pred_mel = pred_mel[:, :, :min_mel_len]
            target_mel = target_mel[:, :, :min_mel_len]

            loss = F.l1_loss(pred_mel, target_mel)

            # Skip inf/nan losses
            if not torch.isfinite(loss):
                continue
            total_loss += loss.item()
            successful_batches += 1

        # Return previous best loss if no successful batches
        if successful_batches == 0:
            print("⚠️ Validation failed - no successful batches!")
            return float('inf')

        return total_loss / successful_batches

    def save_checkpoint(self, path: str):
        """Save LoRA checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        self.model.save_lora_weights(str(path))

        print(f"✅ Checkpoint saved to {path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Kokoro-82M Turkish Fine-Tuning')
    parser.add_argument('--model', type=str, default='models/kokoro-82m/kokoro-v1_0.pth')
    parser.add_argument('--config', type=str, default='models/kokoro-82m/config.json')
    parser.add_argument('--voice', type=str, default='models/kokoro-82m/voices/af_heart.pt')
    parser.add_argument('--limit-samples', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/kokoro_lora')
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    print(f"Using device: {device}")
    
    # Enable TF32 for A100
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("🚀 TF32 enabled for A100 optimization")

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    vocab = config['vocab']

    # Create datasets
    print("\n" + "="*60)
    print("Loading Turkish Dataset")
    print("="*60)

    train_dataset = KokoroTurkishDataset(
        split="train",
        limit_samples=args.limit_samples,
        vocab=vocab
    )

    val_dataset = KokoroTurkishDataset(
        split="test",
        limit_samples=args.limit_samples // 10 if args.limit_samples else 100,
        vocab=vocab
    )

    # Dataloaders
    # Optimize for CUDA (Colab) vs MPS (Mac)
    if device.type == 'cuda':
        # Use all available CPU cores for data loading
        num_workers = os.cpu_count()
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
    
    print(f"DataLoader config: num_workers={num_workers}, pin_memory={pin_memory}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Load model
    print("\n" + "="*60)
    print("Loading Kokoro-82M with LoRA")
    print("="*60)

    lora_config = {
        'r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1
    }

    model = KokoroWithLoRA(
        config=config,
        model_path=args.model,
        lora_config=lora_config,
        device=device
    )

    # Load voice
    print(f"\nLoading voice from {args.voice}...")
    voice = torch.load(args.voice, weights_only=True)
    # Use first frame
    voice_embedding = voice[0, 0]  # (256,)
    print(f"Voice embedding shape: {voice_embedding.shape}")

    # Create trainer
    trainer = KokoroTrainer(
        model=model,
        voice_embedding=voice_embedding,
        device=device,
        lr=args.lr
    )

    # Train
    print("\n" + "="*60)
    print(f"Starting Training ({args.epochs} epochs)")
    print("="*60 + "\n")

    for epoch in range(args.epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        # Validate
        val_loss = trainer.validate(val_loader)
        print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

        trainer.writer.add_scalar('val/loss', val_loss, epoch)

        # Save best
        if val_loss < trainer.best_loss:
            trainer.best_loss = val_loss
            trainer.save_checkpoint(f"{args.checkpoint_dir}/best_model")
            print(f"✅ New best model! Val Loss: {val_loss:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(f"{args.checkpoint_dir}/epoch_{epoch+1}")

    print("\n" + "="*60)
    print("✅ Training Complete!")
    print(f"Best Val Loss: {trainer.best_loss:.4f}")
    print("="*60)

    trainer.writer.close()


if __name__ == "__main__":
    main()
