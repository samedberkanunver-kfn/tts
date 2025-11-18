"""
Training Script for Turkish StyleTTS2 Fine-Tuning

Supports:
- Apple Silicon (MPS) training
- Mixed precision (FP16)
- LoRA parameter-efficient fine-tuning
- Checkpoint saving/loading
- TensorBoard logging
- Early stopping
- Gradient accumulation

Usage:
    python -m src.train --config config.yaml

Author: Claude Code
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional, Any
import time

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import load_dataset

from .phonemizer import TurkishPhonemizer, create_phonemizer_from_config
from .dataset import (
    TurkishTTSDataset,
    create_dataloader,
    build_phoneme_vocab_from_dataset
)
from .model import StyleTTS2WithLoRA, create_model_from_config


class Trainer:
    """
    Trainer for Turkish StyleTTS2 fine-tuning.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: StyleTTS2WithLoRA,
        phonemizer: TurkishPhonemizer,
        device: torch.device,
    ):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
            model: StyleTTS2WithLoRA instance
            phonemizer: TurkishPhonemizer instance
            device: Training device
        """
        self.config = config
        self.model_wrapper = model
        self.model = model.model  # PEFT model
        self.phonemizer = phonemizer
        self.device = device

        # Training config
        self.train_config = config['training']

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler (for FP16)
        self.use_amp = self.train_config.get('mixed_precision', False)
        if self.use_amp and self.device.type == 'mps':
            # MPS has limited AMP support, use with caution
            warnings.warn(
                "Mixed precision on MPS may not work for all operations. "
                "Disabling if errors occur."
            )
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Loss function
        self.loss_type = self.train_config.get('loss_type', 'l1')
        if self.loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # TensorBoard
        if self.train_config.get('use_tensorboard', True):
            log_dir = Path(self.train_config.get('tensorboard_dir', './runs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        # Checkpoint directory
        self.checkpoint_dir = Path(self.train_config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        optimizer_name = self.train_config.get('optimizer', 'AdamW')

        if optimizer_name == 'AdamW':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.train_config['learning_rate'],
                weight_decay=self.train_config.get('weight_decay', 0.01),
                betas=self.train_config.get('betas', [0.9, 0.999]),
                eps=self.train_config.get('eps', 1e-8),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.train_config.get('scheduler', 'linear')

        if scheduler_type == 'linear':
            warmup_steps = self.train_config.get('warmup_steps', 4000)
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        elif scheduler_type == 'constant':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Gradient accumulation
        grad_accum_steps = self.train_config.get('gradient_accumulation_steps', 1)

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            phoneme_ids = batch['phoneme_ids'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            mel_targets = batch['mel'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    mel_pred = self.model(
                        phoneme_ids=phoneme_ids,
                        phoneme_lengths=phoneme_lengths,
                        mel_targets=mel_targets,
                    )
                    # Compute loss (only on non-padded regions)
                    loss = self._compute_loss(mel_pred, mel_targets, mel_lengths)
                    loss = loss / grad_accum_steps
            else:
                mel_pred = self.model(
                    phoneme_ids=phoneme_ids,
                    phoneme_lengths=phoneme_lengths,
                    mel_targets=mel_targets,
                )
                loss = self._compute_loss(mel_pred, mel_targets, mel_lengths)
                loss = loss / grad_accum_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                if self.train_config.get('max_grad_norm'):
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config['max_grad_norm']
                    )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                # Zero gradients
                self.optimizer.zero_grad()

                # Update global step
                self.global_step += 1

                # Logging
                if self.global_step % self.train_config.get('log_interval', 100) == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if self.writer:
                        self.writer.add_scalar('train/loss', loss.item() * grad_accum_steps, self.global_step)
                        self.writer.add_scalar('train/lr', current_lr, self.global_step)

                # Checkpointing
                if self.global_step % self.train_config.get('save_interval', 1000) == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

            # Update progress bar
            total_loss += loss.item() * grad_accum_steps
            num_batches += 1
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        return total_loss / num_batches

    def _compute_loss(
        self,
        mel_pred: torch.Tensor,
        mel_targets: torch.Tensor,
        mel_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with masking for variable-length sequences.

        Args:
            mel_pred: Predicted mel-spectrograms (B, n_mels, T)
            mel_targets: Target mel-spectrograms (B, n_mels, T)
            mel_lengths: Original mel lengths (B,)

        Returns:
            Loss value
        """
        # Create mask
        batch_size, n_mels, max_len = mel_pred.shape
        mask = torch.arange(max_len, device=mel_pred.device).expand(
            batch_size, max_len
        ) < mel_lengths.unsqueeze(1)

        # Expand mask to match mel dimensions
        mask = mask.unsqueeze(1).expand_as(mel_pred)  # (B, n_mels, T)

        # Compute masked loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(mel_pred * mask, mel_targets * mask, reduction='sum')
        else:  # mse
            loss = F.mse_loss(mel_pred * mask, mel_targets * mask, reduction='sum')

        # Normalize by number of non-masked elements
        num_elements = mask.sum()
        loss = loss / num_elements

        return loss

    @torch.no_grad()
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validating"):
            # Move batch to device
            phoneme_ids = batch['phoneme_ids'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            mel_targets = batch['mel'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)

            # Forward pass
            mel_pred = self.model(
                phoneme_ids=phoneme_ids,
                phoneme_lengths=phoneme_lengths,
                mel_targets=mel_targets,
            )

            # Compute loss
            loss = self._compute_loss(mel_pred, mel_targets, mel_lengths)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> None:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        num_epochs = self.train_config.get('num_epochs', 100)
        early_stopping = self.train_config.get('early_stopping', True)
        patience = self.train_config.get('patience', 10)

        print("\n" + "="*60)
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Loss type: {self.loss_type}")
        print("="*60 + "\n")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch}: Train loss = {train_loss:.4f}")

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch}: Val loss = {val_loss:.4f}")

                # Log to TensorBoard
                if self.writer:
                    self.writer.add_scalar('val/loss', val_loss, epoch)

                # Early stopping
                if early_stopping:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        # Save best model
                        self.save_checkpoint('best_model.pt')
                        print(f"New best validation loss: {val_loss:.4f}")
                    else:
                        self.epochs_without_improvement += 1
                        print(f"No improvement for {self.epochs_without_improvement} epochs")

                        if self.epochs_without_improvement >= patience:
                            print(f"Early stopping after {epoch + 1} epochs")
                            break

            # Save epoch checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60 + "\n")

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, filename: str) -> None:
        """
        Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        # Save LoRA weights (PEFT model)
        lora_dir = self.checkpoint_dir / 'lora' / filename.replace('.pt', '')
        self.model_wrapper.save_lora_weights(str(lora_dir))

        # Save training state
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Keep only last N checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        keep_last_n = self.train_config.get('keep_last_n_checkpoints', 3)

        # Get all checkpoint files (excluding best_model.pt)
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob('checkpoint_*.pt')],
            key=lambda x: x.stat().st_mtime
        )

        # Remove old checkpoints
        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint.unlink()
                print(f"Removed old checkpoint: {checkpoint.name}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Turkish StyleTTS2')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--limit-samples', type=int, default=None, help='Limit dataset size (for debugging)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    torch.manual_seed(config['system']['seed'])

    # Setup device
    device_name = config['training']['device']
    if device_name == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        warnings.warn(f"Requested device '{device_name}' not available, using CPU")

    print(f"Using device: {device}")

    # Create phonemizer
    print("\nInitializing phonemizer...")
    phonemizer = create_phonemizer_from_config(config)

    # Load dataset and build vocabulary
    print("\nLoading dataset and building vocabulary...")
    dataset_name = config['data']['dataset_name']

    # Load small subset to build vocabulary
    dataset_for_vocab = load_dataset(dataset_name, split='train[:1000]')
    vocab = build_phoneme_vocab_from_dataset(
        dataset_for_vocab,
        phonemizer,
        save_path=config['phonemizer'].get('phoneme_vocab_path')
    )

    vocab_size = phonemizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TurkishTTSDataset(
        config=config,
        phonemizer=phonemizer,
        split='train',
        limit_samples=args.limit_samples
    )

    val_dataset = TurkishTTSDataset(
        config=config,
        phonemizer=phonemizer,
        split='test',
        limit_samples=args.limit_samples // 10 if args.limit_samples else None
    )

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['validation']['val_batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )

    # Create model
    print("\nCreating model...")
    model = create_model_from_config(config, vocab_size, device)

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(config, model, phonemizer, device)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
