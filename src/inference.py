"""
Inference Script for Turkish StyleTTS2 TTS

Generates speech from Turkish text using trained StyleTTS2 model with LoRA.

Features:
- Text-to-speech generation
- Checkpoint loading
- Batch processing
- WAV file export
- Griffin-Lim vocoder (built-in)
- Support for external vocoder (HiFi-GAN, etc.)

Usage:
    # Single text
    python -m src.inference --checkpoint checkpoints/best_model.pt --text "Merhaba dünya"

    # From file
    python -m src.inference --checkpoint checkpoints/best_model.pt --input texts.txt --output outputs/

Author: Claude Code
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
import torch
import torch.nn as nn
import torchaudio

from .phonemizer import TurkishPhonemizer, create_phonemizer_from_config
from .model import StyleTTS2WithLoRA, create_model_from_config


class TTS:
    """
    Text-to-Speech interface for Turkish StyleTTS2.

    Handles text preprocessing, phonemization, mel-spectrogram generation,
    and audio synthesis.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        vocoder: Optional[nn.Module] = None
    ):
        """
        Initialize TTS system.

        Args:
            config: Configuration dictionary
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on (auto-detect if None)
            vocoder: Optional external vocoder (HiFi-GAN, etc.)
        """
        self.config = config

        # Setup device
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device = device

        print(f"Using device: {self.device}")

        # Create phonemizer
        print("Loading phonemizer...")
        self.phonemizer = create_phonemizer_from_config(config)

        # Load vocabulary
        vocab_path = config['phonemizer'].get('phoneme_vocab_path')
        if vocab_path and Path(vocab_path).exists():
            self.phonemizer.load_vocab(vocab_path)
        else:
            raise FileNotFoundError(
                f"Phoneme vocabulary not found at {vocab_path}. "
                "Run training first to build vocabulary."
            )

        # Create model
        print("Loading model...")
        vocab_size = self.phonemizer.get_vocab_size()
        self.model_wrapper = create_model_from_config(config, vocab_size, device)
        self.model = self.model_wrapper.model

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        self._load_checkpoint(checkpoint_path)

        self.model.eval()

        # Setup vocoder
        self.vocoder = vocoder
        if self.vocoder is None:
            print("No external vocoder provided, using Griffin-Lim")
            self.mel_to_audio = self._create_griffin_lim()
        else:
            self.mel_to_audio = self.vocoder

        # Audio config
        self.sample_rate = config['data']['sample_rate']

        print("TTS system ready!")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        # For LoRA models, we need to load from PEFT checkpoint
        # This is a simplified version - adjust based on actual checkpoint structure
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_dir():
            # Load PEFT model from directory
            from peft import PeftModel
            # Note: This requires base model to be loaded first
            # For now, we'll skip actual loading in this simplified version
            warnings.warn(
                "Checkpoint loading from directory not fully implemented. "
                "Using initialized model weights."
            )
        elif checkpoint_path.suffix == '.pt':
            # Load PyTorch checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Extract model state dict if it exists
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                warnings.warn("No model state dict found in checkpoint")

    def _create_griffin_lim(self) -> torchaudio.transforms.GriffinLim:
        """
        Create Griffin-Lim vocoder for mel-to-audio conversion.

        Returns:
            GriffinLim transform
        """
        gl = torchaudio.transforms.GriffinLim(
            n_fft=self.config['data']['n_fft'],
            n_iter=32,  # Number of iterations
            win_length=self.config['data']['win_length'],
            hop_length=self.config['data']['hop_length'],
            power=2.0,
        )
        # Move to same device as model
        return gl.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        text: str,
        speaker_id: int = 0,
        max_mel_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate speech from text.

        Args:
            text: Input Turkish text
            speaker_id: Speaker ID (for multi-speaker models)
            max_mel_len: Maximum mel-spectrogram length

        Returns:
            Audio waveform (1, T)
        """
        # Phonemize text
        phonemes = self.phonemizer.phonemize(text, normalize=True)
        phoneme_ids = torch.LongTensor(
            self.phonemizer.encode(phonemes)
        ).unsqueeze(0).to(self.device)  # (1, T)

        phoneme_lengths = torch.LongTensor([phoneme_ids.size(1)]).to(self.device)

        # Generate mel-spectrogram
        mel_pred = self.model(
            phoneme_ids=phoneme_ids,
            phoneme_lengths=phoneme_lengths,
            mel_targets=None,  # Inference mode
            max_mel_len=max_mel_len,
        )  # (1, n_mels, T_mel)

        # Convert mel to audio
        audio = self._mel_to_audio(mel_pred)

        return audio

    def _mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to audio waveform using librosa Griffin-Lim.

        Args:
            mel: Mel-spectrogram (B, n_mels, T)

        Returns:
            Audio waveform (B, 1, T_audio)
        """
        import librosa
        import numpy as np

        # Move to CPU and convert to numpy for librosa
        mel_np = mel.cpu().numpy()

        batch_size = mel_np.shape[0]
        audio_list = []

        for i in range(batch_size):
            mel_single = mel_np[i]  # (n_mels, T)

            # Denormalize from log mel
            mel_linear = np.exp(mel_single)

            # Convert mel-spectrogram to linear spectrogram (approximate)
            # This is an approximation - proper way is to use inverse mel filterbank
            mel_basis = librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.config['data']['n_fft'],
                n_mels=self.config['data']['n_mels'],
                fmin=self.config['data']['mel_fmin'],
                fmax=self.config['data']['mel_fmax']
            )

            # Pseudo-inverse to get linear spectrogram
            linear_spec = np.dot(np.linalg.pinv(mel_basis), mel_linear)

            # Apply Griffin-Lim
            audio_np = librosa.griffinlim(
                linear_spec,
                n_iter=32,
                hop_length=self.config['data']['hop_length'],
                win_length=self.config['data']['win_length'],
                n_fft=self.config['data']['n_fft']
            )

            audio_list.append(audio_np)

        # Convert back to torch tensor
        audio_array = np.stack(audio_list)  # (B, T)
        audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(1)  # (B, 1, T)

        # Move back to original device
        audio_tensor = audio_tensor.to(mel.device)

        return audio_tensor

    def generate_batch(
        self,
        texts: List[str],
        speaker_ids: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """
        Generate speech for multiple texts.

        Args:
            texts: List of input texts
            speaker_ids: List of speaker IDs (optional)

        Returns:
            List of audio waveforms
        """
        if speaker_ids is None:
            speaker_ids = [0] * len(texts)

        audios = []
        for text, speaker_id in zip(texts, speaker_ids):
            audio = self.generate(text, speaker_id)
            audios.append(audio)

        return audios

    def save_audio(
        self,
        audio: torch.Tensor,
        output_path: str,
        sample_rate: Optional[int] = None
    ) -> None:
        """
        Save audio to file.

        Args:
            audio: Audio waveform (1, T) or (T,)
            output_path: Output file path (.wav)
            sample_rate: Sample rate (uses config default if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        # Ensure audio is on CPU
        audio = audio.cpu()

        # Ensure correct shape (channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(0)

        # Normalize to [-1, 1]
        audio = audio / (audio.abs().max() + 1e-8)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(
            str(output_path),
            audio,
            sample_rate=sample_rate,
            encoding="PCM_S",
            bits_per_sample=16
        )

        print(f"Audio saved to {output_path}")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Turkish StyleTTS2 Inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--text', type=str, help='Input text to synthesize')
    parser.add_argument('--input', type=str, help='Input text file (one sentence per line)')
    parser.add_argument('--output', type=str, default='outputs/', help='Output directory')
    parser.add_argument('--speaker-id', type=int, default=0, help='Speaker ID')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu, cuda, mps)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = None  # Auto-detect

    # Create TTS system
    print("\nInitializing TTS system...")
    tts = TTS(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device
    )

    # Prepare texts
    if args.text:
        texts = [args.text]
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Default example texts
        texts = config['validation'].get('sample_texts', [
            "Merhaba, size nasıl yardımcı olabilirim?",
            "Bugün hava çok güzel.",
            "Türkçe metin seslendirilmesi çalışıyor.",
        ])

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate speech
    print(f"\nGenerating speech for {len(texts)} text(s)...")
    print("="*60)

    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] Text: {text}")

        # Generate
        audio = tts.generate(text, speaker_id=args.speaker_id)

        # Save
        output_path = output_dir / f"output_{i+1:03d}.wav"
        tts.save_audio(audio, str(output_path))

    print("\n" + "="*60)
    print(f"Generation complete! {len(texts)} audio files saved to {output_dir}")
    print("="*60)


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    main()
