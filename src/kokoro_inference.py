"""
Kokoro-82M Inference Script with ISTFTNet Vocoder

This script provides TTS inference using the pretrained Kokoro-82M model
with optional LoRA fine-tuned weights for Turkish.

Features:
- Pretrained Kokoro-82M model
- ISTFTNet vocoder (high-quality, fast)
- Voice cloning with reference style vectors
- Batch processing
- 24kHz audio output

Usage:
    # Pretrained model
    python -m src.kokoro_inference --text "Hello world" --voice af_heart

    # With fine-tuned LoRA
    python -m src.kokoro_inference --text "Merhaba dünya" \\
        --lora-weights checkpoints/kokoro_lora/best_model

Author: Claude Code
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, List

import torch
import torchaudio
import json

from .kokoro_model import KokoroModel, KokoroWithLoRA
from .phonemizer import TurkishPhonemizer


class KokoroTTS:
    """
    Kokoro-82M TTS inference system.

    Handles text-to-speech generation using pretrained Kokoro-82M
    with optional LoRA fine-tuning for Turkish.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        voice_path: str,
        lora_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Kokoro TTS system.

        Args:
            model_path: Path to kokoro-v1_0.pth
            config_path: Path to config.json
            voice_path: Path to voice .pt file (e.g., af_heart.pt)
            lora_path: Optional path to fine-tuned LoRA weights directory
            device: Device to run inference on (auto-detect if None)
        """
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

        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.vocab = self.config['vocab']
        self.sample_rate = 24000  # Kokoro uses 24kHz

        # Initialize phonemizer
        print("Initializing Turkish Phonemizer...")
        self.phonemizer = TurkishPhonemizer()
        # Load vocab if available in config or default path
        vocab_path = self.config.get('phonemizer', {}).get('phoneme_vocab_path', 'data/phoneme_vocab.json')
        if Path(vocab_path).exists():
            self.phonemizer.load_vocab(vocab_path)
            print(f"Loaded phoneme vocabulary from {vocab_path}")

        # Load base model
        print("Loading Kokoro-82M model...")
        self.model = KokoroModel(
            config=self.config,
            model_path=model_path,
            disable_complex=True  # Use CustomSTFT for MPS compatibility
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load LoRA weights if provided
        if lora_path:
            print(f"Loading LoRA weights from {lora_path}...")
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                print("✅ LoRA weights loaded successfully!")
            except Exception as e:
                warnings.warn(f"Failed to load LoRA weights: {e}")

        # Load voice/style vector
        print(f"Loading voice from {voice_path}...")
        self.voice = torch.load(voice_path, weights_only=True)  # Shape: (T, 1, 256)
        print(f"Voice shape: {self.voice.shape}")

        print("✅ Kokoro TTS system ready!")

    def text_to_phonemes(self, text: str) -> str:
        """
        Convert text to IPA phonemes.

        Args:
            text: Input text

        Returns:
            IPA phoneme string
        """
        return self.phonemizer.phonemize(text, normalize=True)

    def phonemes_to_tokens(self, phonemes: str) -> torch.LongTensor:
        """
        Convert phoneme string to token IDs.

        Args:
            phonemes: Phoneme string (IPA)

        Returns:
            Token IDs tensor (T,)
        """
        # Map each character to vocab ID
        # Unknown characters are skipped
        token_ids = []
        for char in phonemes:
            if char in self.vocab:
                token_ids.append(self.vocab[char])
            else:
                # Skip unknown characters
                # In production, use <UNK> token or better handling
                pass

        if not token_ids:
            raise ValueError(f"No valid phonemes found in: {phonemes}")

        # Add <BOS> and <EOS> tokens (0)
        token_ids = [0] + token_ids + [0]

        return torch.LongTensor(token_ids)

    @torch.no_grad()
    def generate(
        self,
        text: str,
        speed: float = 1.0,
        voice_index: int = 0
    ) -> torch.Tensor:
        """
        Generate speech from text.

        Args:
            text: Input text
            speed: Speaking speed multiplier
            voice_index: Index in voice tensor (for different prosody)

        Returns:
            Audio waveform tensor (T,)
        """
        # Text → Phonemes → Tokens
        phonemes = self.text_to_phonemes(text)
        print(f"Text: {text}")
        print(f"Phonemes: {phonemes}")

        token_ids = self.phonemes_to_tokens(phonemes)
        print(f"Token IDs: {token_ids.tolist()}")

        # Prepare input
        input_ids = token_ids.unsqueeze(0).to(self.device)  # (1, T)

        # Get reference style vector
        # voice shape: (T_voice, 1, 256)
        # We take a single frame (or average)
        voice_index = min(voice_index, self.voice.shape[0] - 1)
        ref_s = self.voice[voice_index].to(self.device)  # (1, 256)

        # Generate audio
        print(f"Generating audio with {input_ids.size(1)} tokens...")
        audio, pred_dur = self.model(input_ids, ref_s=ref_s, speed=speed)

        print(f"Predicted durations: {pred_dur.tolist()}")
        print(f"Audio shape: {audio.shape}")

        return audio.cpu()

    def generate_batch(
        self,
        texts: List[str],
        speed: float = 1.0
    ) -> List[torch.Tensor]:
        """
        Generate speech for multiple texts.

        Args:
            texts: List of input texts
            speed: Speaking speed multiplier

        Returns:
            List of audio waveform tensors
        """
        audios = []
        for text in texts:
            audio = self.generate(text, speed=speed)
            audios.append(audio)
        return audios

    def save_audio(
        self,
        audio: torch.Tensor,
        output_path: str
    ) -> None:
        """
        Save audio to WAV file.

        Args:
            audio: Audio waveform (T,) or (1, T)
            output_path: Output file path
        """
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, T)

        # Normalize to [-1, 1]
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / (max_val + 1e-8)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(
            str(output_path),
            audio,
            sample_rate=self.sample_rate,
            encoding="PCM_S",
            bits_per_sample=16
        )

        print(f"✅ Audio saved to {output_path}")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Kokoro-82M TTS Inference')
    parser.add_argument('--model', type=str, default='models/kokoro-82m/kokoro-v1_0.pth',
                       help='Path to Kokoro model checkpoint')
    parser.add_argument('--config', type=str, default='models/kokoro-82m/config.json',
                       help='Path to config.json')
    parser.add_argument('--voice', type=str, default='models/kokoro-82m/voices/af_heart.pt',
                       help='Path to voice file')
    parser.add_argument('--lora', type=str, default=None,
                       help='Path to fine-tuned LoRA weights directory')
    parser.add_argument('--text', type=str, help='Input text to synthesize')
    parser.add_argument('--input', type=str, help='Input text file (one sentence per line)')
    parser.add_argument('--output', type=str, default='outputs/kokoro/',
                       help='Output directory')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speaking speed multiplier')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cpu, cuda, mps)')
    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = None  # Auto-detect

    # Create TTS system
    print("\n" + "="*60)
    print("Initializing Kokoro-82M TTS System")
    print("="*60 + "\n")

    tts = KokoroTTS(
        model_path=args.model,
        config_path=args.config,
        voice_path=args.voice,
        lora_path=args.lora,
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
        texts = [
            "Hello, how are you?",
            "This is a test of the Kokoro text to speech system.",
        ]

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate speech
    print("\n" + "="*60)
    print(f"Generating speech for {len(texts)} text(s)...")
    print("="*60 + "\n")

    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] Processing...")
        print("-"*60)

        # Generate
        audio = tts.generate(text, speed=args.speed)

        # Save
        output_path = output_dir / f"output_{i+1:03d}.wav"
        tts.save_audio(audio, str(output_path))

    print("\n" + "="*60)
    print(f"✅ Generation complete! {len(texts)} files saved to {output_dir}")
    print("="*60 + "\n")


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    main()
