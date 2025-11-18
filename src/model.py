"""
StyleTTS2 Model with LoRA for Turkish Fine-Tuning

IMPORTANT NOTE:
This is a SIMPLIFIED StyleTTS2 implementation for demonstration purposes.
For production use, you should:
1. Clone the official StyleTTS2 repository: https://github.com/yl4579/StyleTTS2
2. Use this module's LoRA wrapper (StyleTTS2WithLoRA) with the official model
3. Or implement full StyleTTS2 architecture based on the paper

This simplified version provides:
- Basic text-to-mel architecture
- PEFT LoRA integration
- Training-ready structure
- MPS (Apple Silicon) compatibility

For full StyleTTS2 features (diffusion, style encoder, etc.), use the official repo.

Author: Claude Code
"""

import warnings
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    from peft.utils import get_peft_model_state_dict
except ImportError:
    raise ImportError("peft package not found. Install: pip install peft")


# ============================================================================
# Simplified StyleTTS2 Components
# ============================================================================

class TextEncoder(nn.Module):
    """
    Text encoder: Phoneme IDs → Text embeddings

    Simplified version using transformer encoder.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize text encoder.

        Args:
            vocab_size: Phoneme vocabulary size
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.embed_dim = embed_dim

    def forward(
        self,
        phoneme_ids: torch.LongTensor,
        phoneme_lengths: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode phoneme sequence.

        Args:
            phoneme_ids: Phoneme indices (B, T)
            phoneme_lengths: Original lengths before padding (B,)

        Returns:
            Tuple of:
                - Encoded features (B, T, embed_dim)
                - Attention mask (B, T)
        """
        # Embed phonemes
        x = self.embedding(phoneme_ids)  # (B, T, embed_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create attention mask (mask padding positions)
        max_len = phoneme_ids.size(1)
        mask = torch.arange(max_len, device=phoneme_ids.device).expand(
            len(phoneme_lengths), max_len
        ) >= phoneme_lengths.unsqueeze(1)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        return x, mask


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AcousticModel(nn.Module):
    """
    Acoustic model: Text embeddings → Mel-spectrograms

    Simplified version using transformer decoder.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        n_mels: int = 80,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize acoustic model.

        Args:
            embed_dim: Text embedding dimension
            n_mels: Number of mel-spectrogram bins
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        # Transformer decoder (self-attention on mels, cross-attention with text)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Mel projection
        self.mel_projection = nn.Linear(embed_dim, n_mels)

        # Learnable query embeddings (for autoregressive-like generation)
        self.query_embed = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        mel_targets: Optional[torch.Tensor] = None,
        max_mel_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate mel-spectrograms from text features.

        Args:
            text_features: Encoded text features (B, T_text, embed_dim)
            text_mask: Text padding mask (B, T_text)
            mel_targets: Target mel-spectrograms for teacher forcing (B, n_mels, T_mel)
            max_mel_len: Maximum mel length for inference

        Returns:
            Predicted mel-spectrograms (B, n_mels, T_mel)
        """
        batch_size = text_features.size(0)

        if mel_targets is not None:
            # Training: use mel targets as queries (teacher forcing)
            # mel_targets: (B, n_mels, T_mel) → (B, T_mel, n_mels)
            tgt = mel_targets.transpose(1, 2)
            # Project to embed_dim
            tgt = F.linear(tgt, self.mel_projection.weight.t())  # (B, T_mel, embed_dim)
        else:
            # Inference: use learnable query embeddings
            if max_mel_len is None:
                # Estimate mel length from text length
                max_mel_len = text_features.size(1) * 3  # Rough estimate

            tgt = self.query_embed.expand(batch_size, max_mel_len, -1)

        # Transformer decoding (cross-attention with text)
        output = self.transformer(
            tgt=tgt,
            memory=text_features,
            memory_key_padding_mask=text_mask,
        )  # (B, T_mel, embed_dim)

        # Project to mel-spectrogram
        mel_pred = self.mel_projection(output)  # (B, T_mel, n_mels)

        # Transpose to (B, n_mels, T_mel)
        mel_pred = mel_pred.transpose(1, 2)

        return mel_pred


class SimplifiedStyleTTS2(nn.Module):
    """
    Simplified StyleTTS2 model for Turkish TTS.

    WARNING: This is a simplified version for demonstration.
    For production, use the official StyleTTS2 implementation.

    Architecture:
        1. Text Encoder: Phonemes → Text embeddings
        2. Acoustic Model: Text embeddings → Mel-spectrograms
        3. (Vocoder is separate, e.g., HiFi-GAN, not included here)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        n_mels: int = 80,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize SimplifiedStyleTTS2.

        Args:
            vocab_size: Phoneme vocabulary size
            embed_dim: Model embedding dimension
            n_mels: Number of mel-spectrogram bins
            num_encoder_layers: Number of text encoder layers
            num_decoder_layers: Number of acoustic model layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.acoustic_model = AcousticModel(
            embed_dim=embed_dim,
            n_mels=n_mels,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_mels = n_mels

    def forward(
        self,
        phoneme_ids: torch.LongTensor,
        phoneme_lengths: torch.LongTensor,
        mel_targets: Optional[torch.Tensor] = None,
        max_mel_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            phoneme_ids: Phoneme indices (B, T)
            phoneme_lengths: Phoneme lengths (B,)
            mel_targets: Target mel-spectrograms (B, n_mels, T_mel)
            max_mel_len: Maximum mel length for inference

        Returns:
            Predicted mel-spectrograms (B, n_mels, T_mel)
        """
        # Encode text
        text_features, text_mask = self.text_encoder(phoneme_ids, phoneme_lengths)

        # Generate mel-spectrograms
        mel_pred = self.acoustic_model(
            text_features=text_features,
            text_mask=text_mask,
            mel_targets=mel_targets,
            max_mel_len=max_mel_len,
        )

        return mel_pred


# ============================================================================
# LoRA Integration with PEFT
# ============================================================================

class StyleTTS2WithLoRA:
    """
    Wrapper for StyleTTS2 model with PEFT LoRA integration.

    Applies LoRA to attention layers in transformer blocks for parameter-efficient
    fine-tuning on Turkish dataset.

    Usage:
        config = {...}  # Model and LoRA config
        model_wrapper = StyleTTS2WithLoRA(config, vocab_size=200)
        model = model_wrapper.model  # PEFT model with LoRA
        model_wrapper.print_trainable_parameters()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        vocab_size: int,
    ):
        """
        Initialize StyleTTS2 with LoRA.

        Args:
            config: Configuration dictionary with 'model' and 'lora' sections
            vocab_size: Phoneme vocabulary size
        """
        self.config = config
        self.model_config = config['model']
        self.lora_config = config['lora']

        # Create base model
        print("Creating base StyleTTS2 model...")
        self.base_model = SimplifiedStyleTTS2(
            vocab_size=vocab_size,
            embed_dim=self.model_config.get('hidden_dim', 512),
            n_mels=config['data']['n_mels'],
            num_encoder_layers=self.model_config.get('n_layer', 6),
            num_decoder_layers=self.model_config.get('n_layer', 6),
            num_heads=8,
            dropout=0.1,
        )

        # Apply LoRA
        print("Applying LoRA with PEFT...")
        self.model = self._apply_lora(self.base_model)

        print("Model with LoRA created successfully!")

    def _apply_lora(self, model: nn.Module) -> PeftModel:
        """
        Apply LoRA to model using PEFT.

        Args:
            model: Base model

        Returns:
            PEFT model with LoRA
        """
        # Find all Linear modules in the model for LoRA
        # PyTorch Transformer uses different naming than Hugging Face models
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get the last part of the name (e.g., "out_proj" from "encoder.layers.0.self_attn.out_proj")
                module_name = name.split('.')[-1]
                if module_name not in target_modules:
                    target_modules.append(module_name)

        print(f"Auto-detected Linear modules for LoRA: {target_modules}")

        # If no modules found, use default
        if not target_modules:
            target_modules = self.lora_config['target_modules']
            print(f"Using config target modules: {target_modules}")

        # Create LoRA configuration
        peft_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['lora_alpha'],
            lora_dropout=self.lora_config['lora_dropout'],
            target_modules=target_modules,
            bias=self.lora_config.get('bias', 'none'),
            use_rslora=self.lora_config.get('use_rslora', True),
            use_dora=self.lora_config.get('use_dora', False),
        )

        # Apply LoRA
        peft_model = get_peft_model(model, peft_config)

        return peft_model

    def print_trainable_parameters(self) -> None:
        """Print trainable parameter statistics."""
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        trainable_percent = 100 * trainable_params / all_params

        print(f"\n{'='*60}")
        print(f"Trainable Parameters Summary:")
        print(f"{'='*60}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"All params:       {all_params:,}")
        print(f"Trainable%:       {trainable_percent:.4f}%")
        print(f"{'='*60}\n")

    def save_lora_weights(self, path: str) -> None:
        """
        Save only LoRA weights (not full model).

        Args:
            path: Path to save LoRA weights
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        self.model.save_pretrained(save_path)
        print(f"LoRA weights saved to {save_path}")

    def load_lora_weights(self, path: str) -> None:
        """
        Load LoRA weights.

        Args:
            path: Path to LoRA weights
        """
        # This would load LoRA weights into existing model
        # For loading from scratch, use PeftModel.from_pretrained()
        raise NotImplementedError(
            "Use PeftModel.from_pretrained() to load LoRA weights from scratch"
        )

    def to(self, device: torch.device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self

    def train(self):
        """Set model to training mode."""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_model_from_config(
    config: Dict[str, Any],
    vocab_size: int,
    device: Optional[torch.device] = None
) -> StyleTTS2WithLoRA:
    """
    Create StyleTTS2 model with LoRA from configuration.

    Args:
        config: Configuration dictionary
        vocab_size: Phoneme vocabulary size
        device: Device to move model to (optional)

    Returns:
        StyleTTS2WithLoRA instance
    """
    model_wrapper = StyleTTS2WithLoRA(config, vocab_size)

    if device is not None:
        model_wrapper.to(device)

    model_wrapper.print_trainable_parameters()

    return model_wrapper


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    import yaml

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*60)
    print("Testing SimplifiedStyleTTS2 with LoRA")
    print("="*60 + "\n")

    # Create model
    vocab_size = 200  # Example vocab size
    model_wrapper = create_model_from_config(
        config=config,
        vocab_size=vocab_size,
        device=torch.device('cpu')  # Use CPU for testing
    )

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    max_phoneme_len = 50
    max_mel_len = 150

    # Dummy inputs
    phoneme_ids = torch.randint(0, vocab_size, (batch_size, max_phoneme_len))
    phoneme_lengths = torch.LongTensor([40, 30])
    mel_targets = torch.randn(batch_size, config['data']['n_mels'], max_mel_len)

    # Forward
    model_wrapper.model.eval()
    with torch.no_grad():
        mel_pred = model_wrapper.model(
            phoneme_ids=phoneme_ids,
            phoneme_lengths=phoneme_lengths,
            mel_targets=mel_targets,
        )

    print(f"Input phoneme IDs shape: {phoneme_ids.shape}")
    print(f"Target mel shape: {mel_targets.shape}")
    print(f"Predicted mel shape: {mel_pred.shape}")

    print("\n" + "="*60)
    print("Model test completed successfully!")
    print("="*60)
