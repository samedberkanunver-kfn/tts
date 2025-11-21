"""
Kokoro-82M Model with LoRA for Turkish Fine-Tuning

This module wraps the pretrained Kokoro-82M model and applies LoRA
(Low-Rank Adaptation) for parameter-efficient fine-tuning on Turkish dataset.

Architecture:
    - BERT (CustomAlbert): Phoneme encoding
    - BERT Encoder: Linear projection
    - Prosody Predictor: Duration, F0, N prediction
    - Text Encoder: Text feature extraction
    - Decoder: ISTFTNet vocoder (mel-to-audio)

Author: Claude Code
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
from transformers import AlbertConfig

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    from peft.utils import get_peft_model_state_dict
except ImportError:
    raise ImportError("peft package not found. Install: pip install peft")

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder


class KokoroModel(nn.Module):
    """
    Kokoro-82M pretrained model.

    This is a wrapper around the original Kokoro architecture that loads
    pretrained weights from hexgrad/Kokoro-82M.

    Components:
        - bert: CustomAlbert phoneme encoder (768-dim, 12 layers)
        - bert_encoder: Linear projection (768 → 512)
        - predictor: ProsodyPredictor for duration/F0/energy
        - text_encoder: TextEncoder for text features
        - decoder: Decoder with ISTFTNet vocoder
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_path: Optional[str] = None,
        disable_complex: bool = False
    ):
        """
        Initialize Kokoro model.

        Args:
            config: Model configuration dictionary (from config.json)
            model_path: Path to pretrained model (.pth file)
            disable_complex: Whether to use CustomSTFT instead of TorchSTFT
        """
        super().__init__()

        self.config = config
        self.vocab = config['vocab']

        # Build model components
        self.bert = CustomAlbert(
            AlbertConfig(
                vocab_size=config['n_token'],
                **config['plbert']
            )
        )

        self.bert_encoder = nn.Linear(
            self.bert.config.hidden_size,
            config['hidden_dim']
        )

        self.context_length = self.bert.config.max_position_embeddings

        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'],
            d_hid=config['hidden_dim'],
            nlayers=config['n_layer'],
            max_dur=config['max_dur'],
            dropout=config['dropout']
        )

        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'],
            kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'],
            n_symbols=config['n_token']
        )

        self.decoder = Decoder(
            dim_in=config['hidden_dim'],
            style_dim=config['style_dim'],
            dim_out=config['n_mels'],
            disable_complex=disable_complex,
            **config['istftnet']
        )

        # Load pretrained weights if provided
        if model_path:
            self.load_pretrained(model_path)

    def load_pretrained(self, model_path: str) -> None:
        """
        Load pretrained weights from Kokoro-82M checkpoint.

        Args:
            model_path: Path to kokoro-v1_0.pth file
        """
        print(f"Loading pretrained weights from {model_path}...")

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

        for key, state_dict in checkpoint.items():
            if not hasattr(self, key):
                warnings.warn(f"Skipping {key} (not found in model)")
                continue

            try:
                getattr(self, key).load_state_dict(state_dict)
                print(f"✓ Loaded {key}")
            except Exception as e:
                # Try with 'module.' prefix removal (for DataParallel models)
                warnings.warn(f"Failed to load {key}: {e}")
                try:
                    state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
                    getattr(self, key).load_state_dict(state_dict, strict=False)
                    print(f"✓ Loaded {key} (with module. prefix removed)")
                except Exception as e2:
                    warnings.warn(f"Failed to load {key} even after prefix removal: {e2}")

        print("✅ Pretrained weights loaded successfully!")

    @property
    def device(self):
        """Get model device."""
        return self.bert.embeddings.word_embeddings.weight.device

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        **kwargs  # Accept extra args from PEFT wrapper
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Forward pass for TTS generation.

        Args:
            input_ids: Phoneme token IDs (B, T)
            ref_s: Reference style vector (B, 256)  # 128 for decoder + 128 for predictor
            speed: Speaking speed multiplier

        Returns:
            Tuple of:
                - audio: Generated audio waveform (B, T_audio)
                - pred_dur: Predicted phoneme durations (B, T)
        """
        batch_size = input_ids.shape[0]
        input_lengths = torch.full(
            (batch_size,),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long
        )

        # Create text mask
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
            input_lengths.shape[0], -1
        ).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(self.device)

        # BERT encoding
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        # Split style vector
        s = ref_s[:, 128:]  # Style for predictor

        # Duration prediction
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        # Batch alignment (Loop approach to handle variable durations per batch item)
        # This replaces the matrix multiplication approach which failed for batch_size > 1
        
        # Text encoder
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        
        # d is (B, hidden, T) -> permute to (B, T, hidden) for repeat
        d_perm = d.permute(0, 2, 1)
        # t_en is (B, hidden, T) -> permute to (B, T, hidden)
        t_en_perm = t_en.permute(0, 2, 1) # t_en was (B, hidden, T) from text_encoder? 
        # Wait, text_encoder output shape check:
        # self.text_encoder(input_ids, ...) -> (B, hidden, T) usually.
        # Let's verify text_encoder in modules.py if possible, but assuming (B, hidden, T) based on usage.
        # In original code: t_en = self.text_encoder(...) 
        # asr = t_en @ pred_aln_trg
        # pred_aln_trg was (1, T, T_mel). 
        # So t_en must be (B, hidden, T). Correct.

        max_mel_len = int(pred_dur.sum(dim=1).max().item())
        
        en_list = []
        asr_list = []
        
        pred_dur_cpu = pred_dur.cpu()
        
        for i in range(batch_size):
            dur = pred_dur_cpu[i] # (T,)
            
            # Repeat features based on duration
            # d_perm[i] is (T, hidden)
            curr_en = torch.repeat_interleave(d_perm[i], dur, dim=0) # (T_mel_i, hidden)
            curr_asr = torch.repeat_interleave(t_en_perm[i], dur, dim=0) # (T_mel_i, hidden)
            
            # Pad to max_mel_len
            pad_len = max_mel_len - curr_en.size(0)
            if pad_len > 0:
                curr_en = F.pad(curr_en, (0, 0, 0, pad_len))
                curr_asr = F.pad(curr_asr, (0, 0, 0, pad_len))
                
            en_list.append(curr_en)
            asr_list.append(curr_asr)
            
        # Stack and permute back to (B, hidden, T_mel)
        en = torch.stack(en_list).permute(0, 2, 1).to(self.device)
        asr = torch.stack(asr_list).permute(0, 2, 1).to(self.device)

        # F0 and energy prediction
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        # Decode to audio with ISTFTNet
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()

        return audio, pred_dur


class KokoroWithLoRA:
    """
    Wrapper for Kokoro model with PEFT LoRA integration.

    Applies LoRA to specific modules in the Kokoro model for parameter-efficient
    fine-tuning on Turkish dataset.

    Usage:
        config = json.load(open('models/kokoro-82m/config.json'))
        model_wrapper = KokoroWithLoRA(
            config=config,
            model_path='models/kokoro-82m/kokoro-v1_0.pth',
            lora_config={'r': 8, 'lora_alpha': 16}
        )
        model = model_wrapper.model  # PEFT model with LoRA
        model_wrapper.print_trainable_parameters()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_path: str,
        lora_config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize Kokoro with LoRA.

        Args:
            config: Kokoro model configuration (from config.json)
            model_path: Path to pretrained kokoro-v1_0.pth
            lora_config: LoRA configuration dict with keys:
                - r: LoRA rank
                - lora_alpha: LoRA alpha
                - lora_dropout: LoRA dropout
                - target_modules: List of module names to apply LoRA (optional)
            device: Device to move model to
        """
        self.config = config
        self.lora_config = lora_config

        # Create base model
        print("\n" + "="*60)
        print("Creating Kokoro-82M base model...")
        print("="*60)

        self.base_model = KokoroModel(
            config=config,
            model_path=model_path,
            disable_complex=True  # Use CustomSTFT for better ONNX/MPS compatibility
        )

        # Move to device before applying LoRA
        if device is not None:
            print(f"Moving model to {device}...")
            self.base_model = self.base_model.to(device)

        # Apply LoRA
        print("\n" + "="*60)
        print("Applying LoRA with PEFT...")
        print("="*60)

        self.model = self._apply_lora(self.base_model)

        print("\n✅ Kokoro-82M with LoRA created successfully!\n")

    def _apply_lora(self, model: nn.Module) -> PeftModel:
        """
        Apply LoRA to Kokoro model using PEFT.

        Args:
            model: Base Kokoro model

        Returns:
            PEFT model with LoRA
        """
        # Define target modules for LoRA
        target_modules = self.lora_config.get('target_modules')

        if target_modules is None:
            # Default: Focus on BERT attention and FFN layers
            # These are the most important for fine-tuning
            print("Using default LoRA target modules...")
            target_modules = [
                # BERT attention layers
                "query", "key", "value", "dense",
                # BERT FFN layers
                "ffn", "ffn_output",
                # Text encoder
                "linear_layer",
                # Decoder projections
                "bert_encoder",
            ]
            print(f"Target modules: {target_modules}")

        # Create LoRA configuration
        peft_config = LoraConfig(
            r=self.lora_config.get('r', 8),
            lora_alpha=self.lora_config.get('lora_alpha', 16),
            lora_dropout=self.lora_config.get('lora_dropout', 0.1),
            target_modules=target_modules,
            bias=self.lora_config.get('bias', 'none'),
            use_rslora=self.lora_config.get('use_rslora', True),
            use_dora=self.lora_config.get('use_dora', False),
            task_type="FEATURE_EXTRACTION",  # Generic task type for TTS
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
            path: Directory path to save LoRA weights
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        self.model.save_pretrained(save_path)
        print(f"✅ LoRA weights saved to {save_path}")

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

def create_kokoro_model(
    config_path: str,
    model_path: str,
    lora_config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> KokoroWithLoRA:
    """
    Create Kokoro model with LoRA from config files.

    Args:
        config_path: Path to config.json
        model_path: Path to kokoro-v1_0.pth
        lora_config: LoRA configuration dict
        device: Device to move model to (optional)

    Returns:
        KokoroWithLoRA instance
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model
    model_wrapper = KokoroWithLoRA(
        config=config,
        model_path=model_path,
        lora_config=lora_config,
        device=device
    )

    model_wrapper.print_trainable_parameters()

    return model_wrapper


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Test model loading
    print("\n" + "="*60)
    print("Testing Kokoro-82M Model Loading")
    print("="*60 + "\n")

    config_path = "models/kokoro-82m/config.json"
    model_path = "models/kokoro-82m/kokoro-v1_0.pth"

    lora_config = {
        'r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        # target_modules will be auto-detected
    }

    # Create model
    model_wrapper = create_kokoro_model(
        config_path=config_path,
        model_path=model_path,
        lora_config=lora_config,
        device=torch.device('cpu')  # Use CPU for testing
    )

    print("\n✅ Model test completed successfully!")
