# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Turkish Text-to-Speech (TTS) fine-tuning project that adapts a StyleTTS2-based model for Turkish language using LoRA (Low-Rank Adaptation). The project is optimized for Apple Silicon (MPS) devices, specifically MacBook Pro M4, though it supports CUDA and CPU as well.

**Key Technical Points:**
- Uses simplified StyleTTS2 architecture (NOT production-ready; see src/model.py notes)
- PEFT LoRA for parameter-efficient fine-tuning (r=8, alpha=16)
- Dataset: zeynepgulhan/mediaspeech-with-cv-tr (48,781 Turkish audio samples)
- Turkish phonemization via espeak-ng backend
- Mixed precision (FP16) training on MPS

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install espeak-ng (required for phonemization)
brew install espeak-ng

# Verify espeak-ng installation
espeak-ng --version
```

### Testing Individual Components
```bash
# Test phonemizer
python -m src.phonemizer

# Test dataset loading
python -m src.dataset

# Test model initialization
python -c "import yaml; from src.model import create_model_from_config; config = yaml.safe_load(open('config.yaml')); create_model_from_config(config, vocab_size=100, device='cpu')"
```

### Training
```bash
# Full training (48k samples, ~100 epochs)
python -m src.train --config config.yaml

# Debug mode (100 samples)
python -m src.train --config config.yaml --limit-samples 100

# Resume from checkpoint
python -m src.train --config config.yaml --resume checkpoints/checkpoint_step_5000.pt

# Monitor training with TensorBoard
tensorboard --logdir runs/
```

### Inference
```bash
# Generate speech from text
python -m src.inference \
  --checkpoint checkpoints/best_model.pt \
  --text "Merhaba, size nasıl yardımcı olabilirim?"

# Batch inference from file
python -m src.inference \
  --checkpoint checkpoints/best_model.pt \
  --input texts.txt \
  --output outputs/
```

## Architecture

### Core Pipeline Flow
1. **Text → Phonemes** (src/phonemizer.py): Turkish text normalized and converted to phonemes via espeak-ng
2. **Phonemes → IDs** (src/phonemizer.py): Phonemes encoded to integer indices using built vocabulary
3. **Audio → Mel-spectrograms** (src/dataset.py): Audio resampled to 24kHz and converted to 80-bin mel-spectrograms
4. **Training** (src/train.py): Model learns phoneme → mel mapping using L1/MSE loss
5. **Inference** (src/inference.py): Text → phonemes → mel → audio (via vocoder)

### Module Responsibilities

**src/phonemizer.py**: Turkish-specific text processing
- Handles Turkish lowercase (I→ı, İ→i conversion)
- Number expansion (123 → "yüz yirmi üç")
- Abbreviation expansion (Dr. → Doktor)
- Phoneme vocabulary building and encoding/decoding
- Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`

**src/dataset.py**: Data loading and preprocessing
- Loads Hugging Face datasets with automatic caching
- Filters by text length (3-20 words) and audio duration (0.5-15s)
- Resamples audio to 24kHz mono
- Converts audio to log mel-spectrograms
- Handles variable-length sequences with padding in TTSCollator
- Multi-speaker support (checks for 'speaker_id' field)

**src/model.py**: Simplified StyleTTS2 with LoRA
- TextEncoder: Phoneme embeddings → transformer → text features
- AcousticModel: Text features → mel-spectrograms
- LoRA applied to attention layers (q_proj, k_proj, v_proj, o_proj)
- Uses Hugging Face PEFT library
- **IMPORTANT**: This is simplified; production requires full StyleTTS2 from official repo

**src/train.py**: Training loop and optimization
- Gradient accumulation (effective batch size: batch_size × gradient_accumulation_steps)
- Mixed precision training with torch.cuda.amp (limited MPS support)
- Learning rate scheduling (linear warmup)
- Early stopping with validation loss monitoring
- Checkpoint management (keeps only last N checkpoints)
- TensorBoard logging for loss/learning rate
- Masked loss computation for variable-length sequences

### LoRA Integration Details

LoRA weights are separate from base model:
- Base model frozen (requires_grad=False)
- Only LoRA adapters trained (q_proj, k_proj, v_proj, o_proj)
- Saved separately in `checkpoints/lora/` directory
- Loading requires both base model + LoRA weights

To use trained model:
1. Load base StyleTTS2 model
2. Apply LoRA weights from checkpoint
3. Model is in eval mode for inference

### Data Format Expectations

**Dataset structure** (zeynepgulhan/mediaspeech-with-cv-tr):
- `sentence`: Turkish text string
- `audio`: Dict with `array` (waveform) and `sampling_rate`
- Optional `speaker_id`: Speaker identifier for multi-speaker training

**Batch structure** (after collation):
- `phoneme_ids`: Padded tensor (B, max_phoneme_len)
- `phoneme_lengths`: Original lengths (B,)
- `mel`: Padded mel-spectrograms (B, n_mels, max_mel_len)
- `mel_lengths`: Original mel lengths (B,)
- `speaker_ids`: Speaker indices (B,)

### Configuration System

All hyperparameters controlled via `config.yaml`:
- `data`: Dataset name, sample rate, mel-spectrogram params, filtering thresholds
- `phonemizer`: espeak-ng settings, vocabulary path
- `model`: Architecture dimensions, layer counts, speaker embedding
- `lora`: Rank, alpha, dropout, target modules
- `training`: Batch size, learning rate, epochs, device (mps/cuda/cpu), mixed precision
- `validation`: Sample generation during validation
- `system`: Random seed, num_workers

## Common Development Patterns

### Adding New Dataset
1. Update `config.yaml` → `data.dataset_name`
2. Ensure dataset has `sentence` and `audio` fields
3. Rebuild phoneme vocabulary: `build_phoneme_vocab_from_dataset()`
4. Adjust filtering thresholds in config if needed

### Modifying LoRA Configuration
1. Edit `config.yaml` → `lora` section
2. Change `r` (rank) for capacity vs. parameter tradeoff
3. Modify `target_modules` to include/exclude layers
4. Delete old checkpoints when changing architecture

### Debugging Training Issues
1. Use `--limit-samples 100` to test pipeline quickly
2. Check TensorBoard for loss curves: `tensorboard --logdir runs/`
3. Verify device usage: Look for "Using device: mps/cuda/cpu" in logs
4. Monitor memory: Reduce `batch_size` or disable `mixed_precision` if OOM
5. Check gradient flow: Enable gradient logging in trainer

### Memory Optimization for Apple Silicon
- Default batch_size=2 with gradient_accumulation_steps=16
- Mixed precision may not work reliably on MPS (set `mixed_precision: false` if errors)
- Set `num_workers: 4` for data loading (good balance for M4)
- Disable `pin_memory` (not useful for MPS)

## Known Limitations and Warnings

1. **Simplified Model**: This is NOT production StyleTTS2. Missing diffusion model, style encoder, adversarial training. For real TTS, use https://github.com/yl4579/StyleTTS2

2. **Vocoder**: Uses basic Griffin-Lim reconstruction (poor quality). For production, integrate HiFi-GAN or similar neural vocoder.

3. **MPS Mixed Precision**: Apple MPS has limited autocast support. If training crashes with AMP enabled, set `mixed_precision: false` in config.

4. **Multi-speaker**: Dataset may not have explicit speaker IDs. Current implementation defaults to single speaker.

5. **espeak-ng Dependency**: Must be installed system-wide (`brew install espeak-ng`). Python package alone won't work.

6. **Training Time**: Full 48k samples on M4 takes 7-14 days. For faster iteration, use `--limit-samples` or cloud GPU.

## File Locations

- **Source code**: `src/`
- **Configuration**: `config.yaml`
- **Checkpoints**: `checkpoints/` (LoRA weights in `checkpoints/lora/`)
- **Generated audio**: `outputs/`
- **TensorBoard logs**: `runs/`
- **Dataset cache**: `data/cache/`
- **Phoneme vocabulary**: `data/phoneme_vocab.json` (auto-generated)

## Device-Specific Notes

**Apple Silicon (MPS)**:
- Set `device: "mps"` in config
- Use batch_size=1-2 (limited memory)
- Mixed precision is experimental
- Training speed: ~10-15 samples/sec on M4

**NVIDIA GPU (CUDA)**:
- Set `device: "cuda"` in config
- Can increase batch_size to 8-16
- Mixed precision highly recommended
- Training speed: ~50-100 samples/sec on RTX 3090

**CPU**:
- Set `device: "cpu"` in config
- Very slow, not recommended except for debugging
- Reduce batch_size to 1
