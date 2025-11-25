# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Turkish Text-to-Speech fine-tuning using **Kokoro-82M** (primary) or simplified StyleTTS2 with LoRA. Optimized for CUDA (A100/GCP) and Apple Silicon (MPS).

**Key Points:**
- **Kokoro-82M**: Pretrained 82M param model from hexgrad/Kokoro-82M
- PEFT LoRA fine-tuning (r=8, alpha=16, ~109K trainable params)
- Dataset: zeynepgulhan/mediaspeech-with-cv-tr (48k Turkish samples)
- Turkish phonemization via espeak-ng
- ISTFTNet vocoder (built into Kokoro)

## Commands

```bash
# Setup
source venv/bin/activate
pip install -r requirements.txt

# System dependency (REQUIRED)
# Linux (GCP):
sudo apt-get install -y espeak-ng
# macOS:
brew install espeak-ng

# Kokoro Training (RECOMMENDED)
python -m src.kokoro_train --epochs 20 --batch-size 4           # GCP A100
python -m src.kokoro_train --epochs 10 --batch-size 2           # MPS/smaller GPU
python -m src.kokoro_train --limit-samples 100 --epochs 2       # Debug mode

# Kokoro Inference
python -m src.kokoro_inference --checkpoint checkpoints/kokoro_lora/best_model --text "Merhaba"

# Legacy StyleTTS2 Training (simplified, lower quality)
python -m src.train --config config.yaml

# Monitoring
tensorboard --logdir runs/
```

## Architecture

### Kokoro-82M Pipeline (Primary)

```
Text → TurkishPhonemizer → IPA → Kokoro Token IDs → Model → Audio
                ↓
        espeak-ng backend
```

**Model Components** (`src/kokoro_model.py`):
1. **CustomAlbert** (BERT): Phoneme encoding (768-dim, 12 layers)
2. **bert_encoder**: Linear projection (768 → 512)
3. **ProsodyPredictor**: Duration, F0, energy prediction
4. **TextEncoder**: Text feature extraction (CNN + LSTM)
5. **Decoder**: ISTFTNet vocoder (mel → audio)

**Training Flow** (`src/kokoro_train.py`):
1. Load pretrained Kokoro-82M from `models/kokoro-82m/kokoro-v1_0.pth`
2. Apply LoRA to BERT attention layers
3. Freeze base model, train only LoRA adapters
4. Loss: L1 mel-spectrogram reconstruction

### File Structure

```
src/
├── kokoro_train.py      # Main training script (Kokoro)
├── kokoro_model.py      # KokoroModel + KokoroWithLoRA wrapper
├── kokoro_inference.py  # Inference with trained LoRA
├── modules.py           # CustomAlbert, ProsodyPredictor, TextEncoder, DurationEncoder
├── istftnet.py          # Decoder with ISTFTNet vocoder
├── custom_stft.py       # ONNX-compatible STFT implementation
├── phonemizer.py        # TurkishPhonemizer (espeak-ng)
├── dataset.py           # Legacy dataset loader
├── model.py             # Legacy SimplifiedStyleTTS2
├── train.py             # Legacy training script
└── inference.py         # Legacy inference

models/kokoro-82m/
├── config.json          # Model architecture config
├── kokoro-v1_0.pth      # Pretrained weights (327MB)
└── voices/
    └── af_heart.pt      # Voice embedding for inference

checkpoints/kokoro_lora/
└── best_model/          # LoRA weights (saved by trainer)
```

## Configuration

### Kokoro Config (`models/kokoro-82m/config.json`)
- `hidden_dim`: 512
- `style_dim`: 128
- `n_layer`: 3
- `n_token`: 178 (IPA vocab size)
- `plbert.hidden_size`: 768 (BERT dimension)
- `plbert.num_hidden_layers`: 12

### LoRA Config (in `kokoro_train.py`)
- `r`: 8 (rank)
- `lora_alpha`: 16
- `lora_dropout`: 0.1
- Target modules: query, key, value, dense, ffn, bert_encoder

## Critical Dependencies

### System
```bash
# Linux (GCP/Ubuntu)
sudo apt-get install -y espeak-ng libespeak-ng1

# macOS
brew install espeak-ng

# Verify
espeak-ng --version
```

### Python
```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
peft>=0.5.0
datasets>=2.14.0
phonemizer>=3.2.0
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `espeak-ng backend failed` | espeak-ng not installed | `sudo apt-get install espeak-ng` |
| `Expected 640, got 512` | DurationEncoder shape bug | Fixed in modules.py (unsqueeze style) |
| `FileNotFoundError: kokoro-v1_0.pth` | Model not downloaded | Download from HuggingFace |
| `CUDA OOM` | Batch size too large | Reduce `--batch-size` |
| `num_workers error` | DataLoader issue | Set `num_workers=0` |

## Device-Specific

**GCP A100 (CUDA)**:
- batch_size: 32-64
- Mixed precision: enabled (AMP + TF32)
- num_workers: os.cpu_count()

**Apple Silicon (MPS)**:
- batch_size: 2-4
- Mixed precision: experimental
- num_workers: 0

## Model Downloads

Kokoro-82M pretrained model:
```bash
# Manual download from HuggingFace
# https://huggingface.co/hexgrad/Kokoro-82M
# Place files in models/kokoro-82m/
```
