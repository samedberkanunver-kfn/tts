# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Turkish Text-to-Speech fine-tuning using simplified StyleTTS2 with LoRA. Optimized for Apple Silicon (MPS), supports CUDA/CPU.

**Key Points:**
- Simplified StyleTTS2 (NOT production-ready - missing diffusion, style encoder)
- PEFT LoRA fine-tuning (r=8, alpha=16)
- Dataset: zeynepgulhan/mediaspeech-with-cv-tr (48k samples)
- Turkish phonemization via espeak-ng
- Griffin-Lim vocoder (low quality - use HiFi-GAN for production)

## Commands

```bash
# Setup
source venv/bin/activate
pip install -r requirements.txt
brew install espeak-ng  # Required system dependency

# Test components
python -m src.phonemizer
python -m src.dataset

# Training
python -m src.train --config config.yaml                    # Full training
python -m src.train --config config.yaml --limit-samples 100  # Debug mode
python -m src.train --config config.yaml --resume checkpoints/checkpoint_step_5000.pt

# Inference
python -m src.inference --checkpoint checkpoints/best_model.pt --text "Merhaba dünya"
python -m src.inference --checkpoint checkpoints/best_model.pt --input texts.txt --output outputs/

# Monitoring
tensorboard --logdir runs/
```

## Architecture

### Pipeline: Text → Phonemes → Mel-spectrogram → Audio

1. **src/phonemizer.py** (`TurkishPhonemizer`): Text normalization and phoneme conversion
   - Turkish lowercase (I→ı, İ→i), number expansion, abbreviations
   - Vocabulary: `<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3` + phonemes
   - Key functions: `phonemize()`, `encode()`, `decode()`, `build_vocab_from_texts()`

2. **src/dataset.py** (`TurkishTTSDataset`, `TTSCollator`): Data loading
   - Filters: 3-20 words, 0.5-15s audio duration
   - Resamples to 24kHz, converts to 80-bin log mel-spectrogram
   - Batch keys: `phoneme_ids`, `phoneme_lengths`, `mel`, `mel_lengths`, `speaker_ids`

3. **src/model.py** (`SimplifiedStyleTTS2`, `StyleTTS2WithLoRA`): Model
   - TextEncoder: Embedding → Positional Encoding → Transformer Encoder
   - AcousticModel: Transformer Decoder with cross-attention → Mel projection
   - LoRA auto-detects all Linear modules via PEFT

4. **src/train.py** (`Trainer`): Training loop
   - Gradient accumulation, masked L1/MSE loss, early stopping
   - Checkpoints: `checkpoints/*.pt` (metadata), `checkpoints/lora/*/` (LoRA weights)

5. **src/inference.py** (`TTS`): Generation
   - Loads phoneme vocab + LoRA weights
   - Uses librosa Griffin-Lim for mel→audio conversion

### LoRA Details

- Base model frozen, only LoRA adapters trained
- Weights saved separately in `checkpoints/lora/{checkpoint_name}/`
- Loading: metadata from `.pt` file, LoRA from corresponding `lora/` directory

## Configuration (config.yaml)

Key sections:
- `data`: dataset_name, sample_rate (24000), n_mels (80), filtering thresholds
- `model`: hidden_dim (512), n_layer (8)
- `lora`: r (8), lora_alpha (16), target_modules
- `training`: device, batch_size, learning_rate, mixed_precision, early_stopping

## Critical Warnings

1. **espeak-ng**: Must install system-wide (`brew install espeak-ng`), Python package alone fails
2. **MPS AMP**: Mixed precision unreliable on Apple Silicon - set `mixed_precision: false` if crashes
3. **Memory**: Use batch_size=1-2 on M4, increase gradient_accumulation_steps instead
4. **Production**: Use official StyleTTS2 repo + HiFi-GAN vocoder for real applications

## Troubleshooting

- **OOM on MPS**: Reduce batch_size, disable mixed_precision, increase gradient_accumulation_steps
- **espeak-ng errors**: Verify `espeak-ng --version` works, reinstall if needed
- **Quick iteration**: Use `--limit-samples 100` for fast pipeline testing
- **Training time**: Full dataset on M4 = 7-14 days; use cloud GPU for speed
