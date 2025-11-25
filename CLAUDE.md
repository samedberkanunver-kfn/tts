# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Turkish Text-to-Speech fine-tuning using **Kokoro-82M** with LoRA. Optimized for CUDA (A100/GCP) and Apple Silicon (MPS).

**Key Points:**
- **Kokoro-82M**: Pretrained 82M param model from hexgrad/Kokoro-82M
- PEFT LoRA fine-tuning (r=8, alpha=16, ~109K trainable params)
- Dataset: zeynepgulhan/mediaspeech-with-cv-tr (Turkish speech)
- Turkish phonemization via espeak-ng
- ISTFTNet vocoder (built into Kokoro)

## Quick Start

```bash
# 1. Setup (GCP/Linux)
sudo apt-get install -y espeak-ng
python3 -m venv venv && source venv/bin/activate
pip install numpy==1.26.4
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 peft==0.10.0 datasets==2.18.0
pip install soundfile librosa phonemizer tqdm tensorboard huggingface_hub

# 2. Download model
python download_model.py

# 3. Train
python -m src.kokoro_train --epochs 20 --batch-size 4

# 4. Inference
python -m src.kokoro_inference --checkpoint checkpoints/kokoro_lora/best_model --text "Merhaba"
```

## Training Memory Limits

Current settings to prevent OOM:
- **Token length**: 128 (max phoneme sequence)
- **Audio duration**: 0.5s - 5s (samples outside this range are filtered)
- **Text length**: 2-10 words
- **Max audio samples**: 120,000 (5 seconds at 24kHz)

### GPU → Batch Size

| GPU | VRAM | Recommended Batch Size |
|-----|------|------------------------|
| A100 40GB | 40 GB | 4 |
| A100 80GB | 80 GB | 8-16 |
| T4/V100 16GB | 16 GB | 1-2 |
| Apple M1/M2 | 8-16 GB | 1-2 |

## Architecture

```
Text → TurkishPhonemizer → IPA → Kokoro Tokens → Model → Audio
              ↓
        espeak-ng backend
```

**Model Components** (`src/kokoro_model.py`):
1. **CustomAlbert** (BERT): Phoneme encoding (768-dim, 12 layers)
2. **bert_encoder**: Linear projection (768 → 512)
3. **ProsodyPredictor**: Duration, F0, energy prediction
4. **TextEncoder**: Text feature extraction (CNN + LSTM)
5. **Decoder**: ISTFTNet vocoder (mel → audio)

**Training Loss**: L1 log-mel spectrogram reconstruction

## File Structure

```
src/
├── kokoro_train.py      # Main training script
├── kokoro_model.py      # KokoroModel + LoRA wrapper
├── kokoro_inference.py  # Inference script
├── modules.py           # Model components
├── istftnet.py          # ISTFTNet vocoder
├── custom_stft.py       # ONNX-compatible STFT
└── phonemizer.py        # Turkish phonemizer

models/kokoro-82m/
├── config.json
├── kokoro-v1_0.pth      # 327MB pretrained weights
└── voices/af_heart.pt   # Voice embedding

checkpoints/kokoro_lora/
└── best_model/          # LoRA weights
```

## Critical Dependencies

```bash
# System (REQUIRED)
sudo apt-get install -y espeak-ng  # Linux
brew install espeak-ng              # macOS

# Python (pinned versions for compatibility)
numpy==1.26.4
transformers==4.40.0
peft==0.10.0
datasets==2.18.0
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `espeak-ng not found` | `sudo apt-get install espeak-ng` |
| `CUDA OOM` | Reduce `--batch-size` to 2 or 1 |
| `torchcodec` error | Use `datasets==2.18.0` |
| `NumPy incompatible` | `pip install numpy==1.26.4` |
| `inf loss` | Audio normalization is applied automatically |
| `Expected 640, got 512` | Already fixed in modules.py |

## Monitoring

```bash
# GPU usage
watch -n 5 nvidia-smi

# TensorBoard
tensorboard --logdir runs/ --bind_all
```

## See Also

- `INSTANCE.md` - Full GCP setup tutorial
- `download_model.py` - Model download script
- `requirements.txt` - Python dependencies
