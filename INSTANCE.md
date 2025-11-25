# GCP Instance Setup Guide

Bu rehber, Kokoro-82M Turkish TTS modelini GCP üzerinde eğitmek için gerekli adımları içerir.

## Önerilen Instance Konfigürasyonu

| Özellik | Değer |
|---------|-------|
| Machine Type | a2-highgpu-1g (veya üstü) |
| GPU | 1x NVIDIA A100 40GB |
| Boot Disk | 200 GB SSD |
| OS | Debian 11 / Ubuntu 22.04 |

> **Not:** 4x A100 kullanmak isterseniz, şu anda sadece 1 GPU aktif. Multi-GPU desteği henüz eklenmedi.

---

## Kurulum Adımları

### 1. Sistemi Güncelle ve Bağımlılıkları Kur

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y espeak-ng libespeak-ng1 ffmpeg git
```

### 2. NVIDIA Driver Kontrolü

```bash
nvidia-smi
```

Eğer driver yüklü değilse:
```bash
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb
sudo dpkg -i cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y cuda-drivers
sudo reboot
```

### 3. Projeyi Klonla

```bash
cd ~
git clone https://github.com/samedberkanunver-kfn/tts.git
cd tts
```

### 4. Virtual Environment Oluştur

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 5. Paketleri Kur

```bash
# NumPy önce (versiyon uyumu için)
pip install numpy==1.26.4

# PyTorch CUDA ile
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Diğer paketler
pip install transformers==4.40.0 peft==0.10.0
pip install datasets==2.18.0 soundfile librosa huggingface_hub
pip install phonemizer tqdm tensorboard
```

### 6. Kurulumu Doğrula

```bash
# GPU testi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# espeak-ng testi
espeak-ng --version
```

### 7. Modeli İndir

```bash
python download_model.py
```

### 8. Eğitimi Başlat

```bash
# Önerilen konfigürasyon (A100 40GB)
python -m src.kokoro_train --epochs 20 --batch-size 4

# Debug modu (hızlı test)
python -m src.kokoro_train --epochs 2 --batch-size 4 --limit-samples 100
```

---

## Eğitim Parametreleri

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--epochs` | Epoch sayısı | 10 |
| `--batch-size` | Batch boyutu | 1 |
| `--lr` | Learning rate | 1e-4 |
| `--limit-samples` | Debug için sample limiti | None |
| `--checkpoint-dir` | Checkpoint dizini | checkpoints/kokoro_lora |

### GPU'ya Göre Önerilen Batch Size

| GPU | VRAM | Batch Size |
|-----|------|------------|
| A100 40GB | 40 GB | 4-8 |
| A100 80GB | 80 GB | 8-16 |
| T4 | 16 GB | 1-2 |
| V100 | 16/32 GB | 2-4 |

---

## Monitoring

```bash
# GPU kullanımı (başka terminal)
watch -n 5 nvidia-smi

# TensorBoard
tensorboard --logdir runs/ --bind_all --port 6006
```

TensorBoard'a erişim için GCP firewall'da 6006 portunu açın veya SSH tunnel kullanın:
```bash
gcloud compute ssh INSTANCE_NAME --zone ZONE -- -L 6006:localhost:6006
```

---

## Inference (Eğitim Sonrası)

```bash
python -m src.kokoro_inference \
    --checkpoint checkpoints/kokoro_lora/best_model \
    --text "Merhaba, nasılsınız?"
```

---

## Sorun Giderme

### OOM (Out of Memory) Hatası
```bash
# Batch size düşür
python -m src.kokoro_train --batch-size 2

# Veya batch_size=1
python -m src.kokoro_train --batch-size 1
```

### "espeak-ng not found" Hatası
```bash
sudo apt-get install -y espeak-ng libespeak-ng1
```

### "torchcodec" Hatası
```bash
pip uninstall torchcodec -y
pip install datasets==2.18.0
```

### NumPy Uyumsuzluk Hatası
```bash
pip install numpy==1.26.4
```

---

## Dosya Yapısı

```
tts/
├── src/
│   ├── kokoro_train.py      # Ana eğitim scripti
│   ├── kokoro_model.py      # Model + LoRA wrapper
│   ├── kokoro_inference.py  # Inference scripti
│   ├── modules.py           # Model bileşenleri
│   ├── istftnet.py          # Vocoder
│   └── phonemizer.py        # Türkçe phonemizer
├── models/kokoro-82m/
│   ├── config.json
│   ├── kokoro-v1_0.pth
│   └── voices/af_heart.pt
├── checkpoints/
│   └── kokoro_lora/best_model/
├── download_model.py
├── requirements.txt
├── INSTANCE.md              # Bu dosya
└── CLAUDE.md
```

---

## Faydalı Komutlar

```bash
# Disk durumu
df -h

# GPU durumu
nvidia-smi

# Python ortamı
which python
pip list | grep torch

# Eğitim logları
tail -f runs/kokoro/events*

# Checkpoint boyutu
du -sh checkpoints/
```
