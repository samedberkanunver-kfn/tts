# Turkish StyleTTS2 Fine-Tuning Pipeline

Türkçe metin-seslendirilmesi (TTS) için StyleTTS2 modelinin LoRA ile fine-tune edilmesi.

## 🎯 Proje Özeti

Bu proje, StyleTTS2 mimarisini temel alan bir TTS modelini Türkçe konuşacak hale getirmek için hazırlanmıştır. Düşük kaynaklı eğitim için **PEFT LoRA** tekniği kullanılmakta ve **Apple Silicon (MPS)** cihazlarda çalışacak şekilde optimize edilmiştir.

### Özellikler

- ✅ StyleTTS2 tabanlı TTS modeli (basitleştirilmiş implementasyon)
- ✅ PEFT LoRA ile parametre-verimli fine-tuning
- ✅ Apple Silicon (M1/M2/M3/M4) MPS desteği
- ✅ Türkçe phonemization (espeak-ng)
- ✅ Automatic dataset loading (Hugging Face datasets)
- ✅ Mixed precision (FP16) eğitim
- ✅ TensorBoard logging
- ✅ Checkpoint yönetimi ve early stopping
- ✅ Batch inference ve WAV export

### Teknik Detaylar

- **Model**: SimplifiedStyleTTS2 (Text Encoder + Acoustic Model)
- **Dataset**: [zeynepgulhan/mediaspeech-with-cv-tr](https://huggingface.co/datasets/zeynepgulhan/mediaspeech-with-cv-tr) (48,781 samples)
- **Phonemizer**: espeak-ng (Türkçe G2P)
- **Sample Rate**: 24kHz
- **LoRA Config**: r=8, alpha=16, dropout=0.1
- **Training**: AdamW optimizer, L1/MSE loss, gradient accumulation

---

## 📋 Gereksinimler

### Sistem Gereksinimleri

- **OS**: macOS (Apple Silicon önerilen), Linux, Windows
- **RAM**: Minimum 16GB (eğitim için)
- **Disk**: ~5GB (dataset + checkpoints)
- **Python**: 3.8+

### Donanım

- **Eğitim**: MacBook Pro M4 veya üzeri (6-14 gün)
- **Alternatif**: NVIDIA GPU (Tesla T4, RTX 3090, vb.) - daha hızlı eğitim

---

## 🚀 Kurulum

### 1. Repository'yi Klonlayın

```bash
cd /Users/kafein/Desktop/samed/tts
```

### 2. Python Ortamı Oluşturun

```bash
# Python virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows
```

### 3. Bağımlılıkları Yükleyin

```bash
# PyTorch (Apple Silicon için)
pip install torch torchvision torchaudio

# Diğer bağımlılıklar
pip install -r requirements.txt
```

### 4. espeak-ng Kurulumu

**macOS:**
```bash
brew install espeak-ng
```

**Linux:**
```bash
sudo apt-get install espeak-ng
```

**Windows:**
[espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases) sayfasından indirin.

### 5. Kurulumu Test Edin

```bash
# Phonemizer test
python -m src.phonemizer

# Dataset test
python -m src.dataset
```

---

## 📚 Kullanım

### Eğitim (Training)

#### Hızlı Başlangıç

```bash
# Tam eğitim (48k samples, ~100 epoch)
python -m src.train --config config.yaml

# Debug modunda (100 sample ile test)
python -m src.train --config config.yaml --limit-samples 100
```

#### Checkpoint'ten Devam Etme

```bash
python -m src.train --config config.yaml --resume checkpoints/checkpoint_step_5000.pt
```

#### Eğitim Parametreleri

`config.yaml` dosyasında eğitim parametrelerini düzenleyebilirsiniz:

```yaml
training:
  batch_size: 2                      # Batch size (M4 için 2 önerilen)
  gradient_accumulation_steps: 16    # Effective batch: 2 × 16 = 32
  learning_rate: 1.0e-4              # Learning rate
  num_epochs: 100                    # Epoch sayısı
  device: "mps"                      # Device (mps, cuda, cpu)
  mixed_precision: true              # FP16 kullan
  early_stopping: true               # Early stopping aktif
  patience: 10                       # Early stopping patience
```

#### TensorBoard ile İzleme

```bash
# Eğitim sırasında başka bir terminalde:
tensorboard --logdir runs/

# Tarayıcıda açın: http://localhost:6006
```

### Inference (Ses Üretimi)

#### Tek Metin

```bash
python -m src.inference \
  --checkpoint checkpoints/best_model.pt \
  --text "Merhaba, size nasıl yardımcı olabilirim?"
```

#### Toplu İşlem (Batch)

```bash
# texts.txt dosyası oluşturun (her satırda bir cümle)
echo "Merhaba dünya" >> texts.txt
echo "Bugün hava çok güzel" >> texts.txt
echo "Türkçe metin seslendirilmesi" >> texts.txt

# Batch inference
python -m src.inference \
  --checkpoint checkpoints/best_model.pt \
  --input texts.txt \
  --output outputs/
```

#### Python API Kullanımı

```python
from src.inference import TTS
import yaml

# Config yükle
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# TTS sistemi oluştur
tts = TTS(
    config=config,
    checkpoint_path='checkpoints/best_model.pt'
)

# Ses üret
audio = tts.generate("Merhaba, nasılsınız?")

# Kaydet
tts.save_audio(audio, "output.wav")
```

---

## 📁 Proje Yapısı

```
tts/
├── README.md              # Bu dosya
├── plan.md                # Teknik plan (güncellenmiş)
├── config.yaml            # Eğitim konfigürasyonu
├── requirements.txt       # Python bağımlılıkları
│
├── src/                   # Kaynak kod
│   ├── __init__.py
│   ├── phonemizer.py      # Türkçe phonemization
│   ├── dataset.py         # Dataset loading ve preprocessing
│   ├── model.py           # StyleTTS2 + LoRA modeli
│   ├── train.py           # Eğitim scripti
│   └── inference.py       # Inference scripti
│
├── data/                  # Veri dizini
│   ├── cache/             # Hugging Face cache
│   └── phoneme_vocab.json # Phoneme vocabulary
│
├── checkpoints/           # Model checkpoints
│   ├── lora/              # LoRA weights
│   ├── best_model.pt      # En iyi model
│   └── checkpoint_*.pt    # Diğer checkpoints
│
├── outputs/               # Üretilen ses dosyaları
│   └── output_*.wav
│
└── runs/                  # TensorBoard logs
```

---

## ⚙️ Konfigürasyon

### Ana Parametreler

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `data.sample_rate` | Audio sample rate | 24000 |
| `data.n_mels` | Mel-spectrogram bins | 80 |
| `lora.r` | LoRA rank | 8 |
| `lora.lora_alpha` | LoRA alpha | 16 |
| `training.batch_size` | Batch size | 2 |
| `training.learning_rate` | Learning rate | 1e-4 |
| `training.num_epochs` | Epoch sayısı | 100 |

### LoRA Hedef Modüller

```yaml
lora:
  target_modules:
    - "q_proj"    # Query projection
    - "v_proj"    # Value projection
    - "k_proj"    # Key projection
    - "o_proj"    # Output projection
```

---

## 🔧 Troubleshooting

### 1. espeak-ng Bulunamadı

**Hata:**
```
Failed to initialize espeak-ng backend
```

**Çözüm:**
```bash
# macOS
brew install espeak-ng

# Linux
sudo apt-get install espeak-ng

# Test
espeak-ng --version
```

### 2. MPS Out of Memory

**Hata:**
```
RuntimeError: MPS backend out of memory
```

**Çözüm:**
- `batch_size` değerini azaltın (2 → 1)
- `gradient_accumulation_steps` değerini artırın
- `mixed_precision: false` yapın

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 32
  mixed_precision: false
```

### 3. Dataset Format Hatası

**Hata:**
```
WebDataset format error
```

**Çözüm:**
Dataset başarıyla yükleniyorsa sorun yok. Aksi takdirde:
```python
# Manuel yükleme
from datasets import load_dataset
dataset = load_dataset("zeynepgulhan/mediaspeech-with-cv-tr", split="train")
```

### 4. Yavaş Eğitim

**Öneriler:**
- GPU kullanın (AWS, Google Colab, vb.)
- Dataset boyutunu azaltın (`limit_samples` parametresi)
- Epoch sayısını azaltın
- Batch size artırın (GPU varsa)

---

## 📊 Beklenen Performans

### Eğitim Süresi (MacBook Pro M4)

| Konfigürasyon | Süre |
|---------------|------|
| 100 samples (debug) | ~10 dakika |
| 1000 samples | ~2 saat |
| 10,000 samples | ~1 gün |
| 48,000 samples (full) | ~7-14 gün |

### GPU ile Karşılaştırma

| Donanım | Süre (48k samples, 100 epoch) |
|---------|-------------------------------|
| M4 (MPS) | 7-14 gün |
| NVIDIA T4 | 3-5 gün |
| NVIDIA RTX 3090 | 2-3 gün |
| NVIDIA A100 | 1-2 gün |

---

## ⚠️ Önemli Notlar

### 1. Basitleştirilmiş Implementasyon

Bu projede kullanılan **SimplifiedStyleTTS2** modeli, eğitim amaçlı basitleştirilmiş bir versiyondur. Üretim kalitesi için:

- [StyleTTS2 GitHub](https://github.com/yl4579/StyleTTS2) reposunu kullanın
- Bu projedeki LoRA wrapper'ı resmi model ile kullanın
- Diffusion ve style encoder bileşenlerini ekleyin

### 2. Vocoder

Bu implementasyon **Griffin-Lim** vocoder kullanır (düşük kalite). Daha iyi sonuçlar için:

- [HiFi-GAN](https://github.com/jik876/hifi-gan) kullanın
- [Vocoder modelleri](https://huggingface.co/models?search=vocoder) indirin
- `inference.py`'de vocoder parametresini güncelleyin

### 3. Dataset Kalitesi

`mediaspeech-with-cv-tr` dataset'i çeşitli hoparlörler içerir. Daha iyi sonuçlar için:

- Tek hoparlör veri seti kullanın
- Veri filtrelemeyi iyileştirin
- Daha fazla eğitim verisi ekleyin

---

## 📖 Referanslar

### Makaleler

- [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models](https://arxiv.org/abs/2306.07691)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft)

### Kaynaklar

- **StyleTTS2 GitHub**: https://github.com/yl4579/StyleTTS2
- **Hugging Face PEFT**: https://github.com/huggingface/peft
- **Dataset**: https://huggingface.co/datasets/zeynepgulhan/mediaspeech-with-cv-tr
- **espeak-ng**: https://github.com/espeak-ng/espeak-ng

---

## 🤝 Katkıda Bulunma

Bu proje Claude Code tarafından oluşturulmuştur ve eğitim amaçlıdır. Katkılarınızı bekliyoruz:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📄 Lisans

Bu proje eğitim amaçlıdır ve MIT Lisansı altında dağıtılmaktadır.

**NOT**: Bu proje şu bileşenleri kullanır:
- StyleTTS2 (MIT License)
- PEFT (Apache 2.0)
- PyTorch (BSD License)
- Hugging Face datasets (Apache 2.0)

---

## 👨‍💻 Yazar

**Claude Code** tarafından oluşturuldu
- Plan: [plan.md](plan.md)
- Tarih: 2025

---

## 🎓 Sonuç

Bu proje, Türkçe TTS modeli eğitmek için eksiksiz bir pipeline sağlar. **LoRA** ile düşük kaynaklı eğitim mümkündür ve **Apple Silicon** cihazlarda çalışacak şekilde optimize edilmiştir.

Başarılı eğitimler! 🚀

---

## 📞 Destek

Sorularınız için:
1. GitHub Issues açın
2. Config dosyalarınızı kontrol edin
3. TensorBoard loglarını inceleyin
4. Troubleshooting bölümüne bakın

**Happy TTS training!** 🎤
