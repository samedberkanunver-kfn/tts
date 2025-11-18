# GitHub Setup Guide

Bu dosya, projeyi GitHub'a pushlamak için adım adım talimatları içerir.

## Neden Büyük Dosyalar GitHub'a Pushlanmıyor?

**Durum:**
- Model dosyaları: 313 MB (`models/kokoro-82m/`)
- Dataset cache: 1.6 GB (`data/cache/`)
- GitHub limiti: Tek dosya 100 MB

**Çözüm:**
- ✅ Kaynak kodları → GitHub'a push
- ✅ Model ve dataset → Colab'da otomatik indirilir (Hugging Face'ten)
- ✅ Checkpoints → Google Drive'a kaydedilir (Colab'da)

## Adımlar

### 1. GitHub'da Yeni Repo Oluştur

1. https://github.com/new adresine git
2. Repository name: `turkish-tts-training` (ya da istediğiniz isim)
3. Description: "Turkish TTS fine-tuning with Kokoro-82M and LoRA"
4. Public/Private seçin
5. **Initialize this repository with a README** seçmeyin (zaten var)
6. **Create repository** tıklayın

### 2. Local'de Git Komutlarını Çalıştır

Terminal'de bu komutları sırayla çalıştırın:

```bash
cd /Users/kafein/Desktop/samed/tts

# Git zaten init edildi, kontrol edelim
git status

# Tüm dosyaları ekle (models/ ve data/cache/ hariç)
git add .

# İlk commit
git commit -m "Initial commit: Turkish TTS fine-tuning project

- Kokoro-82M model training with LoRA
- Turkish phonemization via espeak-ng
- Google Colab notebook for GPU training
- MPS support for Apple Silicon"

# GitHub remote ekle (BURAYA KENDİ REPO URL'NİZİ YAZIN)
git remote add origin https://github.com/YOUR_USERNAME/turkish-tts-training.git

# Push
git branch -M main
git push -u origin main
```

### 3. GitHub Remote URL Nasıl Bulunur?

GitHub'da yeni oluşturduğunuz repo sayfasında:
- Yeşil **Code** butonuna tıklayın
- HTTPS tab'ını seçin
- URL'yi kopyalayın (örn: `https://github.com/kullanici/turkish-tts-training.git`)

### 4. Colab Notebook'ta Güncelleme

GitHub'a push ettikten sonra, `colab.ipynb` içinde **Hücre 3**'teki URL'yi güncelleyin:

```python
GITHUB_REPO = "https://github.com/YOUR_USERNAME/turkish-tts-training.git"
```

## Push Edilen Dosyalar

```
tts/
├── .gitignore              # Büyük dosyaları hariç tutar
├── CLAUDE.md               # Claude Code için rehber
├── README.md               # Proje dokümantasyonu
├── GITHUB_SETUP.md         # Bu dosya
├── colab.ipynb             # Google Colab training notebook
├── config.yaml             # Training konfigürasyonu
├── plan.md                 # Teknik plan
├── requirements.txt        # Python dependencies
└── src/                    # Kaynak kodlar
    ├── __init__.py
    ├── phonemizer.py
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── inference.py
    ├── kokoro_*.py
    └── ...
```

**PUSHLANMAYAN** (Colab'da otomatik indirilir):
- `models/` - Hugging Face'ten indirilir
- `data/cache/` - Dataset otomatik cache'lenir
- `checkpoints/` - Google Drive'a kaydedilir
- `venv/` - Her ortamda yeniden oluşturulur

## Troubleshooting

### "remote: Repository not found"
→ GitHub repo URL'sini kontrol edin, doğru kullanıcı adı/repo adı olmalı

### "failed to push some refs"
→ GitHub'da repo oluştururken README eklediyseniz:
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### "Large files detected"
→ .gitignore çalışmıyordur, tekrar kontrol edin:
```bash
git rm --cached -r models/
git rm --cached -r data/cache/
git commit -m "Remove large files"
```

## Sonraki Adımlar

1. ✅ GitHub'a push tamamlandı
2. ✅ VS Code'da `colab.ipynb` aç
3. ✅ Hücre 3'teki GitHub URL'sini güncelle
4. ✅ Colab'da çalıştır

## İsterseniz: Git LFS (Large File Storage)

Eğer ileride büyük dosyaları da GitHub'a pushlamak isterseniz:

```bash
# Git LFS kur (macOS)
brew install git-lfs
git lfs install

# Büyük dosya tiplerini track et
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes

# Commit ve push
git add models/
git commit -m "Add model files via LFS"
git push
```

**Not:** GitHub ücretsiz hesaplarda 1GB LFS storage var (sizde 2GB veri var).
