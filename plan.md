📄 Kokoro-82M Türkçe Fine-Tuning Planı (Claude Code için teknik açıklama)

Amaç:
Kokoro-82M adlı açık kaynak TTS modelini Türkçe konuşacak hale getirmek. Model Hugging Face’ten alınacak (hexgrad/Kokoro-82M) ve Turkish_Speech_Corpus veri seti ile MacBook Pro M4 cihazda fine-tune edilecek. Eğitim düşük kaynaklı olduğundan LoRA tekniği kullanılacak.

1. Ortam Bilgisi

Donanım: MacBook Pro M4 (Apple Silicon)

Python 3.10

Eğitim framework’ü: PyTorch + torchaudio

Destek kitaplıklar: Hugging Face datasets, loralib, sentencepiece, soundfile, librosa, einops

2. Model Bilgisi

Kokoro-82M aslında StyleTTS2 mimarisine dayanan bir TTS modelidir (Bark tabanlı DEĞİLDİR).

Hugging Face'de hexgrad/Kokoro-82M adresinde yer alır.

Tam fine-tune yerine LoRA ile hafif fine-tune yapılacaktır (parametre azaltmak için).

Eğitim sırasında yalnızca transformer katmanları açılacak, diğer tüm katmanlar sabit kalacaktır.

3. Veri Seti Bilgisi

Dataset: zeynepgulhan/mediaspeech-with-cv-tr (Hugging Face datasets üzerinden çekilecek)

Veri biçimi: WAV dosyası (24kHz mono) + Türkçe transkript

Filtreleme: Çok kısa cümleler (5 kelime altı) çıkarılacak

Gerekirse tüm WAV dosyaları yeniden 24kHz'e resample edilecek

4. Eğitim Öncesi Adımlar

WAV dosyaları normalize edilecek (mono, 24kHz)

Dataset torch.utils.data.Dataset formatına çevrilecek

text + audio ikilisiyle örnekler oluşturulacak

Tokenizer gerekiyorsa sentencepiece veya Kokoro ile gelen tokenizer kullanılacak

5. LoRA Entegrasyonu

Tüm model parametreleri requires_grad=False olarak ayarlanacak

Sadece Linear katmanlar içindeki lora alt modülleri eğitilebilir hale getirilecek

LoRA parametreleri: r=8, alpha=16

LoRA, Hugging Face PEFT kütüphanesi ile uygulanacak (loralib yerine)

6. Eğitim Ayarları

Optimizer: AdamW

Learning rate: 1e-4

Epochs: 5

Batch size: 1–2

Mixed precision (fp16): Apple MPS destekliyorsa aktif edilebilir

Loss function: MSELoss veya L1 loss (örneğin spectrogram hedefli)

Veriler küçükse (1000–3000 örnek), overfit riskine karşı erken durdurma yapılmalı

7. Inference/Test

Eğitim sonunda model .generate(text) fonksiyonu ile test edilecek

Çıktı audio torchaudio ile .wav olarak kaydedilecek

Basit örnek: "merhaba, size nasıl yardımcı olabilirim?"

8. Alternatif Donanım (Opsiyonel)

Eğitim süresi M4 cihazda 6–12 saat sürebilir

Daha hızlı eğitim için AWS g4dn.xlarge veya g5.xlarge gibi GPU’lu instance’lar önerilir

Bu plana uygun olarak Claude Code’dan aşağıdaki gibi şeyler yazması istenebilir:

LoRA entegre edilmiş KokoroModel sınıfı sarmalayıcı

Turkish_Speech_Corpus veri çekme ve WAV işlemleri

torch.utils.data.Dataset sınıfı

Eğitim döngüsü (train() fonksiyonu)

Model checkpoint kaydı

generate() fonksiyonu ile inference çıktısı