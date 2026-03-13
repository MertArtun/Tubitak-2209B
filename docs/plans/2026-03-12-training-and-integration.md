# Model Eğitimi & Sistem Entegrasyonu — Uygulama Planı

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 4 modeli VS Code + Colab eklentisi ile eğitmek, en iyi modeli seçmek, ONNX export yapmak ve eksik entegrasyon parçalarını tamamlamak.

**Architecture:** VS Code'da Google Colab eklentisi ile GPU runtime'a bağlanarak eğitim yapılacak. Drive mount çalışmadığı için notebook path'leri lokal Colab runtime'a göre düzenlenecek. Eğitim sonrası en iyi ONNX model `models/` dizinine konulacak ve API entegrasyonu tamamlanacak.

**Tech Stack:** PyTorch, timm, albumentations, ONNX Runtime, Flask, Google Colab (T4/A100 GPU)

---

## Mevcut Durum Özeti

### Proje Neredeyiz?

- **Tarih:** 12 Mart 2026
- **Roadmap'e göre:** Faz 2 — Hafta 4 (Sistem entegrasyonu)
- **Deadline:** 12 Nisan 2026 (4 hafta kaldı)
- **Tamamlanan:** Tüm kod bileşenleri (model mimarileri, eğitim pipeline, API, dashboard, yüz tanıma, dikkat puanlama, testler, dokümantasyon)
- **Eksik:** Eğitilmiş model yok + bazı bileşenler API'ye bağlı değil

### Mevcut Veri Durumu

| Veri Seti | Konum | Toplam | Açıklama |
|-----------|-------|--------|----------|
| FER2013 (küçük) | `data/processed/` | ~32K görsel | 7→3 sınıf mapping, hazır |
| Birleşik (büyük) | `data/processed_merged_split/` | ~187K görsel | 10 dataset, stratified split, hazır |
| Birleşik (ham) | `data/processed_merged/` | ~187K görsel | Sadece train/val (test yok) |
| Temizlenmemiş arşiv | `data/processed_merged.tar.gz` | 7.9 GB | Sıkıştırılmış |

**Sınıf dağılımı (processed_merged_split):**

| Split | negative | neutral | positive | Toplam |
|-------|----------|---------|----------|--------|
| train | 59,585 | 43,100 | 46,643 | 149,328 |
| val | 7,448 | 5,388 | 5,831 | 18,667 |
| test | 7,448 | 5,388 | 5,831 | 18,667 |

### Eğitilecek 4 Model

| Model | Feature Dim | Hidden | Params (yaklaşık) | Beklenen Süre (T4) |
|-------|-------------|--------|--------------------|--------------------|
| efficientnet_b3 | 1536 | 512 | ~12M | ~45 dk |
| efficientnet_b0 | 1280 | 256 | ~5M | ~25 dk |
| mobilenet_v3 | 1280 | 256 | ~5M | ~20 dk |
| resnet50_cbam | 2048 | 512 | ~25M | ~50 dk |

### Eğitim Stratejisi

- **Faz 1 (5 epoch):** Backbone donuk, sadece head eğitimi, hafif augmentation, Focal Loss, lr=1e-3
- **Faz 2 (25 epoch):** Tüm katmanlar açık, güçlü augmentation, MixUp/CutMix, SoftTargetCE, cosine annealing lr=1e-4
- **Early stopping:** 7 epoch sabır
- **Checkpoint/resume:** Colab bağlantı koparsa kaldığı yerden devam

---

## BÖLÜM A: VS Code + Colab Ortam Kurulumu

### Task 1: VS Code Colab Eklentisini Kur

**Files:**
- Modify: `notebooks/02_model_training.ipynb` (path düzenlemeleri)

**Step 1: VS Code'da eklentiyi kur**

1. VS Code aç
2. Extensions (Ctrl+Shift+X) → "Google Colab" ara
3. **Google** yayıncısının resmi eklentisini kur
4. Jupyter eklentisi otomatik yüklenecek

**Step 2: Colab runtime'a bağlan**

1. `notebooks/02_model_training.ipynb` dosyasını VS Code'da aç
2. Sağ üstteki kernel seçiciye tıkla
3. "Connect to Google Colab" seç
4. Google hesabınla giriş yap
5. Runtime type: **GPU** (T4 veya daha iyisi)

**Step 3: GPU kontrolü**

İlk hücreyi çalıştır, şu çıktıyı gör:
```
PyTorch: 2.x.x
CUDA: True
GPU: Tesla T4 (veya benzeri)
```

---

## BÖLÜM B: Notebook Path Düzenlemeleri

### Task 2: Eğitim Notebook'unda Drive Path'lerini Güncelle

**Files:**
- Modify: `notebooks/02_model_training.ipynb` (cell-2 ve cell-3)

**Sorun:** VS Code Colab eklentisinde `drive.mount()` çalışmıyor. Tüm path'leri Colab runtime'ın lokal dosya sistemine göre düzenlemek gerekiyor.

**Step 1: cell-2'deki path tanımlarını değiştir**

Mevcut:
```python
DRIVE_ROOT = "/content/drive/MyDrive/Tubitak-2209B"
DATA_DIR = f"{DRIVE_ROOT}/data/cleaned"
CHECKPOINT_DIR = f"{DRIVE_ROOT}/checkpoints"
RESULTS_DIR = f"{DRIVE_ROOT}/results"
ONNX_DIR = f"{DRIVE_ROOT}/onnx_models"
```

Yeni:
```python
# VS Code + Colab: Drive mount yok, lokal runtime kullan
DRIVE_ROOT = "/content/tubitak"
DATA_DIR = f"{DRIVE_ROOT}/data"
CHECKPOINT_DIR = f"{DRIVE_ROOT}/checkpoints"
RESULTS_DIR = f"{DRIVE_ROOT}/results"
ONNX_DIR = f"{DRIVE_ROOT}/onnx_models"
```

**Step 2: cell-3'teki Drive mount hücresini veri yükleme ile değiştir**

Mevcut:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Yeni:
```python
# VS Code Colab: Veriyi lokal runtime'a yükle
import os, zipfile, subprocess

DATA_ROOT = "/content/tubitak/data"
os.makedirs(DATA_ROOT, exist_ok=True)

# SEÇENEK 1: Küçük veri seti (FER2013 ~32K, hızlı başlangıç)
# Projedeki data/processed/ klasörünü zip'le ve Colab'a yükle:
#   Terminalde: cd /Users/halitartun/Tubitak-2209B && zip -r data_processed.zip data/processed/
#   Sonra bu hücrede:
# from google.colab import files
# uploaded = files.upload()  # data_processed.zip seç
# !unzip -q data_processed.zip -d /content/tubitak/

# SEÇENEK 2: Büyük veri seti (~187K, önerilen)
# Projedeki data/processed_merged_split/ klasörünü zip'le ve Colab'a yükle:
#   Terminalde: cd /Users/halitartun/Tubitak-2209B && zip -r data_merged.zip data/processed_merged_split/
#   Sonra bu hücrede:
# from google.colab import files
# uploaded = files.upload()  # data_merged.zip seç
# !unzip -q data_merged.zip -d /content/tubitak/
# DATA_DIR ayarını güncelle:
# DATA_DIR = "/content/tubitak/data/processed_merged_split"

# Veri kontrolü
for split in ["train", "val", "test"]:
    split_dir = os.path.join(DATA_DIR, split)
    if os.path.exists(split_dir):
        counts = {}
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(split_dir, cls)
            counts[cls] = len(os.listdir(cls_dir)) if os.path.exists(cls_dir) else 0
        print(f"{split}: {sum(counts.values())} images — {counts}")
    else:
        print(f"UYARI: {split_dir} bulunamadı!")
```

**Step 3: cell-5'teki preprocessing report hücresini opsiyonel yap**

Bu hücre Drive'daki raporları okumaya çalışıyor. Hata vermemesi için try/except ekle veya atla:
```python
# Preprocessing raporları (opsiyonel — Drive mount yoksa atlanır)
print("Preprocessing raporu Drive'da. Atlanıyor...")
```

**Step 4: Notebook'u kaydet**

---

## BÖLÜM C: Veri Hazırlığı

### Task 3: Eğitim Verisini Zip'le

**Karar: Hangi veri seti ile eğitim yapılacak?**

| Seçenek | Veri | Boyut | Süre | Kalite |
|---------|------|-------|------|--------|
| A: Hızlı | `data/processed/` (~32K) | ~500 MB | ~30 dk/model | Orta (sadece FER2013) |
| B: Tam (önerilen) | `data/processed_merged_split/` (~187K) | ~5 GB | ~2 saat/model | Yüksek (10 dataset) |

**Önerilen strateji:** Önce Seçenek A ile hızlı bir deneme yap, her şey çalışıyorsa Seçenek B ile final eğitimi yap.

**Step 1: Küçük veriyi zip'le (Seçenek A)**

Terminalde çalıştır:
```bash
cd /Users/halitartun/Tubitak-2209B
zip -r data_processed.zip data/processed/
```

Beklenen boyut: ~400-600 MB

**Step 2 (opsiyonel): Büyük veriyi zip'le (Seçenek B)**

```bash
cd /Users/halitartun/Tubitak-2209B
zip -r data_merged.zip data/processed_merged_split/
```

Beklenen boyut: ~4-6 GB

---

## BÖLÜM D: Model Eğitimi

### Task 4: İlk Model Eğitimi — EfficientNet-B3

**Files:**
- Modify: `notebooks/02_model_training.ipynb` (cell-2: MODEL_NAME)

**Step 1: Config hücresinde model adını ayarla**

```python
MODEL_NAME = "efficientnet_b3"
```

**Step 2: Veriyi Colab runtime'a yükle**

- cell-3'teki veri yükleme hücresini çalıştır
- `files.upload()` ile zip dosyasını seç
- Unzip komutunu çalıştır

**Step 3: Tüm hücreleri çalıştır (Run All)**

Beklenen akış:
1. Paket kurulumu (cell-1): ~1 dk
2. Config (cell-2): anlık
3. Veri yükleme (cell-3): ~2-10 dk (boyuta göre)
4. Veri doğrulama (cell-4): anlık
5. Import'lar (cell-6): ~10 sn
6. Transform/Dataset/Model/Loss tanımları (cell-7 → cell-11): anlık
7. **EĞİTİM (cell-15):** ~30-120 dk
   - Faz 1 çıktısı: `[P1] 1/5 | Train: X.XXXX/X.XXXX | Val: X.XXXX/X.XXXX | XXs`
   - Faz 2 çıktısı: `[P2] 1/25 (g:6) | Train: X.XXXX/X.XXXX | Val: X.XXXX/X.XXXX | LR: X.XXe-XX | XXs`
8. Değerlendirme (cell-16): ~2 dk
9. Hardest examples (cell-17): ~2 dk
10. ONNX export (cell-18): ~1 dk
11. Eğitim eğrileri (cell-19): anlık

**Step 4: Sonuçları kontrol et**

Başarı kriterleri:
- Val accuracy: >= %70 (FER2013), >= %75 (merged)
- F1-macro: >= %65 (FER2013), >= %70 (merged)
- ONNX validation: "Validation PASSED"

**Step 5: Sonuçları indir**

```python
# Notebook'un sonuna ekle (yeni hücre)
import shutil
shutil.make_archive('/content/results_efficientnet_b3', 'zip', RESULTS_DIR)
shutil.make_archive('/content/onnx_efficientnet_b3', 'zip', ONNX_DIR)
shutil.make_archive('/content/checkpoints_efficientnet_b3', 'zip', CHECKPOINT_DIR)

from google.colab import files
files.download('/content/results_efficientnet_b3.zip')
files.download('/content/onnx_efficientnet_b3.zip')
```

**Step 6: Lokale kaydet**

İndirilen dosyaları proje dizinine çıkart:
```bash
# results → results/
# onnx_models/{model}.onnx → models/
```

---

### Task 5: Diğer 3 Modeli Eğit

Her model için Task 4'ün Step 1-6'sını tekrarla:

```
Sıra 2: MODEL_NAME = "efficientnet_b0"
Sıra 3: MODEL_NAME = "mobilenet_v3"
Sıra 4: MODEL_NAME = "resnet50_cbam"
```

**ÖNEMLİ:** Her model için:
- Colab runtime sıfırlanmışsa veriyi tekrar yükle
- Checkpoint sistemi sayesinde bağlantı koparsa aynı modeli tekrar başlat, kaldığı yerden devam eder

---

### Task 6: Modelleri Karşılaştır

**Files:**
- Modify: `notebooks/compare_models.ipynb` (path düzenlemeleri)

**Step 1: compare_models.ipynb'deki path'leri düzenle**

Drive path'lerini lokal path'lere çevir (Task 2 ile aynı mantık).

**Step 2: 4 modelin metrics.json dosyalarını Colab'a yükle**

Her modelin `{model}_metrics.json` dosyasını yükle.

**Step 3: Notebook'u çalıştır**

Çıktılar:
- Karşılaştırma tablosu (accuracy, F1, inference süresi, model boyutu)
- Ağırlıklı puanlama ile en iyi model seçimi (F1 %40, Inference %25, Accuracy %20, Boyut %15)
- Bar chart'lar, ROC eğrileri, confusion matrix karşılaştırması

**Step 4: En iyi modeli belirle**

Beklenen sonuç: EfficientNet-B3 muhtemelen en iyi F1'i verir, MobileNetV3 en hızlı inference.

---

## BÖLÜM E: ONNX Modelini Projeye Entegre Et

### Task 7: En İyi Modeli Projeye Koy

**Files:**
- Create: `models/emotion_model.onnx`

**Step 1: En iyi modelin ONNX dosyasını kopyala**

```bash
# Örnek: efficientnet_b3 en iyiyse
cp onnx_models/efficientnet_b3.onnx models/emotion_model.onnx
```

**Step 2: Doğrula**

```bash
python -c "
import onnxruntime as ort
import numpy as np
session = ort.InferenceSession('models/emotion_model.onnx')
inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
out = session.run(None, {session.get_inputs()[0].name: inp})[0]
print(f'Output shape: {out.shape}')  # (1, 3) olmalı
print(f'Predictions: {out}')
print('ONNX model çalışıyor!')
"
```

**Step 3: Commit**

```bash
# .onnx dosyası büyük olabilir, .gitignore'a ekle
echo "models/*.onnx" >> .gitignore
git add .gitignore
git commit -m "chore: add onnx models to gitignore"
```

---

## BÖLÜM F: API Entegrasyonu (Eğitim Sonrası)

### Task 8: Face-to-Face Modu API'ye Bağla

**Files:**
- Modify: `src/api/app.py` — `/analyze` endpoint'ine head pose desteği ekle
- Test: `tests/test_api.py`

**Bağlam:** `src/attention/hybrid_scorer.py` ve `src/attention/head_pose.py` hazır ama `src/api/app.py` sadece emotion-only puanlama yapıyor.

**Step 1: Test yaz**

`/analyze` endpoint'ine `mode=hybrid` parametresi gönderildiğinde response'ta `pose` alanı olmalı.

**Step 2: app.py'de /analyze endpoint'ini güncelle**

- Query param: `mode` (online | hybrid, default: online)
- `mode=hybrid` ise `HybridAttentionScorer` kullan
- Response'a `pose` (yaw, pitch, direction) ekle

**Step 3: Test çalıştır**

```bash
pytest tests/test_api.py -v
```

---

### Task 9: Anomaly Detection'ı Response'a Ekle

**Files:**
- Modify: `src/api/app.py`
- Test: `tests/test_api.py`

**Bağlam:** `AttentionScorer.update()` zaten anomali algılıyor ama API bunu response'a yansıtmıyor.

**Step 1: Test yaz**

Response'ta `anomaly_detected` (bool) alanı olmalı.

**Step 2: app.py güncelle**

Scorer'dan anomali bilgisini al, response'a ekle.

---

### Task 10: StudentAttentionTracker Entegrasyonu

**Files:**
- Modify: `src/api/app.py`
- Test: `tests/test_api.py`

**Bağlam:** `src/attention/tracker.py` öğrenci bazlı kayıt tutuyor ama API kullanmıyor.

**Step 1: Test yaz**

`/analyze` response'ta `student_id` ve `attention_history` alanları olmalı.

**Step 2: app.py güncelle**

Yüz tanıma sonucuyla tracker'ı besle, öğrenci bazlı takip sağla.

---

### Task 11: Response Alanlarını Tamamla

**Files:**
- Modify: `src/api/app.py`
- Test: `tests/test_api.py`

Eksik alanlar:
- `is_known`: Yüz tanınıyor mu (bool)
- `identity_confidence`: Tanıma güven skoru (float)
- `pose`: Baş pozisyonu (yaw, pitch, direction) — Task 8 ile birlikte

---

## BÖLÜM G: Doğrulama ve Commit

### Task 12: Tüm Testleri Çalıştır

**Step 1:**
```bash
pytest tests/ -v
```

Beklenen: Tüm testler geçmeli.

**Step 2: Lint**
```bash
ruff check src/ tests/
```

**Step 3: API'yi başlat ve elle test et**
```bash
python -m src.api.run --model-path models/emotion_model.onnx
# Başka terminalde:
curl http://localhost:5000/health
```

**Step 4: Commit**
```bash
git add -A
git commit -m "feat: complete model training and API integration"
```

---

## Zaman Planı (Önerilen)

| Gün | Task | Süre |
|-----|------|------|
| **1. gün** | Task 1-3: VS Code + Colab kurulum, veri hazırlığı | 1-2 saat |
| **1. gün** | Task 4: İlk model eğitimi (efficientnet_b3) | 1-2 saat |
| **2. gün** | Task 5: Diğer 3 model eğitimi | 3-5 saat |
| **2. gün** | Task 6: Model karşılaştırma | 30 dk |
| **3. gün** | Task 7: ONNX entegrasyonu | 30 dk |
| **3. gün** | Task 8-11: API entegrasyonu | 2-3 saat |
| **3. gün** | Task 12: Doğrulama | 30 dk |

**Toplam: ~3 gün**

---

## Başarı Kriterleri

| Metrik | Minimum | Hedef |
|--------|---------|-------|
| Val Accuracy | %70 | %78+ |
| F1-macro | %65 | %72+ |
| ONNX Inference (CPU) | <100 ms | <50 ms |
| API response time | <500 ms | <200 ms |
| Test pass rate | %100 | %100 |

---

## Notlar

- **Colab bağlantı koparsa:** Notebook'u tekrar çalıştır. Checkpoint sistemi var, kaldığı yerden devam eder.
- **GPU quota biterse:** Birkaç saat bekle veya Colab Pro kullan.
- **Veri yükleme yavaşsa:** Önce küçük veri seti (data/processed, ~32K) ile dene.
- **Model başarısı düşükse:** Büyük veri seti (data/processed_merged_split, ~187K) ile tekrar eğit.
- **files.upload() çalışmazsa:** Alternatif olarak veriyi Google Drive'a yükle, Colab web arayüzünden `drive.mount()` kullan.
