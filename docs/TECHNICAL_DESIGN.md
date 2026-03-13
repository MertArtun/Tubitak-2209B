# Teknik Tasarim - Student Attention Detection System

## 1. Tasarim Hedefleri

Bu dokumanin amaci:
1. Mimariyi netlestirmek,
2. Moduller arasi arayuzleri belirlemek,
3. Uygulama akislarini standardize etmek,
4. Mevcut durumu ve eksikleri acikca belirtmek.

## 2. Referans Kod ve Mevcut Yapi

Mevcut repo modulleri:
1. `src/models`: Model mimarileri (EfficientNet-B3/B0, MobileNetV3, ResNet50+CBAM), egitim, degerlendirme, ONNX export.
2. `src/api`: Flask app (`app.py`), kamera yonetimi (`camera.py`), ONNX inference engine (`inference.py`).
3. `src/face_recognition`: InsightFace detector (`detector.py`), ArcFace recognizer (`recognizer.py`), SQLite database (`database.py`), pipeline (`pipeline.py`).
4. `src/attention`: Emotion-based scorer (`scorer.py`), hybrid scorer (`hybrid_scorer.py`), head pose estimator (`head_pose.py`), per-student tracker (`tracker.py`).
5. `src/dashboard`: HTML/CSS/JS arayuz.
6. `configs/config.py`: Tek kaynakli sabitler.

## 3. Hedef Mimari

### 3.1 Bilesenler

1. **Inference/API Servisi (Flask)**:
   - Request kabul,
   - Pipeline orkestrasyonu,
   - Persistence,
   - Raporlama.
2. **Model Runtime**:
   - ONNX Runtime ile emotion inference (3 sinif: positive/negative/neutral).
3. **Face Runtime**:
   - InsightFace ile detection + ArcFace embedding.
4. **Attention Engine**:
   - `AttentionScorer`: Emotion-based skor (online mod).
   - `HybridAttentionScorer`: Emotion + pose skor (face-to-face mod).
5. **Data Store**:
   - SQLite (students, face_embeddings, sessions, attention_logs).
6. **Dashboard UI**:
   - Webcam frame capture,
   - API entegrasyonu,
   - Canli metrik gosterimi.

### 3.2 Dagitim Topolojisi

1. Tek node:
   - Flask process,
   - Lokal SQLite dosyasi,
   - Opsiyonel GPU runtime (CUDA destegi).
2. Port: `5000` (default, `configs/config.py` uzerinden degistirilebilir).

## 4. Request Akislari

### 4.1 Analyze Akisi (`POST /api/analyze`)

1. **Input decode**:
   - JSON body icinde base64 `image` alani veya multipart form `image` dosyasi.
2. **Validation**:
   - Image varligi kontrolu,
   - Decode basarisi kontrolu.
3. **Face detection**:
   - `FaceDetector.detect(frame)` -> bbox + face_crop listesi.
4. **Identity**:
   - `FaceRecognizer.get_embedding(face_info)` ile embedding cikarimi,
   - `FaceRecognizer.identify(embedding, known_embeddings)` ile cosine similarity eslesmesi.
5. **Emotion**:
   - `EmotionInferenceEngine.preprocess(face_crop)` -> (1, 3, 224, 224) tensor,
   - ONNX `session.run()` -> logits -> softmax -> class + confidence.
6. **Attention score**:
   - `AttentionScorer.engagement_score([prediction])` -> skor,
   - `AttentionScorer.classify_attention(score)` -> seviye.
7. **Persistence**:
   - Aktif session varsa ve ogrenci tanimliysa `attention_logs` tablosuna insert.
8. **Response**:
   - `results` listesi (her yuz icin) + `timestamp` + `frame_count`.

### 4.2 Session Akisi

1. **Start** (`POST /api/sessions/start`):
   - `name` ve `mode` (varsayilan: `"online"`) alarak yeni session olusturur.
   - `active_session_id` global degiskeni guncellenir.
2. **Stop** (`POST /api/sessions/<id>/stop`):
   - `end_time` alani set edilir.
   - `active_session_id` temizlenir.

### 4.3 Student Registration Akisi (`POST /api/students/register`)

1. Form-data: `name` (zorunlu), `email` (opsiyonel), `images[]` (zorunlu).
2. Her image icin:
   - Decode,
   - Face detect,
   - En buyuk yuz sec (bbox alanina gore),
   - ArcFace embedding cikar.
3. En az 1 embedding elde edildiyse commit.
4. Hicbir yuzu tespit edilemezse 400 donulur (ogrenci kaydi yine de olusturulur).

## 5. Mode Mimarisi

### 5.1 Mode Degerleri

1. `online` -- Sadece emotion-based skor.
2. `face-to-face` -- Emotion + head pose hybrid skor.

### 5.2 Skor Mantigi

1. **Emotion score** (`AttentionScorer`):
   - `sum(EMOTION_WEIGHTS[class] * confidence) / n`
   - Agirliklar: positive=0.8, neutral=0.5, negative=0.2.
2. **Pose score** (`HeadPoseEstimator`):
   - forward (yaw < 30, pitch < 25) = 1.0,
   - partial (yaw < 45, pitch < 37.5) = 0.3,
   - away = 0.0.
3. **Hybrid score** (`HybridAttentionScorer`):
   - `0.6 * emotion_score + 0.4 * pose_score`.

### 5.3 Esikler

1. `score > 0.6` -> focused.
2. `score > 0.35` -> moderate.
3. Diger -> distracted.

### 5.4 Anomaly Detection (`AttentionScorer.detect_anomaly`)

Son `ANOMALY_FRAME_WINDOW` (5) frame icinde ardisik iki skor arasindaki fark `ANOMALY_CHANGE_THRESHOLD` (0.4) degerini asarsa anomaly olarak isaretlenir.

## 6. Public Interface ve Type Modeli

### 6.1 API Endpointleri

| Method | Endpoint | Aciklama |
|--------|----------|----------|
| `POST` | `/api/analyze` | Tek frame analizi |
| `GET` | `/api/students` | Tum ogrencileri listele |
| `POST` | `/api/students/register` | Yeni ogrenci kaydi |
| `GET` | `/api/students/<id>/stats` | Ogrenci istatistikleri |
| `POST` | `/api/sessions/start` | Yeni session baslat |
| `POST` | `/api/sessions/<id>/stop` | Session durdur |
| `GET` | `/api/sessions` | Tum sessionlari listele |
| `GET` | `/api/sessions/<id>/stats` | Session istatistikleri |
| `GET` | `/api/export/excel` | Excel raporu indir |
| `GET` | `/dashboard` | Dashboard arayuzu |

### 6.2 Analyze Response Formati

Mevcut kodda (`app.py`) donulen response yapisi:

```json
{
  "results": [
    {
      "student_id": 1,
      "name": "Ali",
      "emotion": "positive",
      "confidence": 0.8532,
      "attention_score": 0.6826,
      "attention_level": "focused",
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "timestamp": "2026-02-18T14:30:00.000000",
  "frame_count": 42
}
```

**Not**: Asagidaki alanlar kodda tanimli ama henuz analyze response'a dahil edilmemistir:
- `is_known: bool` -- Ogrencinin tanilip taninamadigi (`student_id` null kontrolu ile cikarilabilir).
- `identity_confidence: float` -- Cosine similarity skoru.
- `pose: {yaw, pitch, roll, gaze, pose_score}` -- Head pose verileri.
- `anomaly_flag: bool` -- Anomaly detection sonucu.

### 6.3 Inference Engine Predict Ciktisi

```python
{
    "class": "positive" | "negative" | "neutral",
    "confidence": float,          # 0-1
    "probabilities": {
        "negative": float,
        "neutral": float,
        "positive": float
    }
}
```

## 7. Veri Katmani Tasarimi

### 7.1 Veritabani Semasi

Asagidaki tablo yapisi `src/face_recognition/database.py` dosyasindaki `_init_db` metodundan alinmistir:

#### `students`
| Kolon | Tip | Kisitlama |
|-------|-----|-----------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `name` | TEXT | NOT NULL |
| `email` | TEXT | - |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |

#### `face_embeddings`
| Kolon | Tip | Kisitlama |
|-------|-----|-----------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `student_id` | INTEGER | NOT NULL, REFERENCES students(id) |
| `embedding` | BLOB | NOT NULL (512-dim float32) |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |

#### `sessions`
| Kolon | Tip | Kisitlama |
|-------|-----|-----------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `name` | TEXT | - |
| `mode` | TEXT | CHECK(mode IN ('online', 'face-to-face')) |
| `start_time` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |
| `end_time` | TIMESTAMP | - |

#### `attention_logs`
| Kolon | Tip | Kisitlama |
|-------|-----|-----------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `student_id` | INTEGER | NOT NULL, REFERENCES students(id) |
| `session_id` | INTEGER | NOT NULL, REFERENCES sessions(id) |
| `timestamp` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |
| `emotion` | TEXT | - |
| `confidence` | REAL | - |
| `attention_score` | REAL | - |
| `attention_level` | TEXT | - |

**Not**: Foreign key constraint `PRAGMA foreign_keys = ON` ile aktif edilmistir.

## 8. Performans ve Kapasite

1. **Analyze FPS hedefi**: 2 FPS (MVP default, `configs/config.py` icinde `ANALYZE_FPS`).
2. **Frame boyutu**: 640x480 default.
3. **Model input boyutu**: 224x224 (ImageNet normalizasyonu).
4. **Flask serveri**: Tek worker development serveri MVP icin kullanilmaktadir.
5. **ONNX Runtime provider sirasi**:
   - `CUDAExecutionProvider` (GPU varsa),
   - fallback `CPUExecutionProvider`.
6. **Sliding window**: Attention skorlama son 30 frame uzerinden (`ATTENTION_WINDOW_SIZE`).

## 9. Guvenlik ve Gizlilik

1. Ham frame kalici diskte tutulmaz -- sadece bellekte islenir.
2. Export dosyalari gecici dizinde olusturulur (`tempfile`).
3. Yuz embeddingleri 512-boyutlu float32 BLOB olarak saklanir; orijinal goruntu kaydedilmez.

## 10. Mevcut Durum

### 10.1 Tamamlanan ve Calisan Bilesenler

- **Model mimarisi**: 4 farkli backbone (EfficientNet-B3, EfficientNet-B0, MobileNetV3-Large, ResNet50+CBAM) ile 3-sinifli (positive/negative/neutral) emotion classifier. CBAM attention modulu ResNet50 ile entegre. Tum modeller `EmotionClassifier` abstract sinifini implement eder.
- **Egitim altyapisi**: 2 fazli egitim (head-only + full fine-tuning), Focal Loss, early stopping, class-weighted sampling (`BalancedSampler`), kapsamli augmentation pipeline.
- **ONNX export**: `src/models/export_onnx.py` ile PyTorch -> ONNX donusumu.
- **ONNX inference engine**: `EmotionInferenceEngine` sinifi ile preprocess -> infer -> softmax pipeline.
- **Face detection**: InsightFace tabanli `FaceDetector`.
- **Face recognition**: ArcFace tabanli `FaceRecognizer` ile embedding cikarimi ve cosine similarity eslesmesi.
- **Attention scoring (online mod)**: `AttentionScorer` -- emotion agirlikli skor hesabi, sliding window, seviye siniflandirmasi.
- **Hybrid scoring (face-to-face mod)**: `HybridAttentionScorer` sinifi tamamdir.
- **Head pose estimation**: `HeadPoseEstimator` -- MediaPipe Face Mesh + cv2.solvePnP ile yaw/pitch/roll hesaplama, gaze yonu siniflandirmasi.
- **Per-student tracker**: `StudentAttentionTracker` -- bellekte ogrenci bazli zaman serisi birikimi.
- **Anomaly detection**: `AttentionScorer.detect_anomaly()` metodu tamamdir.
- **Veritabani CRUD**: Ogrenci kayit, embedding saklama, session yonetimi, attention log kaydi, istatistik hesabi.
- **API endpointleri**: 9 endpoint + dashboard sayfasi (Bolum 6.1'deki tablo).
- **Excel export**: Coklu sayfa (Overview, Per Student, Detailed Logs) ile raporlama.
- **Dashboard**: Canli webcam, session yonetimi, ogrenci listesi, istatistik paneli.
- **FER2013 veri hazirligi**: 7 siniftan 3 sinifa mapping, surprise sinifi dislanmasi, train/val/test bolunmesi.

### 10.2 Henuz Entegre Edilmemis / Eksik Parcalar

- **Egitilmis ONNX model dosyasi yok**: Model mimarisi ve export kodu hazir ancak henuz bir ONNX model dosyasi uretilmemistir. Sistem calismasi icin Google Colab'da egitim yapilip `models/emotion_model.onnx` dosyasi olusturulmalidir.
- **Face-to-face mod API'ye baglanmamis**: `HybridAttentionScorer` ve `HeadPoseEstimator` siniflari tamamdir, ancak `app.py` icindeki `/api/analyze` endpointi sadece `AttentionScorer` (emotion-only) kullaniyor. Session mode'u `face-to-face` olarak secilse bile head pose hesaplanmiyor.
- **Anomaly detection response'a yansitilmiyor**: `detect_anomaly()` metodu kodda mevcut ancak `/api/analyze` response'unda `anomaly_flag` alani donulmuyor.
- **Tracker API'ye baglanmamis**: `StudentAttentionTracker` sinifi mevcut ama `app.py` tarafindan kullanilmiyor.
- **Ek response alanlari eksik**: `is_known`, `identity_confidence`, `pose` alanlari analyze response'unda yer almiyor (Bolum 6.2 notu).

### 10.3 Yapilacaklar (TODO)

1. Google Colab'da model egitimi ve ONNX export.
2. `/api/analyze` icinde session mode'una gore `HybridAttentionScorer` + `HeadPoseEstimator` entegrasyonu.
3. Anomaly detection sonucunun response'a eklenmesi.
4. `StudentAttentionTracker` kullanilarak bellekte canli takip.
5. Analyze response'una `is_known`, `identity_confidence`, `pose` alanlarinin eklenmesi.

## 11. Model Mimarisi Detayi

### 11.1 Desteklenen Modeller

| Model | timm adi | Feature dim | Hidden dim | Dropout |
|-------|----------|-------------|------------|---------|
| `efficientnet_b3` | `efficientnet_b3` | 1536 | 512 | 0.3 / 0.2 |
| `efficientnet_b0` | `efficientnet_b0` | 1280 | 256 | 0.3 / 0.2 |
| `mobilenet_v3` | `mobilenetv3_large_100` | 1280 | 256 | 0.3 / 0.2 |
| `resnet50_cbam` | `resnet50` | 2048 | 512 | 0.3 / 0.2 |

### 11.2 Classifier Head Yapisi

Tum modeller ortak bir head kullanir:
```
AdaptiveAvgPool2d(1) -> Flatten -> Dropout(p1) -> Linear(feature_dim, hidden_dim) -> ReLU -> Dropout(p2) -> Linear(hidden_dim, 3)
```

### 11.3 Egitim Stratejisi

- **Faz 1**: Backbone dondurulur, sadece head egitilir (5 epoch, lr=1e-3).
- **Faz 2**: Tum agirliklar acilir, full fine-tuning (25 epoch, lr=1e-4).
- **Loss**: Focal Loss (gamma=2.0) ile class imbalance problemi ele alinir.
- **Early stopping**: 7 epoch patience.
- **Veri**: FER2013 -> 3 sinif (surprise dislanir), 70/15/15 train/val/test split.
