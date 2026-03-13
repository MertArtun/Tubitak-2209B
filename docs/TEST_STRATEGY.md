# Test Stratejisi

## 1. Test Amaclari

Bu belge, ogrenci dikkat tespit sisteminin temel islevselligini dogrulamak icin
uygulanan test stratejisini tanimlar. Amac:

1. Algoritma ve utility fonksiyonlarinin birim duzeyinde dogrulanmasi.
2. API, veritabani ve pipeline bilesenleri arasindaki entegrasyonun test edilmesi.

## 2. Test Kapsami

### 2.1 Dahil
- **Unit test:** Attention scorer, face recognition, inference, veri hazirlama,
  loss fonksiyonlari ve mode-aware scoring modulleri.
- **Integration test:** Ogrenci kayit akisi, session yasam dongusu, export,
  face-to-face mode ve unknown face senaryolari.

### 2.2 Haric
- Ucuncu parti kutuphanelerin ic testleri (ONNX Runtime, InsightFace vb.).
- End-to-end browser/kamera testleri (ortam siniri nedeniyle).
- Performance ve load testleri (tek node gelistirme ortami).

## 3. Test Veri Setleri

| Veri Seti | Aciklama |
|---|---|
| Synthetic | Random image/tensor fixture uretimi |
| Controlled face samples | Bilinen ve bilinmeyen yuz embedding setleri |
| Boundary payload | Bos image, bozuk base64, unsupported mode |

## 4. Unit Testler

| Test ID | Modul | Kapsam |
|---|---|---|
| TC-ALG-001 | `attention.scorer` | Esik siniflandirma |
| TC-ALG-002 | `attention.scorer` | Sliding window davranisi |
| TC-ALG-003 | `attention.scorer` | Anomaly detection |
| TC-ALG-004 | `attention.hybrid_scorer` | Agirlikli kombinasyon |
| TC-ALG-005 | `face_recognition.recognizer` | Cosine similarity |
| TC-ALG-006 | `face_recognition.recognizer` | Threshold matching |
| TC-ALG-007 | `api.inference` | Preprocess shape ve range |
| TC-ALG-008 | `data.prepare_data` | FER 7->3 map dogrulama |
| TC-ALG-009 | `models.losses` | Focal loss / cross-entropy uyumu |
| TC-ALG-010 | mode-aware scoring | Online vs face-to-face ayrimi |

## 5. Integration Testler

| Test ID | Senaryo | Beklenti |
|---|---|---|
| TC-INT-001 | Ogrenci kayit -> embedding save | En az 1 embedding yoksa fail |
| TC-INT-002 | Session start -> analyze -> stop | Loglar dogru sessiona yazilir |
| TC-INT-004 | Session stats -> export excel | Sayfalar ve sayilar tutarli |
| TC-INT-005 | Face-to-face mode | Pose alanlari response'ta dolar |
| TC-INT-006 | Unknown face akisi | `student_id=null`, `is_known=false` |

## 6. Testleri Calistirma

Tum testleri calistirmak icin:

```bash
pytest tests/ -v
```

Lint kontrolu:

```bash
ruff check src/ tests/
```

Belirli bir test grubunu calistirmak icin:

```bash
# Sadece unit testler
pytest tests/unit/ -v

# Sadece integration testler
pytest tests/integration/ -v
```

## 7. Bilinen Kisitlamalar

1. **Domain shift:** Egitim verisindeki yuz dagilimi ile gercek sinif ortami
   arasinda farklilik olabilir; bu durum model dogrulugunu dusurur.
2. **Aydinlatma ve kamera acisi:** Duzensiz isik kosullari ve farkli kamera
   acilari emotion/pose tahminlerini olumsuz etkileyebilir.
3. **Tek node siniri:** Sistem tek bir makine uzerinde calisacak sekilde
   tasarlanmistir; buyuk siniflar icin olcekleme planlanmamistir.
