# Proje Yol Haritasi — TUBITAK 2209-B

> Tek kisi (lisans ogrencisi + danisman), 8 hafta, Nisan 2026 teslim.

## Genel Bakis

| Bilgi | Deger |
|---|---|
| Baslangic | 18 Subat 2026 |
| Bitis (deadline) | 12 Nisan 2026 |
| Toplam sure | 8 hafta, 3 faz |
| Calisan | 1 kisi |
| Hedef | Ogrenci dikkat tespiti sistemi (duygu analizi + yuz tanima) |

---

## Faz 1 — Model Egitimi ve Dogrulama

**Tarih:** Hafta 1-3 (18 Subat — 8 Mart)

**Amac:** FER2013 veri seti uzerinde duygu siniflandirma modeli egitmek,
degerlendirmek ve ONNX formatinda cikarmak.

### Gorevler

| # | Gorev | Tahmini Sure |
|---|---|---|
| 1.1 | FER2013 veri setini indirip Colab'a yuklemek | 1 gun |
| 1.2 | `src/data/` altindaki transform ve loader kodlarini dogrulamak | 1 gun |
| 1.3 | EfficientNet-B3 ile 2 asamali egitim (frozen backbone → full fine-tune) | 3 gun |
| 1.4 | Alternatif modelleri denemek (EfficientNet-B0, MobileNetV3) | 2 gun |
| 1.5 | Degerlendirme: confusion matrix, per-class F1, accuracy | 2 gun |
| 1.6 | En iyi modeli ONNX formatinda export etmek | 1 gun |
| 1.7 | Egitim notebook'unu ve sonuclari dokumante etmek | 1 gun |

### Komutlar

```bash
# Egitim
python -m src.models.train --model efficientnet_b3 --data_dir data/processed

# ONNX export
# (notebook icerisinde veya ayri script ile)
```

### Ciktilar

- `models/emotion_model.onnx` — uretim modeli
- Degerlendirme raporu (confusion matrix, F1 skorlari, accuracy)
- Egitim notebook'u (`notebooks/` altinda)

### Basari Kriteri

- 3-sinif (Positive / Negative / Neutral) F1-macro >= 0.75
- ONNX modeli ONNX Runtime ile sorunsuz yukleniyor ve inference veriyor

---

## Faz 2 — Sistem Entegrasyonu ve Demo

**Tarih:** Hafta 4-6 (9 Mart — 29 Mart)

**Amac:** Egitilen modeli Flask sunucusuyla entegre etmek, yuz tanima ile
ogrenci bazli takip saglamak ve canli demo hazirlayabilir duruma gelmek.

### Gorevler

| # | Gorev | Tahmini Sure |
|---|---|---|
| 2.1 | Flask sunucusunu ONNX modeliyle ayaga kaldirmak | 2 gun |
| 2.2 | Test ogrencilerini kaydetmek (5-10 kisi, InsightFace/ArcFace) | 2 gun |
| 2.3 | Dashboard canli akis: kamera → analiz → gorsellestirrme | 3 gun |
| 2.4 | Session akisi: baslat → analiz et → durdur → export | 2 gun |
| 2.5 | Dashboard CSS/JS tutarsizliklarini gidermek | 2 gun |
| 2.6 | Mevcut testleri calistirmak, kiriklari onarmak | 2 gun |
| 2.7 | **[BONUS]** Yuz yuze mod entegrasyonu (head pose + gaze) | 3 gun |

### Komutlar

```bash
# Sunucu baslatma
python -m src.api.run --model-path models/emotion_model.onnx

# Testler
pytest tests/ -v

# Lint
ruff check src/ tests/
```

### Ciktilar

- Calisan Flask sunucusu + Dashboard demo
- Ogrenci yuz veritabani (SQLite)
- Gecen test suite
- (Bonus) Yuz yuze mod destegi

### Basari Kriteri

- Sunucu kamera girdisinden duygu tahmini donduruyor
- Dashboard'da canli analiz goruntuleniyor
- Session export JSON/CSV olarak indirilebiliyor
- `pytest tests/ -v` tamami geciyor

---

## Faz 3 — Sonuc Toplama ve Rapor

**Tarih:** Hafta 7-8 (30 Mart — 12 Nisan)

**Amac:** Sistemi farkli senaryolarda test etmek, sonuclari toplamak,
TUBITAK final raporunu ve demo videosunu hazirlamak.

### Gorevler

| # | Gorev | Tahmini Sure |
|---|---|---|
| 3.1 | Farkli senaryolarda test oturumlari duzenlemek | 2 gun |
| 3.2 | Sonuc verilerini export etmek ve analiz etmek | 2 gun |
| 3.3 | Performans tablolari ve grafikleri olusturmak | 2 gun |
| 3.4 | Demo videosu kaydetmek | 1 gun |
| 3.5 | TUBITAK final raporunu yazmak | 4 gun |
| 3.6 | Teslim oncesi son kontrol ve duzeltmeler | 1 gun |

### Ciktilar

- TUBITAK final raporu
- Demo videosu
- Test sonuc verileri (tablolar, grafikler)

### Basari Kriteri

- En az 3 farkli senaryoda test oturumu tamamlanmis
- Final raporu TUBITAK formatina uygun
- Demo videosu sistemin uctan uca calistigini gosteriyor

---

## Haftalik Zaman Cizelgesi

```
Hafta 1  (18-23 Sub)  : Veri hazirligi + ilk egitim denemeleri
Hafta 2  (24 Sub-1 Mar): EfficientNet-B3 egitimi + alternatif modeller
Hafta 3  (2-8 Mar)    : Degerlendirme + ONNX export + dokumantasyon
Hafta 4  (9-15 Mar)   : Flask sunucu + ONNX entegrasyonu
Hafta 5  (16-22 Mar)  : Dashboard canli demo + session akisi
Hafta 6  (23-29 Mar)  : Test/fix + bonus yuz yuze mod
Hafta 7  (30 Mar-5 Nis): Test oturumlari + veri analizi
Hafta 8  (6-12 Nis)   : Final raporu + demo videosu + teslim
```

---

## Risk ve Tampon Plani

| Risk | Etki | Onlem |
|---|---|---|
| Model dogrulugu dusuk cikiyor | Faz 2'ye gecis gecikir | Daha basit model (B0, MobileNetV3) ile devam et |
| Colab GPU kotasi bitiyor | Egitim yavaslar | Epoch sayisini azalt, kucuk model sec |
| InsightFace kurulumu sorun cikartiyor | Yuz tanima calismiyor | dlib veya OpenCV alternatifine gec |
| Dashboard entegrasyonu beklenenden uzun suruyor | Demo gecikir | Minimal UI ile devam et, goruntuyu sonra duzelt |
| Hastalik / sinav haftasi | Zaman kaybi | Faz 3'te 2 gunluk tampon mevcut |

---

## Notlar

- Bu yol haritasi tek kisinin (lisans ogrencisi) gercekci is yukune gore planlanmistir.
- Gunluk calismalar ortalama 3-4 saat uzerine hesaplanmistir.
- **[BONUS]** ile isaretlenen gorevler zaman kalirsa yapilacaktir, zorunlu degildir.
- Danisman ile haftalik ilerleme gorusmesi yapilmasi onerilir.
- Tum egitim isleri Google Colab (GPU) uzerinde, diger isler lokal makinede yurutulur.
