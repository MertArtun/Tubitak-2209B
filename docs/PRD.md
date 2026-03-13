# Ogrenci Dikkat Tespit Sistemi - Urun Gereksinim Dokumani (PRD)

## 1. Proje Bilgisi

| Alan | Deger |
|------|-------|
| Program | TUBITAK 2209-B Sanayiye Yonelik Lisans Arastirma Projesi |
| Proje Adi | Ogrenci Dikkat Tespit Sistemi |
| Arastirmaci | Lisans ogrencisi (tek gelistirici) |
| Danisman | Proje danismani (akademik yonetici) |
| Baslangic | Subat 2026 |
| Teslim Tarihi | Nisan 2026 |
| Referans Kod Tabani | `src/api/app.py`, `src/attention/scorer.py`, `src/face_recognition/database.py`, `src/attention/head_pose.py`, `src/dashboard/` |

## 2. Problem Tanimi

Sinif ici veya online derslerde ogrenci dikkat seviyesini surekli, olculebilir ve raporlanabilir sekilde takip etmek gereklidir. Manuel gozlem:
- olceklenemez,
- subjektiftir,
- zaman serisi analizi sunmaz.

Sistem; duygu analizi, yuz tanima ve oturum bazli loglama ile bu boslugu doldurur.

## 3. Cozum Yaklasimi

Tek kamera akisindan asagidaki adimlar gerceklestirilir:

1. Ogrenciyi tespit et,
2. Biliniyorsa kimligini esle,
3. Duygu sinifini tahmin et,
4. Dikkat skorunu hesapla,
5. Oturum bazli takip ve rapor uret.

### Faz 1: Online Mod (Temel)
- 3 sinifli duygu tahmini (`negative`, `neutral`, `positive`) ile dikkat skoru hesaplama.
- EfficientNet tabanli model, ONNX Runtime ile CPU uzerinde inference.

### Faz 2: Yuz Yuze Mod (Bonus)
- Head pose + gaze direction ile fiziksel dikkat olcumu.
- Emotion + head pose hybrid skoru.
- MediaPipe FaceMesh tabanli aci hesaplama.

## 4. Temel Gereksinimler

Asagidaki 9 gereksinim projenin cekirdek kapsamini olusturur.

| ID | Gereksinim | Endpoint | Kabul Kriteri |
|------|------|------|------|
| FR-001 | Session baslat / durdur | `POST /api/sessions/start`, `POST /api/sessions/<id>/stop` | Session state gecisleri dogru calisir; aktif session listelenebilir (`GET /api/sessions`) |
| FR-002 | Ogrenci kayit (coklu goruntu) | `POST /api/students/register` | En az 1 embedding kaydedilmeden kayit basarili sayilmaz |
| FR-003 | Ogrenci listeleme | `GET /api/students` | Kayitli ogrenciler dogru sekilde listelenir |
| FR-004 | Frame analiz endpointi | `POST /api/analyze` | Gecersiz inputta 4xx, basarili analizde yuz/duygu/skor bilgisi doner |
| FR-005 | Yuz tespiti ve kimlik esleme | (FR-004 icinde) | Bilinen yuzlerde threshold ustu esleme, bilinmeyende `student_id=null` |
| FR-006 | Duygu tahmini (3 sinif) | (FR-004 icinde) | Sonuc `class`, `confidence`, `probabilities` alanlarini icerir |
| FR-007 | Dikkat skoru ve seviye | (FR-004 icinde) | Skor [0,1] araliginda; seviyeler (`focused`, `moderate`, `distracted`) esiklere gore belirlenir |
| FR-010 | Ogrenci/session istatistikleri | `GET /api/students/<id>/stats`, `GET /api/sessions/<id>/stats` | Ortalama skor, dagilim ve toplam entry alanlari doner |
| FR-011 | Excel export | `GET /api/export/excel` | Overview, Per Student, Detailed Logs sayfalari uretilir |

## 5. Bonus Hedefler

Zaman kalirsa gerceklestirilecek hedefler. Kod altyapisi mevcuttur ancak API'ye tam entegre degildir.

| ID | Hedef | Mevcut Durum | Aciklama |
|------|------|------|------|
| FR-008 | Yuz yuze mod (hybrid skor) | `HeadPoseEstimator` ve `HybridAttentionScorer` siniflari mevcut | API'de `mode` parametresi ile etkinlestirilmesi gerekiyor |
| FR-012 | Dashboard canli izleme iyilestirmesi | Dashboard calisiyor | FPS ayari ve grafik guncellemelerinin ince ayari |
| FR-017 | Anomali tespiti | `detect_anomaly()` fonksiyonu mevcut | Analiz response'una `anomaly_flag` eklenmesi gerekiyor |

## 6. Basari Kriterleri

Akademik proje baglaminda gercekci hedefler:

| Metrik | Hedef | Olcum Yontemi |
|------|------|------|
| Model kalitesi | Macro F1 >= 0.75 | FER2013 test seti uzerinde degerlendirme |
| Inference hizi | Tek yuz analizi < 1 saniye | CPU uzerinde (ONNX Runtime), tek frame olcumu |
| Sistem kararliligi | Demo suresince kesintisiz calisma | Canli sunum sirasinda hata almama |
| Yuz esleme dogrulugu | Kontrollu ortamda (5-10 kisi) basarili esleme | Kayitli ogrencilerin dogru tanimlanmasi |
| Excel export | Hatasiz dosya uretimi | Export endpoint'inin gecerli `.xlsx` dosyasi donmesi |

## 7. Is Akislari

### 7.1 Canli Analiz Akisi
1. Dashboard frame yakalar.
2. `POST /api/analyze` ile frame gonderir.
3. API:
   - yuzleri tespit eder,
   - kimlik esler,
   - duygu tahmini yapar,
   - skor hesaplar,
   - aktif session varsa log yazar.
4. UI overlay + grafikler guncellenir.

### 7.2 Ogrenci Kayit Akisi
1. Kullanici isim + coklu goruntu gonderir.
2. Sistem her goruntude en buyuk yuzu secer.
3. Embedding cikarir ve veritabanina saklar.
4. En az 1 embedding varsa kayit basarili olur.

### 7.3 Session Raporlama Akisi
1. Session baslatilir (`POST /api/sessions/start`).
2. Session boyunca analiz loglari toplanir.
3. Session durdurulur (`POST /api/sessions/<id>/stop`).
4. Session bazli istatistik (`GET /api/sessions/<id>/stats`) ve Excel export (`GET /api/export/excel`) alinabilir.

## 8. Teknik Bagimliliklar

| Kategori | Kutuphaneler |
|------|------|
| Egitim (Colab) | `torch`, `timm`, `onnx` |
| Inference / API | `flask`, `onnxruntime`, `opencv-python`, `insightface`, `mediapipe` |
| Veri | `sqlite3` (stdlib), `openpyxl` |
| Test / Lint | `pytest`, `pytest-cov`, `ruff` |

## 9. Kapsam Disi

Asagidakiler bu projenin kapsaminda degildir:

- **API versiyonlama** (`/api/v1` prefix) - tek versiyon yeterli
- **Health / metrics / config endpointleri** - enterprise monitoring gereksimi yok
- **Veri saklama politikasi (retention / purge)** - akademik projede gereksiz
- **Ogrenci soft delete ve consent yonetimi** - KVKK kapsami bu proje icin gecerli degil
- **Standart hata modeli (`request_id`, `code`, `details`)** - basit hata mesajlari yeterli
- **Session pause / resume** - baslat/durdur yeterli
- **LMS entegrasyonu** - kapsam disi
- **Rol bazli kimlik dogrulama (RBAC)** - tek kullanici sistemi
- **Dagitik veritabani / migration** - SQLite yeterli
- **Mobil uygulama** - web dashboard yeterli
- **SLA / uptime hedefleri** - akademik demo ortami
- **Ops runbook / incident yonetimi** - tek gelistirici projesi

## 10. Varsayimlar

1. Sistem tek servis olarak calisir (monolit Flask).
2. Lokal agda veya tek makinede konumlanir; dis erisim beklenmez.
3. Kameradan gelen frame kalitesi minimum 640x480 seviyesindedir.
4. Kullanici sayisi sinirlidir (5-10 ogrenci, demo ortami).
5. Egitim Google Colab GPU ortaminda, inference yerel CPU'da yapilir.
6. Proje TUBITAK 2209-B teslim tarihine (Nisan 2026) kadar tamamlanir.
