# API Contract - Student Attention Detection System

## 1. Genel Ilkeler

1. **Base path:** `/api/*` (tum endpointler bu prefix altindadir).
2. **Format:** Request/response `application/json` (file upload ve export endpointleri haric).
3. **Saat formati:** ISO-8601.
4. **Kimlik dogrulama:** Bu surumde zorunlu auth yok; intranet trusted ortam varsayilir.
5. **CORS:** Tum originler icin aktif.

## 2. Hata Formati

Tum hata cevaplari asagidaki formatta doner:

```json
{
  "error": "Hata aciklamasi burada yer alir."
}
```

Uygun HTTP status kodu ile birlikte doner (400, 404, 500 vb.).

## 3. Tip Sozlesmeleri

### 3.1 AnalyzeResult

`POST /api/analyze` endpointinin `results` dizisindeki her eleman:

```json
{
  "student_id": 12,
  "name": "Ayse Yilmaz",
  "emotion": "positive",
  "confidence": 0.9312,
  "attention_score": 0.79,
  "attention_level": "focused",
  "bbox": [100, 72, 260, 280]
}
```

| Alan | Tip | Aciklama |
|---|---|---|
| `student_id` | `int \| null` | Taninmis ogrenci ID'si; taninamazsa `null` |
| `name` | `string \| null` | Ogrenci adi; taninamazsa `null` |
| `emotion` | `string` | Tahmin edilen duygu sinifi (`positive`, `negative`, `neutral`) |
| `confidence` | `float` | Duygu tahmin guveni, [0, 1] araliginda |
| `attention_score` | `float` | Dikkat skoru, [0, 1] araliginda |
| `attention_level` | `string` | `focused`, `moderate` veya `distracted` |
| `bbox` | `[int, int, int, int]` | Yuz bounding box `[x1, y1, x2, y2]` |

### 3.2 AnalyzeResponse

```json
{
  "results": [],
  "timestamp": "2026-02-17T12:34:56.123456",
  "frame_count": 42
}
```

| Alan | Tip | Aciklama |
|---|---|---|
| `results` | `AnalyzeResult[]` | Tespit edilen yuzlerin analiz sonuclari |
| `timestamp` | `string` | ISO-8601 formatinda islem zamani |
| `frame_count` | `int` | Sunucu basladigindan beri islenen toplam frame sayisi |

## 4. Endpointler

### 4.1 Frame Analizi

#### `POST /api/analyze`

Tek bir frame'i analiz eder: yuz tespiti, ogrenci tanima, duygu tahmini ve dikkat skorlamasi yapar.

**Request (JSON):**

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

`image` alani base64 encoded goruntu verisi icerir. Opsiyonel `data:image/...;base64,` prefix'i desteklenir.

**Request (multipart/form-data):**

| Alan | Tip | Zorunlu | Aciklama |
|---|---|---|---|
| `image` | file | Evet | Goruntu dosyasi |

**Response 200:**

```json
{
  "results": [
    {
      "student_id": null,
      "name": null,
      "emotion": "neutral",
      "confidence": 0.6132,
      "attention_score": 0.31,
      "attention_level": "distracted",
      "bbox": [12, 50, 120, 175]
    }
  ],
  "timestamp": "2026-02-17T12:34:56.123456",
  "frame_count": 123
}
```

**Hata cevaplari:**

| HTTP | Kosul | Ornek mesaj |
|---|---|---|
| 400 | JSON body'de `image` alani yok | `"Missing 'image' field in JSON body."` |
| 400 | Multipart'ta `image` dosyasi yok | `"No image provided."` |
| 500 | Beklenmeyen sunucu hatasi | Hata detayi |

---

### 4.2 Ogrenci Endpointleri

#### `GET /api/students`

Kayitli tum ogrencileri listeler.

**Response 200:**

```json
{
  "students": [
    {
      "id": 1,
      "name": "Ali Veli",
      "email": "ali@example.com",
      "created_at": "2026-02-17T10:00:00"
    }
  ]
}
```

**Hata cevaplari:**

| HTTP | Kosul |
|---|---|
| 500 | Veritabani veya sunucu hatasi |

---

#### `POST /api/students/register`

Yuz goruntuleriyle yeni bir ogrenci kaydeder.

**Request:** `multipart/form-data`

| Alan | Tip | Zorunlu | Aciklama |
|---|---|---|---|
| `name` | string | Evet | Ogrenci adi |
| `email` | string | Hayir | Ogrenci e-posta adresi |
| `images` | file[] | Evet | En az 1 yuz goruntus dosyasi |

**Response 200:**

```json
{
  "student_id": 15,
  "name": "Yeni Ogrenci",
  "message": "Student registered with 4 embedding(s)."
}
```

**Hata cevaplari:**

| HTTP | Kosul | Ornek mesaj |
|---|---|---|
| 400 | `name` alani eksik | `"Missing 'name' field."` |
| 400 | Goruntu dosyasi gonderilmemis | `"No images provided."` |
| 400 | Hicbir goruntude yuz tespit edilemedi | `"No face could be detected in the provided images."` |
| 500 | Beklenmeyen sunucu hatasi | Hata detayi |

> **Not:** Yuz tespit edilemeyen gorseller sessizce atlanir. Hicbir goruntude yuz bulunamazsa ogrenci kaydedilir fakat embedding kaydedilmez ve 400 hatasi doner. Bu durumda response body icinde `student_id` alani da bulunur.

---

#### `GET /api/students/<student_id>/stats`

Belirtilen ogrencinin dikkat istatistiklerini doner.

**Query parametreleri:**

| Parametre | Tip | Zorunlu | Aciklama |
|---|---|---|---|
| `session_id` | int | Hayir | Belirli bir oturuma filtrelemek icin |

**Response 200:**

```json
{
  "student_id": 1,
  "avg_score": 0.58,
  "dominant_emotion": "neutral",
  "total_entries": 53,
  "attention_distribution": {
    "focused": 12,
    "moderate": 28,
    "distracted": 13
  }
}
```

**Hata cevaplari:**

| HTTP | Kosul | Ornek mesaj |
|---|---|---|
| 404 | Ogrenci bulunamadi | `"Student not found."` |
| 500 | Beklenmeyen sunucu hatasi | Hata detayi |

---

### 4.3 Oturum (Session) Endpointleri

#### `POST /api/sessions/start`

Yeni bir oturum olusturur ve aktif hale getirir. Aktif oturum varken yapilan `analyze` istekleri, taninan ogrencilerin dikkat verilerini veritabanina kaydeder.

**Request:** `application/json`

```json
{
  "name": "Matematik 101",
  "mode": "online"
}
```

| Alan | Tip | Zorunlu | Varsayilan | Aciklama |
|---|---|---|---|---|
| `name` | string | Evet | - | Oturum adi |
| `mode` | string | Hayir | `"online"` | `"online"` veya `"face-to-face"` |

**Response 200:**

```json
{
  "session_id": 9,
  "name": "Matematik 101",
  "mode": "online"
}
```

**Hata cevaplari:**

| HTTP | Kosul | Ornek mesaj |
|---|---|---|
| 400 | `name` alani eksik | `"Missing 'name' field."` |
| 400 | Gecersiz `mode` degeri | `"Invalid mode. Use 'online' or 'face-to-face'."` |
| 500 | Beklenmeyen sunucu hatasi | Hata detayi |

---

#### `POST /api/sessions/<session_id>/stop`

Aktif bir oturumu durdurur.

**Response 200:**

```json
{
  "session_id": 9,
  "message": "Session stopped."
}
```

**Hata cevaplari:**

| HTTP | Kosul |
|---|---|
| 500 | Beklenmeyen sunucu hatasi |

---

#### `GET /api/sessions`

Tum oturumlari listeler (en yeni ilk sirada).

**Response 200:**

```json
{
  "sessions": [
    {
      "id": 9,
      "name": "Matematik 101",
      "mode": "online",
      "started_at": "2026-02-17T10:00:00",
      "ended_at": null
    }
  ]
}
```

**Hata cevaplari:**

| HTTP | Kosul |
|---|---|
| 500 | Veritabani veya sunucu hatasi |

---

#### `GET /api/sessions/<session_id>/stats`

Belirtilen oturumun istatistiklerini doner. Her ogrenci icin ayri istatistik ve oturumun genel ozeti bulunur.

**Response 200:**

```json
{
  "session_id": 9,
  "students": {
    "1": {
      "avg_score": 0.65,
      "dominant_emotion": "positive",
      "total_entries": 37,
      "attention_distribution": {
        "focused": 20,
        "moderate": 12,
        "distracted": 5
      }
    }
  },
  "session": {
    "avg_score": 0.65,
    "total_students": 1
  }
}
```

**Hata cevaplari:**

| HTTP | Kosul |
|---|---|
| 500 | Beklenmeyen sunucu hatasi |

---

### 4.4 Raporlama

#### `GET /api/export/excel`

Dikkat verilerini Excel dosyasi olarak indirir.

**Query parametreleri:**

| Parametre | Tip | Zorunlu | Aciklama |
|---|---|---|---|
| `session_id` | int | Hayir | Belirli bir oturumun verilerini export etmek icin |

**Response 200:**

- Content-Type: `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`
- Dosya adi: `attention_report.xlsx` veya `attention_report_session_{id}.xlsx`

**Hata cevaplari:**

| HTTP | Kosul |
|---|---|
| 500 | Excel olusturma veya sunucu hatasi |

---

### 4.5 Dashboard

#### `GET /dashboard`

Dashboard HTML sayfasini doner. Tarayicida acilmak uzere tasarlanmistir.

**Response 200:** HTML sayfasi.

#### `GET /`

Root path, `/dashboard` adresine yonlendirir (302 redirect).

## 5. Deger Kisitlamalari

| Alan | Kisit |
|---|---|
| `attention_score` | `[0, 1]` araliginda float |
| `attention_level` | `"focused"`, `"moderate"` veya `"distracted"` |
| `mode` | `"online"` veya `"face-to-face"` |
| `emotion` | `"positive"`, `"negative"` veya `"neutral"` |
| `bbox` | `[x1, y1, x2, y2]` formatinda integer dizi, `x2 > x1` ve `y2 > y1` |
| `confidence` | `[0, 1]` araliginda float, 4 ondalik basamaga yuvarlanir |
