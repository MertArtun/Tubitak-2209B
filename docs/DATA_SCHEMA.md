# Veritabani Semasi

## 1. Amac

Bu dokuman, Student Attention Detection System projesinin SQLite veritabani semasini tanimlar. Tum tablo ve alan tanimlari `src/face_recognition/database.py` dosyasindaki `_init_db()` metoduyla birebir eslesir.

## 2. Tablolar

### 2.1 `students`

Ogrenci kayitlarini tutar.

| Alan | Tip | Kisitlama | Aciklama |
|---|---|---|---|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Ogrenci kimligi |
| name | TEXT | NOT NULL | Ogrenci adi |
| email | TEXT | | Email adresi |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Kayit zamani |

### 2.2 `face_embeddings`

ArcFace modeli tarafindan uretilen yuz embedding vektorlerini saklar. Her ogrencinin birden fazla embedding kaydi olabilir; tanima sirasinda ortalamalari alinarak normalize edilir (`get_all_embeddings`).

| Alan | Tip | Kisitlama | Aciklama |
|---|---|---|---|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Kayit kimligi |
| student_id | INTEGER | NOT NULL, FK -> students(id) | Ogrenci iliskisi |
| embedding | BLOB | NOT NULL | 512 boyutlu float32 vektor (2048 byte) |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Kayit zamani |

### 2.3 `sessions`

Ders oturumlarini temsil eder. Bir session `online` veya `face-to-face` modunda olabilir.

| Alan | Tip | Kisitlama | Aciklama |
|---|---|---|---|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Session kimligi |
| name | TEXT | | Session adi |
| mode | TEXT | CHECK(mode IN ('online', 'face-to-face')) | Ders modu |
| start_time | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Baslangic zamani |
| end_time | TIMESTAMP | | Bitis zamani (`end_session` ile set edilir) |

### 2.4 `attention_logs`

Her analiz frame'i icin uretilen dikkat kaydini tutar.

| Alan | Tip | Kisitlama | Aciklama |
|---|---|---|---|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Log kimligi |
| student_id | INTEGER | NOT NULL, FK -> students(id) | Ogrenci |
| session_id | INTEGER | NOT NULL, FK -> sessions(id) | Session |
| timestamp | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Kayit zamani |
| emotion | TEXT | | Tespit edilen duygu (Positive/Negative/Neutral) |
| confidence | REAL | | Duygu tahmini confidence degeri |
| attention_score | REAL | | Dikkat skoru [0, 1] araliginda |
| attention_level | TEXT | | focused / moderate / distracted |

## 3. Iliskiler

```
students (1) ──── (*) face_embeddings
    |
    └── (1) ──── (*) attention_logs (*) ──── (1) sessions
```

- Bir **ogrencinin** birden fazla **face_embeddings** kaydi olabilir (farkli acilarda cekilmis yuz verileri).
- Bir **ogrencinin** birden fazla **attention_logs** kaydi olabilir.
- Bir **session** birden fazla **attention_logs** kaydina sahip olabilir.
- Foreign key iliskileri `REFERENCES` ile tanimlidir ve `PRAGMA foreign_keys = ON` ile aktif edilir.

## 4. Veri Erisim Katmani

Tum veritabani islemleri `src/face_recognition/database.py` dosyasindaki `StudentDatabase` sinifi uzerinden yapilir. Dogrudan SQL sorgusu calistirmak yerine bu sinifin metodlari kullanilir.

Temel metodlar:

| Metod | Islem |
|---|---|
| `add_student(name, email)` | Yeni ogrenci ekler |
| `get_student(student_id)` | Ogrenci bilgisini getirir |
| `list_students()` | Tum ogrencileri listeler |
| `save_embedding(student_id, embedding)` | Yuz embedding kaydeder |
| `get_embeddings(student_id)` | Bir ogrencinin embedding listesini doner |
| `get_all_embeddings()` | Tum ogrenciler icin normalize ortalama embedding doner |
| `create_session(name, mode)` | Yeni session olusturur |
| `end_session(session_id)` | Session bitis zamanini set eder |
| `log_attention(...)` | Dikkat logu ekler |
| `get_student_stats(student_id, session_id)` | Ogrenci dikkat istatistiklerini hesaplar |
| `get_session_stats(session_id)` | Session geneli istatistik doner |
| `export_to_excel(filepath, session_id)` | Verileri Excel dosyasina aktarir |

Sinif context manager olarak da kullanilabilir:

```python
with StudentDatabase() as db:
    db.add_student("Ali Veli", "ali@example.com")
```

## 5. Notlar

- **Veritabani dosyasi:** `data/students.db` (yol `configs/config.py` icindeki `DB_PATH` ile belirlenir).
- **Foreign key zorlama:** Baglanti acildiginda `PRAGMA foreign_keys = ON` calistirilir. Bu olmadan SQLite foreign key kisitlamalarini denetlemez.
- **Embedding boyutu:** ArcFace 512 boyutlu float32 vektor uretir. BLOB olarak saklanir (`512 * 4 = 2048 byte`).
- **Row factory:** Baglanti `sqlite3.Row` row factory kullanir; sorgular dict-benzeri erisim saglar.
