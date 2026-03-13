# Kurulum ve Calistirma Kilavuzu

## 1. Ortam Gereksinimleri

- **Python**: 3.10 veya uzeri
- **Bagimliliklar**:
  - `requirements-server.txt` — sunucu (inference) icin
  - `requirements-train.txt` — egitim icin (Google Colab + GPU onerilir)
- **Donanim**: CPU zorunlu, GPU opsiyonel (egitim haricinde gerekli degil)
- **Onemli dosya yollari**:
  - Model: `models/emotion_model.onnx`
  - Veritabani: `data/students.db`

## 2. Kurulum Adimlari

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-server.txt
```

Egitim ortami icin (Colab uzerinde):

```bash
pip install -r requirements-train.txt
```

## 3. Calistirma Komutlari

Sunucuyu baslatmak icin:

```bash
python -m src.api.run --model-path models/emotion_model.onnx --host 0.0.0.0 --port 5000
```

Baslatma sonrasi erisim adresleri:

- Dashboard: `http://localhost:5000/dashboard`
- API: `http://localhost:5000/api/students` (test icin)

Testleri calistirmak icin:

```bash
pytest tests/ -v
```

Lint kontrolu:

```bash
ruff check src/ tests/
```

## 4. Hizli Saglik Kontrolu

Sunucu basladiktan sonra asagidaki kontrol listesini gozden gecirin:

| Kontrol | Nasil dogrulanir |
| --- | --- |
| Sunucu calisiyor mu? | `http://localhost:5000/dashboard` aciliyor |
| Analyze endpoint calisiyor mu? | Bir test frame gondererek `200` yaniti alinmali |
| Veritabani erisimi var mi? | `GET /api/students` bos liste donmeli (`{"students": []}`) |

Hizli kontrol icin:

```bash
curl -s http://localhost:5000/api/students
```
