# ADR-003: Privacy, Retention, and Deletion Policy

## Durum
Accepted

## Tarih
2026-02-17

## Baglam
Sistem ogrenci kimlik bilgisi ve dikkat logu gibi hassas veri isler. Akademik standarda uygun veri yasam dongusu zorunludur.

## Karar
1. Veri minimizasyonu:
- Ham frame kalici saklanmaz.
2. Consent:
- `students.consent_at` alani tutulur.
3. Soft delete:
- `students.is_active=0`, `deleted_at` set edilir.
4. Retention:
- attention logs varsayilan 180 gun.
5. Purge:
- gunluk otomatik job + manuel endpoint.
6. Embedding silme:
- soft delete sonrasi 7 gun icinde purge.
7. Audit:
- purge islemleri actor/time/count bilgisiyle loglanir.

## Sonuclar
Olumlu:
1. Veri saklama suresi denetlenebilir hale gelir.
2. Silme talepleri operasyonel olarak uygulanabilir olur.

Olumsuz:
1. Purge operasyonu ek operasyonel yuk getirir.
2. Audit ihlallerinde ek izleme ihtiyaci dogar.

## Uygulama Notlari
1. Purge endpoint dry-run destekler.
2. Retention policy degisikligi config uzerinden yonetilir.
3. FR eslesmesi: FR-015, FR-016.

