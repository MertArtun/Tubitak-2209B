# ADR-001: Mode-Aware Architecture (online vs face-to-face)

## Durum
Accepted

## Tarih
2026-02-17

## Baglam
Sistem iki ayri kullanim ortamina hizmet eder:
1. Online ders (kamera karsisinda ekran odakli),
2. Sinif ici yuz yuze ders (bakis yonu daha kritik).

Tek skorlama mantigi her iki ortamda ayni performansi vermemektedir.

## Karar
Pipeline mode-aware calisacak:
1. `online`:
- attention score sadece emotion tabanli hesaplanacak.
2. `face-to-face`:
- emotion + head pose hybrid skor kullanilacak.

API kontrati:
1. Analyze request `mode` kabul edecek.
2. Analyze response `mode` ve pose alanlarini donecek.

## Sonuclar
Olumlu:
1. Ortama uygun sinyal kullanimi.
2. Face-to-face senaryoda daha aciklanabilir skor.

Olumsuz:
1. Kod ve test kompleksitesi artar.
2. Mode bazli regresyon riski olusur.

## Uygulama Notlari
1. Varsayilan mode `online`.
2. Gecersiz mode 400 `INVALID_MODE`.
3. FR eslesmesi: FR-008.

