# ADR-002: Attention Scoring and Threshold Policy

## Durum
Accepted

## Tarih
2026-02-17

## Baglam
Sistem davranisi tek bir sayisal skora indirgenmeli ve UI/raporda tutarli sekilde siniflandirilmalidir.

Mevcut kod:
1. Emotion agirliklari (`positive=0.8`, `neutral=0.5`, `negative=0.2`)
2. Esikler:
- focused > 0.6
- moderate > 0.35
- aksi distracted

## Karar
Skor politikasi asagidaki sekilde standardize edildi:
1. Emotion-only score:
- `sum(weight[class] * confidence) / n`
2. Hybrid score:
- `0.6 * emotion_score + 0.4 * pose_score`
3. Level thresholds:
- focused if `score > 0.6`
- moderate if `score > 0.35`
- distracted otherwise
4. Anomaly flag:
- son 5 frame icindeki ardisik skor farki > 0.4 ise `anomaly_flag=true`

## Sonuclar
Olumlu:
1. Deterministik ve izlenebilir kural seti.
2. Contract test yazimi kolaylasir.

Olumsuz:
1. Sabit esikler domain degisiminde hassas olabilir.
2. Kalibrasyon ihtiyaci dogabilir.

## Uygulama Notlari
1. Score [0,1] araliginda clamp edilir.
2. Tum response ve loglarda ayni level kurali kullanilir.
3. FR eslesmesi: FR-007, FR-008, FR-017.

