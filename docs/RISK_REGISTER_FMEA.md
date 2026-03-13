# Risk Analizi

| # | Risk | Aciklama | Azaltim |
|---|------|----------|---------|
| 1 | Model performansi yetersiz kalabilir | FER2013 uzerinde egitilen model hedef Macro F1 >= 0.75'i tutturamayabilir | Veri augmentasyonu, farkli mimariler deneme (EfficientNet-B0, MobileNetV3), class weights ayarlama |
| 2 | Domain shift (FER2013 vs gercek sinif ortami) | Egitim verisi (FER2013) ile gercek sinif ortami arasinda aydinlatma, kamera acisi, yuz ifadesi farkliliklari olabilir | Gercek sinif ortaminda test oturumlari yaparak modelin gercek performansini olcme |
| 3 | Kamera kalitesi ve acisi etkisi | Dusuk kalite veya uygunsuz aci yuz tespitini ve duygu tahminini olumsuz etkileyebilir | Farkli kosullarda (aydinlatma, mesafe, aci) test yapma, minimum gereksinimleri belirleme |
| 4 | Zaman yetersizligi | Nisan 2026 deadline'ina kadar tum ozellikleri tamamlayamama riski | MVP'ye odaklanma (9 temel gereksinim), bonus hedefleri (face-to-face, anomali) erteleme |
| 5 | Head pose tahmini kararsizligi | MediaPipe ile head pose tahmini tutarsiz veya gurultulu olabilir | Face-to-face modu bonus hedef olarak birakma, oncelikle emotion-only skora odaklanma |
