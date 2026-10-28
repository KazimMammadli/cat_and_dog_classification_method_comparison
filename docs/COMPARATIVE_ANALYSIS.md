# Pişiklər və İtlər Şəkil Təsnifatı: Müqayisəli Təhlil Hesabatı

## 1. İcra xülasəsi
Bu hesabat kiçik bir məlumat toplusunda (~1000 şəkil) Pişiklər və İtlər şəkil təsnifatı tapşırığı üçün iki fərqli dərin öyrənmə yanaşmasının müqayisəli təhlilini təqdim edir:
1. **Xüsusi CNN (sıfırdan)**: Tamamilə sıfırdan qurulmuş və öyrədilmiş Konvolyusiya Neyron Şəbəkəsi (CNN).
2. **Transfer Learning (MobileNetV2)**: Bu spesifik ikili təsnifat tapşırığı üçün uyğunlaşdırılmış və incə tənzimlənmiş (fine-tuned) əvvəlcədən öyrədilmiş MobileNetV2 arxitekturası.

Təhlil açıq şəkildə göstərir ki, Transfer Learning yanaşması bütün göstəricilər üzrə xüsusi CNN-i böyük fərqlə üstələyir. O, xüsusi CNN-in **54.67%** nəticəsi ilə müqayisədə **96.67%** mükəmməl test dəqiqliyi əldə edir və həmçinin daha az təlim epoxası (epoch) tələb edir.

---

## 2. Metodologiya və Arxitektura

### Yanaşma A: Xüsusi CNN (Sıfırdan)
- **Arxitektura**: Conv2D, BatchNormalization, MaxPooling2D və tam əlaqəli (dense) təbəqələrdən istifadə edərək qurulmuş ardıcıl CNN.
- **Müntəzəmləşdirmə (Regularization)**: Həddindən artıq öyrənmənin (overfitting) qarşısını almaq üçün Dropout (0.3-dən 0.5-ə qədər) və L2 regularization istifadə edilmişdir. Məlumat toplusunun (dataset) məhdud ölçüsünə görə ağır şəkil artırması (image augmentation - RandomFlip, RandomRotation, RandomZoom və s.) tətbiq olunmuşdur.
- **Təlim Konfiqurasiyası**: Adam optimizatoru və İkili Çarpaz Entropiya (Binary Crossentropy) itkisi istifadə edilərək 60 epoxa ərzində öyrədilib. Callbacks olaraq EarlyStopping, ModelCheckpoint və ReduceLROnPlateau daxil edilmişdir.

### Yanaşma B: Transfer Learning (MobileNetV2)
- **Arxitektura**: Əvvəlcədən öyrədilmiş `MobileNetV2` əsas xüsusiyyət çıxarıcısı (feature extractor) kimi istifadə olunur (çəkilər dondurulub) və son ikili təsnifat üçün GlobalAveragePooling2D təbəqəsi, Dropout və Dense təbəqələri ilə birləşdirilmişdir.
- **İlkin emal (Preprocessing)**: Giriş şəkilləri incə tənzimləmə (fine-tuning) üçün hazırlanmış data artırma boru kəməri ilə yanaşı, MobileNetV2 üçün xüsusi olan Keras-ın daxili `preprocess_input` funksiyası istifadə edilərək ilkin emal edilmişdir.
- **Təlim Konfiqurasiyası**: Maksimum 15 epoxa ərzində öyrədilib. Callbacks olaraq ModelCheckpoint, ReduceLROnPlateau və EarlyStopping daxil edilmişdir.

---

## 3. Performans Göstəriciləri və Nəticələr

| Metrika | Xüsusi CNN (Sıfırdan) | MobileNetV2 (Transfer Learning) |
| :--- | :--- | :--- |
| **Ən Yaxşı Epoxa** | 59 | 4 (6-cı epoxada dayandırıldı) |
| **Validasiya Dəqiqliyi** | ~53.33% | ~99.33% |
| **Test Dəqiqliyi** | **54.67%** | **96.67%** |
| **Test İtkisi** | 0.7211 | 0.0993 |
| **Makro F1-Score** | 0.44 | 0.97 |

### Ətraflı Müşahidələr

#### Xüsusi CNN Nəticələri:
- Sıfırdan qurulmuş model mənalı xüsusiyyətləri öyrənməkdə çətinlik çəkdi və təsadüfi təxminlərdən (54.67%) çətinliklə daha yaxşı nəticə göstərdi.
- **Qarışıqlıq Matrisi (Confusion Matrix) və Təsnifat Hesabatı**: Model "pişiklər"i təxmin etməyə doğru ciddi bir qərəz (bias) inkişaf etdirdi. İtlər üçün dəqiqlik (precision) yüksək (0.89) olsa da, geri çağırma (recall) çox pis idi (0.11), yəni demək olar ki, bütün it şəkillərini qaçırdı. Əksinə olaraq, pişiklər üçün geri çağırma (recall) 0.99 idi, lakin dəqiqlik (precision) aşağı idi (0.52).
- **Öyrənmə Əyrisi (Learning Curve)**: Təlim tarixçəsi həm təlim, həm də validasiya dəqiqliyinin çox erkən, təxminən 50% civarında dayandığını (plateau) göstərdi ki, bu da modelin əvvəlcədən bilik olmadan mürəkkəb fərqləndirici xüsusiyyətləri öyrənmək üçün kifayət qədər tutumu olmadığını (və ya məlumat toplusunun kifayət qədər nümunədən məhrum olduğunu) göstərir.

#### MobileNetV2 Nəticələri:
- Transfer learning modeli, ilk bir neçə epoxada 98%-dən çox validasiya dəqiqliyinə çatmaqla, demək olar ki, dərhal görkəmli performans əldə etdi.
- **Qarışıqlıq Matrisi və Təsnifat Hesabatı**: Model inanılmaz dərəcədə balanslaşdırılmış və yüksək performans nümayiş etdirdi. Həm pişiklər, həm də itlər üçün dəqiqlik (precision) və geri çağırma (recall) 0.96 və 0.97 arasında idi.
- **Öyrənmə Əyrisi**: Model sürətlə cəmləşdi (converged) və 6-cı epoxada EarlyStopping işə düşdü. Beləliklə test dəstində müstəsna ümumiləşdirmə (generalization) təmin edilərkən lazımsız hesablamaların qarşısı alındı.

---

## 4. Nəticə və Tövsiyələr
Nəticələr fundamental bir dərin öyrənmə (deep learning) paradiqmasını vurğulayır: **çox kiçik məlumat toplularında (dataset) dərin CNN-ləri sıfırdan öyrətmək son dərəcə çətindir** və ağır artırma (augmentation) və müntəzəmləşdirməyə (regularization) baxmayaraq zəif ümumiləşdirməyə və ya az öyrənməyə (underfitting) / həddindən artıq öyrənməyə (overfitting) meyllidir.

**Transfer Learning**-dən istifadə edərək, MobileNetV2 modeli nəhəng ImageNet məlumat toplusundan öyrənilmiş zəngin, ümumiləşdirilmiş xüsusiyyətlərdən istifadə edə bildi. Bu isə aşağıdakılarla nəticələndi:
- **Əhəmiyyətli dərəcədə daha yüksək dəqiqlik** (54.67%-ə qarşı 96.67%).
- **Daha sürətli cəmləşmə (convergence)** (optimal çəkilər 60 epoxaya deyil, 4 epoxada tapıldı).
- Təlim mərhələsində **daha az hesablama xərci**.

**Tövsiyə**: Kiçik və orta ölçülü məlumat toplularında şəkil təsnifatı tapşırıqları üçün MobileNetV2 kimi səmərəli arxitekturalardan istifadə edən Transfer Learning defolt (standart) yanaşma olmalıdır. Xüsusi CNN yanaşması yalnız nəhəng məlumat toplusu mövcud olduqda və ya şəkil domeni (domain) standart əvvəlcədən öyrədilmə məlumat toplularından köklü şəkildə fərqləndikdə (məsələn, spesifik tibbi təsvirlər və ya peyk təsvirləri) nəzərdən keçirilməlidir.
