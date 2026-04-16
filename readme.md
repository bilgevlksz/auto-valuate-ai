# ⚡ Auto Valuate AI+ | Multi-Modal Araç Değerleme Motoru

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Vision%20Transformer-orange.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-Regression-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B.svg)

Auto Valuate AI+, ikinci el araçların piyasa değerini tahmin etmek için sadece geleneksel tablosal verileri (Kilometre, Marka, Yıl vb.) değil, aynı zamanda **aracın fotoğraflarını da analiz eden** uçtan uca (end-to-end) bir yapay zeka ve karar destek sistemidir. 

Sıradan değerleme algoritmalarının aksine, fotoğraflardaki hasar, kozmetik durum ve donanım paketlerini anlamlandırarak fiyat tahminindeki yanılma payını minimize eder.

---

## 🧠 Hibrid Yapay Zeka Mimarisi (Multi-Modal Fusion)

Proje, birbirine entegre çalışan iki farklı yapay zeka beyninden oluşur:

1. **Görsel Algı (Vision Transformer - ViT):** - Araç fotoğrafları, Google tarafından geliştirilen modern ViT (ViT_B_16) mimarisinden geçirilir. 
   - Model, fotoğraflardan 768 boyutlu (768-D) "Feature Extraction" (öznitelik vektörü) çıkararak aracın görsel kondisyonunu matematiksel bir veriye dönüştürür.

2. **Fiyatlama Motoru (CatBoost Regressor):** - ViT'ten gelen görsel vektörler ve kullanıcının girdiği teknik parametreler (Marka, KM, Kasa Tipi) tek bir potada eritilir (Multi-modal Data Fusion).
   - Bu karmaşık veri seti, kategorik verilerdeki üstün başarısıyla bilinen CatBoost algoritması ile işlenerek nihai fiyat hesaplanır.
   - *Not: Fiyat verilerindeki çarpıklığı (skewness) gidermek için eğitim aşamasında Log-Transform uygulanmış, sonuç gösterilirken Exponentiyel dönüşümle (np.expm1) gerçek TL değerine çevrilmiştir.*

---

## 🚀 Öne Çıkan Özellikler

* **📷 Çoklu Fotoğraf Analizi:** Aynı anda 10 farklı araç fotoğrafını işleyerek ortalama bir kozmetik skoru çıkarır.
* **📊 Plotly Karar Destek Göstergesi:** Çıkan sonucu sadece bir rakam olarak değil; dinamik olarak hesaplanan Ucuz/Normal/Pahalı piyasa koridoru (Gauge Chart) içinde gösterir.
* **🎨 Modern SaaS Arayüzü:** Standart veri bilimi panellerinden farklı olarak; Glassmorphism (Buzlu Cam) efektleri, karanlık tema (Dark Mode) ve neon detaylara sahip, ticarileşmeye hazır (MVP) bir web arayüzü sunar.
* **⚡ Yüksek Hız (Edge Inference):** ViT ağırlıkları lokalde önbelleğe alınarak (caching) tahmin süresi saniyeler seviyesine indirilmiştir.

---

## 💻 Kurulum ve Çalıştırma (Localhost)

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

**1. Depoyu Klonlayın:**
```bash
git clone [https://github.com/kullaniciadiniz/auto-valuate-ai.git](https://github.com/kullaniciadiniz/auto-valuate-ai.git)
cd auto-valuate-ai