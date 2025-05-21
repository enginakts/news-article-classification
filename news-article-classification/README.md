# Haber Kategorisi Tahmin Sistemi

Bu proje, Türkçe haber metinlerini kategorize etmek için makine öğrenmesi modelleri kullanan bir web uygulamasıdır. İki farklı model (Naive Bayes ve SVM) kullanılarak haber metinlerinin kategorilerini tahmin eder.

## Veri Seti

Projede kullanılan veri seti Hugging Face'den alınmıştır:
[Interpress News Category TR Lite Dataset](https://huggingface.co/datasets/yavuzkomecoglu/interpress_news_category_tr_lite)

Veri setini indirmek için:

```python
from datasets import load_dataset

# Veri setini indir
dataset = load_dataset("yavuzkomecoglu/interpress_news_category_tr_lite")

# Veri setini TSV formatında kaydet
import pandas as pd

# Eğitim setini al
train_df = pd.DataFrame(dataset['train'])
# Test setini al
test_df = pd.DataFrame(dataset['test'])

# TSV formatında kaydet
train_df.to_csv('datasets/dataset1.tsv', sep='\t', index=False)
test_df.to_csv('datasets/dataset2.tsv', sep='\t', index=False)
```

## Özellikler

- Türkçe haber metinlerini 10 farklı kategoriye sınıflandırma
- İki farklı model (Naive Bayes ve SVM) ile tahmin yapma
- Kullanıcı dostu web arayüzü
- Gerçek zamanlı tahmin sonuçları
- Metin ön işleme ve temizleme

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. NLTK gerekli dosyalarını indirin:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

3. Uygulamayı çalıştırın:
```bash
python app.py
```

4. Web tarayıcınızda `http://localhost:5000` adresine gidin.

## Proje Yapısı

```
├── app.py                  # Flask web uygulaması
├── deneme3.py             # Model eğitim kodları
├── models/                # Eğitilmiş modeller
│   ├── naive_bayes_model.pkl
│   ├── svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── kategoriler.pkl
├── templates/             # HTML şablonları
│   └── index.html
├── requirements.txt       # Gerekli Python paketleri
└── README.md             # Bu dosya
```

## Kullanım

1. Web arayüzünde metin giriş alanına haber metnini yapıştırın
2. "Tahmin Et" butonuna tıklayın
3. Her iki modelin tahmin sonuçlarını görüntüleyin

## Modeller

- **Naive Bayes**: Metin sınıflandırma için kullanılan olasılıksal bir model
- **SVM (Support Vector Machine)**: Yüksek boyutlu veri uzayında sınıflandırma yapan bir model

## Teknolojiler

- Python 3.x
- Flask
- scikit-learn
- NLTK
- Bootstrap 5
- jQuery
- Hugging Face Datasets 