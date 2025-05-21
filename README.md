# 📊 Turkish News Classification using NLP

This project applies Natural Language Processing (NLP) techniques to classify Turkish news articles into different categories using machine learning algorithms.

## 🔍 Project Highlights
- Language: Turkish 🇹🇷
- Preprocessing steps:
  - Lowercasing, punctuation and special character removal
  - Stopword removal using NLTK's Turkish corpus
  - Word stemming using TurkishStemmer
- Feature extraction with TF-IDF (including bigrams)
- Classification using:
  - Multinomial Naive Bayes
  - Linear Support Vector Machine (SVM)
- 5-Fold Cross Validation to evaluate model performance
- Evaluation metrics: Accuracy, Classification Report, Confusion Matrix
- Dataset: TSV formatted Turkish news data

## 🚀 Technologies
- Python
- Scikit-learn
- NLTK
- TurkishStemmer
- Matplotlib & Seaborn (for visualization)

## 📁 Dataset
The dataset (`dataset1.tsv`) contains Turkish news articles with their corresponding labels. After preprocessing, it's saved as `processed_dataset2.csv`.

## 📊 Results
The SVM model outperformed Naive Bayes in accuracy. See the confusion matrices and classification reports in the notebook for detailed insights.

## 📂 How to Run
1. Install requirements
2. Run the Jupyter notebook or script
3. Analyze the results and visualizations

---

## 🗣️ Türkçe Açıklama

Bu proje, Türkçe haber verilerini doğal dil işleme (NLP) yöntemleriyle analiz ederek kategorilere ayırmayı amaçlamaktadır.  
Veri temizleme, kök indirgeme, TF-IDF vektörleme ve makine öğrenmesi modelleri ile başarılı sonuçlar elde edilmiştir.

Model karşılaştırmaları, çapraz doğrulama (cross-validation) ve başarı oranları görselleştirme ile desteklenmiştir.

---

✅ Bu proje Türkçe NLP üzerine çalışanlar için sağlam bir temel sağlayabilir.

