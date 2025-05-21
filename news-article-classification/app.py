from flask import Flask, render_template, request, jsonify
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# NLTK gerekli dosyaları indirme
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Modelleri ve vectorizer'ı yükle
def load_models():
    with open('models/naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    with open('models/svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('models/kategoriler.pkl', 'rb') as f:
        kategoriler = pickle.load(f)
    return nb_model, svm_model, tfidf_vectorizer, kategoriler

# Metin ön işleme fonksiyonu
def preprocess_text(text):
    # Küçük harfe çevirme
    text = text.lower()
    # Noktalama işaretlerini kaldırma
    text = re.sub(r'[^\w\s]', '', text)
    # Stopwords'leri kaldırma
    stop_words = set(stopwords.words('turkish'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

# Modelleri yükle
nb_model, svm_model, tfidf_vectorizer, kategoriler = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Metni ön işleme
        processed_text = preprocess_text(text)
        
        # TF-IDF dönüşümü
        text_tfidf = tfidf_vectorizer.transform([processed_text])
        
        # Tahminler
        nb_pred = nb_model.predict(text_tfidf)[0]
        svm_pred = svm_model.predict(text_tfidf)[0]
        
        # Kategori isimlerini al
        nb_category = kategoriler[nb_pred]
        svm_category = kategoriler[svm_pred]
        
        return jsonify({
            'nb_prediction': nb_category,
            'svm_prediction': svm_category
        })

if __name__ == '__main__':
    app.run(debug=True)