# =============================================================================
# chatbot_logic.py — Modul Logika NLP Chatbot MedicBot
# =============================================================================
# Modul ini menangani seluruh pipeline pemrosesan bahasa alami (NLP)
# untuk chatbot MedicBot, mulai dari preprocessing teks hingga menghasilkan
# respons yang sesuai berdasarkan intent pengguna.
#
# Pipeline:
#   Input teks user
#       → remove_punctuation()   : normalisasi & lowercase
#       → vectorization()        : tokenisasi → sequence → padding
#       → predict()              : klasifikasi intent dengan model Keras
#       → generate_response()    : pilih respons acak dari tag intent
# =============================================================================

import json       # Parsing file dataset respons (himedic.json)
import random     # Memilih respons secara acak dari list respons per intent
import nltk       # Natural Language Toolkit: lemmatisasi, tokenisasi
import string     # Konstanta karakter tanda baca untuk pembersihan teks
import numpy as np  # Operasi array numerik (reshape vector)
import pickle     # Deserialize objek tokenizer dan label encoder dari file .pkl
import tensorflow # Backend deep learning (digunakan Keras di atasnya)
import sklearn    # Scikit-learn (diimpor untuk kompatibilitas, tidak dipakai langsung di sini)
from nltk.stem import WordNetLemmatizer  # Lemmatizer: mengubah kata ke bentuk dasar
import keras      # High-level API deep learning untuk load model dan pad_sequences

# Deklarasi variabel global yang digunakan lintas fungsi dalam modul ini
global responses, lemmatizer, tokenizer, le, model, input_shape

# Panjang sequence input yang diharapkan model (jumlah token per kalimat setelah padding)
input_shape = 9


# =============================================================================
# FUNGSI: load_response
# =============================================================================
def load_response():
    """
    Membaca file dataset JSON yang berisi daftar intent beserta respons.
    
    Struktur himedic.json:
        {
          "intents": [
            {
              "tag": "nama_intent",       ← label/kelas intent
              "patterns": ["..."],        ← contoh kalimat pengguna
              "responses": ["..."]        ← daftar kemungkinan jawaban
            },
            ...
          ]
        }
    
    Hasil: Dict `responses` dengan format { "tag": ["respons1", "respons2", ...] }
    """
    global responses
    responses = {}

    # Buka dan parse file JSON dataset
    with open('dataset/himedic.json') as content:
        data = json.load(content)

    # Buat dictionary: tag → list respons
    for intent in data['intents']:
        responses[intent['tag']] = intent['responses']


# =============================================================================
# FUNGSI: preparation
# =============================================================================
def preparation():
    """
    Fungsi inisialisasi yang dipanggil SATU KALI saat server Flask startup.
    
    Memuat semua komponen model ke memori:
      - tokenizer   : Keras Tokenizer yang sudah difit ke data training
                      (mengubah kata → integer index)
      - le          : Scikit-learn LabelEncoder
                      (mengubah label string intent → integer, dan sebaliknya)
      - model       : Model Keras Sequential (arsitektur Neural Network)
                      yang sudah dilatih untuk klasifikasi intent
      - lemmatizer  : NLTK WordNetLemmatizer untuk normalisasi kata
    
    Juga mendownload resource NLTK yang diperlukan:
      - punkt    : Tokenizer kalimat/kata
      - wordnet  : Database leksikal untuk lemmatisasi
      - omw-1.4  : Open Multilingual Wordnet (ekstensi wordnet)
    """
    load_response()

    global lemmatizer, tokenizer, le, model

    # Load tokenizer yang disimpan saat training model
    tokenizer = pickle.load(open('model/tokenizers.pkl', 'rb'))

    # Load label encoder untuk mapping integer → nama intent
    le = pickle.load(open('model/labelencoder.pkl', 'rb'))

    # Load model neural network yang sudah dilatih (.h5 = format HDF5 Keras)
    model = keras.models.load_model('model/model.h5')

    # Inisialisasi lemmatizer untuk preprocessing teks
    lemmatizer = WordNetLemmatizer()

    # Download resource NLTK (quiet=True agar tidak mencetak output)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


# =============================================================================
# FUNGSI: remove_punctuation
# =============================================================================
def remove_punctuation(text):
    """
    Membersihkan teks input dari tanda baca dan mengubahnya ke huruf kecil.
    
    Langkah:
      1. Iterasi setiap karakter dalam teks
      2. Hanya pertahankan karakter yang BUKAN tanda baca (string.punctuation)
      3. Ubah ke lowercase dengan .lower()
      4. Gabungkan kembali menjadi string
    
    Args:
        text (str): Kalimat input dari pengguna
    
    Returns:
        list[str]: List berisi satu string yang sudah dibersihkan
                   (format list diperlukan oleh tokenizer.texts_to_sequences)
    
    Contoh:
        "Apa itu Kolesterol?!" → ["apa itu kolesterol"]
    """
    texts_p = []
    text = [letters.lower() for letters in text if letters not in string.punctuation]
    text = ''.join(text)
    texts_p.append(text)
    return texts_p


# =============================================================================
# FUNGSI: vectorization
# =============================================================================
def vectorization(texts_p):
    """
    Mengubah teks yang sudah dibersihkan menjadi vektor numerik berpadding.
    
    Pipeline:
      1. texts_to_sequences : Ubah kata → integer berdasarkan vocabulary tokenizer
      2. reshape(-1)        : Ratakan array agar berbentuk 1D
      3. pad_sequences      : Pad/truncate ke panjang input_shape=9
                              (agar dimensi input konsisten dengan model)
    
    Args:
        texts_p (list[str]): Output dari remove_punctuation()
    
    Returns:
        np.ndarray: Array 2D shape (1, input_shape) siap menjadi input model
    
    Contoh:
        ["apa itu kolesterol"] → [[3, 7, 12, 0, 0, 0, 0, 0, 0]]  (padded ke 9)
    """
    # Konversi teks ke sequence integer menggunakan vocabulary tokenizer
    vector = tokenizer.texts_to_sequences(texts_p)

    # Ratakan array menjadi 1D
    vector = np.array(vector).reshape(-1)

    # Pad sequence ke panjang yang diharapkan model (input_shape=9)
    vector = keras.utils.pad_sequences([vector], input_shape)

    return vector


# =============================================================================
# FUNGSI: predict
# =============================================================================
def predict(vector):
    """
    Menjalankan inferensi model untuk mengklasifikasikan intent pengguna.
    
    Langkah:
      1. model.predict(vector) → output probabilitas untuk setiap kelas intent
      2. argmax()              → ambil index kelas dengan probabilitas tertinggi
      3. le.inverse_transform  → konversi index → nama tag intent (string)
    
    Args:
        vector (np.ndarray): Vektor input shape (1, 9) dari vectorization()
    
    Returns:
        str: Nama tag intent yang diprediksi (contoh: "apa_itu_kolesterol")
    
    Contoh output model.predict(): [[0.02, 0.87, 0.01, ...]]
    Setelah argmax: 1 → inverse_transform → "apa_itu_kolesterol"
    """
    # Jalankan forward pass model; output = array probabilitas per intent
    output = model.predict(vector)

    # Ambil index kelas dengan probabilitas tertinggi (argmax = argumen maksimum)
    output = output.argmax()

    # Decode integer index → nama tag intent menggunakan label encoder
    response_tag = le.inverse_transform([output])[0]

    return response_tag


# =============================================================================
# FUNGSI: generate_response (PUBLIC API)
# =============================================================================
def generate_response(text):
    """
    Fungsi utama yang dipanggil dari app.py untuk menghasilkan respons chatbot.
    
    Pipeline lengkap end-to-end:
        Teks → Preprocessing → Vektorisasi → Prediksi Intent → Pilih Respons
    
    Args:
        text (str): Pesan teks mentah dari pengguna
    
    Returns:
        str: Satu respons teks yang dipilih secara acak dari daftar respons intent
    
    Contoh:
        Input:  "Apa itu kolesterol?"
        Intent: "apa_itu_kolesterol"
        Output: "Kolesterol adalah zat lemak yang ada di dalam darah..."
    """
    # Langkah 1: Bersihkan teks dari tanda baca & lowercase
    texts_p = remove_punctuation(text)

    # Langkah 2: Ubah teks menjadi vektor numerik berpadding
    vector = vectorization(texts_p)

    # Langkah 3: Klasifikasikan intent dengan model neural network
    response_tag = predict(vector)

    # Langkah 4: Pilih satu respons secara acak dari daftar respons intent ini
    answer = random.choice(responses[response_tag])

    return answer


# =============================================================================
# INISIALISASI OTOMATIS SAAT MODUL DIIMPOR
# =============================================================================
# preparation() dipanggil di sini agar model sudah siap ketika modul ini
# pertama kali diimpor oleh app.py. Ini berarti model hanya dimuat SEKALI
# (tidak per request), yang jauh lebih efisien.
preparation()
