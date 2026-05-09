# =============================================================================
# app.py — Entry Point Utama Aplikasi HiMedic
# =============================================================================
# File ini adalah titik masuk (entry point) dari aplikasi web HiMedic.
# Bertanggung jawab atas:
#   1. Inisialisasi server Flask
#   2. Preprocessing data untuk sistem rekomendasi (dijalankan saat startup)
#   3. Pendefinisian semua route (URL) aplikasi
#   4. Manajemen state sesi pengguna menggunakan global variables
# =============================================================================

from flask import Flask, render_template, request  # Framework web Python
from chatbot_medicbot.chatbot_logic import generate_response, preparation  # Modul chatbot NLP
from sistem_rekomendasi.rekomendasi_logic import cosim_diagnosis  # Modul diagnosa cosine similarity

import pandas as pd                              # Manipulasi dan analisis data tabular
import math                                      # Operasi matematika (ceil)
from sklearn.metrics.pairwise import cosine_similarity  # Algoritma kemiripan vektor
import re                                        # Regular expressions untuk pembersihan string
import os
import nltk

# Download NLTK data otomatis jika belum ada (untuk environment production)
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


# =============================================================================
# BAGIAN 1: PREPROCESSING DATA SISTEM REKOMENDASI
# =============================================================================
# Blok ini dieksekusi SATU KALI saat server pertama kali dijalankan.
# Tujuannya adalah mempersiapkan semua struktur data yang dibutuhkan
# untuk proses diagnosa berbasis cosine similarity.

# --- 1.1 Load Dataset ---

# Membaca dataset gejala penyakit dari file CSV (separator titik koma)
# Kolom: "penyakit" (nama penyakit) dan "gejala" (nama gejala)
data_symptoms = pd.read_csv("dataset/desease_symptoms.csv", sep=";")

# Membaca dataset rekomendasi gaya hidup dan makanan untuk setiap penyakit
# Kolom: "Nama Penyakit", "Penjelasan Singkat", "Saran Gaya Hidup", "Saran Makanan"
data_food_style = pd.read_csv("dataset/food_n_style.csv", sep=";")

# Membuat list unik dari semua gejala yang ada di seluruh dataset
# (digunakan sebagai header kolom pada matriks encoding gejala)
gejala_all = list(set(data_symptoms["gejala"]))

# Membuat array unik nama-nama penyakit yang ada di dataset
penyakit_label = data_symptoms["penyakit"].unique()

# --- 1.2 Encoding Variabel Laboratorium (Kolesterol, Asam Urat, Gula Darah) ---

# Membuat DataFrame kosong untuk menampung hasil encoding tiga variabel lab.
# Encoding menggunakan skala kategorik: 0 = tidak ada hubungan, 1 = rendah, 2 = tinggi
data_autocheck = pd.DataFrame(columns=["kolesterol", "asam urat", "gula darah"])

# Iterasi setiap penyakit untuk menentukan hubungannya dengan 3 variabel lab
for penyakit in penyakit_label:

    # Ambil list gejala milik penyakit ini
    gejala_temp = list(data_symptoms.loc[data_symptoms["penyakit"] == penyakit]["gejala"])

    # Vector sementara: [nilai_kolesterol, nilai_asam_urat, nilai_gula_darah]
    temp = [0, 0, 0]

    for gejala in gejala_temp:

        # --- Mapping hubungan penyakit dengan kolesterol ---
        # 2 = kolesterol tinggi, 1 = kolesterol rendah, 0 = tidak ada hubungan
        if "kolesterol (tinggi)" in gejala_temp:
            temp[0] = 2
        elif "kolesterol (rendah)" in gejala_temp:
            temp[0] = 1
        elif "kolesterol (tidak ada hubungan)" in gejala_temp:
            temp[0] = 0

        # --- Mapping hubungan penyakit dengan asam urat ---
        if "asam urat (tinggi)" in gejala_temp:
            temp[1] = 2
        elif "asam urat (rendah)" in gejala_temp:
            temp[1] = 1
        elif "asam urat (tidak ada hubungan)" in gejala_temp:
            temp[1] = 0

        # --- Mapping hubungan penyakit dengan gula darah ---
        if "gula darah (tinggi)" in gejala_temp:
            temp[2] = 2
        elif "gula darah (rendah)" in gejala_temp:
            temp[2] = 1
        elif "gula darah (tidak ada hubungan)" in gejala_temp:
            temp[2] = 0

    # Tambahkan vector hasil encoding ke DataFrame data_autocheck
    data_autocheck.loc[len(data_autocheck)] = temp

# --- 1.3 Hapus Kolom Variabel Lab dari data_symptoms ---

# Setelah variabel lab diencode ke data_autocheck, hapus entri terkait
# dari dataset gejala agar tidak terjadi duplikasi saat encoding gejala klinis
list_autocheck_del = [
    "kolesterol (tinggi)", "kolesterol (rendah)", "kolesterol (tidak ada hubungan)",
    "asam urat (tinggi)", "asam urat (rendah)", "asam urat (tidak ada hubungan)",
    "gula darah (tinggi)", "gula darah (rendah)", "gula darah (tidak ada hubungan)"
]

for word in list_autocheck_del:
    data_symptoms = data_symptoms[data_symptoms.gejala != word]

# Reset index DataFrame agar berurutan kembali setelah penghapusan baris
data_symptoms = data_symptoms.reset_index(drop=True)

# --- 1.4 One-Hot Encoding Gejala Klinis ---

# Membuat matriks binary (0/1) penyakit × gejala
# Baris = setiap penyakit, Kolom = setiap gejala unik
# Nilai 1 = penyakit ini memiliki gejala tersebut, 0 = tidak
data_enc = pd.DataFrame(columns=gejala_all)

for penyakit in penyakit_label:
    gejala_temp = list(data_symptoms.loc[data_symptoms["penyakit"] == penyakit]["gejala"])
    temp = []

    for gejala in gejala_all:
        if gejala in gejala_temp:
            temp.append(1)  # Penyakit memiliki gejala ini
        else:
            temp.append(0)  # Penyakit tidak memiliki gejala ini

    data_enc.loc[len(data_enc)] = temp

# --- 1.5 Pengurutan Kolom Berdasarkan Frekuensi Gejala ---

# Urutkan kolom (gejala) dari yang paling sering muncul ke yang paling jarang.
# Strategi ini memastikan gejala paling diskriminatif ditanyakan lebih awal.
name = list(data_enc.sum(axis=0, skipna=True).sort_values(ascending=False).index)
data_enc = data_enc[name]

# --- 1.6 Gabungkan Variabel Lab dengan Encoding Gejala ---

# Satukan DataFrame data_autocheck (variabel lab) dengan data_enc (gejala klinis)
# menjadi satu matriks fitur lengkap yang siap digunakan cosine similarity
data_enc = pd.concat([data_autocheck, data_enc.reindex(data_autocheck.index)], axis=1)


# =============================================================================
# BAGIAN 2: INISIALISASI STATE GLOBAL SESI PENGGUNA
# =============================================================================
# Karena Flask tidak menggunakan session per pengguna secara default,
# state diagnosa disimpan sebagai variabel global.
# CATATAN: Ini adalah pendekatan single-user; tidak cocok untuk produksi multi-user.

flag_counter = 0          # Penanda tahap diagnosa: 0=awal input lab, 3+=pertanyaan gejala, -1=hasil
list_gejala_user = []     # Akumulasi jawaban gejala user sebagai vektor input cosine similarity
data_enc2 = data_enc      # Salinan kerja data_enc yang dipersempit setiap iterasi diagnosa
kolesterol_user = 0       # Nilai kadar kolesterol yang diinput user (mg/dL)
asam_urat_user = 0        # Nilai kadar asam urat yang diinput user (mg/dL)
gula_darah_user = 0       # Nilai kadar gula darah yang diinput user (mg/dL)
input_gejala = 0          # Jawaban user terakhir untuk pertanyaan gejala (0=tidak, 1=ya)
hasil_diagnosa = ""       # List nama penyakit hasil diagnosa akhir
persentase = ""           # List nilai persentase kemiripan cosine per penyakit
penjelasan_penyakit = ""  # List penjelasan singkat per penyakit hasil diagnosa
gaya_hidup_penyakit = ""  # List saran gaya hidup per penyakit
makanan_penyakit = ""     # List saran makanan per penyakit
reset_counter = 0         # Flag untuk memicu reset state setelah diagnosa selesai


# =============================================================================
# BAGIAN 3: INISIALISASI APLIKASI FLASK
# =============================================================================

app = Flask(__name__)  # Buat instance aplikasi Flask

# Load model chatbot saat server start (bukan per request, agar efisien)
preparation()


# =============================================================================
# BAGIAN 4: ROUTE — HALAMAN UTAMA
# =============================================================================

@app.route("/")
def home():
    """Menampilkan halaman landing page utama."""
    return render_template("home.html")


# =============================================================================
# BAGIAN 5: ROUTE — CHATBOT MEDICBOT
# =============================================================================

@app.route("/chatbot_medicbot")
def chatbot_page():
    """Menampilkan halaman antarmuka chat MedicBot."""
    return render_template("chatbot_templates.html")


@app.route("/get")
def get_bot_response():
    """
    Endpoint AJAX untuk mendapatkan respons chatbot.
    
    Method: GET
    Query Parameter: msg (str) — pesan teks dari pengguna
    Returns: str — respons teks dari model NLP chatbot
    """
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)  # Klasifikasi intent → pilih respons acak
    return result


# =============================================================================
# BAGIAN 6: ROUTE — SISTEM REKOMENDASI (DIAGNOSA)
# =============================================================================

@app.route('/sistem_rekomendasi')
def rekomendasi_page():
    """
    Menampilkan halaman awal sistem diagnosa.
    Selalu menampilkan form input laboratorium (flag_counter=0).
    """
    global flag_counter
    return render_template(
        'rekomendasi_templates.html',
        flag_counter=flag_counter,
        flag_counter2=flag_counter,
        # Hitung progress: berapa persen penyakit sudah berhasil difilter
        progress=(1 - len(data_enc2.index) / len(data_enc.index)) * 100
    )


deteksi = 0  # Variabel auxiliary (tidak digunakan secara aktif)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint utama yang menangani seluruh alur diagnosa step-by-step.
    
    Alur kerja:
      - flag_counter == 0  → Terima input lab, mulai diagnosa, tampilkan pertanyaan gejala pertama
      - flag_counter >= 3  → Terima jawaban gejala, lanjutkan iterasi cosine similarity
      - flag_counter == -1 → Diagnosa selesai, tampilkan hasil akhir 2-3 penyakit
      - reset_counter == 1 → Reset semua state ke kondisi awal
    
    Method: POST
    """
    global flag_counter, list_gejala_user, data_enc2, kolesterol_user, \
           asam_urat_user, gula_darah_user, data_enc, input_gejala, \
           hasil_diagnosa, persentase, penjelasan_penyakit, gaya_hidup_penyakit, \
           makanan_penyakit, reset_counter

    # --- TAHAP 1: Input Laboratorium (flag_counter == 0) ---
    if flag_counter == 0:
        # Ambil tiga nilai lab dari form HTML
        kolesterol_user = int(request.form['kadar_kolseterol'])
        asam_urat_user = int(request.form['kadar_asam_urat'])
        gula_darah_user = int(request.form['kadar_gula_darah'])

        # Panggil fungsi diagnosa; kembalikan semua state yang diperbarui
        (reset_counter, flag_counter, list_gejala_user, data_enc2,
         kolesterol_user, asam_urat_user, gula_darah_user, data_enc,
         input_gejala, hasil_diagnosa, persentase, penjelasan_penyakit,
         gaya_hidup_penyakit, makanan_penyakit) = cosim_diagnosis(
            reset_counter, flag_counter, list_gejala_user, data_enc2,
            kolesterol_user, asam_urat_user, gula_darah_user, data_enc,
            input_gejala, hasil_diagnosa, persentase, penjelasan_penyakit,
            gaya_hidup_penyakit, makanan_penyakit, penyakit_label, data_food_style)

        print(flag_counter)

        # Tampilkan pertanyaan gejala pertama setelah input lab diterima
        return render_template(
            'rekomendasi_templates.html',
            pertanyaan_gejala=data_enc.columns[flag_counter],
            flag_counter=flag_counter,
            flag_counter2=flag_counter,
            flag_counter3=flag_counter,
            progress=(1 - len(data_enc2.index) / len(data_enc.index)) * 100
        )

    # --- TAHAP 2: Pertanyaan Gejala Berulang (flag_counter >= 3) ---
    elif flag_counter >= 3:
        # Ambil jawaban user: 1 = ya (punya gejala), 0 = tidak
        input_gejala = int(request.form['is_menderita_gejala'])

        # Lanjutkan proses diagnosa dengan jawaban baru
        (reset_counter, flag_counter, list_gejala_user, data_enc2,
         kolesterol_user, asam_urat_user, gula_darah_user, data_enc,
         input_gejala, hasil_diagnosa, persentase, penjelasan_penyakit,
         gaya_hidup_penyakit, makanan_penyakit) = cosim_diagnosis(
            reset_counter, flag_counter, list_gejala_user, data_enc2,
            kolesterol_user, asam_urat_user, gula_darah_user, data_enc,
            input_gejala, hasil_diagnosa, persentase, penjelasan_penyakit,
            gaya_hidup_penyakit, makanan_penyakit, penyakit_label, data_food_style)

        # --- TAHAP 3: Diagnosa Selesai (flag_counter == -1) ---
        if flag_counter == -1:
            reset_counter = 1  # Tandai bahwa state perlu di-reset di request berikutnya

            if len(hasil_diagnosa) == 2:
                # Kasus 2 hasil diagnosa (penyakit ke-3 tidak ada atau sama)
                return render_template(
                    'rekomendasi_templates.html',
                    flag_counter=flag_counter, flag_counter2=flag_counter,
                    flag_counter3=flag_counter, flag_counter4=flag_counter,
                    reset_counter=reset_counter,
                    hasil_diagnosa1=hasil_diagnosa[0], hasil_diagnosa2=hasil_diagnosa[1],
                    persentase1=int(persentase[0] * 100), persentase2=int(persentase[1] * 100),
                    penjelasan1=penjelasan_penyakit[0], penjelasan2=penjelasan_penyakit[1],
                    gaya_hidup1=gaya_hidup_penyakit[0], gaya_hidup2=gaya_hidup_penyakit[1],
                    makanan_penyakit1=makanan_penyakit[0], makanan_penyakit2=makanan_penyakit[1],
                    len_output=len(hasil_diagnosa)
                )
            else:
                # Kasus 3 hasil diagnosa (umum)
                return render_template(
                    'rekomendasi_templates.html',
                    flag_counter=flag_counter, flag_counter2=flag_counter,
                    flag_counter3=flag_counter, flag_counter4=flag_counter,
                    reset_counter=reset_counter,
                    hasil_diagnosa1=hasil_diagnosa[0], hasil_diagnosa2=hasil_diagnosa[1], hasil_diagnosa3=hasil_diagnosa[2],
                    persentase1=int(persentase[0] * 100), persentase2=int(persentase[1] * 100), persentase3=int(persentase[2] * 100),
                    penjelasan1=penjelasan_penyakit[0], penjelasan2=penjelasan_penyakit[1], penjelasan3=penjelasan_penyakit[2],
                    gaya_hidup1=gaya_hidup_penyakit[0], gaya_hidup2=gaya_hidup_penyakit[1], gaya_hidup3=gaya_hidup_penyakit[2],
                    makanan_penyakit1=makanan_penyakit[0], makanan_penyakit2=makanan_penyakit[1], makanan_penyakit3=makanan_penyakit[2],
                    len_output=len(hasil_diagnosa)
                )

        else:
            # Masih ada penyakit > 3 yang tersisa, lanjutkan pertanyaan gejala
            return render_template(
                'rekomendasi_templates.html',
                pertanyaan_gejala=data_enc.columns[flag_counter],
                flag_counter=flag_counter,
                flag_counter2=flag_counter,
                flag_counter3=flag_counter,
                progress=(1 - len(data_enc2.index) / len(data_enc.index)) * 100
            )

    # --- TAHAP 4: Reset State Setelah Diagnosa Selesai ---
    if reset_counter == 1:
        # Kembalikan semua variabel global ke kondisi awal
        reset_counter = 0
        flag_counter = 0
        list_gejala_user = []
        data_enc2 = data_enc
        kolesterol_user = 0
        asam_urat_user = 0
        gula_darah_user = 0
        input_gejala = 0
        hasil_diagnosa = ""
        persentase = ""
        penjelasan_penyakit = ""
        gaya_hidup_penyakit = ""
        makanan_penyakit = ""

        # Arahkan kembali ke halaman awal diagnosa
        return render_template(
            'rekomendasi_templates.html',
            flag_counter=flag_counter,
            flag_counter2=flag_counter,
            progress=(1 - len(data_enc2.index) / len(data_enc.index)) * 100
        )


@app.route("/reset", methods=["POST"])
def reset():
    """
    Endpoint untuk mereset paksa semua state diagnosa.
    Dipanggil dari tombol 'Reset' di halaman hasil diagnosa.
    
    Method: POST
    Returns: Render halaman awal diagnosa dengan state bersih
    """
    global flag_counter, list_gejala_user, data_enc2, kolesterol_user, \
           asam_urat_user, gula_darah_user, data_enc, input_gejala, \
           hasil_diagnosa, persentase, penjelasan_penyakit, gaya_hidup_penyakit, \
           makanan_penyakit, reset_counter

    # Reset semua variabel ke nilai awal
    reset_counter = 0
    flag_counter = 0
    list_gejala_user = []
    data_enc2 = data_enc
    kolesterol_user = 0
    asam_urat_user = 0
    gula_darah_user = 0
    input_gejala = 0
    hasil_diagnosa = ""
    persentase = ""
    penjelasan_penyakit = ""
    gaya_hidup_penyakit = ""
    makanan_penyakit = ""

    return render_template(
        'rekomendasi_templates.html',
        flag_counter=flag_counter, flag_counter2=flag_counter,
        flag_counter3=flag_counter, flag_counter4=flag_counter,
        reset_counter=reset_counter,
        progress=(1 - len(data_enc2.index) / len(data_enc.index)) * 100
    )


# =============================================================================
# BAGIAN 7: ENTRYPOINT — JALANKAN SERVER
# =============================================================================

if __name__ == "__main__":
    # Gunakan PORT dari environment variable jika ada (Railway/production)
    # Fallback ke port 5000 untuk development lokal
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
