# =============================================================================
# rekomendasi_logic.py — Modul Diagnosa Berbasis Cosine Similarity
# =============================================================================
# Modul ini mengimplementasikan algoritma diagnosa penyakit menggunakan
# pendekatan Content-Based Filtering dengan Cosine Similarity.
#
# Konsep Utama:
#   - Setiap penyakit direpresentasikan sebagai vektor biner (0/1) fitur gejala.
#   - Input user (jawaban gejala + nilai lab) juga dibentuk menjadi vektor.
#   - Cosine Similarity mengukur "sudut" antara vektor user dan vektor penyakit.
#   - Nilai mendekati 1.0 = sangat mirip; mendekati 0.0 = tidak mirip.
#
# Strategi Diagnosa (Iteratif / Narrowing):
#   Alih-alih menghitung similarity terhadap semua penyakit sekaligus,
#   sistem mengeliminasi penyakit yang tidak relevan secara bertahap
#   setiap 2 jawaban baru. Proses berhenti saat tersisa ≤ 3 penyakit.
# =============================================================================

import pandas as pd                              # Manipulasi data tabular
import math                                      # math.ceil untuk perhitungan proporsi
from sklearn.metrics.pairwise import cosine_similarity  # Algoritma cosine similarity
import re                                        # Regex untuk membersihkan string nama penyakit


# =============================================================================
# FUNGSI: cosim_diagnosis
# =============================================================================
def cosim_diagnosis(reset_counter, flag_counter, list_gejala_user,
                    data_enc2, kolesterol_user, asam_urat_user, gula_darah_user,
                    data_enc, input_gejala, hasil_diagnosa, persentase,
                    penjelasan_penyakit, gaya_hidup_penyakit, makanan_penyakit,
                    penyakit_label, data_food_style):
    """
    Fungsi inti diagnosa berbasis cosine similarity yang berjalan secara iteratif.
    
    Fungsi ini dipanggil berulang kali (per request POST) dan menggunakan
    pola stateful melalui parameter yang di-pass dari variabel global app.py.
    Setiap pemanggilan merupakan SATU LANGKAH dalam proses diagnosa multi-step.
    
    Args:
        reset_counter    (int)          : Flag reset (1=sedang proses hasil, 0=normal)
        flag_counter     (int)          : Penanda tahap: 0=input lab, 3+=gejala, -1=selesai
        list_gejala_user (list[int])    : Akumulasi vector input user [lab(3) + gejala(n)]
        data_enc2        (DataFrame)    : Subset penyakit yang masih aktif (belum tereliminasi)
        kolesterol_user  (int)          : Nilai kolesterol yang sudah dikategorikan (1 atau 2)
        asam_urat_user   (int)          : Nilai asam urat yang sudah dikategorikan (1 atau 2)
        gula_darah_user  (int)          : Nilai gula darah yang sudah dikategorikan (1 atau 2)
        data_enc         (DataFrame)    : Matriks encoding penuh (referensi asli, tidak berubah)
        input_gejala     (int)          : Jawaban gejala terbaru user (0=tidak, 1=ya)
        hasil_diagnosa   (list/str)     : Akumulasi nama penyakit hasil akhir
        persentase       (list/str)     : Akumulasi nilai similarity hasil akhir
        penjelasan_penyakit (list/str)  : Penjelasan singkat per penyakit
        gaya_hidup_penyakit (list/str)  : Saran gaya hidup per penyakit
        makanan_penyakit    (list/str)  : Saran makanan per penyakit
        penyakit_label   (ndarray)      : Array semua nama penyakit
        data_food_style  (DataFrame)    : Dataset penjelasan & saran (Penjelasan Singkat, Saran Gaya Hidup, Saran Makanan)
    
    Returns:
        tuple: (reset_counter, flag_counter, list_gejala_user, data_enc2,
                kolesterol_user, asam_urat_user, gula_darah_user, data_enc,
                input_gejala, hasil_diagnosa, persentase, penjelasan_penyakit,
                gaya_hidup_penyakit, makanan_penyakit)
        Semua nilai diperbarui dan dikembalikan ke app.py untuk di-assign ke global variables.
    """

    # Guard: Hentikan eksekusi jika sudah ditandai selesai
    if flag_counter == "stop":
        print("udah bos")
        return

    # =========================================================================
    # TAHAP A: Input Laboratorium (flag_counter == 0)
    # =========================================================================
    if flag_counter == 0:
        # --- Konversi Nilai Numerik Lab ke Kategorik ---
        # Nilai numerik (mg/dL) diubah ke skala kategorik untuk dimasukkan ke vektor:
        #   1 = kadar normal/rendah (tidak mencapai threshold)
        #   2 = kadar tinggi (melewati threshold)
        #
        # Threshold yang digunakan:
        #   Kolesterol : < 240 mg/dL = normal (1), ≥ 240 = tinggi (2)
        #   Asam Urat  : < 6.5 mg/dL = normal (1), ≥ 6.5 = tinggi (2)
        #   Gula Darah : < 180 mg/dL = normal (1), ≥ 180 = tinggi (2)
        kolesterol_user = 1 if kolesterol_user < 240 else 2
        asam_urat_user  = 1 if asam_urat_user  < 6.5  else 2
        gula_darah_user = 1 if gula_darah_user < 180  else 2

        # Inisialisasi vektor user dengan 3 nilai lab sebagai titik awal
        # Format: [kolesterol_kategori, asam_urat_kategori, gula_darah_kategori]
        list_gejala_user = [kolesterol_user, asam_urat_user, gula_darah_user]

        # Pindahkan flag_counter ke 3 (menandakan 3 fitur pertama sudah diisi)
        flag_counter += 3
        print(flag_counter)

    # =========================================================================
    # TAHAP B: Pemrosesan Jawaban Gejala (flag_counter > 0)
    # =========================================================================
    elif flag_counter > 0:

        # --- Cek Kolom Saat Ini Apakah Perlu Ditanyakan ---

        # Hitung total nilai di kolom gejala saat ini (posisi = len(list_gejala_user))
        total = sum(data_enc2.iloc[:, len(list_gejala_user)])

        # total_index = jumlah penyakit yang masih aktif di data_enc2
        total_index = len(data_enc2.index)

        if total == total_index:
            # SKIP: Semua penyakit aktif memiliki gejala ini.
            # Jawaban selalu "ya" → tidak mendiskriminasi → tambahkan 1, skip pertanyaan
            list_gejala_user.append(1)
            flag_counter += 1

            # Rekursi untuk memproses kolom berikutnya tanpa menunggu input
            cosim_diagnosis(reset_counter, flag_counter, list_gejala_user,
                            data_enc2, kolesterol_user, asam_urat_user, gula_darah_user,
                            data_enc, input_gejala, hasil_diagnosa, persentase,
                            penjelasan_penyakit, gaya_hidup_penyakit, makanan_penyakit,
                            penyakit_label, data_food_style)
            return

        elif total == 0:
            # SKIP: Tidak ada penyakit aktif yang memiliki gejala ini.
            # Jawaban selalu "tidak" → tidak mendiskriminasi → tambahkan 0, skip pertanyaan
            list_gejala_user.append(0)
            flag_counter += 1

            # Rekursi untuk memproses kolom berikutnya
            cosim_diagnosis(reset_counter, flag_counter, list_gejala_user,
                            data_enc2, kolesterol_user, asam_urat_user, gula_darah_user,
                            data_enc, input_gejala, hasil_diagnosa, persentase,
                            penjelasan_penyakit, gaya_hidup_penyakit, makanan_penyakit,
                            penyakit_label, data_food_style)
            return

        else:
            # TANYA: Gejala ini bersifat diskriminatif (sebagian penyakit punya, sebagian tidak)
            # Tambahkan jawaban user (0 atau 1) ke vektor
            list_gejala_user.append(int(input_gejala))
            flag_counter += 1

        # --- Kalkulasi Cosine Similarity (setiap 2 jawaban baru) ---
        len_input = len(list_gejala_user)

        if len_input % 2 == 0:
            # Setiap 2 jawaban terkumpul, jalankan similarity filtering
            print("yokai")

            # Ambil subset kolom data_enc2 sesuai panjang vektor user saat ini
            data_temp = data_enc2.iloc[:, :len_input]

            # Tambahkan vektor user sebagai baris baru di bawah data penyakit
            # (dibutuhkan untuk menghitung similarity antara user vs setiap penyakit)
            data_temp = data_temp.append(
                pd.DataFrame([list_gejala_user], columns=list(data_temp.columns)),
                ignore_index=False
            )

            # Hitung matriks cosine similarity (ukuran: n_penyakit+1 × n_penyakit+1)
            # Baris/kolom terakhir = vektor user
            similarity = pd.DataFrame(cosine_similarity(data_temp))

            # Ambil hanya baris penyakit (kecuali baris terakhir = user)
            similarity = similarity.iloc[0:len(similarity) - 1]

            # Restore index agar sesuai dengan index data_enc2
            similarity = similarity.set_index(pd.Series(list(data_enc2.index)))

            # Urutkan berdasarkan kolom terakhir (similarity dengan user) dari tertinggi
            similarity = similarity.sort_values(by=len(similarity), ascending=False)

            # Ambil TOP 60% penyakit dengan similarity tertinggi (eliminasi 40% terendah)
            # math.ceil memastikan minimal 1 penyakit dipertahankan
            index_similarity = similarity.iloc[0:math.ceil(0.6 * len(similarity))].index

            # Perbarui data_enc2 hanya dengan penyakit yang lolos seleksi
            data_enc2 = data_enc2.loc[list(index_similarity)]

        # --- Cek Apakah Proses Diagnosa Sudah Bisa Dihentikan ---

        # Ambil nama penyakit yang masih aktif berdasarkan index
        hasil_diagnosa = penyakit_label[list(data_enc2.index)]
        hasil_diagnosa_awal = hasil_diagnosa  # Simpan untuk referensi (tidak dipakai lebih lanjut)

        # Bersihkan angka dari nama penyakit menggunakan regex
        # Contoh: "asam urat1" → "asam urat"
        order = r'[0123456789]'
        temp_diagnosa = []
        for j in hasil_diagnosa:
            filtered_string = re.sub(order, '', j)
            temp_diagnosa.append(filtered_string)
        hasil_diagnosa = temp_diagnosa

        # Jika tersisa ≤ 3 penyakit, proses diagnosa selesai
        if len(hasil_diagnosa) <= 3:

            # Ambil nilai similarity akhir untuk penyakit yang tersisa
            persentase = list(similarity.iloc[:, -1].loc[list(data_enc2.index)])

            # Ambil data rekomendasi dari dataset food_n_style berdasarkan index penyakit
            penjelasan_penyakit = list(data_food_style.loc[list(data_enc2.index), "Penjelasan Singkat"])
            gaya_hidup_penyakit = list(data_food_style.loc[list(data_enc2.index), "Saran Gaya Hidup"])
            makanan_penyakit    = list(data_food_style.loc[list(data_enc2.index), "Saran Makanan"])

            # --- Hapus Duplikat Nama Penyakit ---
            # Buat DataFrame sementara untuk deduplication berdasarkan nama penyakit
            dict_hasil = {
                'hasil_diagnosa':     hasil_diagnosa,
                'persentase':         persentase,
                'penjelasan_penyakit':penjelasan_penyakit,
                'gaya_hidup_penyakit':gaya_hidup_penyakit,
                'makanan_penyakit':   makanan_penyakit
            }

            df_hasil = pd.DataFrame(dict_hasil)

            # Hapus baris dengan nama penyakit duplikat, pertahankan kemunculan pertama
            df_hasil = df_hasil.drop_duplicates(subset='hasil_diagnosa', keep="first")

            # Konversi kembali ke list untuk dikembalikan ke app.py
            hasil_diagnosa      = df_hasil["hasil_diagnosa"].values.tolist()
            persentase          = df_hasil["persentase"].values.tolist()
            penjelasan_penyakit = df_hasil["penjelasan_penyakit"].values.tolist()
            gaya_hidup_penyakit = df_hasil["gaya_hidup_penyakit"].values.tolist()
            makanan_penyakit    = df_hasil["makanan_penyakit"].values.tolist()

            # Tandai diagnosa selesai dengan flag_counter = -1
            flag_counter = -1

    # =========================================================================
    # RETURN: Kembalikan semua state yang telah diperbarui ke app.py
    # =========================================================================
    print("sebelum return", flag_counter)
    return (
        reset_counter,
        flag_counter,
        list_gejala_user,
        data_enc2,
        kolesterol_user,
        asam_urat_user,
        gula_darah_user,
        data_enc,
        input_gejala,
        hasil_diagnosa,
        persentase,
        penjelasan_penyakit,
        gaya_hidup_penyakit,
        makanan_penyakit
    )
