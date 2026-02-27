import pandas as pd
import math
from sklearn.metrics.pairwise import cosine_similarity
import re



def cosim_diagnosis(reset_counter, flag_counter, list_gejala_user, 
                    data_enc2, kolesterol_user, asam_urat_user, gula_darah_user, 
                    data_enc, input_gejala, hasil_diagnosa, persentase, 
                    penjelasan_penyakit, gaya_hidup_penyakit, makanan_penyakit, 
                    penyakit_label, data_food_style):
  if flag_counter == "stop":
    print("udah bos")
    return
  if flag_counter == 0:
    # meminta inputan awal berupa data kolesterol, asam urat, gula darah
    # kolesterol_user = int(input(f"masukkan kadar kolesterol : "))
    # asam_urat_user = int(input(f"masukkan kadar asam urat : "))
    # gula_darah_user = int(input(f"masukkan kadar gula darah : "))

    # mengkonversi data jumlah kedalam data kategorik karena dalam dataframe kolesterol, asam urat dan gula darah memiliki nilai kategorik (0,1,2)
    kolesterol_user = 1 if kolesterol_user < 240 else 2
    asam_urat_user = 1 if asam_urat_user < 6.5 else 2
    gula_darah_user = 1 if gula_darah_user < 180 else 2

    # list gejala user merupakan vector inputan yang akan dihitung bersamaan dengan datframe data_enc2 menggunakan cosine similarity
    # telah terdapat 3 nilai awal yang merupakan inputan pada baris sebelumnya
    list_gejala_user = [kolesterol_user, asam_urat_user, gula_darah_user]
    flag_counter += 3
    print(flag_counter)
  elif flag_counter > 0:
    total = sum(data_enc2.iloc[:,len(list_gejala_user)])
    total_index =  len(data_enc2.index)
    if total == total_index:
      list_gejala_user.append(1)
      flag_counter += 1
      cosim_diagnosis()
      return
    elif total == 0:
      list_gejala_user.append(0)
      flag_counter += 1
      cosim_diagnosis()
      return
    else:
      # input_gejala = input(f"apakah anda merasa {data_enc2.columns[flag_counter]} ")
      list_gejala_user.append(int(input_gejala))   
      flag_counter += 1

    len_input = len(list_gejala_user)
    if len_input%2 == 0:
      print("yokai")
      data_temp = data_enc2.iloc[:,:len_input]


      data_temp = data_temp.append(pd.DataFrame([list_gejala_user], columns=list(data_temp.columns)), ignore_index=False)


      similarity = pd.DataFrame(cosine_similarity(data_temp))


      similarity = similarity.iloc[0:len(similarity)-1]

      similarity = similarity.set_index(pd.Series(list(data_enc2.index)))

      similarity = similarity.sort_values(by=len(similarity), ascending=False)

      index_similarity = similarity.iloc[0:math.ceil(0.6*len(similarity))].index

      data_enc2 = data_enc2.loc[list(index_similarity)]

    # jika list penyakit tersisa 3 saja, maka perulangan berhenti
    
    hasil_diagnosa = penyakit_label[list(data_enc2.index)]
    hasil_diagnosa_awal = hasil_diagnosa
    order = r'[0123456789]'
    temp_diagnosa = []
    for j in hasil_diagnosa:
      filtered_string = re.sub(order, '', j)
      temp_diagnosa.append(filtered_string)

    hasil_diagnosa = temp_diagnosa

    if len(hasil_diagnosa) <= 3:

      persentase = list(similarity.iloc[:,-1].loc[list(data_enc2.index)])
      penjelasan_penyakit = list(data_food_style.loc[list(data_enc2.index), "Penjelasan Singkat"])
      gaya_hidup_penyakit = list(data_food_style.loc[list(data_enc2.index), "Saran Gaya Hidup"])
      makanan_penyakit = list(data_food_style.loc[list(data_enc2.index), "Saran Makanan"])


      # dictionary of lists 
      dict = {'hasil_diagnosa': hasil_diagnosa, 'persentase': persentase, 'penjelasan_penyakit': penjelasan_penyakit,
              'gaya_hidup_penyakit' : gaya_hidup_penyakit, 'makanan_penyakit' : makanan_penyakit} 
          
      df_hasil = pd.DataFrame(dict)
      df_hasil = df_hasil.drop_duplicates(subset='hasil_diagnosa', keep="first")

      hasil_diagnosa = df_hasil["hasil_diagnosa"].values.tolist()
      persentase = df_hasil["persentase"].values.tolist()
      penjelasan_penyakit = df_hasil["penjelasan_penyakit"].values.tolist()
      gaya_hidup_penyakit = df_hasil["gaya_hidup_penyakit"].values.tolist()
      makanan_penyakit = df_hasil["makanan_penyakit"].values.tolist()

      #print(hasil_diagnosa)
      #print(persentase)
      #print(penjelasan_penyakit)
      #print(gaya_hidup_penyakit)
      #print(makanan_penyakit)
      
      flag_counter = -1

  print("sebelum return", flag_counter)
  return reset_counter, flag_counter, list_gejala_user, data_enc2, kolesterol_user, asam_urat_user, gula_darah_user, data_enc, input_gejala, hasil_diagnosa, persentase, penjelasan_penyakit, gaya_hidup_penyakit, makanan_penyakit