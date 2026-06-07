# Draft Penelitian: Model Bahasa SNN dengan Local Error Learning dan Spiking Self-Attention

## 1. Pendahuluan & Latar Belakang
Penelitian ini mengusulkan arsitektur model bahasa baru yang berlandaskan pada **Spiking Neural Networks (SNN)**, dengan tujuan menghadirkan kecerdasan buatan yang amat hemat energi dan efisien secara memori. Terobosan utama (novelty) dalam penelitian ini dibagi menjadi dua komponen krusial:
1. **Spike-driven Self-Attention**: Menggantikan mekanisme Softmax dan perkalian presisi tinggi (float) tradisional dengan operasi logika murni (*bitwise addition / add-only*) serta lapisan Leaky Integrate-and-Fire (LIF).
2. **Local Error Learning**: Sebuah metode pelatihan yang terinspirasi dari *Forward-Forward Algorithm* (gagasan Geoffrey Hinton) yang difungsikan sebagai alternatif modern pengganti *Backpropagation Through Time* (BPTT).

## 2. Kebaruan (Novelty): Local Error Learning pada SNN
Algoritma Backpropagation konvensional—khususnya BPTT pada pemrosesan SNN—menuntut penyimpanan jejak gradien (*gradient tracking*) secara terus-menerus ke seluruh *time-steps* jaringan (melintasi grafik komputasi dari ujung akhir ke lapisan terawal). Pendekatan ini secara drastis menghabiskan ketersediaan memori GPU dan di saat bersamaan dikritik karena tidak relevan secara biologis (*not bio-plausible*).

Sebagai alternatif, penelitian ini mengusung **Local Error Learning** dengan mekanisme berikut:

### a. Mekanisme
- Jaringan **tidak menghitung error/loss gabungan di lapisan paling ujung** lalu melemparkannya secara mundur (backward) menembus batas-batas layer.
- Sebaliknya, **setiap lapisan memiliki fungsi kerugian (loss function) sendiri-sendiri**. Pembelajaran dipartisi sedemikian rupa sehingga suatu layer dievaluasi kesuksesannya sesaat setelah ia merampungkan komputasinya.

### b. Proses Pembelajaran Instan Mengalir ke Depan (Forward)
Proses pelatihan pada sistem berjalan selaras dengan aliran propagasi datanya:
1. **Pembelajaran Layer Pertama (Representasi Awal):** 
   Lapisan pertama bertugas belajar untuk merepresentasikan fitur kalimat awal dengan baik. Berdasarkan output *spike* yang ia cetak, sebuah *local loss* akan dihitung. Penyesuaian bobotnya kemudian dilakukan secara **mandiri dan instan** (menggunakan *Hebbian/Delta Rule* atau *Surrogate Gradient* lokal).
2. **Propagasi Spike Maju:**
   Tanpa harus menunggu instruksi dari ujung jaringan, *spike* yang telah disempurnakan tersebut langsung disalurkan ke lapisan kedua.
3. **Pembelajaran Layer Kedua (Konteks Global via Self-Attention):**
   Lapisan kedua (yakni layer *Spiking Self-Attention*) menerima *spike* dari layer pertama, melakukan kalkulasi atensi murni-addisi (*match scores*), dan menghitung kerugian representasi konseptualnya sendiri. Pembaruan bobot layer ini kembali dilakukan secara mandiri di lokasinya sendiri.

## 3. Keunggulan Teoritis dan Praktis
1. **Penghematan Memori Ekstrem:** Karena putusnya ikatan rantai gradien global (*no backpropagation chain*), *engine* pembelajaran tidak perlu lagi menyimpan *activation maps* raksasa di memori saat mengurai rentetan waktu (*time-steps*). Konsumsi memori VRAM / GPU dapat ditekan hingga berlipat ganda, memampukan ukuran batch yang lebih besar di perangkat komoditas.
2. **Komputasi Asinkron / Edge Computing Friendly:** Pembaruan bobot yang bersifat instan di saat *forward pass* memungkinkan paralelisasi perangkat keras yang sesungguhnya. Edge device maupun arsitektur chip neuromorfik modern sangat diuntungkan dari tidak adanya rutinitas *wait-and-backward* ini.
3. **Bio-Plausibility Maksimal:** *Local Error Learning* sangat mengamini cara otak biologis belajar. Sinapsis serebral manusia menyesuaikan kekuatan sambungannya secara independen dipengaruhi oleh penembakan (*firing*) neuron terdekat yang terhubung secara langsung (Heabian Plasticity), bukan dikoordinasikan oleh agen pusat di ujung jalur saraf.

## 4. Implementasi Arsitektural dalam Oxide-JS
- **Integrasi Self-Attention Bebas Softmax**: Menggunakan `SpikingSelfAttention` berbasis *add-only computation* murni.
- **Modifikasi Forward Loop**: Pemanggilan rutin pembelajaran secara langsung diinjeksikan pada tahapan komputasi forward. Fungsi-fungsi semacam `learnHidden` tidak lagi membutuhkan turunan gradien paksaan (`errorFromNext`), melainkan akan menghitung kecocokannya pada fasa yang sama.

## 5. Teknik Pelatihan: Unsupervised Contrastive Learning (Self-Supervised)
Guna mengatasi keterbatasan perangkat keras dalam memproses dataset *triplet* (Q&A terlabel) yang masif dan berat, penelitian ini mengadopsi pendekatan **Self-Supervised Learning** menggunakan teknik **Unsupervised Contrastive Learning** (terinspirasi dari metode SimCSE).

### a. Sumber Data
Pelatihan tidak membutuhkan dataset berlabel rumit. Sumber data cukup menggunakan *dump* artikel tekstual mentah tanpa label, seperti kumpulan artikel Wikipedia (baik bahasa Inggris maupun Indonesia).

### b. Mekanisme Representasi
- **Query ($Q$)**: Sebuah kalimat acak diekstraksi dari artikel dan dijadikan sebagai titik tumpu (*anchor* atau Query).
- **Positive Sample ($P^+$)**: Salinan dari $Q$ yang telah diberikan sedikit trik modifikasi. Dalam ranah SNN, modifikasi ini dapat berupa penyembunyian beberapa kata (*word masking*) atau penyuntikan sedikit *noise* pada pola *spike* (misalnya dengan memodifikasi threshold secara dinamis).
- **Negative Sample ($P^-$)**: Sebuah kalimat yang diambil secara acak dari artikel lain yang sama sekali berbeda konteksnya.

### c. Keunggulan Metode Pelatihan Ini
1. **Biaya Pelatihan Sangat Murah:** Menghapus sepenuhnya kebergantungan pada proses *data labeling* yang memakan waktu dan biaya mahal.
2. **Ketersediaan Data Tidak Terbatas:** Dengan memakan teks mentah dari internet, model memiliki suplai informasi linguistik tanpa batas.
3. **Membentuk Topologi Semantik Secara Alami:** Memaksa SNN untuk memisahkan representasi kalimat acak dan mendekatkan kalimat yang identik/bernoise, yang pada akhirnya sukses melatih SNN untuk mampu menangkap struktur semantik dasar dari berbagai macam tata bahasa secara mandiri.
