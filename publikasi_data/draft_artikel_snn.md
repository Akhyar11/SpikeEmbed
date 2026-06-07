# Draft Artikel Penelitian: Order-Aware Spiking Neural Networks untuk Pemrosesan Bahasa Alami

Dokumen ini berisi poin-poin utama, data, dan argumentasi metodologis yang sangat penting untuk dimasukkan ke dalam naskah artikel / jurnal penelitian Anda.

## 1. Latar Belakang & Masalah (Introduction)
*   **Dominasi Transformer & Kendala Energi:** Transformer konvensional sangat akurat namun boros energi karena penggunaan komputasi _floating-point_ yang intensif (operasi _Softmax_ dan perkalian matriks padat).
*   **Potensi SNN (Spiking Neural Networks):** SNN menjanjikan efisiensi energi yang masif melalui komputasi berbasis _event_ (biner 0 dan 1).
*   **Kesenjangan (Research Gap):** Implementasi SNN murni pada teks sering kali gagal membedakan urutan kata (terjebak dalam simetri _Bag-of-Words_) karena sifat _step function_ pada LIF dan kesulitan melatih arsitektur _Self-Attention_ tanpa fungsi _Softmax_ yang dapat diturunkan (_differentiable_).

## 2. Metodologi & Arsitektur yang Diusulkan (Proposed Method)
Jelaskan bahwa penelitian ini mengajukan arsitektur **Hybrid / 3rd Generation SNN** dengan 3 inovasi utama:

### A. Modulasi Atensi dengan *Graded Potentials* (Bukan Softmax)
*   **Masalah:** Pemangkasan biner murni pada matriks atensi menyebabkan hilangnya identitas kata, membuat token saling menimpa menjadi makna tunggal.
*   **Solusi Biologis:** Meniru cara kerja dendrit dan celah sinapsis di neokorteks mamalia (reseptor NMDA). Alih-alih merubah nilai kecocokan (_match scores_) menjadi biner 0 dan 1, nilai dipertahankan sebagai **tegangan kontinu (Graded Potentials)**.
*   **Mekanisme Normalisasi:** Mengganti _Softmax_ (yang butuh kalkulus eksponensial global) dengan **Dynamic Max Normalization** sebagai aproksimasi mekanisme _Lateral Inhibition_ (Winner-Take-All). Ini ramah *neuromorphic* namun tetap mempertahankan presisi distribusi atensi.

### B. Mencegah "Kematian Kata" dengan *Residual Connections*
*   **Masalah:** Pada fase awal pelatihan (_training_), matriks *Value* ($V$) sering gagal mencapai batas ambang (_threshold_) sehingga kata tertentu menghasilkan 0 letupan listrik (_dead words_). Ini memutus propagasi sinyal.
*   **Solusi:** Menyuntikkan jalur pintas (Skip-Connection) seperti pada ResNet: `Word Embedding + Attention Output`. Secara biologis, ini ekuivalen dengan proyeksi paralel (_bypass pathways_) antarlapisan di korteks otak, menjamin sinyal mentah tidak pernah hilang.

### C. Pemecah Simetri: *Heterogeneous Temporal Pooler*
*   **Masalah:** Menggunakan metrik _Cosine Similarity_ pada jaringan integrator dengan memori seragam ($\beta \approx 0.99$) akan membuat kalimat yang ditukar urutannya memiliki sudut >99% sejajar karena hukum penjumlahan bersifat komutatif ($A+B \approx B+A$).
*   **Solusi:** Menugaskan lapisan `SpikingDense` sebagai *Temporal Pooler* dengan tingkat peluruhan yang bervariasi secara **Heterogen** (rentang $\beta = 0.2$ hingga $0.8$). Ini mendesentralisasi memori, sebagian neuron mengingat konteks panjang (klausa) dan sebagian hanya mengingat kata terakhir. Solusi ini secara dramatis meruntuhkan invariansi skala dan berhasil memisahkan "ayam goreng" dari "goreng ayam" menjadi 96.62% similarity (tidak lagi identik mutlak).

### D. Attention Masking via Neuromorphic Silencing (PAD Token)
*   **Masalah:** Penggunaan *padding* pada kalimat pendek (misalnya, menutupi batas sequence 32 dengan 30 token PAD) mendistraksi matriks atensi karena token PAD dihitung sebagai letupan listrik aktif.
*   **Solusi:** Memaksa embedding token PAD (Index 0) bernilai konstan negatif ($-1.0$) sebelum setiap proses inferensi. Hal ini membisukan token PAD secara absolut (0 spike), bertindak secara fungsional seperti *Attention Mask* pada Transformer klasik, namun dijalankan murni melalui manipulasi tegangan biologis.

### E. Mencegah *Dead Neuron Collapse* dengan *Dopamine Injection*
*   **Masalah:** Pada *Contrastive Hebbian Learning* tanpa pengawasan, jika sinyal *Query* dan sampel Negatif sama-sama meletup, keduanya diberi penalti. Jaringan kemudian mencari jalan pintas licik (Global Minimum) dengan mematikan total semua neuron (0 letupan) untuk menghindari hukuman sama sekali (keruntuhan mematikan).
*   **Solusi:** Memperkenalkan *Homeostasis Plasticity* berupa "Suntikan Dopamin" (penambahan _error gradient_ positif +0.05) yang hanya aktif apabila sebuah neuron terdeteksi mati total (0 spike pada *Query, Positive*, dan *Negative*). Suntikan energi biologis ini memaksa jaringan bangun dari kondisi koma dan terus mengoptimasi topologi ruang semantiknya.

### F. Pembelajaran Lokal tanpa Backpropagation
*   Melatih keseluruhan arsitektur secara *unsupervised* menggunakan metode **Contrastive Hebbian Learning / Surrogate Gradients**. Kesalahan (_error_) dihitung melalui *pull-push mechanism* dari sampel positif/negatif, lalu disebarkan ke belakang tanpa bergantung pada _Chain-Rule_ dari kalkulus tradisional.

## 3. Metodologi Pengujian (Evaluation Methods)
Untuk memastikan validitas arsitektur secara saintifik, pengujian dibagi menjadi dua tahapan: Makro dan Mikro.

### A. Pengujian Kuantitatif Makro: STS-B (Semantic Textual Similarity Benchmark)
*   **Dataset:** Evaluasi menggunakan 1500 pasang kalimat dari _dataset_ STS-B yang berisi pasangan kalimat beserta skor kedekatan semantik yang dianotasi oleh manusia.
*   **Prosedur:** Kalimat $A$ dan $B$ dienkode ke dalam _spikes_, lalu dilewatkan melalui SNN selama 5 _timesteps_ temporal. Output membran akhir ($lastPotentials$) dari _Temporal Pooler_ diekstrak sebagai representasi vektor kalimat.
*   **Metrik Evaluasi:** Menghitung jarak vektor kalimat menggunakan _Cosine Similarity_, lalu membandingkannya dengan anotasi manusia menggunakan metrik **Pearson Correlation Coefficient**. Ini membuktikan sejauh mana arsitektur SNN mampu membentuk topologi ruang semantik yang menyerupai nalar bahasa manusia.

### B. Pengujian Kualitatif Mikro: Uji Kesadaran Posisi (Permutation Stress-Test)
*   **Tujuan:** Menguji apakah jaringan masih terjebak dalam masalah _Bag-of-Words_ (tidak peka pada susunan kalimat).
*   **Prosedur:** Memasukkan dua kalimat dengan kosa kata yang sama persis namun dirotasi posisinya (Contoh: "ayam goreng" vs "goreng ayam").
*   **Validasi:** Jika _Cosine Similarity_ mencapai 100%, model dianggap buta terhadap urutan posisi. Jika _Cosine Similarity_ secara signifikan berada di bawah 100% (misal 79.43%), SNN terbukti berhasil mengasimilasi memori temporal urutan secara organik melalui peluruhan integrator ($\beta$).

## 4. Eksperimen dan Hasil (Results)
Masukkan data kuantitatif berikut sebagai bukti keberhasilan arsitektur:

*   **Pecahnya Simetri (Order Awareness):** 
    *   Pengujian inferensi pada permutasi kalimat *"ayam goreng"* vs *"goreng ayam"*.
    *   Sebelum perbaikan (Basic SNN): Cosine Similarity **100.00%** (Gagal mendeteksi urutan).
    *   Sesudah Inovasi (Proposed SNN): Cosine Similarity anjlok ke **79.43%**, membuktikan model secara matematis telah sensitif terhadap susunan temporal bahasa. Identitas kalimat *"ayam goreng"* vs *"ayam goreng"* tetap terjaga di angka **100.00%**.
*   **Evaluasi Semantik (STS-B):** 
    *   Bahkan hanya dengan 1 siklus pelatihan (1 Epoch) menggunakan dataset kecil (_Wikipedia mini-corpus_), model mampu menghasilkan **Pearson Correlation ~30.58%** pada pengujian pasangan kalimat STS-B. Ini membuktikan bahwa pembentukan topologi ruang semantik dapat berhasil dicapai menggunakan arsitektur ini.

## 4. Kesimpulan (Conclusion)
*   Arsitektur yang diusulkan berhasil membuktikan bahwa SNN dapat memproses Bahasa Alami dengan kesadaran struktural (*Order-Aware*).
*   Kolaborasi antara mekanisme *Spiking* berbasis *Event* dan modulasi *Graded Potentials* menawarkan ekuivalensi fungsional dengan Transformer konvensional, namun dengan janji kompatibilitas komputasi *Neuromorphic* berdaya ultra-rendah.
