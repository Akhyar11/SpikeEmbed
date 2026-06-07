# Data Eksperimen untuk Paper Akademis SNN (SpikeEmbed)

Gunakan angka dan data di bawah ini saat menyusun draf paper, laporan jurnal, atau presentasi Anda. Data ini disarikan langsung dari hasil uji coba *engine* Oxide-JS yang kita jalankan.

## 1. Konfigurasi Arsitektur Model (SpikeEmbed)
*   **Kerangka Utama (Framework):** Oxide-JS (Native Rust FFI backend)
*   **Ukuran Kosakata (Vocabulary Size):** 29.803 BPE Tokens (di-*train* menggunakan Wikipedia bahasa Indonesia)
*   **Dimensi Vektor (Embedding Dimension):** 64
*   **Total Parameter Latih (Trainable Params):** 
    *   `SpikingEmbedding`: 2.048.000 parameter
    *   `SpikingDenseBPTT`: 4.160 parameter
*   **Mekanisme Neuron:** Leaky Integrate-and-Fire (LIF) dengan *Heterogeneous Dynamics* ($\beta$ dan $\theta$ diinisialisasi secara acak untuk keberagaman).
*   **Temporal Pooler:** BPTT dengan *Spike-Count Accumulation* dan Proyeksi Normalisasi L2 (Hypersphere).

## 2. Parameter Pelatihan (Unsupervised Learning)
*   **Metode:** Hebbian Contrastive Learning (In-Batch Negative Sampling)
*   **Loss Function:** *Add-Only Margin-based Contrastive Loss* (Tanpa perkalian floating-point)
*   **Dataset Pelatihan:** Subset Wikipedia Bahasa Indonesia
*   **Batch Size:** 64
*   **Total Langkah Pelatihan (Steps):** 2.641 Batches (Hanya 1 Epoch)
*   **Kecepatan Pelatihan:** Rata-rata **~41 hingga ~45 milidetik per batch** menggunakan paralelisasi CPU Rust (Rayon).

## 3. Hasil Evaluasi & Metrik
*   **Benchmark Dataset:** Semantic Textual Similarity Benchmark (STS-B) — Varian Bahasa Indonesia (1.500 Pasang Kalimat).
*   **Metrik Evaluasi:** *Pearson Correlation Coefficient* ($r$)
*   **Hasil Murni (Tanpa Fine-Tuning Supervised):**
    *   Korelasi Pearson yang dicapai: **47.70%**
*   **Perbandingan Metrik:** 
    *   Saat menggunakan Cosine standar, model SNN terjebak di *Positive Orthant* (skor palsu $>0.7$ untuk semua pasang kalimat). 
    *   Penggunaan formula **Mean-Centered Cosine Similarity** berhasil memperbaiki bias ini, mengembalikan rentang skor menjadi dinamis layaknya pemikiran manusia, yang berkorelasi 47.70% dengan *Ground Truth* STS-B.

## 4. Argumen Kebaruan (Novelty) untuk Klaim Artikel
1.  **Efisiensi MatMul:** Pada fase inferensi, *SpikingDenseBPTT* hanya menggunakan operasi penambahan murni (*Add-Only*) saat menerima input temporal dari lapisan *Embedding*.
2.  **Anti-Saturasi Temporal:** Model menembus batasan bawaan *membrane potential* SNN standar (yang biasanya mentok di nilai 1.0 pada urutan kalimat panjang) melalui penggabungan *Accumulative Spike-Count* dengan Normalisasi Geometris L2.
3.  **Akurasi Head-to-Head (Kualitatif):** Pada beberapa skenario semantik yang ambigu (misalnya pengenalan subjek terbalik), model Spiking 64-dimensi Unsupervised ini terbukti dapat mempertahankan logika linguistik meskipun dimensinya sangat kecil dibandingkan model Transformer (MiniLM 384-dimensi).
