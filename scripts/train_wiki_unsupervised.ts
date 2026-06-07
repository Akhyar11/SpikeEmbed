import { Matrix, BPETokenizer } from "@oxide-js/core";
import { SpikingSentenceEmbedder } from "../models/SpikingSentenceEmbedder.js";
import * as fs from 'fs';
import * as readline from 'readline';

// Fungsi Trik Unsupervised: Membuat Positive Sample (P+) dengan Noise/Masking (Aggressive)
function corruptSentence(sentence: string): string {
    const words = sentence.split(" ");
    // Tingkatkan kerusakan menjadi 25% (sangat kuat)
    if (words.length > 3) {
        const dropCount = Math.max(2, Math.floor(words.length * 0.25));
        for (let i = 0; i < dropCount; i++) {
            if (words.length <= 2) break;

            const dropType = Math.random();
            const targetIdx = Math.floor(Math.random() * words.length);

            if (dropType < 0.4) {
                // 40% Peluang: Hapus 1 kata penuh
                words.splice(targetIdx, 1);
            } else if (dropType < 0.8) {
                // 40% Peluang: Hapus 1 huruf di dalam kata (Typo)
                const targetWord = words[targetIdx];
                if (targetWord.length > 3) {
                    const charIdx = Math.floor(Math.random() * targetWord.length);
                    words[targetIdx] = targetWord.slice(0, charIdx) + targetWord.slice(charIdx + 1);
                } else {
                    words.splice(targetIdx, 1);
                }
            } else {
                // 20% Peluang: Duplikasi/Pengulangan kata (Gagap)
                words.splice(targetIdx, 0, words[targetIdx]);
            }
        }
    }
    return words.join(" ");
}

async function main() {
    console.log("Memuat BPETokenizer lokal dari vocab.json...");
    const tokenizer = BPETokenizer.load("./models/vocab.json");
    const vocabSize = tokenizer.getVocabularyCapacity();

    // Hyperparameters
    const d_model = 64;
    const sequenceLength = 64;
    const numPairs = 32; // Jumlah pasangan Q dan P+ per batch
    const batchSize = numPairs * 2; // Total kalimat dalam satu batch (contoh: 64)
    const timeSteps = 5;
    const learningRate = 0.01;
    const epochs = 1;

    const padTokenId = tokenizer.getPadId();
    console.log(`Menginisialisasi SpikingSentenceEmbedder (d_model=${d_model}, seqLen=${sequenceLength}, padId=${padTokenId})...`);
    const model = new SpikingSentenceEmbedder(vocabSize, d_model, sequenceLength, padTokenId);
    model.build(batchSize);
    model.summary();

    // Matriks Broadcast B untuk Local Learning Layer 1 (Identity Matrix)
    const B_emb_data = new Float32Array(d_model * d_model).fill(0);
    for (let i = 0; i < d_model; i++) {
        B_emb_data[i * d_model + i] = 1.0;
    }
    const B_emb = Matrix.fromFlat(B_emb_data, [d_model, d_model]);

    console.log("\nMemproses dataset dan Tokenisasi (Pre-tokenize)...");
    const datasetInputs: Matrix[] = [];
    const fileStream = fs.createReadStream('./dataset/mini_corpus20mb.txt');
    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

    let lineCount = 0;
    let batchPairs: { q: number[], p: number[] }[] = [];

    for await (const line of rl) {
        let Q = line.trim();
        if (!Q || Q.length < 20) continue; // Lewati kalimat kosong atau terlalu pendek

        // POTONG KALIMAT AGAR SESUAI DENGAN WINDOW TOKENIZER (20 kata ~ 32 token)
        let words = Q.split(" ");
        if (words.length > 20) {
            // Ambil jendela 20 kata secara ACAK (bisa di awal, tengah, atau akhir kalimat)
            const startIdx = Math.floor(Math.random() * (words.length - 20));
            words = words.slice(startIdx, startIdx + 20);
        }
        Q = words.join(" ");

        const P_plus = corruptSentence(Q);  // Positive Sample

        // 1. Tokenisasi Teks (Menggunakan BPETokenizer lokal)
        const encodedQ = tokenizer.encode(Q);
        const encodedP = tokenizer.encode(P_plus);

        // Pad sequence agar panjangnya persis sequenceLength
        const tokensQ = tokenizer.padSequence(encodedQ, sequenceLength);
        const tokensP = tokenizer.padSequence(encodedP, sequenceLength);

        batchPairs.push({ q: tokensQ as number[], p: tokensP as number[] });

        // Jika sudah mencapai numPairs, bentuk 1 matrix batch
        if (batchPairs.length === numPairs) {
            const inputData = new Float32Array(batchSize * sequenceLength);
            for (let i = 0; i < numPairs; i++) {
                // Set Q di paruh pertama batch
                inputData.set(batchPairs[i].q, i * sequenceLength);
                // Set P+ di paruh kedua batch
                inputData.set(batchPairs[i].p, (numPairs + i) * sequenceLength);
            }
            datasetInputs.push(Matrix.fromFlat(inputData, [batchSize * sequenceLength]));
            batchPairs = [];
        }

        lineCount++;
        if (lineCount % 5000 === 0) {
            console.log(`  > Pre-tokenized ${lineCount} kalimat...`);
        }
    }

    console.log(`Selesai Pre-tokenize! Total data latih: ${datasetInputs.length} batches.\n`);
    console.log("Memulai Pelatihan: Unsupervised Contrastive Learning & Local Error Learning");

    let bestLoss = Infinity;
    let patienceCounter = 0;
    const patienceLimit = 2; // Berhenti jika 2 epoch tidak ada peningkatan loss

    for (let epoch = 1; epoch <= epochs; epoch++) {
        let epochLossL1 = 0;
        let epochLossL2 = 0;
        let epochLossPooler = 0;
        let iterCount = 0;
        let t0_epoch = performance.now();
        let t0_batch = performance.now();

        // Loop melintasi dataset yang sudah di-pretokenize di memori
        for (let i = 0; i < datasetInputs.length; i++) {
            const inputs = datasetInputs[i];

            // 3. Reset State Potensial Membran SNN sebelum menyuapi sequence baru
            model.resetState();

            // 4. Single-Shot Forward & BPTT (Dinamika Temporal terpusat di Sekuens BPTT)
            const result = model.forwardAndLearnLocal(inputs, B_emb, learningRate);

            epochLossL1 += result.localLoss1;
            epochLossL2 += result.localLoss2;
            epochLossPooler += result.poolerLoss;
            iterCount++;

            if (iterCount % 10 === 0) {
                const t1 = performance.now();
                const msPerIter = ((t1 - t0_batch) / 10).toFixed(1);
                const progress = ((iterCount / datasetInputs.length) * 100).toFixed(1);

                // Menggunakan \r untuk menimpa baris yang sama, sehingga terminal tetap bersih
                process.stdout.write(`\r[Epoch ${epoch}/${epochs}] Progress: ${progress}% (${iterCount}/${datasetInputs.length}) | L1: ${(epochLossL1 / iterCount).toFixed(2)} | L2: ${(epochLossL2 / iterCount).toFixed(2)} | BPTT/L3: ${(epochLossPooler / iterCount).toFixed(2)} | ${msPerIter} ms/step `);

                t0_batch = performance.now();
            }
        }

        const totalEpochTime = ((performance.now() - t0_epoch) / 1000).toFixed(2);
        const avgLossL2 = epochLossL2 / iterCount;

        console.log(`\n[HASIL] Epoch ${epoch}/${epochs} | Rata-rata L1 Loss: ${(epochLossL1 / iterCount).toFixed(4)} | Rata-rata L2 Loss: ${avgLossL2.toFixed(4)} | Rata-rata BPTT Loss: ${(epochLossPooler / iterCount).toFixed(4)} | Waktu Total: ${totalEpochTime} s\n`);

        console.log(`Menyimpan checkpoint model untuk Epoch ${epoch}...`);
        const modelConfig = {
            embedding_weights: Array.from(model.embedding.embeddings!._data),
            kernelQ: Array.from(model.attention.kernelQ!._data),
            kernelK: Array.from(model.attention.kernelK!._data),
            kernelV: Array.from(model.attention.kernelV!._data),
            d_model,
            sequenceLength,
            vocabSize
        };
        fs.writeFileSync('./models/spiking_model_weights.json', JSON.stringify(modelConfig));

        // Early Stopping Logic
        // Kita pantau L2 Loss karena ini merepresentasikan ketajaman jarak Contrastive Learning
        if (avgLossL2 < bestLoss) {
            bestLoss = avgLossL2;
            patienceCounter = 0;
        } else {
            patienceCounter++;
            console.log(`[Early Stopping Warning] Tidak ada peningkatan signifikan. Patience: ${patienceCounter}/${patienceLimit}`);
            if (patienceCounter >= patienceLimit) {
                console.log(`\n🚨 [Early Stopping] Menghentikan pelatihan di Epoch ${epoch}! Loss tidak membaik selama ${patienceLimit} Epoch.`);
                break;
            }
        }
    }

    console.log("\nProses penulisan script implementasi selesai.");
}

main().catch(console.error);
