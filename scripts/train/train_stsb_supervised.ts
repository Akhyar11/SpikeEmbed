import { Matrix, BPETokenizer } from "@oxide-js/core";
import { SpikingSentenceEmbedder } from "../../models/SpikingSentenceEmbedder.js";
import * as fs from 'fs';

async function main() {
    console.log("=== SUPERVISED FINE-TUNING (HUMAN LABELS STS-B) ===");
    console.log("Memuat BPETokenizer lokal dari vocab.json...");
    const tokenizer = BPETokenizer.load("./models/vocab.json");
    const vocabSize = tokenizer.getVocabularyCapacity();

    // Hyperparameters
    const d_model = 64;
    const sequenceLength = 64;
    const learningRate = 0.005; 
    const epochs = 10; // Kita perbanyak epoch karena datanya kecil (5.700 pasang)
    const numPairs = 32; 
    const batchSize = numPairs * 2; 

    // 1. Memuat Model SNN
    const padTokenId = tokenizer.getPadId();
    const model = new SpikingSentenceEmbedder(vocabSize, d_model, sequenceLength, padTokenId);
    model.build(batchSize);

    // Muat Bobot Unsupervised (Transfer Learning)
    // KITA MATIKAN SEMENTARA SESUAI PERMINTAAN: Latih murni dari NOL (From Scratch)
    /*
    if (fs.existsSync('./models/spiking_model_weights.json')) {
        console.log("Memuat Pre-trained Weights dari Unsupervised Learning...");
        const weightsStr = fs.readFileSync('./models/spiking_model_weights.json', 'utf8');
        const config = JSON.parse(weightsStr);
        model.embedding.embeddings!._data.set(config.embedding_weights);
        model.attention.kernelQ!._data.set(config.kernelQ);
        model.attention.kernelK!._data.set(config.kernelK);
        model.attention.kernelV!._data.set(config.kernelV);
    }
    */

    const B_emb_data = new Float32Array(d_model * d_model).fill(0);
    for (let i = 0; i < d_model; i++) B_emb_data[i * d_model + i] = 1.0;
    const B_emb = Matrix.fromFlat(B_emb_data, [d_model, d_model]);

    // 2. Memuat Dataset Supervised (STS-B Manusia)
    console.log("\nMemuat dataset Human Labels (STS-B Train)...");
    const csvData = fs.readFileSync('./dataset/data_stsb.train.modified_indo.csv', 'utf8');
    const lines = csvData.split('\n').slice(1); // Buang header

    const dataset: any[] = [];
    for (const line of lines) {
        if (!line.trim()) continue;
        // Parsing CSV sederhana (asumsi tidak ada koma di dalam teks)
        // Jika ada koma di teks, kita ambil kolom terakhir sebagai skor
        const parts = line.split(',');
        const scoreStr = parts.pop();
        if (!scoreStr) continue;
        
        const score = parseFloat(scoreStr.trim());
        if (isNaN(score)) continue;

        // Gabungkan kembali sisa array jika ada koma di dalam kalimat
        const half = Math.floor(parts.length / 2);
        const s1 = parts.slice(0, half).join(',');
        const s2 = parts.slice(half).join(',');

        if (s1.length > 5 && s2.length > 5) {
            dataset.push({ q: s1, p: s2, score: score });
        }
    }

    console.log(`Selesai! Total data latih Supervised: ${dataset.length} pasang kalimat.\n`);
    console.log("Memulai Pelatihan Supervised Sebenarnya...");

    for (let epoch = 1; epoch <= epochs; epoch++) {
        let totalLoss = 0;
        let t0_epoch = performance.now();
        let batchCount = 0;

        const shuffledData = dataset.sort(() => 0.5 - Math.random());

        for (let i = 0; i < shuffledData.length; i += numPairs) {
            const batchPairs = shuffledData.slice(i, i + numPairs);
            if (batchPairs.length < numPairs) break;

            const inputData = new Float32Array(batchSize * sequenceLength);
            
            for (let j = 0; j < numPairs; j++) {
                const pair = batchPairs[j];
                const tQ = tokenizer.padSequence(tokenizer.encode(pair.q), sequenceLength) as number[];
                const tP = tokenizer.padSequence(tokenizer.encode(pair.p), sequenceLength) as number[];
                
                inputData.set(tQ, j * sequenceLength);
                inputData.set(tP, (numPairs + j) * sequenceLength);
            }

            const inputs = Matrix.fromFlat(inputData, [batchSize * sequenceLength]);
            model.resetState();
            
            // Ekstrak Human Score sebagai Target Loss
            const humanScores = batchPairs.map(t => t.score);
            
            // Asumsi kalimat acak di dalam batch memiliki kemiripan 0.0 (Negatif sempurna)
            const assumedNegScores = new Array(numPairs).fill(0.0);

            // Injeksi Kunci Jawaban Manusia!
            const result = model.forwardAndLearnLocal(inputs, B_emb, learningRate, humanScores, assumedNegScores);
            
            totalLoss += result.poolerLoss;
            batchCount++;

            if (batchCount % 10 === 0) {
                process.stdout.write(`\r  [Epoch ${epoch}] Batch ${batchCount} | Rata-rata Loss: ${(totalLoss / batchCount).toFixed(4)}`);
            }
        }

        const msEpoch = ((performance.now() - t0_epoch) / 1000).toFixed(2);
        console.log(`\n>>> SELESAI EPOCH ${epoch} | Total Loss Akhir: ${(totalLoss / batchCount).toFixed(4)} | Waktu: ${msEpoch} detik <<<\n`);
    }

    // 4. Simpan Bobot (Supervised Weights)
    console.log("Menyimpan bobot Human-Supervised SNN...");
    const configToSave = {
        d_model: d_model,
        sequenceLength: sequenceLength,
        vocabSize: vocabSize,
        embedding_weights: Array.from(model.embedding.embeddings!._data),
        kernelQ: Array.from(model.attention.kernelQ!._data),
        kernelK: Array.from(model.attention.kernelK!._data),
        kernelV: Array.from(model.attention.kernelV!._data),
    };
    fs.writeFileSync('./models/spiking_model_weights_supervised.json', JSON.stringify(configToSave));
    console.log("Berhasil disimpan di: ./models/spiking_model_weights_supervised.json");
}

main().catch(console.error);
