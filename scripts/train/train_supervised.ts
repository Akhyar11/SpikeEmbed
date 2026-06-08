import { Matrix, BPETokenizer } from "@oxide-js/core";
import { SpikingSentenceEmbedder } from "../../models/SpikingSentenceEmbedder.js";
import * as fs from 'fs';

async function main() {
    console.log("=== KNOWLEDGE DISTILLATION (SUPERVISED TRAINING) ===");
    console.log("Memuat BPETokenizer lokal dari vocab.json...");
    const tokenizer = BPETokenizer.load("./models/vocab.json");
    const vocabSize = tokenizer.getVocabularyCapacity();

    // Hyperparameters
    const d_model = 64;
    const sequenceLength = 64;
    const learningRate = 0.005; // Learning rate lebih kecil untuk Fine-Tuning
    const epochs = 5;
    const numPairs = 32; // Jumlah pasangan Q dan P per batch
    const batchSize = numPairs * 2; // Total 64 kalimat per batch

    // 1. Memuat Model SNN
    const padTokenId = tokenizer.getPadId();
    console.log(`Menginisialisasi SpikingSentenceEmbedder (d_model=${d_model}, seqLen=${sequenceLength})...`);
    const model = new SpikingSentenceEmbedder(vocabSize, d_model, sequenceLength, padTokenId);

    model.build(batchSize);

    // Muat Bobot Unsupervised (Transfer Learning / Pre-training)
    // if (fs.existsSync('./models/spiking_model_weights.json')) {
    //     console.log("Memuat Pre-trained Weights dari Unsupervised Learning...");
    //     const weightsStr = fs.readFileSync('./models/spiking_model_weights.json', 'utf8');
    //     const config = JSON.parse(weightsStr);
    //     model.embedding.embeddings!._data.set(config.embedding_weights);
    //     model.attention.kernelQ!._data.set(config.kernelQ);
    //     model.attention.kernelK!._data.set(config.kernelK);
    //     model.attention.kernelV!._data.set(config.kernelV);
    // }

    const B_emb_data = new Float32Array(d_model * d_model).fill(0);
    for (let i = 0; i < d_model; i++) B_emb_data[i * d_model + i] = 1.0;
    const B_emb = Matrix.fromFlat(B_emb_data, [d_model, d_model]);

    // 2. Memuat Dataset Supervised (Triplet dari Guru)
    console.log("\nMemuat dataset Teacher Labels dari teacher_distillation_dataset.json...");
    if (!fs.existsSync('./dataset/teacher_distillation_dataset.json')) {
        console.error("Dataset Distilasi belum ada! Harap jalankan 'generate_distillation_data.ts' terlebih dahulu.");
        process.exit(1);
    }
    const rawData = fs.readFileSync('./dataset/teacher_distillation_dataset.json', 'utf8');
    const dataset = JSON.parse(rawData);

    console.log(`Selesai! Total data latih: ${dataset.length} triplets.\n`);
    console.log("Memulai Pelatihan Supervised: Knowledge Distillation...");

    for (let epoch = 1; epoch <= epochs; epoch++) {
        let totalLoss = 0;
        let t0_epoch = performance.now();
        let batchCount = 0;

        // Acak urutan data
        const shuffledData = dataset.sort(() => 0.5 - Math.random());

        for (let i = 0; i < shuffledData.length; i += numPairs) {
            const batchTriplets = shuffledData.slice(i, i + numPairs);
            if (batchTriplets.length < numPairs) break; // Abaikan sisa data yang tidak genap 1 batch

            const inputData = new Float32Array(batchSize * sequenceLength);

            for (let j = 0; j < numPairs; j++) {
                const triplet = batchTriplets[j];
                const tQ = tokenizer.padSequence(tokenizer.encode(triplet.q), sequenceLength) as number[];
                const tP = tokenizer.padSequence(tokenizer.encode(triplet.p), sequenceLength) as number[];

                // Susunan In-Batch Negative: Paruh pertama adalah Q, paruh kedua adalah P
                inputData.set(tQ, j * sequenceLength);
                inputData.set(tP, (numPairs + j) * sequenceLength);
            }

            const inputs = Matrix.fromFlat(inputData, [batchSize * sequenceLength] as any);

            model.resetState();

            // Ekstrak skor guru
            const teacherPosScores = batchTriplets.map((t: any) => t.teacher_pos_score);
            const teacherNegScores = batchTriplets.map((t: any) => t.teacher_neg_score);

            const result = model.forwardAndLearnLocal(inputs, B_emb, learningRate, teacherPosScores, teacherNegScores);

            totalLoss += result.poolerLoss;
            batchCount++;

            if (batchCount % 10 === 0) {
                process.stdout.write(`\r  [Epoch ${epoch}] Batch ${batchCount} | Rata-rata Loss: ${(totalLoss / batchCount).toFixed(4)}`);
            }
        }

        const msEpoch = ((performance.now() - t0_epoch) / 1000).toFixed(2);
        console.log(`\n>>> SELESAI EPOCH ${epoch} | Total Loss Akhir: ${(totalLoss / batchCount).toFixed(4)} | Waktu: ${msEpoch} detik <<<\n`);
    }

    // 4. Simpan Bobot (Distilled Weights)
    console.log("Menyimpan bobot Supervised (Distilled) SNN...");
    const configToSave = {
        d_model: d_model,
        sequenceLength: sequenceLength,
        vocabSize: vocabSize,
        embedding_weights: Array.from(model.embedding.embeddings!._data),
        kernelQ: Array.from(model.attention.kernelQ!._data),
        kernelK: Array.from(model.attention.kernelK!._data),
        kernelV: Array.from(model.attention.kernelV!._data),
    };
    fs.writeFileSync('./models/spiking_model_weights_distilled.json', JSON.stringify(configToSave));
    console.log("Berhasil disimpan di: ./models/spiking_model_weights_distilled.json");
}

main().catch(console.error);
