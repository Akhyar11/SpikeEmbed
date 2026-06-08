import { Matrix, BPETokenizer } from "@oxide-js/core";
import { SpikingSentenceEmbedder } from "../../models/SpikingSentenceEmbedder.js";
import * as fs from 'fs';
import * as readline from 'readline';



function cosineSimilarity(vecA: Float32Array, vecB: Float32Array): number {
    // MEAN-CENTERING: Menggeser vektor agar SNN bisa memiliki kemiripan negatif/rendah
    // Karena Spike selalu >= 0, vektor SNN terjebak di kuadran positif, membuat skor selalu tinggi secara semu.
    let meanA = 0, meanB = 0;
    for (let i = 0; i < vecA.length; i++) { meanA += vecA[i]; meanB += vecB[i]; }
    meanA /= vecA.length; meanB /= vecB.length;

    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        const valA = vecA[i] - meanA;
        const valB = vecB[i] - meanB;
        dotProduct += valA * valB;
        normA += valA * valA;
        normB += valB * valB;
    }
    if (normA === 0 || normB === 0) return 0;
    // Gunakan cosine murni (di-clamp ke 0 agar tidak ada persentase negatif)
    const centeredCosine = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return Math.max(0, centeredCosine);
}

async function main() {
    console.log("Memuat BPETokenizer lokal dari vocab.json...");
    const tokenizer = BPETokenizer.load("./models/vocab.json");

    console.log("Memuat bobot model SNN (SpikingSentenceEmbedder)...");
    const weightsStr = fs.readFileSync('./models/spiking_model_weights_distilled.json', 'utf8');
    const config = JSON.parse(weightsStr);
    const { d_model, sequenceLength, vocabSize, embedding_weights, pos_embedding_weights, kernelQ, kernelK, kernelV } = config;

    const padTokenId = tokenizer.getPadId();
    const model = new SpikingSentenceEmbedder(vocabSize, d_model, sequenceLength, padTokenId);
    // Batch size 2 karena kita akan membandingkan 2 kalimat
    model.build(2);

    // Muat bobot yang sudah dilatih
    model.embedding.embeddings!._data.set(embedding_weights);
    if (pos_embedding_weights) {
        model.posEmbedding.embeddings!._data.set(pos_embedding_weights);
    }
    model.attention.kernelQ!._data.set(kernelQ);
    model.attention.kernelK!._data.set(kernelK);
    model.attention.kernelV!._data.set(kernelV);

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const askQuestion = (query: string): Promise<string> => {
        return new Promise(resolve => rl.question(query, resolve));
    };

    console.log("\n==============================================");
    console.log("TEST INFERENSI SPIKING NEURAL NETWORK (SNN)");
    console.log("==============================================\n");

    while (true) {
        const sentenceA = await askQuestion("Kalimat A (atau ketik 'exit' untuk keluar): ");
        if (sentenceA.toLowerCase() === 'exit') break;

        const sentenceB = await askQuestion("Kalimat B: ");

        // 1. Tokenisasi
        const tokensA = tokenizer.padSequence(tokenizer.encode(sentenceA.toLowerCase()), sequenceLength);
        const tokensB = tokenizer.padSequence(tokenizer.encode(sentenceB.toLowerCase()), sequenceLength);

        // 2. Siapkan Matrix Input [Batch=2, SeqLen]
        const inputData = new Float32Array(2 * sequenceLength);
        inputData.set(tokensA, 0);
        inputData.set(tokensB, sequenceLength);
        const inputs = Matrix.fromFlat(inputData, [2 * sequenceLength]);

        // 3. Reset state LIF (potensial membran)
        model.resetState();

        // 4. Inference (Internal Time-Stepping + Pooling)
        const finalSpikes = model.infer(inputs); // [batchSize, d_model]

        // 5. Ekstraksi vektor kalimat (langsung dari output SpikingDense)
        const repA = finalSpikes._data.slice(0, d_model);
        const repB = finalSpikes._data.slice(d_model, 2 * d_model);

        // 6. Hitung Cosine Similarity
        const similarity = cosineSimilarity(repA, repB);

        console.log("\n----------------------------------------------");
        console.log("repA (5 nilai pertama):", repA.slice(0, 5));
        console.log("repB (5 nilai pertama):", repB.slice(0, 5));
        console.log(`Skor Kemiripan Semantik (Cosine): ${(similarity * 100).toFixed(2)}%`);
        console.log("----------------------------------------------\n");
    }

    rl.close();
}

main().catch(console.error);
