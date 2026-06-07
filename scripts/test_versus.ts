import { Matrix, BPETokenizer } from "@oxide-js/core";
import { SpikingSentenceEmbedder } from "../models/SpikingSentenceEmbedder.js";
import { pipeline } from '@xenova/transformers';
import * as fs from 'fs';

// Helper: Cosine Similarity untuk Transformer (murni)
function standardCosine(vecA: any, vecB: any): number {
    let dot = 0.0, normA = 0.0, normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        dot += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Helper: Cosine Similarity untuk SNN (Mean-Centered untuk mengkoreksi Positive Orthant)
function snnCosineSimilarity(vecA: Float32Array, vecB: Float32Array): number {
    let meanA = 0, meanB = 0;
    for (let i = 0; i < vecA.length; i++) { meanA += vecA[i]; meanB += vecB[i]; }
    meanA /= vecA.length; meanB /= vecB.length;

    let dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        const valA = vecA[i] - meanA;
        const valB = vecB[i] - meanB;
        dotProduct += valA * valB;
        normA += valA * valA;
        normB += valB * valB;
    }
    if (normA === 0 || normB === 0) return 0;
    const centeredCosine = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return Math.max(0, centeredCosine);
}

async function main() {
    console.log("============================================================");
    console.log("          PERSIAPAN PERTANDINGAN: SNN vs MINI-LM            ");
    console.log("============================================================");

    // 1. MEMUAT SNN (LOKAL BUATAN SENDIRI)
    console.log("\n[1/2] Memuat Spiking Neural Network (SNN) Anda...");
    const tokenizer = BPETokenizer.load("./models/vocab.json");
    const weightsStr = fs.readFileSync('./models/spiking_model_weights.json', 'utf8');
    const config = JSON.parse(weightsStr);
    const { d_model, sequenceLength, vocabSize, embedding_weights, kernelQ, kernelK, kernelV } = config;

    const padTokenId = tokenizer.getPadId();
    const snnModel = new SpikingSentenceEmbedder(vocabSize, d_model, sequenceLength, padTokenId);
    snnModel.build(2);
    
    snnModel.embedding.embeddings!._data.set(embedding_weights);
    snnModel.attention.kernelQ!._data.set(kernelQ);
    snnModel.attention.kernelK!._data.set(kernelK);
    snnModel.attention.kernelV!._data.set(kernelV);
    console.log("      -> SNN Siap! (Dimensi: 64, Unsupervised 1 Epoch)");

    // 2. MEMUAT TRANSFORMER (STATE-OF-THE-ART DARI INTERNET)
    console.log("\n[2/2] Mengunduh/Memuat MiniLM-L6-v2 (Transformer State-of-the-Art)...");
    const hfExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log("      -> MiniLM Siap! (Dimensi: 384, Pre-trained Jutaan Kalimat)");

    const testPairs = [
        { s1: "Sebuah pesawat sedang lepas landas.", s2: "Sebuah pesawat terbang sedang lepas landas.", truth: 1.0 },
        { s1: "Seorang pria sedang memainkan seruling besar.", s2: "Seorang pria sedang memainkan seruling.", truth: 0.76 },
        { s1: "Seorang pria sedang mengoleskan keju parut di atas pizza.", s2: "Seorang pria sedang mengoleskan keju parut di atas pizza yang belum matang.", truth: 0.76 },
        { s1: "Tiga pria sedang bermain catur.", s2: "Dua orang pria sedang bermain catur.", truth: 0.52 },
        { s1: "Seorang pria sedang memainkan cello.", s2: "Seorang pria yang sedang duduk sedang memainkan cello.", truth: 0.85 },
        { s1: "Beberapa orang berkelahi.", s2: "Dua orang berkelahi.", truth: 0.85 },
        { s1: "Seorang pria sedang merokok.", s2: "Seorang pria sedang berseluncur.", truth: 0.1 },
        { s1: "Pria itu sedang memainkan piano.", s2: "Pria itu sedang memainkan gitar.", truth: 0.32 },
        { s1: "Seorang pria sedang bermain gitar dan bernyanyi.", s2: "Seorang wanita memainkan gitar akustik dan bernyanyi.", truth: 0.44 },
        { s1: "Seseorang melempar seekor kucing ke langit-langit.", s2: "Seseorang melempar kucing ke langit-langit.", truth: 1.0 },
        { s1: "Pria itu memukul pria lainnya dengan tongkat.", s2: "Pria itu memukul pria lainnya dengan tongkat.", truth: 1.0 }
    ];

    console.log("\n============================================================");
    console.log("                 HASIL PERTANDINGAN (SNN vs SOTA)           ");
    console.log("============================================================\n");

    for (let i = 0; i < testPairs.length; i++) {
        const pair = testPairs[i];

        // --- PREDIKSI SNN ---
        const tokensA = tokenizer.padSequence(tokenizer.encode(pair.s1.toLowerCase()), sequenceLength);
        const tokensB = tokenizer.padSequence(tokenizer.encode(pair.s2.toLowerCase()), sequenceLength);

        const inputData = new Float32Array(2 * sequenceLength);
        inputData.set(tokensA, 0);
        inputData.set(tokensB, sequenceLength);
        const inputs = Matrix.fromFlat(inputData, [2 * sequenceLength]);

        snnModel.resetState();
        const finalSpikes = snnModel.infer(inputs);

        const repA = finalSpikes._data.slice(0, d_model);
        const repB = finalSpikes._data.slice(d_model, 2 * d_model);
        const simSNN = snnCosineSimilarity(repA, repB);

        // --- PREDIKSI HUGGINGFACE TRANSFORMER (MiniLM) ---
        const hfOutA = await hfExtractor(pair.s1, { pooling: 'mean', normalize: true });
        const hfOutB = await hfExtractor(pair.s2, { pooling: 'mean', normalize: true });
        const simHF = standardCosine(hfOutA.data, hfOutB.data);

        console.log(`[Kasus ${i + 1}]`);
        console.log(`A: "${pair.s1}"`);
        console.log(`B: "${pair.s2}"`);
        console.log(`=> Ground Truth STS-B   : ${(pair.truth * 100).toFixed(2)}%`);
        console.log(`=> SNN Buatan Anda      : ${(simSNN * 100).toFixed(2)}%`);
        console.log(`=> MiniLM (Internet)    : ${(simHF * 100).toFixed(2)}%`);
        console.log("------------------------------------------------------------");
    }
}

main().catch(console.error);
