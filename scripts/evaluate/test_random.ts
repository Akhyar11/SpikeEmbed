import { Matrix, BPETokenizer } from "@oxide-js/core";
import { SpikingSentenceEmbedder } from "../../models/SpikingSentenceEmbedder.js";
import { pipeline } from '@xenova/transformers';
import * as fs from 'fs';

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

function meanCenteredCosine(vecA: Float32Array, vecB: Float32Array): number {
    let meanA = 0, meanB = 0;
    for (let i = 0; i < vecA.length; i++) { meanA += vecA[i]; meanB += vecB[i]; }
    meanA /= vecA.length; meanB /= vecB.length;

    let dot = 0.0, normA = 0.0, normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        const valA = vecA[i] - meanA;
        const valB = vecB[i] - meanB;
        dot += valA * valB;
        normA += valA * valA;
        normB += valB * valB;
    }
    if (normA === 0 || normB === 0) return 0;
    const centeredCosine = dot / (Math.sqrt(normA) * Math.sqrt(normB));
    return Math.max(0, centeredCosine);
}

function pearsonCorrelation(x: number[], y: number[]): number {
    const n = x.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (let i = 0; i < n; i++) {
        sumX += x[i]; sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i]; sumY2 += y[i] * y[i];
    }
    const num = n * sumXY - sumX * sumY;
    const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    return den === 0 ? 0 : num / den;
}

async function main() {
    console.log("=== BENCHMARK ROBUST: SNN (RANDOM/UNTRAINED) vs MINILM ===");
    const tokenizer = BPETokenizer.load("./models/vocab.json");
    
    // Gunakan dimensi standar karena kita tidak memuat config
    const d_model = 64;
    const sequenceLength = 512;
    const vocabSize = tokenizer.getVocabularyCapacity();

    const snnModel = new SpikingSentenceEmbedder(vocabSize, d_model, sequenceLength, tokenizer.getPadId());
    snnModel.build(2);
    
    // TIDAK ADA pemuatan bobot (murni inisialisasi acak dari nol)
    console.log("-> SNN diinisialisasi dengan bobot acak (Belum Dilatih Sama Sekali)!");

    const hfExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    const stsbRaw = fs.readFileSync('./dataset/sts-b_valid.json', 'utf8');
    const stsbPairs = JSON.parse(stsbRaw);

    const humanScores: number[] = [];
    const snnScores: number[] = [];
    const hfScores: number[] = [];

    let snnTime = 0;
    let hfTime = 0;
    const totalData = stsbPairs.length;

    console.log(`Memulai evaluasi penuh pada seluruh ${totalData} pasangan kalimat...`);

    for (let i = 0; i < totalData; i++) {
        const pair = stsbPairs[i];
        humanScores.push(pair.score);

        // --- SNN INFERENCE ---
        const startSNN = performance.now();
        const encoded1 = tokenizer.encode(pair.sentence1.toLowerCase());
        const encoded2 = tokenizer.encode(pair.sentence2.toLowerCase());
        const tokens1 = tokenizer.padSequence(encoded1, sequenceLength);
        const tokens2 = tokenizer.padSequence(encoded2, sequenceLength);

        const inputData = new Float32Array(2 * sequenceLength);
        inputData.set(tokens1, 0); inputData.set(tokens2, sequenceLength);
        const inputs = Matrix.fromFlat(inputData, [2 * sequenceLength] as any);

        snnModel.resetState();
        const finalSpikes = snnModel.infer(inputs);
        const simSNN = meanCenteredCosine(finalSpikes._data.slice(0, d_model), finalSpikes._data.slice(d_model, 2 * d_model));
        snnTime += (performance.now() - startSNN);
        snnScores.push(simSNN);

        // --- TRANSFORMER INFERENCE ---
        const startHF = performance.now();
        const hfOutA = await hfExtractor(pair.sentence1, { pooling: 'mean', normalize: true });
        const hfOutB = await hfExtractor(pair.sentence2, { pooling: 'mean', normalize: true });
        const simHF = standardCosine(hfOutA.data, hfOutB.data);
        hfTime += (performance.now() - startHF);
        hfScores.push(simHF);

        if ((i + 1) % 100 === 0) console.log(`  > Dievaluasi ${i + 1} pasangan...`);
    }

    const pearsonSNN = pearsonCorrelation(snnScores, humanScores);
    const pearsonHF = pearsonCorrelation(hfScores, humanScores);

    console.log(`\n=== HASIL AKHIR (KUANTITATIF) ===`);
    console.log(`[Pearson Correlation STS-B]`);
    console.log(`- SNN Buatan Anda : ${(pearsonSNN * 100).toFixed(2)}%`);
    console.log(`- MiniLM (SOTA)   : ${(pearsonHF * 100).toFixed(2)}%`);
    console.log(`\n[Latency Rata-Rata (per pasangan kalimat)]`);
    console.log(`- SNN Buatan Anda : ${(snnTime / totalData).toFixed(2)} ms`);
    console.log(`- MiniLM (SOTA)   : ${(hfTime / totalData).toFixed(2)} ms`);
    console.log(`=================================`);
}

main().catch(console.error);
