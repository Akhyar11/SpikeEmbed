import { Matrix, BPETokenizer } from "@oxide-js/core";
import { SpikingSentenceEmbedder } from "../../models/SpikingSentenceEmbedder.js";
import * as fs from 'fs';

function cosineSimilarity(vecA: Float32Array, vecB: Float32Array): number {
    let meanA = 0, meanB = 0;
    for (let i = 0; i < vecA.length; i++) { meanA += vecA[i]; meanB += vecB[i]; }
    meanA /= vecA.length; meanB /= vecB.length;

    let dot = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        const valA = vecA[i] - meanA;
        const valB = vecB[i] - meanB;
        dot += valA * valB;
        normA += valA * valA;
        normB += valB * valB;
    }
    if (normA === 0 || normB === 0) return 0;
    const centeredCosine = dot / (Math.sqrt(normA) * Math.sqrt(normB));
    return Math.max(0, centeredCosine); // Hindari persentase negatif
}


// Helper: Pearson Correlation (Metrik Standar STS-B)
function pearsonCorrelation(x: number[], y: number[]): number {
    const n = x.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
    }
    const num = n * sumXY - sumX * sumY;
    const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    if (den === 0) return 0;
    return num / den;
}

async function main() {
    console.log("Memuat BPETokenizer...");
    const tokenizer = BPETokenizer.load("./models/vocab.json");
    
    console.log("Memuat bobot model SNN (SpikingSentenceEmbedder)...");
    const weightsStr = fs.readFileSync('./models/spiking_model_weights.json', 'utf8');
    const config = JSON.parse(weightsStr);
    
    const { d_model, sequenceLength, vocabSize, embedding_weights, pos_embedding_weights, kernelQ, kernelK, kernelV } = config;
    
    // Inisiasi Model
    const padTokenId = tokenizer.getPadId();
    const model = new SpikingSentenceEmbedder(vocabSize, d_model, sequenceLength, padTokenId);
    model.build(2); // batch size = 2 (sentence 1 & sentence 2)
    
    // Injeksi Bobot (Load Weights)
    model.embedding.embeddings!._data.set(embedding_weights);
    if (pos_embedding_weights) {
        model.embedding.embeddings!._data.set(pos_embedding_weights);
    }
    model.attention.kernelQ!._data.set(kernelQ);
    model.attention.kernelK!._data.set(kernelK);
    model.attention.kernelV!._data.set(kernelV);
    
    console.log("Memuat dataset evaluasi Semantic Textual Similarity (STS-B)...");
    const stsbRaw = fs.readFileSync('./dataset/sts-b_valid.json', 'utf8');
    const stsbPairs = JSON.parse(stsbRaw);
    
    const predictedScores: number[] = [];
    const humanScores: number[] = [];
    
    let evalCount = 0;
    const maxEval = 1500; // Kita batasi sebagian tes agar evaluasi tidak terlalu lama
    
    console.log(`Memulai evaluasi inference SNN pada ${Math.min(stsbPairs.length, maxEval)} pasangan kalimat...`);

    for (const pair of stsbPairs) {
        if (evalCount >= maxEval) break;

        const sent1 = pair.sentence1;
        const sent2 = pair.sentence2;
        // STS-B dataset ini terlihat menggunakan rentang [0, 1]
        const trueScore = pair.score; 

        // 1. Tokenisasi
        const encoded1 = tokenizer.encode(sent1);
        const encoded2 = tokenizer.encode(sent2);
        
        const tokens1 = tokenizer.padSequence(encoded1, sequenceLength);
        const tokens2 = tokenizer.padSequence(encoded2, sequenceLength);
        
        // 2. Format Input
        const inputData = new Float32Array(2 * sequenceLength);
        inputData.set(tokens1, 0);
        inputData.set(tokens2, sequenceLength);
        const inputs = Matrix.fromFlat(inputData, [2 * sequenceLength] as any);
        
        // 3. Reset Neuron
        model.resetState();
        
        // 4. Inference (Internal Time-Stepping + Pooling)
        const finalSpikes = model.infer(inputs); // [2, d_model]
        
        // 5. Ekstraksi Vektor (langsung dari output SpikingDense)
        const rep1 = finalSpikes._data.slice(0, d_model);
        const rep2 = finalSpikes._data.slice(d_model, 2 * d_model);
        
        // 6. Hitung Cosine Similarity
        const sim = cosineSimilarity(rep1, rep2);
        
        predictedScores.push(sim);
        humanScores.push(trueScore);
        
        evalCount++;
        if (evalCount % 100 === 0) {
            console.log(`  > Dievaluasi ${evalCount} pasangan kalimat...`);
        }
    }
    
    // 7. Hitung Pearson Correlation
    const pearson = pearsonCorrelation(predictedScores, humanScores);
    console.log(`\n==============================================`);
    console.log(`EVALUASI SELESAI`);
    console.log(`Total Pasangan: ${evalCount}`);
    console.log(`Pearson Correlation (STS-B): ${(pearson * 100).toFixed(2)}%`);
    console.log(`==============================================\n`);
}

main().catch(console.error);
