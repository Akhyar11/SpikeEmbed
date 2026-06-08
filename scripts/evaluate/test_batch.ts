import { Matrix, BPETokenizer } from "@oxide-js/core";
import { SpikingSentenceEmbedder } from "../../models/SpikingSentenceEmbedder.js";
import * as fs from 'fs';

function cosineSimilarity(vecA: Float32Array, vecB: Float32Array): number {
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
    console.log("Memuat BPETokenizer...");
    const tokenizer = BPETokenizer.load("./models/vocab.json");

    console.log("Memuat bobot model SNN...");
    const weightsStr = fs.readFileSync('./models/spiking_model_weights_distilled.json', 'utf8');
    const config = JSON.parse(weightsStr);
    const { d_model, sequenceLength, vocabSize, embedding_weights, pos_embedding_weights, kernelQ, kernelK, kernelV } = config;

    const padTokenId = tokenizer.getPadId();
    const model = new SpikingSentenceEmbedder(vocabSize, d_model, sequenceLength, padTokenId);
    model.build(2);

    model.embedding.embeddings!._data.set(embedding_weights);
    if (pos_embedding_weights) {
        // Just in case it's in the config, though we removed it
        // model.posEmbedding.embeddings!._data.set(pos_embedding_weights);
    }
    model.attention.kernelQ!._data.set(kernelQ);
    model.attention.kernelK!._data.set(kernelK);
    model.attention.kernelV!._data.set(kernelV);

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
        { s1: "Pria itu memukul pria lainnya dengan tongkat.", s2: "Pria itu memukul pria lainnya dengan tongkat.", truth: 1 }
    ];

    console.log("\n============================================================");
    console.log("HASIL PREDIKSI BATCH SNN vs GROUND TRUTH STS-B");
    console.log("============================================================\n");

    for (let i = 0; i < testPairs.length; i++) {
        const pair = testPairs[i];
        // SNN lebih baik menerima huruf kecil karena vocabulary kita tidak terlalu besar
        const tokensA = tokenizer.padSequence(tokenizer.encode(pair.s1.toLowerCase()), sequenceLength);
        const tokensB = tokenizer.padSequence(tokenizer.encode(pair.s2.toLowerCase()), sequenceLength);

        const inputData = new Float32Array(2 * sequenceLength);
        inputData.set(tokensA, 0);
        inputData.set(tokensB, sequenceLength);
        const inputs = Matrix.fromFlat(inputData, [2 * sequenceLength]);

        model.resetState();
        const finalSpikes = model.infer(inputs);

        const repA = finalSpikes._data.slice(0, d_model);
        const repB = finalSpikes._data.slice(d_model, 2 * d_model);

        const sim = cosineSimilarity(repA, repB);

        console.log(`[Pasangan ${i + 1}]`);
        console.log(`A: ${pair.s1}`);
        console.log(`B: ${pair.s2}`);
        console.log(`Ground Truth : ${(pair.truth * 100).toFixed(2)}%`);
        console.log(`Prediksi SNN : ${(sim * 100).toFixed(2)}%`);
        console.log("------------------------------------------------------------");
    }
}

main().catch(console.error);
