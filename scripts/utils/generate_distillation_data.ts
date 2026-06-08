import { pipeline } from '@xenova/transformers';
import * as fs from 'fs';

// Helper: Cosine Similarity
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

// Helper: Trik Unsupervised (Corrupt Sentence untuk Positive Sample)
function corruptSentence(sentence: string): string {
    const words = sentence.split(" ");
    if (words.length > 3) {
        const dropCount = Math.max(2, Math.floor(words.length * 0.25));
        for (let i = 0; i < dropCount; i++) {
            if (words.length <= 2) break;
            const targetIdx = Math.floor(Math.random() * words.length);
            words.splice(targetIdx, 1); // Hapus kata
        }
    }
    return words.join(" ");
}

async function main() {
    console.log("=== MEMBANGUN DATASET DISTILASI (FAST METHOD) ===");
    
    // 1. Muat Model Guru (Teacher)
    console.log("Memuat Teacher Model (MiniLM-L6-v2)...");
    const hfExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    // 2. Memuat Kalimat Mentah dari Korpus Anda
    console.log("Membaca corpus lokal (mini_corpus20mb.txt)...");
    const corpusRaw = fs.readFileSync('./dataset/mini_corpus20mb.txt', 'utf8');
    const allLines = corpusRaw.split('\n').filter((l: string) => l.trim().length > 20); // Ambil kalimat minimal 20 karakter
    
    const TOTAL_TRIPLETS = 15000; 
    const dataset: any[] = [];

    console.log(`Mulai memproses ${TOTAL_TRIPLETS} data Triplet (Q, P, N) dengan skor Guru...`);

    let t0 = performance.now();

    for (let i = 0; i < TOTAL_TRIPLETS; i++) {
        // 1. Tentukan Q (Anchor) secara acak
        const qIdx = Math.floor(Math.random() * allLines.length);
        const Q = allLines[qIdx].trim();

        // 2. Tentukan P (Positive) dengan cara merusak Q sedikit (Data Augmentation)
        const P = corruptSentence(Q);

        // 3. Tentukan N (Negative) dengan mengambil kalimat acak lain dari korpus
        let nIdx = Math.floor(Math.random() * allLines.length);
        while (nIdx === qIdx) { nIdx = Math.floor(Math.random() * allLines.length); }
        const N = allLines[nIdx].trim();

        // 4. Hitung Embedding menggunakan Guru (Hanya 3 kali inference per iterasi)
        const embQ = await hfExtractor(Q, { pooling: 'mean', normalize: true });
        const embP = await hfExtractor(P, { pooling: 'mean', normalize: true });
        const embN = await hfExtractor(N, { pooling: 'mean', normalize: true });

        // 5. Minta Guru menilai seberapa mirip P dan N terhadap Q
        const scorePos = standardCosine(embQ.data, embP.data);
        const scoreNeg = standardCosine(embQ.data, embN.data);

        // Simpan hasil
        dataset.push({
            q: Q,
            p: P,
            n: N,
            teacher_pos_score: scorePos,
            teacher_neg_score: scoreNeg
        });

        if ((i + 1) % 100 === 0) {
            const msPerTriplets = (performance.now() - t0) / 100;
            console.log(`  > Berhasil menghasilkan ${i + 1} triplet... (Kecepatan: ${msPerTriplets.toFixed(2)} ms/triplet)`);
            t0 = performance.now();
        }
    }

    // 3. Simpan ke JSON
    const outputPath = './dataset/teacher_distillation_dataset.json';
    fs.writeFileSync(outputPath, JSON.stringify(dataset, null, 2));
    console.log(`\nSELESAI! Dataset berhasil disimpan di: ${outputPath}`);
    console.log(`Format data siap digunakan untuk Supervised Triplet Loss SNN Anda.`);
}

main().catch(console.error);
