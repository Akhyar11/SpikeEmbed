import * as fs from 'fs';
import { BPETokenizer, Matrix } from "@oxide-js/core";
import { SpikingSentenceEmbedder } from "../../models/SpikingSentenceEmbedder.js";

// Helper fungsi untuk menghitung Cosine Similarity
function cosineSimilarity(vecA: number[], vecB: number[]): number {
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
    return centeredCosine;
}

// Fungsi utama Sliding Window Chunking
function embedLongText(
    text: string, 
    tokenizer: BPETokenizer, 
    model: SpikingSentenceEmbedder, 
    sequenceLength: number = 512,
    stride: number = 24
): number[] {
    const padTokenId = tokenizer.getPadId();
    const tokens = tokenizer.encode(text);
    
    // Jika kalimat pendek, kita proses langsung (tanpa sliding window)
    if (tokens.length <= sequenceLength) {
        while (tokens.length < sequenceLength) tokens.push(padTokenId);
        const inputMatrix = Matrix.fromFlat(new Float32Array(tokens), [sequenceLength]);
        model.build(1);
        const outputSpikes = model.infer(inputMatrix);
        return Array.from(outputSpikes._data);
    }
    
    // Jika kalimat panjang, kita gunakan Sliding Window
    const chunks: number[][] = [];
    let startIdx = 0;
    
    while (startIdx < tokens.length) {
        let chunk = tokens.slice(startIdx, startIdx + sequenceLength);
        
        // Pad chunk terakhir jika kurang dari sequenceLength
        while (chunk.length < sequenceLength) {
            chunk.push(padTokenId);
        }
        
        chunks.push(chunk);
        
        // Jika sisa teks habis, berhentikan loop
        if (startIdx + sequenceLength >= tokens.length) break;
        
        startIdx += stride; // Bergeser maju sesuai langkah stride (tumpang tindih)
    }
    
    const numChunks = chunks.length;
    console.log(`      [Sliding Window] Teks dipotong menjadi ${numChunks} chunks (total ${tokens.length} token).`);
    
    // Siapkan Batch untuk memproses semua chunk SEKALIGUS (Sangat Cepat!)
    const flatTokens = new Float32Array(numChunks * sequenceLength);
    for (let i = 0; i < numChunks; i++) {
        flatTokens.set(chunks[i], i * sequenceLength);
    }
    
    const inputMatrix = Matrix.fromFlat(flatTokens, [numChunks * sequenceLength]);
    
    // Prediksi semua chunk sekaligus (Paralel)
    model.build(numChunks);
    const chunkEmbeddings = model.infer(inputMatrix);
    const d_model = 64;
    
    // Lakukan Average Pooling (Menggabungkan makna dari seluruh chunks)
    const finalEmbedding = new Array(d_model).fill(0);
    const outData = chunkEmbeddings._data;
    
    for (let i = 0; i < numChunks; i++) {
        for (let d = 0; d < d_model; d++) {
            finalEmbedding[d] += outData[i * d_model + d];
        }
    }
    
    for (let d = 0; d < d_model; d++) {
        finalEmbedding[d] /= numChunks;
    }
    
    return finalEmbedding;
}

async function main() {
    console.log("=== UJI COBA SNN: SLIDING WINDOW UNTUK TEKS SUPER PANJANG ===");
    const tokenizer = BPETokenizer.load("./models/vocab.json");
    
    const d_model = 64;
    const sequenceLength = 512; // Batas asli SNN
    const vocabSize = tokenizer.getVocabularyCapacity();
    const padTokenId = tokenizer.getPadId();
    
    const snnModel = new SpikingSentenceEmbedder(vocabSize, d_model, sequenceLength, padTokenId);
    
    if (fs.existsSync('./models/spiking_model_weights.json')) {
        const weightsStr = fs.readFileSync('./models/spiking_model_weights.json', 'utf8');
        const config = JSON.parse(weightsStr);
        // Kita hanya build 1 untuk inisiasi awal alokasi bobot
        snnModel.build(1); 
        snnModel.embedding.embeddings!._data.set(config.embedding_weights);
        snnModel.attention.kernelQ!._data.set(config.kernelQ);
        snnModel.attention.kernelK!._data.set(config.kernelK);
        snnModel.attention.kernelV!._data.set(config.kernelV);
        console.log("-> SNN & Bobot berhasil dimuat!\n");
    } else {
        console.log("-> [PERINGATAN] File bobot tidak ditemukan, akan gagal memuat semantik.\n");
        process.exit(1);
    }

    const docA = "Perubahan iklim adalah ancaman terbesar bagi umat manusia di abad ke-21. Efek rumah kaca menyebabkan suhu rata-rata permukaan bumi terus meningkat dari tahun ke tahun. Akibatnya, gletser di kutub utara mulai mencair, yang memicu kenaikan permukaan air laut. Banyak negara kepulauan yang kini terancam tenggelam dalam beberapa dekade mendatang jika tidak ada langkah drastis untuk mengurangi emisi karbon dari kendaraan berbahan bakar fosil dan pabrik industri.";
    
    const docB = "Pemanasan global telah menjadi krisis lingkungan yang sangat genting bagi planet kita saat ini. Tingkat karbon dioksida di atmosfer menahan panas dari matahari, membuat suhu global memanas dengan cepat. Dampak paling nyata adalah mencairnya es di antartika dan kutub lainnya, yang membuat air laut naik drastis. Kota-kota pesisir berada dalam bahaya besar terendam banjir abadi apabila kita gagal beralih ke energi terbarukan dan meninggalkan penggunaan minyak bumi atau batu bara.";
    
    const docC = "Resep membuat nasi goreng yang lezat sangatlah mudah dan bisa dilakukan oleh pemula di dapur. Pertama, siapkan nasi putih dingin yang sudah dibiarkan semalaman agar teksturnya tidak lembek. Tumis bawang putih, bawang merah, dan cabai cincang hingga harum menggunakan sedikit minyak sayur. Masukkan telur lalu orak-arik hingga matang. Terakhir, masukkan nasi, tambahkan kecap manis, garam, dan penyedap rasa secukupnya, aduk rata dengan api besar hingga bumbu meresap sempurna.";

    console.log("--------------------------------------------------");
    console.log("[Dokumen A] (Isu Pemanasan Global, panjang ~60 token)\n" + docA + "\n");
    console.log("[Dokumen B] (Krisis Iklim & Es Mencair, panjang ~60 token)\n" + docB + "\n");
    console.log("[Dokumen C] (Resep Nasi Goreng, panjang ~60 token)\n" + docC + "\n");
    console.log("--------------------------------------------------\n");

    console.log(">> Meng-encode Dokumen A...");
    const startA = performance.now();
    const vecA = embedLongText(docA, tokenizer, snnModel, sequenceLength, 24);
    const timeA = performance.now() - startA;

    console.log(">> Meng-encode Dokumen B...");
    const startB = performance.now();
    const vecB = embedLongText(docB, tokenizer, snnModel, sequenceLength, 24);
    const timeB = performance.now() - startB;

    console.log(">> Meng-encode Dokumen C...");
    const startC = performance.now();
    const vecC = embedLongText(docC, tokenizer, snnModel, sequenceLength, 24);
    const timeC = performance.now() - startC;

    console.log("\n================ HASIL ANALISIS ==================");
    console.log(`[Kecepatan] Rata-rata inferensi per dokumen raksasa: ${((timeA + timeB + timeC) / 3).toFixed(2)} ms`);
    
    const simAB = cosineSimilarity(vecA, vecB) * 100;
    const simAC = cosineSimilarity(vecA, vecC) * 100;
    
    console.log("\n[Uji Kedekatan Semantik via Sliding Window]");
    console.log(`Dokumen A vs Dokumen B (Topik Sama) : ${simAB.toFixed(2)}% Mirip`);
    console.log(`Dokumen A vs Dokumen C (Beda Jauh)  : ${simAC.toFixed(2)}% Mirip`);
    console.log("==================================================");
}

main().catch(console.error);
