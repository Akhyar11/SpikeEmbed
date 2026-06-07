import { SpikingEmbedding, SpikingSelfAttention, SpikingDenseBPTT, contrastiveHebbianNativeWrapper, isNativeAvailable } from "@oxide-js/spiking";
import { Matrix } from "@oxide-js/core";

export class SpikingSentenceEmbedder {
   public embedding: SpikingEmbedding;
   public attention: SpikingSelfAttention;
   public temporalPooler: SpikingDenseBPTT;
   public d_model: number;
   public sequenceLength: number;
   public padTokenId: number;
   
   private errEmbDataBuffer?: Float32Array;
   private spikes2DataBuffer?: Float32Array;
   private errAttDataBuffer?: Float32Array;

   constructor(vocabSize: number, d_model: number, sequenceLength: number, padTokenId: number = 0) {
       this.d_model = d_model;
       this.sequenceLength = sequenceLength;
       this.padTokenId = padTokenId;

       // 1. Layer Representasi Awal (Embedding SNN)
       this.embedding = new SpikingEmbedding({ inputDim: vocabSize, outputDim: d_model });
       
       // 2. Layer Atensi Bebas Softmax (Pengolahan Konteks Kalimat)
       this.attention = new SpikingSelfAttention({ d_model, sequenceLength });

       // 3. Temporal Pooler (LIF Integrator) menggunakan SpikingDenseBPTT
       // Bertugas mengonsumsi sequence satu per satu untuk menciptakan jejak temporal dan belajar mundur (BPTT).
       this.temporalPooler = new SpikingDenseBPTT({ units: d_model });
   }

   public summary(): void {
       console.log("\n==========================================================");
       console.log("                 SPIKING BPTT MODEL SUMMARY               ");
       console.log("==========================================================");
       console.log(`- d_model        : ${this.d_model}`);
       console.log(`- sequenceLength : ${this.sequenceLength}`);
       console.log(`- padTokenId     : ${this.padTokenId}`);
       console.log("----------------------------------------------------------");
       this.embedding.summary();
       this.attention.summary();
       this.temporalPooler.summary();
       console.log("==========================================================\n");
   }

   public build(batchSize: number) {
       // Matriks SNN di-flatten menjadi 2D: [batchSize * sequenceLength, d_model]
       const batchSeq = batchSize * this.sequenceLength;
       this.embedding.build([batchSeq]);
       this.attention.build([batchSeq, this.d_model]);
       
       this.errEmbDataBuffer = new Float32Array(batchSeq * this.d_model);
       this.spikes2DataBuffer = new Float32Array(batchSeq * this.d_model);
       this.errAttDataBuffer = new Float32Array(batchSeq * this.d_model);
       
       // Temporal Pooler beroperasi per batch (tidak dikalikan sequenceLength karena sequence di-loop)
       this.temporalPooler.build([batchSize, this.d_model]);
       
       // Inisialisasi bobot Temporal Pooler sebagai Identity Matrix (agar bekerja murni sebagai Integrator murni)
       const identityWeights = new Float32Array(this.d_model * this.d_model).fill(0);
       for (let i = 0; i < this.d_model; i++) {
           identityWeights[i * this.d_model + i] = 1.0;
       }
       this.temporalPooler.kernel!._data.set(identityWeights);

       // HETEROGENEOUS TIME CONSTANTS secara otomatis sudah dibuat via bit-shift dinamis di dalam SpikingDenseBPTT.
       // Rentang secara otomatis 0.75 hingga 0.96875 untuk merepresentasikan multi-timescale memori.
   }

    public resetState() {
        this.embedding.resetState();
        this.attention.resetState();
        this.temporalPooler.resetSequence(this.sequenceLength);
    }

   /**
    * Forward Pass dengan LOCAL ERROR LEARNING (In-Batch Negative Contrastive Loss)
    * Jaringan akan dilatih untuk mendekatkan Q ke P+ (Positive) dan menjauhkan Q dari N (Negative).
    */
    /**
     * Mencegah Token PAD (ID: 0) meletup/spike. 
     * Ini adalah cara Neuromorphic untuk melakukan "Attention Masking".
     */
    private zeroPadToken() {
        const embData = this.embedding.embeddings!._data;
        for (let d = 0; d < this.d_model; d++) {
            embData[d] = -1.0; // Paksa nilai negatif agar tidak pernah spike
        }
    }

   public forwardAndLearnLocal(inputs: Matrix, B_emb: Matrix, learningRate: number) {
       this.zeroPadToken();
       const batchSeq = inputs._shape[0];
       const numPairs = batchSeq / (2 * this.sequenceLength);
       
       if (!Number.isInteger(numPairs) || numPairs < 2) {
           throw new Error("[SpikingSentenceEmbedder] Batch harus berisi setidaknya 2 pasang Q dan P+ untuk In-Batch Negative Sampling.");
       }

       // ============================================
       // LAYER 1: REPRESENTASI AWAL
       // ============================================

       const spikes1 = this.embedding.forward(inputs) as Matrix;

       const errEmbData = this.errEmbDataBuffer!;
       errEmbData.fill(0);
       let localLoss1 = 0;

       if (isNativeAvailable()) {
           localLoss1 = contrastiveHebbianNativeWrapper(spikes1._data, errEmbData, numPairs, this.sequenceLength, this.d_model);
       } else {
           for (let i = 0; i < numPairs; i++) {
               const qOffset = i * this.sequenceLength * this.d_model;
               const pOffset = (numPairs + i) * this.sequenceLength * this.d_model;
               
               // In-Batch Negative Sampling: Ambil kalimat dari pasangan lain di batch yang sama
               const negIdx = (i + 1 + Math.floor(Math.random() * (numPairs - 1))) % numPairs;
               const nOffset = (numPairs + negIdx) * this.sequenceLength * this.d_model;

               for(let s = 0; s < this.sequenceLength; s++) {
                   for(let d = 0; d < this.d_model; d++) {
                       const idxQ = qOffset + s * this.d_model + d;
                       const idxP = pOffset + s * this.d_model + d;
                       const idxN = nOffset + s * this.d_model + d;

                       const qSpike = spikes1._data[idxQ];
                       const pSpike = spikes1._data[idxP];
                       const nSpike = spikes1._data[idxN];
                       
                       // OPTIMIZATION: Skip math if all are zero
                       let pull = pSpike - qSpike; 
                       if (qSpike === 0 && pSpike === 0 && nSpike === 0) {
                           pull = 0.05; // Suntik energi agar hidup
                       }
                       
                       // Daya Tolak (Repulsion): N menolak Q (hanya menolak jika keduanya spike)
                       const push = (qSpike * nSpike) * 0.2; // 1 * 0.2 jika keduanya identik/kolaps

                       if (pull !== 0 || push !== 0) {
                           errEmbData[idxQ] += pull - push; // Q ditarik ke P, didorong oleh N
                           errEmbData[idxP] += -pull;       // P ditarik ke Q
                           errEmbData[idxN] += -push;       // N didorong oleh Q
                           localLoss1 += Math.abs(pull) + push;
                       }
                   }
               }
           }
       }
       
       const errEmb = Matrix.fromFlat(errEmbData, [batchSeq, this.d_model]);
       this.embedding.learnEmbedding(errEmb, B_emb, learningRate);

       // ============================================
       // LAYER 2: SPIKING SELF-ATTENTION
       // ============================================
        const attSpikes = this.attention.forward(spikes1) as Matrix;
        const spikes2Data = this.spikes2DataBuffer!;
        for(let i=0; i<spikes2Data.length; i++) {
            spikes2Data[i] = spikes1._data[i] + attSpikes._data[i];
        }
        const spikes2 = Matrix.fromFlat(spikes2Data, [batchSeq, this.d_model]);
       const errAttData = this.errAttDataBuffer!;
       errAttData.fill(0);
       let localLoss2 = 0;

       if (isNativeAvailable()) {
           localLoss2 = contrastiveHebbianNativeWrapper(spikes2._data, errAttData, numPairs, this.sequenceLength, this.d_model);
       } else {
           for (let i = 0; i < numPairs; i++) {
               const qOffset = i * this.sequenceLength * this.d_model;
               const pOffset = (numPairs + i) * this.sequenceLength * this.d_model;
               
               // Gunakan pasangan negatif yang sama untuk stabilitas
               const negIdx = (i + 1) % numPairs; 
               const nOffset = (numPairs + negIdx) * this.sequenceLength * this.d_model;

               for(let s = 0; s < this.sequenceLength; s++) {
                   for(let d = 0; d < this.d_model; d++) {
                       const idxQ = qOffset + s * this.d_model + d;
                       const idxP = pOffset + s * this.d_model + d;
                       const idxN = nOffset + s * this.d_model + d;

                       const qSpike = spikes2._data[idxQ];
                       const pSpike = spikes2._data[idxP];
                       const nSpike = spikes2._data[idxN];
                       
                       let pull = pSpike - qSpike; 
                       if (qSpike === 0 && pSpike === 0 && nSpike === 0) {
                           pull = 0.05;
                       }
                       const push = (qSpike * nSpike) * 0.2; 

                       if (pull !== 0 || push !== 0) {
                           errAttData[idxQ] += pull - push;
                           errAttData[idxP] += -pull;
                           errAttData[idxN] += -push;
                           localLoss2 += Math.abs(pull) + push;
                       }
                   }
               }
           }
       }
       
       const errAtt = Matrix.fromFlat(errAttData, [batchSeq, this.d_model]);
       this.attention.learnAttention(errAtt, learningRate);

       // ============================================
       // LAYER 3: TEMPORAL POOLER (BPTT)
       // ============================================
       const batchSize = inputs._shape[0] / this.sequenceLength;
       this.temporalPooler.resetSequence(this.sequenceLength);
       
       const finalOutData = new Float32Array(batchSize * this.d_model); // Akumulasi total aktivitas
       const validTokenCounts = new Int32Array(batchSize).fill(0);

       for (let s = 0; s < this.sequenceLength; s++) {
           const tokenData = new Float32Array(batchSize * this.d_model);
           let hasValidTokens = false;

           for (let b = 0; b < batchSize; b++) {
               const tokenId = inputs._data[b * this.sequenceLength + s];
               if (tokenId !== this.padTokenId) { // Lewati Padding Token
                   hasValidTokens = true;
                   validTokenCounts[b]++;
                   const srcOffset = (b * this.sequenceLength + s) * this.d_model;
                   const dstOffset = b * this.d_model;
                   for (let d = 0; d < this.d_model; d++) {
                       tokenData[dstOffset + d] = spikes2Data[srcOffset + d];
                   }
               }
           }
           const tokenMatrix = Matrix.fromFlat(tokenData, [batchSize, this.d_model]);
           
           // Pooler tetap berjalan untuk memproses leak/decay, tapi input padding adalah 0
           const outSpikes = this.temporalPooler.computeStep(tokenMatrix, s);
           
           for (let b = 0; b < batchSize; b++) {
               const tokenId = inputs._data[b * this.sequenceLength + s];
               if (tokenId !== this.padTokenId) {
                   const offset = b * this.d_model;
                   for (let d = 0; d < this.d_model; d++) {
                       // Akumulasi aktivitas neuron (Jumlah tembakan / tingkat potensial melintasi waktu)
                       finalOutData[offset + d] += outSpikes._data[offset + d];
                   }
               }
           }
       }

       // NORMALISASI L2 TERHADAP AKUMULASI (Untuk Mencegah Error Exploding & Thresholding)
       const normalizedOutData = new Float32Array(batchSize * this.d_model);
       for (let b = 0; b < batchSize; b++) {
           let sqSum = 0;
           const offset = b * this.d_model;
           for (let d = 0; d < this.d_model; d++) {
               sqSum += finalOutData[offset + d] * finalOutData[offset + d];
           }
           const norm = Math.sqrt(sqSum) || 1e-8;
           for (let d = 0; d < this.d_model; d++) {
               normalizedOutData[offset + d] = finalOutData[offset + d] / norm;
           }
       }

       // MENGHITUNG CONTRASTIVE LOSS PADA TOKEN VALID TERAKHIR (Dynamic Padding Avoidance)
       const errorFinalData = new Float32Array(batchSize * this.d_model).fill(0);
       let poolerLoss = 0;
       
       for (let i = 0; i < numPairs; i++) {
           const idxQ = i;
           const idxP = numPairs + i;
           const idxN = (numPairs + ((i + 1) % numPairs));
           
           for (let d = 0; d < this.d_model; d++) {
               const qSpike = normalizedOutData[idxQ * this.d_model + d];
               const pSpike = normalizedOutData[idxP * this.d_model + d];
               const nSpike = normalizedOutData[idxN * this.d_model + d];
               
               let pull = pSpike - qSpike;
               if (qSpike === 0 && pSpike === 0 && nSpike === 0) pull = 0.05;
               const push = (qSpike * nSpike) * 0.2;

               if (pull !== 0 || push !== 0) {
                   errorFinalData[idxQ * this.d_model + d] += pull - push;
                   errorFinalData[idxP * this.d_model + d] += -pull;
                   errorFinalData[idxN * this.d_model + d] += -push;
                   poolerLoss += Math.abs(pull) + push;
               }
           }
       }

       // BPTT Backward Pass - Distribusi Error ke Semua Waktu Token Valid (Kalkulus Sum Rule)
       const errorsSequence = [];
       for(let s = 0; s < this.sequenceLength; s++) {
           const stepErrorData = new Float32Array(batchSize * this.d_model);
           for (let b = 0; b < batchSize; b++) {
               const tokenId = inputs._data[b * this.sequenceLength + s];
               if (tokenId !== this.padTokenId) {
                   const offset = b * this.d_model;
                   for (let d = 0; d < this.d_model; d++) {
                       // Gradient of Sum is 1, so the error passes through identically to all valid timesteps
                       stepErrorData[offset + d] = errorFinalData[offset + d];
                   }
               }
           }
           errorsSequence.push(Matrix.fromFlat(stepErrorData, [batchSize, this.d_model]));
       }

       this.temporalPooler.learnThroughTime(errorsSequence, undefined, learningRate);

       return { spikes2, localLoss1, localLoss2, poolerLoss };
   }

    /**
     * Inference: SNN Sequence Processing (BPTT-Style)
     * 1. Layer Spasial (Embedding + Attention) dieksekusi secara instan (Direct Firing).
     * 2. Integrasikan urutan fitur melalui Temporal Pooler melintasi waktu sesuai panjang sekuens.
     */
    public infer(inputs: Matrix): Matrix {
        const batchSeq = inputs._shape[0];
        const batchSize = batchSeq / this.sequenceLength;
        
        this.embedding.resetState();
        this.attention.resetState();
        
        // Tahap 1: SNN Spatial Forward (Direct Firing)
        this.zeroPadToken();
        const spikes1 = this.embedding.forward(inputs) as Matrix;
        const attSpikes = this.attention.forward(spikes1) as Matrix;
        
        const aggregatedFeatures = new Float32Array(batchSeq * this.d_model);
        for (let i = 0; i < aggregatedFeatures.length; i++) {
            aggregatedFeatures[i] = spikes1._data[i] + attSpikes._data[i];
        }
        
        // Tahap 2: Order-Aware Sequence Integration
        // temporalPooler memiliki metode resetSequence tersendiri yang dipanggil secara eksplisit nanti
        const finalDenseOutData = new Float32Array(batchSize * this.d_model);
        
        for (let s = 0; s < this.sequenceLength; s++) {
            const tokenData = new Float32Array(batchSize * this.d_model);
            let hasValidTokens = false;
            
            for (let b = 0; b < batchSize; b++) {
                const tokenId = inputs._data[b * this.sequenceLength + s];
                if (tokenId !== this.padTokenId) {
                    hasValidTokens = true;
                    const srcOffset = (b * this.sequenceLength + s) * this.d_model;
                    const dstOffset = b * this.d_model;
                    for (let d = 0; d < this.d_model; d++) {
                        tokenData[dstOffset + d] = aggregatedFeatures[srcOffset + d];
                    }
                }
            }
            
            if (hasValidTokens) {
                const tokenMatrix = Matrix.fromFlat(tokenData, [batchSize, this.d_model]);
                this.temporalPooler.computeStep(tokenMatrix, s);
                
                for (let b = 0; b < batchSize; b++) {
                    const tokenId = inputs._data[b * this.sequenceLength + s];
                    if (tokenId !== this.padTokenId) {
                        const dstOffset = b * this.d_model;
                        for (let d = 0; d < this.d_model; d++) {
                            // Akumulasi potensial melintasi waktu
                            finalDenseOutData[dstOffset + d] += this.temporalPooler.historyPotentials[s]._data[dstOffset + d];
                        }
                    }
                }
            }
        }
        
        // NORMALISASI L2 UNTUK HASIL INFERENCE AGAR SKALA TETAP KONSISTEN
        for (let b = 0; b < batchSize; b++) {
            let sqSum = 0;
            const offset = b * this.d_model;
            for (let d = 0; d < this.d_model; d++) {
                sqSum += finalDenseOutData[offset + d] * finalDenseOutData[offset + d];
            }
            const norm = Math.sqrt(sqSum) || 1e-8;
            for (let d = 0; d < this.d_model; d++) {
                finalDenseOutData[offset + d] /= norm;
            }
        }

        return Matrix.fromFlat(finalDenseOutData, [batchSize, this.d_model]);
    }
}
