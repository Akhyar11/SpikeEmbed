import { SpikingEmbedding, SpikingDenseBPTT, contrastiveHebbianNativeWrapper, isNativeAvailable } from "@oxide-js/spiking";
import { Matrix } from "@oxide-js/core";

export class SpikingSentenceEmbedder {
   public embedding: SpikingEmbedding;
   public temporalPooler: SpikingDenseBPTT;
   public d_model: number;
   public sequenceLength: number;
   public padTokenId: number;
   
   private errEmbDataBuffer?: Float32Array;

   constructor(vocabSize: number, d_model: number, sequenceLength: number, padTokenId: number = 0) {
       this.d_model = d_model;
       this.sequenceLength = sequenceLength;
       this.padTokenId = padTokenId;

       // 1. Layer Representasi Awal (Embedding SNN biner stateless)
       this.embedding = new SpikingEmbedding({ inputDim: vocabSize, outputDim: d_model });

       // 2. Temporal Pooler (LIF Integrator multi-timescale)
       this.temporalPooler = new SpikingDenseBPTT({ units: d_model });
   }

   public summary(): void {
       console.log("\n==========================================================");
       console.log("             ATTENTION-LESS SPIKING BPTT MODEL            ");
       console.log("==========================================================");
       console.log(`- d_model        : ${this.d_model}`);
       console.log(`- sequenceLength : ${this.sequenceLength}`);
       console.log(`- padTokenId     : ${this.padTokenId}`);
       console.log("----------------------------------------------------------");
       this.embedding.summary();
       this.temporalPooler.summary();
       console.log("==========================================================\n");
   }

   public build(batchSize: number) {
       // Matriks SNN di-flatten menjadi 2D: [batchSize * sequenceLength, d_model]
       const batchSeq = batchSize * this.sequenceLength;
       this.embedding.build([batchSeq]);
       
       this.errEmbDataBuffer = new Float32Array(batchSeq * this.d_model);
       
       // Temporal Pooler beroperasi per batch
       this.temporalPooler.build([batchSize, this.d_model]);
       
       // Inisialisasi bobot Temporal Pooler sebagai Identity Matrix
       const identityWeights = new Float32Array(this.d_model * this.d_model).fill(0);
       for (let i = 0; i < this.d_model; i++) {
           identityWeights[i * this.d_model + i] = 1.0;
       }
       this.temporalPooler.kernel!._data.set(identityWeights);
   }

    public resetState() {
        this.embedding.resetState();
        this.attention.resetState();
        this.temporalPooler.resetSequence(this.sequenceLength);
    }

    private zeroPadToken() {
        const embData = this.embedding.embeddings!._data;
        for (let d = 0; d < this.d_model; d++) {
            embData[d] = -1.0; // Paksa nilai negatif agar tidak meletup (0)
        }
    }

   public forwardAndLearnLocal(inputs: Matrix, B_emb: Matrix, learningRate: number, teacherPosScores?: number[], teacherNegScores?: number[]) {
       this.zeroPadToken();
       const batchSeq = inputs._shape[0];
       const numPairs = batchSeq / (2 * this.sequenceLength);
       
       if (!Number.isInteger(numPairs) || numPairs < 2) {
           throw new Error("[SpikingSentenceEmbedder] Batch harus berisi setidaknya 2 pasang Q dan P+.");
       }

       // ============================================
       // LAYER 1: REPRESENTASI AWAL (BINARY)
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
                       
                       let pull = pSpike - qSpike; 
                       if (qSpike === 0 && pSpike === 0 && nSpike === 0) {
                           pull = 0.05; 
                       }
                       
                       const push = (qSpike * nSpike) * 0.2; 

                       if (pull !== 0 || push !== 0) {
                           errEmbData[idxQ] += pull - push; 
                           errEmbData[idxP] += -pull;       
                           errEmbData[idxN] += -push;       
                           localLoss1 += Math.abs(pull) + push;
                       }
                   }
               }
           }
       }
       
       const errEmb = Matrix.fromFlat(errEmbData, [batchSeq, this.d_model]);
       this.embedding.learnEmbedding(errEmb, B_emb, learningRate);

       // ============================================
       // LAYER 2: TEMPORAL POOLER (BPTT)
       // ============================================
       const batchSize = inputs._shape[0] / this.sequenceLength;
       this.temporalPooler.resetSequence(this.sequenceLength);
       
       const finalOutData = new Float32Array(batchSize * this.d_model); 
       const validTokenCounts = new Int32Array(batchSize).fill(0);

       for (let s = 0; s < this.sequenceLength; s++) {
           const tokenData = new Float32Array(batchSize * this.d_model);
           let hasValidTokens = false;

           for (let b = 0; b < batchSize; b++) {
               const tokenId = inputs._data[b * this.sequenceLength + s];
               if (tokenId !== this.padTokenId) {
                   hasValidTokens = true;
                   validTokenCounts[b]++;
                   const srcOffset = (b * this.sequenceLength + s) * this.d_model;
                   const dstOffset = b * this.d_model;
                   for (let d = 0; d < this.d_model; d++) {
                       // Bypass Attention: langsung ambil dari spikes1
                       tokenData[dstOffset + d] = spikes1._data[srcOffset + d];
                   }
               }
           }
           const tokenMatrix = Matrix.fromFlat(tokenData, [batchSize, this.d_model]);
           
           const outSpikes = this.temporalPooler.computeStep(tokenMatrix, s);
           
           for (let b = 0; b < batchSize; b++) {
               const tokenId = inputs._data[b * this.sequenceLength + s];
               if (tokenId !== this.padTokenId) {
                   const offset = b * this.d_model;
                   for (let d = 0; d < this.d_model; d++) {
                       finalOutData[offset + d] += outSpikes._data[offset + d];
                   }
               }
           }
       }

       // NORMALISASI L2
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

       // CONTRASTIVE LOSS
       const errorFinalData = new Float32Array(batchSize * this.d_model).fill(0);
       let poolerLoss = 0;
       
       for (let i = 0; i < numPairs; i++) {
            const idxQ = i;
            const idxP = numPairs + i;
            const idxN = (numPairs + ((i + 1) % numPairs));
            
            const posScale = teacherPosScores ? Math.max(0, teacherPosScores[i]) : 1.0;
            const negScale = teacherNegScores ? Math.max(0, 1.0 - teacherNegScores[i]) : 0.2;

            for (let d = 0; d < this.d_model; d++) {
                const qSpike = normalizedOutData[idxQ * this.d_model + d];
                const pSpike = normalizedOutData[idxP * this.d_model + d];
                const nSpike = normalizedOutData[idxN * this.d_model + d];
                
                let pull = pSpike - qSpike;
                if (qSpike === 0 && pSpike === 0 && nSpike === 0) pull = 0.05;
                
                pull *= posScale;
                const push = (qSpike * nSpike) * negScale;

                if (pull !== 0 || push !== 0) {
                    errorFinalData[idxQ * this.d_model + d] += pull - push;
                    errorFinalData[idxP * this.d_model + d] += -pull;
                    errorFinalData[idxN * this.d_model + d] += -push;
                    poolerLoss += Math.abs(pull) + push;
                }
            }
        }

       // BPTT Backward Pass
       const errorsSequence = [];
       for(let s = 0; s < this.sequenceLength; s++) {
           const stepErrorData = new Float32Array(batchSize * this.d_model);
           for (let b = 0; b < batchSize; b++) {
               const tokenId = inputs._data[b * this.sequenceLength + s];
               if (tokenId !== this.padTokenId) {
                   const offset = b * this.d_model;
                   for (let d = 0; d < this.d_model; d++) {
                       stepErrorData[offset + d] = errorFinalData[offset + d];
                   }
               }
           }
           errorsSequence.push(Matrix.fromFlat(stepErrorData, [batchSize, this.d_model]));
       }

       this.temporalPooler.learnThroughTime(errorsSequence, undefined, learningRate);

       // Kembalikan localLoss2 = 0 untuk backwards compatibility dengan script pelatih
       return { spikes1, localLoss1, localLoss2: 0, poolerLoss };
   }

    public infer(inputs: Matrix): Matrix {
        const batchSeq = inputs._shape[0];
        const batchSize = batchSeq / this.sequenceLength;
        
        this.embedding.resetState();
        this.attention.resetState();
        this.temporalPooler.resetSequence(this.sequenceLength);
        
        // Tahap 1: SNN Spatial Forward (Bypass Attention)
        this.zeroPadToken();
        const spikes1 = this.embedding.forward(inputs) as Matrix;
        
        const aggregatedFeatures = new Float32Array(batchSeq * this.d_model);
        for (let i = 0; i < aggregatedFeatures.length; i++) {
            aggregatedFeatures[i] = spikes1._data[i];
        }
        
        // Tahap 2: Order-Aware Sequence Integration
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
                            finalDenseOutData[dstOffset + d] += this.temporalPooler.historyPotentials[s]._data[dstOffset + d];
                        }
                    }
                }
            }
        }
        
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
