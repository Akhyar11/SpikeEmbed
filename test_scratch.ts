import { BPETokenizer, Matrix } from '@oxide-js/core';
import { SpikingSentenceEmbedder } from './models/SpikingSentenceEmbedder.js';
const tokenizer = BPETokenizer.load('./models/vocab.json');
const d_model = 64;
const sequenceLength = 64;
const batchSize = 64;
const padTokenId = tokenizer.getPadId();
const model = new SpikingSentenceEmbedder(tokenizer.getVocabularyCapacity(), d_model, sequenceLength, padTokenId);
model.build(batchSize);
const B_emb = Matrix.fromFlat(new Float32Array(d_model * d_model).fill(0), [d_model, d_model]);
for (let i = 0; i < d_model; i++) B_emb._data[i * d_model + i] = 1.0;
const inputData = new Float32Array(batchSize * sequenceLength).fill(0);
const inputs = Matrix.fromFlat(inputData, [batchSize * sequenceLength]);
model.resetState();
try {
    model.forwardAndLearnLocal(inputs, B_emb, 0.005, new Array(32).fill(1.0), new Array(32).fill(0.0));
    console.log('Success');
} catch(e) {
    console.log('Error:', e.message);
    let isKernelBinary = true;
    for (let v of model.temporalPooler.kernel._data) {
        if (v !== 0 && v !== 1) isKernelBinary = false;
    }
    console.log('Is kernel binary at crash time?', isKernelBinary);
}
