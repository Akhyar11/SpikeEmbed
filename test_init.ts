import { SpikingSentenceEmbedder } from "./models/SpikingSentenceEmbedder.js";
const model = new SpikingSentenceEmbedder(100, 64, 64, 0);
model.build(2);
console.log('Embedding weights (first 5):', Array.from(model.embedding.embeddings._data).slice(0, 5));
console.log('Attention Q (first 5):', Array.from(model.attention.kernelQ._data).slice(0, 5));
