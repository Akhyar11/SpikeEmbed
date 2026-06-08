import { SpikingSentenceEmbedder } from "./models/SpikingSentenceEmbedder.js";
import { Matrix } from "@oxide-js/core";
const model = new SpikingSentenceEmbedder(100, 64, 64, 0);
model.build(2);
const inputs = Matrix.fromFlat(new Float32Array(2*64).fill(1), [2*64]);
const spikes1 = model.embedding.forward(inputs);
const { spikes: spikes2 } = model.attention.forward(spikes1);
console.log('Spikes2 (first 10):', Array.from(spikes2._data).slice(0, 10));
let allBinary = true;
for(let v of spikes2._data) {
  if (v !== 0 && v !== 1) allBinary = false;
}
console.log('Is spikes2 all binary?', allBinary);
