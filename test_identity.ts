import { SpikingSentenceEmbedder } from "./models/SpikingSentenceEmbedder.js";
const model = new SpikingSentenceEmbedder(100, 64, 64, 0);
model.build(2);
let isBinary = true;
for (let v of model.temporalPooler.kernel._data) {
    if (v !== 0 && v !== 1) {
        isBinary = false;
        break;
    }
}
console.log('Is temporalPooler.kernel binary?', isBinary);
