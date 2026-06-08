import { Matrix } from "@oxide-js/core";
import { isNativeAvailable, lifStepNativeWrapper } from "@oxide-js/spiking";

function lifStepJS(
    potentials: Float32Array,
    dot: Float32Array,
    spikes: Float32Array,
    lastPotentials: Float32Array,
    beta: Float32Array,
    threshold: Float32Array
) {
    const units = beta.length;
    const batch = potentials.length / units;
    for (let b = 0; b < batch; b++) {
        const offset = b * units;
        for (let i = 0; i < units; i++) {
            const idx = offset + i;
            potentials[idx] = Math.min((potentials[idx] * beta[i]) + dot[idx], 1.0);
            lastPotentials[idx] = potentials[idx];
        }
        for (let i = 0; i < units; i++) {
            const idx = offset + i;
            if (potentials[idx] >= threshold[i]) {
                spikes[idx] = 1.0;
                potentials[idx] -= threshold[i];
            } else {
                spikes[idx] = 0.0;
            }
        }
    }
}

const units = 1000;
const batch = 10;
const pot1 = new Float32Array(batch * units);
const pot2 = new Float32Array(batch * units);
const dot = new Float32Array(batch * units);
const spikes1 = new Float32Array(batch * units);
const spikes2 = new Float32Array(batch * units);
const lp1 = new Float32Array(batch * units);
const lp2 = new Float32Array(batch * units);
const beta = new Float32Array(units);
const threshold = new Float32Array(units);

for (let i = 0; i < units; i++) {
    beta[i] = 0.9 + Math.random() * 0.1;
    threshold[i] = 0.1 + Math.random() * 0.9;
}
for (let i = 0; i < dot.length; i++) {
    dot[i] = Math.random();
    pot1[i] = Math.random();
    pot2[i] = pot1[i];
}

lifStepJS(pot1, dot, spikes1, lp1, beta, threshold);
lifStepNativeWrapper(pot2, dot, spikes2, lp2, beta, threshold);

let diff = 0;
for (let i = 0; i < spikes1.length; i++) {
    if (spikes1[i] !== spikes2[i]) diff++;
}
console.log("Spikes diff:", diff);

let potDiff = 0;
for (let i = 0; i < pot1.length; i++) {
    if (Math.abs(pot1[i] - pot2[i]) > 1e-6) potDiff++;
}
console.log("Potentials diff (>1e-6):", potDiff);
