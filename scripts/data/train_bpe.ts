import * as fs from "node:fs";
import { BPETokenizer, NativeBpeTrainer } from "@oxide-js/core";

// Konfigurasi Training
const DATASET_PATH = "dataset/mini_corpus.txt";
const SAVE_PATH = "models/vocab.json";
const VOCAB_SIZE = 32000;

async function trainBPE() {
  console.log(`=== Memulai Pelatihan BPE Tokenizer (RUST NATIVE ENGINE) ===`);
  console.log(`Target Vocab Size: ${VOCAB_SIZE}`);
  console.log(`Dataset: ${DATASET_PATH}`);

  if (!NativeBpeTrainer) {
    console.error("NativeBpeTrainer tidak tersedia. Pastikan @oxide-js/core sudah di-build rust.");
    return;
  }

  // 1. Inisialisasi Native Trainer (C++)
  const specialTokens = ["<UNK>", "<PAD>", "<BOS>", "<EOS>"];
  const trainer = new NativeBpeTrainer(VOCAB_SIZE, specialTokens);

  console.log("\nMemproses file dan menyimpan model...");
  const startTime = Date.now();

  // 2. Training dan Save (ZERO COPY IPC)
  // Menulis langsung vocab.json dari C++/Rust memory ke disk
  const success = trainer.trainAndSave(DATASET_PATH, SAVE_PATH);

  const endTime = Date.now();
  console.log(`\nTraining & Saving selesai dalam ${((endTime - startTime) / 1000).toFixed(2)} detik!`);

  if (success) {
    console.log(`\n✅ Model BPE berhasil diselesaikan secara zero-copy!`);
  }
}

trainBPE().catch(console.error);
