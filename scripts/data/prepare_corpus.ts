import * as fs from "node:fs";
import * as readline from "node:readline";
import { parquetRead } from "hyparquet";

const WIKI_TRAIN_PATH = "dataset/wiki.train.tokens";
const INDO_WIKI_PATH = "dataset/indonesia_wiki.parquet";
const OUTPUT_PATH = "dataset/mini_corpus.txt";

const TARGET_BYTES_PER_DATASET = 75 * 1024 * 1024; // 10 MB

async function prepareDataset() {
    console.log("=== Mempersiapkan Dataset 20MB (Simetris 10MB + 10MB) ===");

    // 1. Ekstrak 10MB dari wiki.train.tokens
    console.log(`Mengekstrak 10MB dari ${WIKI_TRAIN_PATH}...`);
    const enLines: string[] = [];
    let enBytes = 0;

    const enStream = fs.createReadStream(WIKI_TRAIN_PATH, { encoding: 'utf-8' });
    const enRl = readline.createInterface({ input: enStream });

    for await (const line of enRl) {
        if (line.trim().length > 10) {
            enLines.push(line.trim());
            enBytes += Buffer.byteLength(line, 'utf-8');
            if (enBytes >= TARGET_BYTES_PER_DATASET) break;
        }
    }
    enRl.close();
    console.log(`Selesai: Terkumpul ${enLines.length} baris bahasa Inggris.`);

    // 2. Ekstrak 10MB dari indonesia_wiki.parquet
    console.log(`Mengekstrak 10MB dari ${INDO_WIKI_PATH}...`);
    const idLines: string[] = [];
    let idBytes = 0;

    const fileBuffer = fs.readFileSync(INDO_WIKI_PATH);
    const arrayBuffer = fileBuffer.buffer.slice(fileBuffer.byteOffset, fileBuffer.byteOffset + fileBuffer.byteLength);

    await parquetRead({
        file: arrayBuffer,
        columns: ["text"],
        onComplete: (data) => {
            if (data && data.length > 0) {
                // data is an array of rows, where each row is an array of column values
                for (let i = 0; i < data.length; i++) {
                    const row = data[i];
                    if (!row || row.length === 0) continue;

                    const text = row[0]; // because we only requested ["text"]
                    if (text && typeof text === 'string' && text.trim().length > 10) {
                        // Pecah menjadi kalimat (opsional) atau masukkan per paragraf
                        const sentences = text.split(/(?<=[.!?])\s+/);
                        for (const sentence of sentences) {
                            if (sentence.trim().length > 10) {
                                idLines.push(sentence.trim());
                                idBytes += Buffer.byteLength(sentence, 'utf-8');
                            }
                            if (idBytes >= TARGET_BYTES_PER_DATASET) break;
                        }
                    }
                    if (idBytes >= TARGET_BYTES_PER_DATASET) break;
                }
            }
        }
    });
    console.log(`Selesai: Terkumpul ${idLines.length} baris bahasa Indonesia.`);

    // 3. Gabungkan secara simetris (interleaved)
    console.log("Menggabungkan kedua dataset secara simetris...");
    const mergedLines: string[] = [];
    const maxLen = Math.max(enLines.length, idLines.length);

    for (let i = 0; i < maxLen; i++) {
        if (i < enLines.length) mergedLines.push(enLines[i]);
        if (i < idLines.length) mergedLines.push(idLines[i]);
    }

    // 4. Tulis ke file output
    console.log(`Menyimpan ke ${OUTPUT_PATH}...`);
    fs.writeFileSync(OUTPUT_PATH, mergedLines.join('\n'), 'utf-8');
    console.log("Selesai! File dataset siap digunakan.");
}

prepareDataset().catch(console.error);
