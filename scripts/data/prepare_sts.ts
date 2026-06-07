import { parquetRead } from "hyparquet";
import * as fs from "node:fs";

async function prepareDataset() {
    console.log("Membaca file Parquet STS-B menggunakan hyparquet...");
    const fileBuffer = fs.readFileSync("dataset/valid_semantik.parquet");
    
    await parquetRead({
        file: fileBuffer.buffer.slice(fileBuffer.byteOffset, fileBuffer.byteOffset + fileBuffer.byteLength),
        onComplete: (data) => {
            const dataset = data.map((row: any) => ({
                sentence1: row[0],
                sentence2: row[1],
                score: row[2]
            }));
            
            const outputPath = "dataset/sts-b_valid.json";
            fs.writeFileSync(outputPath, JSON.stringify(dataset, null, 2));
            console.log(`Berhasil mengekstrak ${dataset.length} pasang kalimat STS-B.`);
            console.log(`Disimpan ke: ${outputPath}`);
        }
    });
}

prepareDataset().catch(console.error);
