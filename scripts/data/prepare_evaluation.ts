import { parquetRead } from "hyparquet";
import * as fs from "node:fs";

async function prepareDataset() {
    console.log("Membaca file Parquet STS-B menggunakan hyparquet...");
    const fileBuffer = fs.readFileSync("dataset/train_semantik.parquet");
    
    await parquetRead({
        file: fileBuffer.buffer.slice(fileBuffer.byteOffset, fileBuffer.byteOffset + fileBuffer.byteLength),
        onComplete: (data) => {
            console.log("Data loaded, first 2 rows:");
            console.log(data.map(col => col.slice(0, 2)));
        }
    });
}

prepareDataset().catch(console.error);
