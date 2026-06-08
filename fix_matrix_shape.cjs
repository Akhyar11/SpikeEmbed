const fs = require('fs');
const path = require('path');

function walk(dir) {
    let results = [];
    const list = fs.readdirSync(dir);
    list.forEach(function(file) {
        file = path.join(dir, file);
        const stat = fs.statSync(file);
        if (stat && stat.isDirectory()) { 
            results = results.concat(walk(file));
        } else { 
            if (file.endsWith('.ts')) results.push(file);
        }
    });
    return results;
}

const files = walk('scripts');
let changed = 0;

for (const file of files) {
    let content = fs.readFileSync(file, 'utf8');
    let original = content;
    
    // Revert Matrix.fromFlat(data, [size, 1]) to Matrix.fromFlat(data, [size] as any)
    content = content.replace(/Matrix\.fromFlat\(([^,]+),\s*\[([^,\]]+),\s*1\]\)/g, "Matrix.fromFlat($1, [$2] as any)");

    if (content !== original) {
        fs.writeFileSync(file, content);
        changed++;
        console.log("Fixed", file);
    }
}
console.log("Modified", changed, "files.");
