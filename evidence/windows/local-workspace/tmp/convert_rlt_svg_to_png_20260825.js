const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

async function main() {
  const dir = process.argv[2];
  if (!dir) throw new Error('usage: node convert_rlt_svg_to_png_20260825.js <dir>');
  for (const name of fs.readdirSync(dir).filter((x) => x.endsWith('.svg'))) {
    const source = path.join(dir, name);
    const target = path.join(dir, name.replace(/\.svg$/, '.png'));
    await sharp(source, { density: 144 }).png().toFile(target);
    console.log(`${source} -> ${target}`);
  }
}

main().catch((error) => { console.error(error); process.exit(1); });
