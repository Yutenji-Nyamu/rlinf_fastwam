const sharp = require('sharp');
const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('usage: node svg_to_png_sharp_20260828.js input.svg output.png');
sharp(input).png().toFile(output).then(info => console.log(JSON.stringify(info)));
