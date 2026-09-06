const fs = require("fs");

const path = "E:/Codex/home/visualizations/2026/07/28/019fa85b-0b0c-72d0-a9ea-a9d88c48e8aa/qam-live-cycle-238.html";
const text = fs.readFileSync(path, "utf8");
const match = text.match(/<script>([\s\S]*?)<\/script>/);
if (!match) throw new Error("missing script");
new Function(match[1]);
console.log("JS_SYNTAX_OK");
