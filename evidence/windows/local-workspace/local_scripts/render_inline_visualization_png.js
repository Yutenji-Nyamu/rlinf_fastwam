const fs = require('fs');
const { chromium } = require('playwright');

async function main() {
  const [fragmentPath, stylesheetPath, outputPath] = process.argv.slice(2);
  if (!fragmentPath || !stylesheetPath || !outputPath) {
    throw new Error('usage: node render_inline_visualization_png.js <fragment> <visualize.css> <output.png>');
  }
  const fragment = fs.readFileSync(fragmentPath, 'utf8');
  const stylesheet = fs.readFileSync(stylesheetPath, 'utf8');
  const executablePath = process.env.PLAYWRIGHT_CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  const browser = await chromium.launch({ headless: true, executablePath });
  try {
    const page = await browser.newPage({
      viewport: { width: 1200, height: 1000 },
      colorScheme: 'light',
      deviceScaleFactor: 1.5,
    });
    await page.setContent(`<!doctype html><html><head><meta charset="utf-8"><style>${stylesheet}\nhtml,body{margin:0;background:var(--background)}body{padding:18px}</style></head><body>${fragment}</body></html>`, { waitUntil: 'load' });
    await page.waitForTimeout(200);
    const height = await page.evaluate(() => Math.ceil(document.documentElement.scrollHeight));
    await page.setViewportSize({ width: 1200, height: Math.max(1000, Math.min(height, 5000)) });
    await page.screenshot({ path: outputPath, fullPage: true });
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
