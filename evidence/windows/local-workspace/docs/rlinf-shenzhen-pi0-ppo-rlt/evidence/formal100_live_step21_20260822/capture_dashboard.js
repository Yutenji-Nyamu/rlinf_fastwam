const path = require('path');
const { chromium } = require('playwright');
const sharp = require('sharp');

(async () => {
  const dir = __dirname;
  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  });
  const page = await browser.newPage({
    viewport: { width: 1180, height: 900 },
    deviceScaleFactor: 2,
  });
  const errors = [];
  page.on('pageerror', err => errors.push(String(err)));
  await page.goto('file:///' + path.join(dir, 'ppo_step22_dashboard.html').replace(/\\/g, '/'));
  await page.waitForLoadState('load');
  await page.waitForTimeout(250);

  const requested = process.argv[2] || 'all';
  const outputs = [
    ['#training-export', 'ppo_step22_training_trends_mobile.png'],
    ['#resource-export', 'ppo_step22_resource_snapshot_mobile.png'],
  ].filter(([selector]) => requested === 'all' || selector === `#${requested}-export`);
  for (const [selector, filename] of outputs) {
    await page.locator(selector).screenshot({ path: path.join(dir, filename) });
    const meta = await sharp(path.join(dir, filename)).metadata();
    console.log(`${filename}\t${meta.width}x${meta.height}\t${meta.format}`);
  }

  const successBox = await page.locator('#success-chart').boundingBox();
  await page.mouse.move(successBox.x + successBox.width * 0.74, successBox.y + successBox.height * 0.45);
  const hoverCheck = await page.locator('#tooltip').evaluate(node => ({
    visible: getComputedStyle(node).display !== 'none',
    text: node.textContent.trim(),
  }));

  const checks = await page.evaluate(() => ({
    title: document.title,
    charts: [...document.querySelectorAll('svg.chart')].map(x => ({ id: x.id, paths: x.querySelectorAll('path.series').length })),
    summaryCards: document.querySelectorAll('#summary-metrics .metric').length,
    gpuRows: document.querySelectorAll('#gpu-list .gpu-row').length,
    trainingHeight: document.querySelector('#training-export').getBoundingClientRect().height,
    resourceHeight: document.querySelector('#resource-export').getBoundingClientRect().height,
  }));
  console.log(JSON.stringify({ checks, hoverCheck, errors }, null, 2));
  await browser.close();
})().catch(err => { console.error(err); process.exit(1); });
