const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require('C:/Users/86136/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = path.resolve(__dirname, '../docs/rlinf-shenzhen-multitask-pi05/evidence/formal100-summary-20260905');
const qa = base + '-qa';
(async () => {
  const browser = await chromium.launch({headless:true, channel:'msedge'});
  const page = await browser.newPage({viewport:{width:1200,height:950}});
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push(request.url()));
  await page.goto(pathToFileURL(path.join(base,'dashboard.html')).href);
  const options = await page.locator('#metric option').count();
  for (let i=0;i<options;i++) {
    await page.locator('#metric').selectOption(String(i));
    await page.locator('canvas').hover({position:{x:350,y:160}});
    if (!(await page.locator('#detail').innerText()).includes(':')) throw new Error('No tooltip for metric '+i);
  }
  await page.locator('#metric').selectOption('0');
  await page.locator('#legend input').first().uncheck();
  await page.locator('#legend input').first().check();
  const links = await page.locator('a').evaluateAll(items=>items.map(item=>item.getAttribute('href')));
  for (const link of links) if (!fs.existsSync(path.resolve(base,link))) throw new Error('Broken link: '+link);
  fs.mkdirSync(qa,{recursive:true});
  await page.mouse.move(3,3);
  await page.screenshot({path:path.join(qa,'desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  const overflow = await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  await page.screenshot({path:path.join(qa,'mobile.png'),fullPage:true});
  const externalRequests = requests.filter(url=>/^https?:/.test(url));
  if (overflow || errors.length || externalRequests.length) throw new Error(JSON.stringify({overflow,errors,externalRequests}));
  const result = {metricOptions:options,linksVerified:links.length,pageErrors:errors,mobileOverflow:overflow,externalRequests,qa};
  fs.writeFileSync(path.join(qa,'checks.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(error=>{console.error(error);process.exit(1)});
