// Run with Playwright installed, or set PLAYWRIGHT_MODULE to its module path.
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'outputs/map-aggregation-comparison');
fs.mkdirSync(output, { recursive: true });
const html = `<!doctype html><html><head><meta charset="utf-8">
<style>body{margin:0;padding:20px;background:white;font-family:Arial}#map{width:900px}</style>
<script src="https://cdn.jsdelivr.net/npm/vega@5/build/vega.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5/build/vega-lite.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6/build/vega-embed.min.js"></script>
</head><body><div id="map"></div></body></html>`;
const server = http.createServer((request, response) => {
  if (request.url === '/__benchmark__') {
    response.setHeader('Content-Type', 'text/html');
    response.end(html);
    return;
  }
  const file = path.resolve(root, '.' + decodeURIComponent(request.url.split('?')[0]));
  if (!file.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
  fs.readFile(file, (error, data) => {
    if (error) { response.writeHead(404).end(); return; }
    response.setHeader('Cache-Control', 'no-store');
    response.setHeader('Content-Type', file.endsWith('.json') ? 'application/json' : 'text/csv');
    response.end(data);
  });
});

(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 940, height: 560 }, deviceScaleFactor: 1 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base + '/__benchmark__');
    const session = await page.context().newCDPSession(page);
    await session.send('Network.setCacheDisabled', { cacheDisabled: true });
    const runs = [], hovers = [];
    for (let trial = 0; trial < 6; trial++) {
      for (const kind of trial % 2 ? ['aggregated', 'original'] : ['original', 'aggregated']) {
        const specPath = kind === 'original'
          ? 'charts/heatwave_frequency_map.json'
          : 'charts/heatwave_frequency_map_reduced_test.json';
        const timing = await page.evaluate(async ({ base, specPath }) => {
          if (window.currentView) window.currentView.finalize();
          document.querySelector('#map').innerHTML = '';
          const spec = await (await fetch(base + '/' + specPath)).json();
          performance.clearResourceTimings();
          const start = performance.now();
          const result = await vegaEmbed('#map', spec, {
            renderer: 'canvas', actions: false, loader: { baseURL: base + '/' }
          });
          window.currentView = result.view;
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
          const elapsed = performance.now() - start;
          const csv = performance.getEntriesByType('resource').find(r => r.name.includes('.csv'));
          return { elapsed_ms: elapsed, csv_fetch_ms: csv.duration, decoded_bytes: csv.decodedBodySize };
        }, { base, specPath });
        runs.push({ kind, trial, warmup: trial === 0, ...timing });
        console.log(JSON.stringify(runs.at(-1)));
        if (trial === 5) {
          await page.locator('#map').screenshot({ path: path.join(output, kind + '.png') });
          const hover = await page.evaluate(async () => {
            const canvas = document.querySelector('#map canvas');
            const rect = canvas.getBoundingClientRect();
            const samples = [];
            // Sweep through mainland interiors; synthetic pointermove dispatch
            // measures synchronous Canvas hit testing and tooltip DOM updates.
            for (let i = 0; i < 20; i++) {
              const x = 330 + (i % 10) * 24;
              const y = 205 + Math.floor(i / 10) * 25;
              const before = document.querySelector('#vg-tooltip-element')?.textContent || '';
              const start = performance.now();
              canvas.dispatchEvent(new PointerEvent('pointermove', {
                clientX: rect.left + x, clientY: rect.top + y,
                bubbles: true, view: window, pointerType: 'mouse', pointerId: 1
              }));
              const dispatch_ms = performance.now() - start;
              const tooltip = document.querySelector('#vg-tooltip-element');
              const visible = !!tooltip?.classList.contains('visible');
              const changed = !!tooltip && tooltip.textContent !== before;
              await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
              const to_frame_ms = performance.now() - start;
              samples.push({ dispatch_ms, to_frame_ms, visible, changed });
            }
            return samples;
          });
          hovers.push({ kind, samples: hover });
          if (!hover.some(sample => sample.visible && sample.changed)) {
            throw Error('Hover benchmark did not trigger tooltips for ' + kind);
          }
          await page.screenshot({ path: path.join(output, kind + '-hover.png') });
        }
      }
    }
    const report = {
      browser: browser.version(),
      method: 'Local HTTP, cache disabled, Canvas at 900px/DPR 1; one warm-up and five measured alternating trials. Load-to-paint includes CSV fetch/parse, transforms, rendering and two animation frames; excludes CDN/spec fetch. Hover: 20 synthetic pointermove events over mainland positions; dispatch_ms measures synchronous hit testing and tooltip update; to_frame_ms includes two animation frames and intervening redraw work. Approximate browser responsiveness, not physical input/display latency.',
      runs, hovers, errors
    };
    fs.writeFileSync(path.join(output, 'benchmark.json'), JSON.stringify(report, null, 2) + '\n');
    if (errors.length) throw Error(errors.join('\n'));
  } finally {
    await browser.close();
    server.close();
  }
})().catch(error => { console.error(error); server.close(); process.exitCode = 1; });
