/**
 * Server-renders every view with real exporter documents (viz/data) and with the fixtures, to catch render-time crashes
 * without a browser (host rule: no browsers on the host). Effects (fetch, three.js, timers) do not run under SSR.
 * Build + run: RRP_NO_SNAPSHOT=1 npx vite build --ssr scripts/render-check.tsx --outDir <tmp> && node <tmp>/render-check.js <repo>
 */
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { resolve } from 'node:path';
import type { ComponentType } from 'react';
import { renderToString } from 'react-dom/server';

const repo = process.argv[2] || resolve(process.cwd(), '../..');
const g = globalThis as unknown as Record<string, unknown>;
g.window = {
  location: { hash: '', search: '' }, history: { replaceState() {}, pushState() {} }, addEventListener() {}, removeEventListener() {},
  setTimeout, clearTimeout, setInterval: () => 0, clearInterval() {}, matchMedia: () => ({ matches: false }),
};
g.localStorage = { getItem: () => null, setItem() {} };

// [label, component, hashes (view?query)]: every consolidated view and every lens
const VIEWS: [string, () => Promise<{ default: ComponentType }>, string[]][] = [
  ['ticker', () => import('../src/components/Ticker'), ['board']],
  ['overview', () => import('../src/views/OverviewView'), ['overview', 'claims']],
  ['evaluations', () => import('../src/views/Evidence'), ['results', 'results?data=1', 'results?data=1&mode=delta&dd=grasp_version', 'results?agg=pool&r1=route&c=metric&metric=', 'results?q=compare_gc2_final&metric=grasp_v2',
    'radar', 'edits', 'edits?body=go2', 'robustness', 'robustness?m=motion.joint_jerk_rms', 'physics', 'physics?data=1', 'psi0', 'psi0?data=1', 'edits?data=1', 'robustness?data=1']],
  ['runs', () => import('../src/views/RunHistory'), ['runs', 'runs?list=videos', 'runs?demo=1', 'runs?env=legged&res=failure', 'runs?env=physics&q=body:go2', 'runs?env=dual&task=handover%20%C2%B7%20teacher']],
  ['training', () => import('../src/views/TrainingView'), ['training', 'training?data=1', 'live']],
  ['overview-docs', () => import('../src/views/OverviewView'), ['knowledge', 'knowledge?tab=crosswalk', 'knowledge?tab=roadmap&q=%2313', 'knowledge?tab=backlog', 'knowledge?tab=strategy', 'knowledge?tab=status', 'knowledge?tab=docs', 'knowledge?tab=decisions&d=D-100']],
];
const DOCS = ['radar', 'overview', 'live', 'dags', 'results', 'edits', 'training', 'robustness', 'physics', 'psi0', 'knowledge', 'replays', 'videos'];

function load(dir: string) {
  const out: Record<string, unknown> = {};
  for (const d of DOCS) { const f = resolve(dir, `${d}.json`); if (existsSync(f)) out[d] = JSON.parse(readFileSync(f, 'utf8')); }
  return out;
}

const paneCount = new Map<string, number>();
async function main() {
  let failures = 0;
  const lensMods = await Promise.all([import('../src/views/Evidence'), import('../src/views/TrainingView'), import('../src/views/OverviewView')]);
  g.__RRP_LENS__ = {};
  for (const l of [...lensMods[0].EVIDENCE, ...lensMods[1].TRAINING, ...lensMods[2].OVERVIEW]) (g.__RRP_LENS__ as Record<string, unknown>)[l.id] = (await l.load()).default;
  const sets: [string, Record<string, unknown>][] = [
    ['real', load(resolve(repo, 'viz/data'))],
    ['fixture', load(resolve(repo, 'viz/room/fixtures'))],
  ];
  // the theatre with fixture replays preloaded (index + both replays)
  const fx = sets[1][1];
  for (const f of readdirSync(resolve(repo, 'viz/room/fixtures/replays'))) fx[`replay:${f.replace(/\.json$/, '')}`] = JSON.parse(readFileSync(resolve(repo, 'viz/room/fixtures/replays', f), 'utf8'));
  // optional: real replay files (REPLAY_DIR=<dir of *.json.gz>) rendered in the theatre with an index built from their meta
  const rdir = process.env.REPLAY_DIR;
  if (rdir) {
    const docs: Record<string, unknown> = { ...sets[0][1] };
    const idx: Record<string, unknown>[] = [];
    const files: string[] = [];
    const walk = (d: string) => { for (const e of readdirSync(d, { withFileTypes: true })) { const p = resolve(d, e.name); if (e.isDirectory()) walk(p); else if (e.name.endsWith('.json.gz')) files.push(p); } };
    walk(rdir);
    for (const f of files) {
      const r = JSON.parse(gunzipSync(readFileSync(f)).toString('utf8'));
      docs[`replay:${r.id}`] = r;
      const m = r.meta;
      idx.push({ id: r.id, family: m.family, task: m.task, body: m.body, route: m.route, source_label: m.source_label, variant: m.variant, seed: m.seed, condition: m.condition, success: m.success, n_frames: r.n_frames, fps: r.fps });
    }
    docs.replays = { schema: 'rrp-viz/replays/v1', replays: idx };
    // geometry: every geom must build a three.js mesh with a finite bounding box (no WebGL needed)
    const { geomObject } = await import('../src/components/Stage');
    for (const [k, r] of Object.entries(docs)) {
      if (!k.startsWith('replay:')) continue;
      const geoms = (r as { geoms: Parameters<typeof geomObject>[0][] }).geoms;
      let built = 0, bad = 0;
      for (const gm of geoms) {
        const o = geomObject(gm);
        if (!o) { bad++; continue; }
        o.geometry.computeBoundingBox();
        const b = o.geometry.boundingBox!;
        if (![b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z].every(Number.isFinite)) bad++; else built++;
      }
      console.log(`geom ${k.slice(7)}: ${built} built, ${bad} skipped/bad of ${geoms.length}`);
      if (bad) failures++;
    }
    sets.push(['replays', docs]);
    VIEWS.splice(0, VIEWS.length, ['runs', () => import('../src/views/RunHistory'), idx.flatMap((a, i) => [`runs?a=${a.id}`, `runs?a=${a.id}&b=${idx[(i + 1) % idx.length].id}`])]);
  }
  for (const [label, docs] of sets) {
    if (rdir && label !== 'replays') continue; // REPLAY_DIR mode renders only the real runs
    g.__RRP_PRELOAD__ = docs;
    g.__RRP_EAGER__ = true;
    for (const [name, loader, queries] of VIEWS) {
      const View = (await loader()).default;
      for (const q of queries) {
        const extra = label === 'fixture' && name === 'runs' ? '&a=fixture-arm-unedited&b=fixture-arm-halt' : '';
        (g.window as { location: { hash: string } }).location.hash = `#${q}${q.includes('?') ? '' : '?'}${extra}`;
        const t0 = performance.now();
        try {
          const html = renderToString(<View />);
          const ms = performance.now() - t0;
          const text = html.replace(/<[^>]+>/g, ' ');
          // type scale (IBM-2 10/11/12): no inline font size above 12 px; no SVG that can scale its text up
          const big = [...html.matchAll(/font-size(?:="|:\s*)([\d.]+)(?:px)?/g)].map((m) => Number(m[1])).filter((v) => v > 12);
          const scaled = [...html.matchAll(/<svg[^>]*viewBox[^>]*>/g)].filter((m) => !/\sheight="\d/.test(m[0]));
          if (big.length || scaled.length) { failures++; console.log(`FAIL type ${label} ${name} ${q}: font sizes ${big.join(',')} · scalable svgs ${scaled.length}`); }
          const flags = [/No data/.test(html) ? 'NO-DATA' : '', /Could not load/.test(html) ? 'ERROR-STATE' : '',
            /\bNaN\b/.test(text) ? `NaN×${(text.match(/\bNaN\b/g) || []).length}` : '', /\bundefined\b/.test(text) ? `undefined×${(text.match(/\bundefined\b/g) || []).length}` : '',
            /\[object Object\]/.test(text) ? 'OBJECT-STRING' : ''].filter(Boolean).join(' ');
          if (name === 'runs') for (const m of html.matchAll(/class="rh-pane"><header>([^<]+)/g)) paneCount.set(`${label}:${m[1]}`, (paneCount.get(`${label}:${m[1]}`) || 0) + 1);
          if (process.env.DUMP === `${label}:${name}:${q}`) console.log(text.replace(/\s+/g, ' ').slice(0, 20000));
          console.log(`ok   ${label.padEnd(7)} ${name.padEnd(10)} ${q.padEnd(40)} ${String(html.length).padStart(8)} chars ${ms.toFixed(0).padStart(5)} ms ${flags}`);
        } catch (e) {
          failures++;
          console.log(`FAIL ${label.padEnd(7)} ${name.padEnd(10)} ${q.padEnd(40)} ${(e as Error).stack?.split('\n').slice(0, 6).join('\n    ')}`);
        }
      }
    }
  }
  if (paneCount.size) console.log('run-history panels rendered (count over all run renders):', [...paneCount.entries()].map(([k, v]) => `${k} ${v}`).join(' · '));
  // CSS audit: every font size in styles.css ≤ 12 px except the one headline token (used only by the headline rule)
  const css = readFileSync(resolve(repo, 'viz/room/src/styles.css'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
  const cssBad: string[] = [];
  for (const m of css.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
    const sel = m[1].trim();
    for (const d of m[2].split(';')) {
      const [prop, val = ''] = d.split(':').map((x) => x.trim());
      if (prop === '--fs-headline') continue;
      if (!/^font(-size)?$/.test(prop) && !/^--fs-/.test(prop)) continue;
      const px = [...val.matchAll(/([\d.]+)px/g)].map((x) => Number(x[1])).filter((v) => v > 12);
      if (px.length) cssBad.push(`${sel} { ${prop}: ${val} }`);
      if (/--fs-headline/.test(val) && !/\.big|\.headline/.test(sel)) cssBad.push(`${sel} uses the headline size`);
    }
  }
  console.log(`${cssBad.length ? 'FAIL' : 'ok  '} css type scale: ${cssBad.length ? cssBad.join(' | ') : 'all font sizes ≤ 12 px except .big/.headline'}`);
  if (cssBad.length) failures++;
  // Board budget at 1440×900: every panel's content must fit its fixed cell (no vertical overflow)
  const bm = await import('../src/views/Board');
  for (const [label, docs] of sets.slice(0, 2)) {
    const needs = bm.panelNeeds(docs as Record<string, never>);
    for (const [panel, { need, cells }] of Object.entries(needs)) {
      const avail = cells * bm.CELL_H - bm.HEAD_H;
      const fits = need <= avail;
      console.log(`${fits ? 'ok  ' : 'FAIL'} budget  ${label.padEnd(7)} board ${panel.padEnd(11)} needs ${need}px of ${avail}px (${cells} cell${cells > 1 ? 's' : ''}; ${bm.KPI_H}px KPI strip; ${bm.VIEWPORT_H}px screen)`);
      if (!fits) failures++;
    }
  }
  process.exit(failures ? 1 : 0);
}
main();
