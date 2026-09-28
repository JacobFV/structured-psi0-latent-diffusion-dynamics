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
  ['board', () => import('../src/views/Board'), ['board']],
  ['evidence', () => import('../src/views/Evidence'), ['results', 'results?data=1', 'results?data=1&mode=delta&dd=grasp_version', 'results?agg=pool&r1=route&c=metric&metric=', 'results?q=compare_gc2_final&metric=grasp_v2',
    'edits', 'edits?body=go2', 'robustness', 'robustness?m=motion.joint_jerk_rms', 'physics', 'physics?data=1', 'psi0', 'psi0?data=1', 'edits?data=1', 'robustness?data=1']],
  ['runs', () => import('../src/views/RunHistory'), ['runs', 'runs?list=videos', 'runs?demo=1']],
  ['ops', () => import('../src/views/Ops'), ['live', 'training']],
  ['library', () => import('../src/views/Library'), ['knowledge', 'knowledge?tab=crosswalk', 'knowledge?tab=roadmap&q=%2313', 'knowledge?tab=backlog', 'knowledge?tab=strategy', 'knowledge?tab=status', 'knowledge?tab=docs', 'knowledge?tab=decisions&d=D-100', 'overview']],
];
const DOCS = ['overview', 'live', 'dags', 'results', 'edits', 'training', 'robustness', 'physics', 'psi0', 'knowledge', 'replays', 'videos'];

function load(dir: string) {
  const out: Record<string, unknown> = {};
  for (const d of DOCS) { const f = resolve(dir, `${d}.json`); if (existsSync(f)) out[d] = JSON.parse(readFileSync(f, 'utf8')); }
  return out;
}

async function main() {
  let failures = 0;
  const lensMods = await Promise.all([import('../src/views/Evidence'), import('../src/views/Ops'), import('../src/views/Library')]);
  g.__RRP_LENS__ = {};
  for (const l of [...lensMods[0].EVIDENCE, ...lensMods[1].OPS, ...lensMods[2].LIBRARY]) (g.__RRP_LENS__ as Record<string, unknown>)[l.id] = (await l.load()).default;
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
    for (const f of readdirSync(rdir).filter((x) => x.endsWith('.json.gz'))) {
      const r = JSON.parse(gunzipSync(readFileSync(resolve(rdir, f))).toString('utf8'));
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
          const flags = [/No data/.test(html) ? 'NO-DATA' : '', /Could not load/.test(html) ? 'ERROR-STATE' : '',
            /\bNaN\b/.test(text) ? `NaN×${(text.match(/\bNaN\b/g) || []).length}` : '', /\bundefined\b/.test(text) ? `undefined×${(text.match(/\bundefined\b/g) || []).length}` : '',
            /\[object Object\]/.test(text) ? 'OBJECT-STRING' : ''].filter(Boolean).join(' ');
          if (process.env.DUMP === `${label}:${name}:${q}`) console.log(text.replace(/\s+/g, ' ').slice(0, 20000));
          console.log(`ok   ${label.padEnd(7)} ${name.padEnd(10)} ${q.padEnd(40)} ${String(html.length).padStart(8)} chars ${ms.toFixed(0).padStart(5)} ms ${flags}`);
        } catch (e) {
          failures++;
          console.log(`FAIL ${label.padEnd(7)} ${name.padEnd(10)} ${q.padEnd(40)} ${(e as Error).stack?.split('\n').slice(0, 6).join('\n    ')}`);
        }
      }
    }
  }
  // Board budget at 1440×900: every tile's chart must fit its tile (no vertical overflow).
  const bm = await import('../src/views/Board');
  for (const [label, docs] of sets.slice(0, 2)) {
    const needs = bm.tileNeeds(docs as Record<string, never>);
    for (const [tile, need] of Object.entries(needs)) {
      const total = bm.TILE_CHROME + need;
      const fits = total <= bm.TILE_H;
      console.log(`${fits ? 'ok  ' : 'FAIL'} budget  ${label.padEnd(7)} board tile ${tile.padEnd(11)} needs ${total}px of ${bm.TILE_H}px (2 rows in ${bm.VIEWPORT_H}px)`);
      if (!fits) failures++;
    }
  }
  process.exit(failures ? 1 : 0);
}
main();
