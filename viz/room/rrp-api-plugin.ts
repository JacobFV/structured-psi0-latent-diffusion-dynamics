/**
 * Read-only local data API for the rrp visualization room (viz/CONTRACT.md).
 *
 * - `/api/<doc>` runs the exporter (`python -m rrp.viz.export --only <doc> [--live] --out viz/data`) at most every
 *   15 s (live docs: 10 s), one export at a time, niced, and serves `viz/data/<doc>.json`.
 *   If an export fails but an older file exists, that file is served with `X-RRP-Export-Error` and its mtime, so the
 *   UI can say it is stale. Nothing is invented: a missing document is a 503 naming the path that was expected.
 * - `/api/replay/<id>` serves `~/work/rrp-data/viz/replays/<id>.json[.gz]` (id allowlist regex, path containment).
 * - `/media/<name>` serves `artifacts/video/<name>` (name regex, extension allowlist, Range support for seeking).
 * - `/api/doc?path=` serves allowlisted markdown directly.
 * - `/api/meta` reports what exists, so the UI can decide between live data and clearly labelled fixtures.
 * Only GET/HEAD. Request values never reach subprocess arguments except the doc name from a fixed list.
 */
import { execFile } from 'node:child_process';
import { createReadStream, existsSync } from 'node:fs';
import { copyFile, mkdir, readFile, readdir, stat } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { homedir } from 'node:os';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';

export const DOCS = [
  'overview', 'live', 'dags', 'results', 'edits', 'training', 'robustness', 'physics', 'psi0', 'knowledge', 'replays', 'videos',
] as const;
const LIVE_DOCS = new Set(['live']);
const TTL_MS = 15_000;
const LIVE_TTL_MS = 10_000;
const EXPORT_TIMEOUT_MS = 30_000; // contract: 20 s exporter timeout (+ margin for a cold first export)

type Paths = {
  root: string; python: string; exporter: string; out: string; replays: string; video: string; psi1z: string;
};

function paths(): Paths {
  const root = resolve(process.env.RRP_ROOT || resolve(dirname(fileURLToPath(import.meta.url)), '../..'));
  const home = homedir();
  const candidates = [
    process.env.RRP_PYTHON,
    resolve(root, '.venv/bin/python'),
    resolve(home, 'work/relational-robot-policy/.venv/bin/python'),
  ].filter(Boolean) as string[];
  return {
    root,
    python: candidates.find((p) => existsSync(p)) || candidates[candidates.length - 1],
    exporter: resolve(root, 'src/rrp/viz/api.py'),
    out: resolve(root, 'viz/data'),
    replays: resolve(process.env.RRP_REPLAYS || resolve(home, 'work/rrp-data/viz/replays')),
    video: resolve(root, 'artifacts/video'),
    psi1z: resolve(process.env.RRP_PSI1Z || resolve(home, 'work/psi1z')),
  };
}

function within(base: string, target: string) {
  return target === base || target.startsWith(base + sep);
}

function send(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  for (const [k, v] of Object.entries(headers)) res.setHeader(k, v);
  res.end(typeof body === 'string' ? body : JSON.stringify(body));
}

/** One exporter at a time (host-light, D-127). */
let chain: Promise<unknown> = Promise.resolve();
function serial<T>(job: () => Promise<T>): Promise<T> {
  const next = chain.then(job, job);
  chain = next.catch(() => undefined);
  return next;
}

function helper(p: Paths, args: string[], timeout = EXPORT_TIMEOUT_MS): Promise<string> {
  return new Promise((ok, fail) => {
    execFile('nice', ['-n', '10', p.python, '-m', 'rrp.viz.api', ...args], {
      cwd: p.root,
      timeout,
      maxBuffer: 1 << 20,
      env: { ...process.env, PYTHONPATH: resolve(p.root, 'src'), OMP_NUM_THREADS: '1', OPENBLAS_NUM_THREADS: '1', MKL_NUM_THREADS: '1' },
    }, (error, stdout, stderr) => {
      if (error) fail(new Error(`${error.message}\n${String(stderr).slice(-2000)}`));
      else ok(String(stdout).trim());
    });
  });
}

/** `python -m rrp.viz.api get <doc> [--live] --max-age S` re-exports only when older than S and prints the file path. */
async function runExport(p: Paths, doc: string): Promise<void> {
  const live = LIVE_DOCS.has(doc);
  const args = ['get', doc, '--max-age', String((live ? LIVE_TTL_MS : TTL_MS) / 1000)];
  if (live) args.push('--live');
  await helper(p, args);
}

type Entry = { at: number; pending?: Promise<void>; error?: string };

function docApi(p: Paths) {
  const state = new Map<string, Entry>();
  async function refresh(doc: string) {
    const entry = state.get(doc) || { at: 0 };
    state.set(doc, entry);
    const ttl = LIVE_DOCS.has(doc) ? LIVE_TTL_MS : TTL_MS;
    if (Date.now() - entry.at < ttl) return entry;
    if (!entry.pending) {
      entry.pending = serial(() => runExport(p, doc))
        .then(() => { entry.error = undefined; })
        .catch((e: Error) => { entry.error = e.message; })
        .finally(() => { entry.at = Date.now(); entry.pending = undefined; });
    }
    await entry.pending;
    return entry;
  }
  return async (doc: string, req: IncomingMessage, res: ServerResponse) => {
    const file = resolve(p.out, `${doc}.json`);
    const available = existsSync(p.exporter);
    const entry = available ? await refresh(doc) : { at: 0, error: `exporter not found: ${p.exporter}` };
    try {
      const info = await stat(file);
      const etag = `"${Math.round(info.mtimeMs)}-${info.size}"`;
      const headers: Record<string, string> = {
        'X-RRP-Source-File': file, 'X-RRP-File-Mtime': new Date(info.mtimeMs).toISOString(), ETag: etag,
      };
      if (entry.error) headers['X-RRP-Export-Error'] = encodeURIComponent(entry.error.slice(0, 600));
      if (req.headers['if-none-match'] === etag && !entry.error) {
        res.statusCode = 304;
        for (const [k, v] of Object.entries(headers)) res.setHeader(k, v);
        res.setHeader('Cache-Control', 'no-cache');
        return res.end();
      }
      send(res, 200, await readFile(file, 'utf8'), { ...headers, 'Cache-Control': 'no-cache' });
    } catch {
      send(res, 503, {
        error: 'no data', doc, expected: file, exporter: p.exporter, exporter_available: available,
        detail: entry.error ? entry.error.slice(-1500) : 'the exporter produced no file',
      });
    }
  };
}

const ID_RE = /^[A-Za-z0-9][A-Za-z0-9._=+-]{0,200}$/;
/** id -> file under the replay root (the recorder may nest files in subdirectories); rescanned at most every 15 s. */
let replayMap: { at: number; map: Map<string, string> } = { at: 0, map: new Map() };
async function replayFiles(p: Paths) {
  if (Date.now() - replayMap.at < TTL_MS) return replayMap.map;
  const map = new Map<string, string>();
  async function walk(dir: string, depth: number) {
    if (depth > 4) return;
    let entries: import('node:fs').Dirent[] = [];
    try { entries = await readdir(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      const full = resolve(dir, e.name);
      if (e.isDirectory()) await walk(full, depth + 1);
      else if (e.name.endsWith('.json.gz') || (e.name.endsWith('.json') && e.name !== 'index.json')) {
        const id = e.name.replace(/\.json(\.gz)?$/, '');
        if (!map.has(id) || e.name.endsWith('.json.gz')) map.set(id, full);
      }
    }
  }
  await walk(p.replays, 0);
  replayMap = { at: Date.now(), map };
  return map;
}
async function serveReplay(p: Paths, id: string, res: ServerResponse) {
  if (!ID_RE.test(id)) return send(res, 400, { error: 'bad replay id' });
  const file = (await replayFiles(p)).get(id);
  if (!file || !within(p.replays, file) || !existsSync(file))
    return send(res, 404, { error: 'no data', expected: resolve(p.replays, `**/${id}.json[.gz]`) });
  const info = await stat(file);
  if (info.size > 256 * 1024 * 1024) return send(res, 413, { error: 'replay too large', file });
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-RRP-Source-File', file);
  if (file.endsWith('.gz')) res.setHeader('Content-Encoding', 'gzip');
  res.setHeader('Content-Length', String(info.size));
  createReadStream(file).pipe(res);
}

/** One training series: viz/data/training/<id>.json (written by the training export). */
async function serveTraining(p: Paths, id: string, res: ServerResponse) {
  if (!ID_RE.test(id)) return send(res, 400, { error: 'bad training id' });
  const file = resolve(p.out, 'training', `${id}.json`);
  if (!within(resolve(p.out, 'training'), file) || !existsSync(file)) return send(res, 404, { error: 'no data', expected: file });
  send(res, 200, await readFile(file, 'utf8'), { 'X-RRP-Source-File': file });
}

const MEDIA_RE = /^[A-Za-z0-9][A-Za-z0-9._+=-]{0,250}$/;
const MEDIA_TYPES: Record<string, string> = {
  '.mp4': 'video/mp4', '.webm': 'video/webm', '.gif': 'image/gif', '.png': 'image/png', '.jpg': 'image/jpeg',
};
async function serveMedia(p: Paths, name: string, req: IncomingMessage, res: ServerResponse) {
  const type = MEDIA_TYPES[extname(name).toLowerCase()];
  const file = resolve(p.video, name);
  if (!MEDIA_RE.test(name) || !type || !within(p.video, file)) return send(res, 400, { error: 'not an allowlisted media file' });
  let size = 0;
  try { size = (await stat(file)).size; } catch {
    const alt = resolve(homedir(), 'work/relational-robot-policy/artifacts/video', name);
    try { size = (await stat(alt)).size; return serveFile(alt, size, type, req, res); } catch { return send(res, 404, { error: 'no data', expected: file }); }
  }
  return serveFile(file, size, type, req, res);
}
function serveFile(file: string, size: number, type: string, req: IncomingMessage, res: ServerResponse) {
  res.setHeader('Content-Type', type);
  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Cache-Control', 'no-store');
  const range = /^bytes=(\d*)-(\d*)$/.exec(String(req.headers.range || ''));
  if (range) {
    let start = range[1] ? Number(range[1]) : NaN;
    let end = range[2] ? Number(range[2]) : size - 1;
    if (Number.isNaN(start)) { start = Math.max(0, size - end); end = size - 1; }
    end = Math.min(end, size - 1);
    if (start > end || start >= size) {
      res.statusCode = 416;
      res.setHeader('Content-Range', `bytes */${size}`);
      return res.end();
    }
    res.statusCode = 206;
    res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
    res.setHeader('Content-Length', String(end - start + 1));
    if (req.method === 'HEAD') return res.end();
    createReadStream(file, { start, end }).pipe(res);
    return;
  }
  res.statusCode = 200;
  res.setHeader('Content-Length', String(size));
  if (req.method === 'HEAD') return res.end();
  createReadStream(file).pipe(res);
}

/** Markdown allowlist from the contract: docs/, research/, STATUS.md, README.md, AGENTS.md, psi1z notes. */
function docPath(p: Paths, raw: string): string | null {
  if (!raw || raw.includes('\0') || raw.includes('..') || !raw.endsWith('.md')) return null;
  if (raw.startsWith('psi1z/')) {
    const rel = raw.slice('psi1z/'.length);
    const file = resolve(p.psi1z, rel);
    const ok = /^(README\.md|research\/notes\.md|research\/decisions\.md)$/.test(rel);
    return ok && within(p.psi1z, file) ? file : null;
  }
  if (!/^(STATUS\.md|README\.md|AGENTS\.md|(docs|research)\/[A-Za-z0-9._/ -]+\.md)$/.test(raw)) return null;
  const file = resolve(p.root, raw);
  return within(p.root, file) ? file : null;
}

async function serveDoc(p: Paths, raw: string, res: ServerResponse) {
  const file = docPath(p, raw);
  if (!file) return send(res, 400, { error: 'path not in the allowlist', path: raw });
  try {
    const info = await stat(file);
    if (info.size > 4 * 1024 * 1024) return send(res, 413, { error: 'document too large', path: raw });
    const markdown = await readFile(file, 'utf8');
    send(res, 200, {
      schema: 'rrp-viz/doc/v1', generated_at: new Date().toISOString(), git_sha: null, sources: [file],
      path: raw, modified: new Date(info.mtimeMs).toISOString(), markdown,
    });
  } catch {
    send(res, 404, { error: 'no data', expected: file, path: raw });
  }
}

async function listDocs(p: Paths): Promise<string[]> {
  const out: string[] = [];
  for (const top of ['STATUS.md', 'README.md', 'AGENTS.md']) if (existsSync(resolve(p.root, top))) out.push(top);
  async function walk(rel: string, depth: number) {
    if (depth > 4) return;
    let entries: import('node:fs').Dirent[] = [];
    try { entries = await readdir(resolve(p.root, rel), { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      const child = `${rel}/${e.name}`;
      if (e.isDirectory()) await walk(child, depth + 1);
      else if (e.name.endsWith('.md')) out.push(child);
    }
  }
  await walk('docs', 0);
  await walk('research', 0);
  for (const f of ['README.md', 'research/notes.md', 'research/decisions.md'])
    if (existsSync(resolve(p.psi1z, f))) out.push(`psi1z/${f}`);
  return out.sort();
}

export function rrpApi(): Plugin {
  const p = paths();
  const docs = docApi(p);
  return {
    name: 'rrp-viz-api',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url || '/', 'http://127.0.0.1');
        const path = decodeURIComponent(url.pathname);
        if (!path.startsWith('/api/') && !path.startsWith('/media/')) return next();
        if (req.method !== 'GET' && req.method !== 'HEAD') return send(res, 405, { error: 'read-only API' });
        try {
          if (path.startsWith('/media/')) return await serveMedia(p, path.slice('/media/'.length), req, res);
          const rest = path.slice('/api/'.length);
          if (rest === 'meta') {
            return send(res, 200, {
              schema: 'rrp-viz/meta/v1', generated_at: new Date().toISOString(),
              exporter_available: existsSync(p.exporter), exporter: p.exporter, python: p.python, root: p.root,
              out: p.out, replays_dir: p.replays, replays_dir_exists: existsSync(p.replays), video_dir: p.video,
              cache_s: TTL_MS / 1000, live_cache_s: LIVE_TTL_MS / 1000, docs: DOCS,
            });
          }
          if (rest === 'doc') return await serveDoc(p, url.searchParams.get('path') || '', res);
          if (rest === 'doclist') return send(res, 200, { schema: 'rrp-viz/doclist/v1', generated_at: new Date().toISOString(), docs: await listDocs(p) });
          if (rest.startsWith('replay/')) return await serveReplay(p, rest.slice('replay/'.length), res);
          if (rest.startsWith('training/')) return await serveTraining(p, rest.slice('training/'.length), res);
          if ((DOCS as readonly string[]).includes(rest)) return await docs(rest, req, res);
          return send(res, 404, { error: 'unknown API path', path });
        } catch (e) {
          return send(res, 500, { error: String((e as Error).message || e).slice(0, 500) });
        }
      });
    },
    /** Static builds carry a snapshot of viz/data (if present) under dist/data; the UI labels it "snapshot". */
    async writeBundle(options) {
      const dir = options.dir;
      if (!dir || !existsSync(p.out)) return;
      await mkdir(resolve(dir, 'data'), { recursive: true });
      for (const name of await readdir(p.out)) {
        if (name.endsWith('.json')) await copyFile(resolve(p.out, name), resolve(dir, 'data', name));
      }
    },
  };
}
