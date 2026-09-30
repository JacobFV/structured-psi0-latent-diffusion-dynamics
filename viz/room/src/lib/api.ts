/**
 * Data access for the room. Four honest modes, always shown to the viewer:
 *  - live:     the Vite plugin ran the exporter (≤15 s cache; live ops ≤10 s) and served viz/data/<doc>.json
 *  - stale:    the plugin served an older viz/data file because the latest export failed (the error is shown)
 *  - snapshot: a static build with no API; files come from ./data/<doc>.json ("snapshot as of generated_at")
 *  - fixture:  the exporter module does not exist yet (or ?fixture=1): small hand-written FIXTURE JSON, never real results
 * A document that cannot be found is "no data" with the path that was expected. Nothing is filled in.
 */
import { useCallback, useEffect, useState } from 'react';

export type Mode = 'live' | 'stale' | 'snapshot' | 'fixture';
export type Envelope = { schema?: string; generated_at?: string; git_sha?: string | null; sources?: string[]; stale?: boolean } & Record<string, unknown>;
export type Meta = {
  exporter_available: boolean; exporter: string; root: string; out: string; replays_dir: string; replays_dir_exists: boolean;
  video_dir: string; cache_s: number; live_cache_s: number;
};
export type DocResult<T> =
  | { status: 'loading' }
  | { status: 'ok'; data: T; mode: Mode; sourceFile?: string; fileMtime?: string; exportError?: string; fetchedAt: number; etag?: string }
  | { status: 'missing'; expected: string; detail?: string; mode: Mode | 'api' }
  | { status: 'error'; message: string };

const fixtureDocs = import.meta.glob('../../fixtures/*.json', { import: 'default' });
const fixtureReplays = import.meta.glob('../../fixtures/replays/*.json', { import: 'default' });

export function forceFixture() {
  return /(^|[?&])fixture=1(&|$)/.test(window.location.hash.split('?')[1] || '') || /[?&]fixture=1/.test(window.location.search);
}

let metaPromise: Promise<Meta | null> | null = null;
let metaAt = 0;
/** null => no API (static build). */
export function getMeta(): Promise<Meta | null> {
  if (!metaPromise || Date.now() - metaAt > 60_000) {
    metaAt = Date.now();
    metaPromise = fetch('/api/meta', { cache: 'no-store' })
      .then(async (r) => {
        if (!r.ok) return null;
        const ct = r.headers.get('content-type') || '';
        return ct.includes('json') ? ((await r.json()) as Meta) : null;
      })
      .catch(() => null);
  }
  return metaPromise;
}

export function useMeta() {
  const [meta, setMeta] = useState<Meta | null | undefined>(undefined);
  useEffect(() => {
    let live = true;
    getMeta().then((m) => live && setMeta(m));
    return () => { live = false; };
  }, []);
  return meta;
}

async function loadFixture<T>(name: string): Promise<T | null> {
  const key = `../../fixtures/${name}.json`;
  const loader = fixtureDocs[key];
  return loader ? ((await loader()) as T) : null;
}

export async function fetchDoc<T = Envelope>(name: string, prevEtag?: string): Promise<DocResult<T>> {
  const meta = await getMeta();
  const useFixture = forceFixture() || (meta !== null && !meta.exporter_available);
  if (useFixture) {
    const data = await loadFixture<T>(name);
    if (data) return { status: 'ok', data, mode: 'fixture', sourceFile: `viz/room/fixtures/${name}.json`, fetchedAt: Date.now() };
    return { status: 'missing', mode: 'fixture', expected: `viz/room/fixtures/${name}.json`, detail: 'no fixture for this document' };
  }
  if (meta === null) {
    try {
      const r = await fetch(`./data/${name}.json`, { cache: 'no-store' });
      if (!r.ok || !(r.headers.get('content-type') || '').includes('json'))
        return { status: 'missing', mode: 'snapshot', expected: `data/${name}.json (static snapshot; run npm run snapshot before npm run build)` };
      return { status: 'ok', data: (await r.json()) as T, mode: 'snapshot', sourceFile: `data/${name}.json`, fetchedAt: Date.now() };
    } catch (e) {
      return { status: 'error', message: String(e) };
    }
  }
  try {
    // no-cache: the browser revalidates with the plugin's ETag (file mtime + size), so an unchanged 10 MB document is not re-sent
    const r = await fetch(`/api/${name}`, { cache: 'no-cache' });
    if (r.status === 503 || r.status === 404) {
      const body = await r.json().catch(() => ({}));
      return { status: 'missing', mode: 'api', expected: body.expected || `/api/${name}`, detail: body.detail };
    }
    if (!r.ok) return { status: 'error', message: `HTTP ${r.status} for /api/${name}` };
    const etag = r.headers.get('ETag') || undefined;
    if (prevEtag && etag === prevEtag) return { status: 'unchanged' } as unknown as DocResult<T>;
    const data = (await r.json()) as T;
    const exportError = r.headers.get('X-RRP-Export-Error');
    return {
      status: 'ok', data, mode: exportError ? 'stale' : 'live', fetchedAt: Date.now(), etag,
      sourceFile: r.headers.get('X-RRP-Source-File') || undefined,
      fileMtime: r.headers.get('X-RRP-File-Mtime') || undefined,
      exportError: exportError ? decodeURIComponent(exportError) : undefined,
    };
  } catch (e) {
    return { status: 'error', message: String(e) };
  }
}

/** Test hook only: scripts/render-check.tsx seeds documents here to server-render every view with real data. */
declare global { var __RRP_PRELOAD__: Record<string, unknown> | undefined }
function preloaded<T>(name: string): DocResult<T> | null {
  const d = globalThis.__RRP_PRELOAD__?.[name];
  return d ? { status: 'ok', data: d as T, mode: 'live', fetchedAt: 0 } : null;
}

/** Shared per-document store: several panels (ticker, board, views) polling one document share one request. */
const store = new Map<string, { at: number; res: DocResult<unknown>; pending?: Promise<DocResult<unknown>> }>();
const listeners = new Map<string, Set<() => void>>();
async function refreshShared(name: string, maxAgeMs: number) {
  const cur = store.get(name);
  if (cur && Date.now() - cur.at < maxAgeMs) return;
  if (cur?.pending) { await cur.pending; return; }
  const etag = cur && cur.res.status === 'ok' ? (cur.res as { etag?: string }).etag : undefined;
  const pending = fetchDoc<unknown>(name, etag);
  store.set(name, { at: cur?.at ?? 0, res: cur?.res ?? { status: 'loading' }, pending });
  const next = await pending;
  const prev = store.get(name)!;
  let res = prev.res;
  if ((next.status as string) !== 'unchanged' && !(next.status === 'error' && prev.res.status === 'ok')) res = next;
  store.set(name, { at: Date.now(), res });
  listeners.get(name)?.forEach((f) => f());
}

/** Polls a document (default: 60 s; live ops pass 10 s). Keeps the last good value while refreshing. */
export function useDoc<T = Envelope>(name: string, pollMs = 60_000) {
  const initial = (): DocResult<T> => preloaded<T>(name) ?? (store.get(name)?.res as DocResult<T> | undefined) ?? { status: 'loading' };
  const [result, setResult] = useState<DocResult<T>>(initial);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let set = listeners.get(name);
    if (!set) listeners.set(name, (set = new Set()));
    const on = () => setResult((store.get(name)?.res as DocResult<T>) ?? { status: 'loading' });
    set.add(on);
    const tick = (force = false) => { setBusy(true); refreshShared(name, force ? 0 : Math.min(pollMs, 60_000) / 2).finally(() => { setBusy(false); on(); }); };
    tick();
    const id = pollMs > 0 ? window.setInterval(() => tick(), pollMs) : undefined;
    return () => { set!.delete(on); if (id) window.clearInterval(id); };
  }, [name, pollMs]);
  const reload = useCallback(async () => { setBusy(true); await refreshShared(name, 0); setBusy(false); }, [name]);
  return { result, reload, busy };
}

export async function fetchReplay<T>(id: string): Promise<DocResult<T>> {
  const meta = await getMeta();
  if (id.startsWith('fixture-') || forceFixture() || (meta !== null && !meta.exporter_available)) {
    const key = `../../fixtures/replays/${id}.json`;
    const loader = fixtureReplays[key];
    if (loader) return { status: 'ok', data: (await loader()) as T, mode: 'fixture', sourceFile: `viz/room/fixtures/replays/${id}.json`, fetchedAt: Date.now() };
    if (id.startsWith('fixture-')) return { status: 'missing', mode: 'fixture', expected: key };
  }
  if (meta === null) return { status: 'missing', mode: 'snapshot', expected: `replays are not included in static snapshots (${id})` };
  try {
    const r = await fetch(`/api/replay/${encodeURIComponent(id)}`, { cache: 'no-store' });
    if (r.status === 404) {
      const body = await r.json().catch(() => ({}));
      return { status: 'missing', mode: 'api', expected: body.expected || `/api/replay/${id}` };
    }
    if (!r.ok) return { status: 'error', message: `HTTP ${r.status} for replay ${id}` };
    return { status: 'ok', data: (await r.json()) as T, mode: 'live', sourceFile: r.headers.get('X-RRP-Source-File') || undefined, fetchedAt: Date.now() };
  } catch (e) {
    return { status: 'error', message: String(e) };
  }
}

export function useReplay<T>(id: string | undefined) {
  const [result, setResult] = useState<DocResult<T> | null>(() => (id ? preloaded<T>(`replay:${id}`) : null));
  useEffect(() => {
    if (!id) { setResult(null); return; }
    let live = true;
    setResult({ status: 'loading' });
    fetchReplay<T>(id).then((r) => live && setResult(r));
    return () => { live = false; };
  }, [id]);
  return result;
}


const seriesCache = new Map<string, Promise<DocResult<Record<string, unknown>>>>();
/** One training series (`rrp-viz/training-series/v1`) by run id; fixtures embed their series in the index. */
export function fetchTrainingSeries(id: string): Promise<DocResult<Record<string, unknown>>> {
  if (!seriesCache.has(id)) {
    seriesCache.set(id, (async () => {
      const meta = await getMeta();
      const url = meta === null ? `./data/training/${encodeURIComponent(id)}.json` : `/api/training/${encodeURIComponent(id)}`;
      try {
        const r = await fetch(url, { cache: 'no-cache' });
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          return { status: 'missing', mode: meta === null ? 'snapshot' : 'api', expected: body.expected || url } as DocResult<Record<string, unknown>>;
        }
        return { status: 'ok', data: await r.json(), mode: meta === null ? 'snapshot' : 'live', fetchedAt: Date.now() } as DocResult<Record<string, unknown>>;
      } catch (e) {
        return { status: 'error', message: String(e) } as DocResult<Record<string, unknown>>;
      }
    })());
    setTimeout(() => seriesCache.delete(id), 60_000);
  }
  return seriesCache.get(id)!;
}
