/**
 * Deep links: `#<view>?key=value&...`. Every view keeps its selection (filters, replay ids, time, run, doc) in the
 * hash query so a link reproduces what the viewer saw. Scrubbing writes are debounced replaceState; view changes push.
 */
import { useCallback, useEffect, useState } from 'react';

function split(hash: string) {
  const raw = hash.replace(/^#/, '');
  const at = raw.indexOf('?');
  return { path: at < 0 ? raw : raw.slice(0, at), query: new URLSearchParams(at < 0 ? '' : raw.slice(at + 1)) };
}
export function currentView() {
  return decodeURIComponent(split(window.location.hash).path);
}
export function readParam(key: string) {
  return split(window.location.hash).query.get(key) ?? undefined;
}

let pending: Record<string, string | undefined> = {};
let timer: number | undefined;
function flush() {
  const { path, query } = split(window.location.hash);
  for (const [k, v] of Object.entries(pending)) {
    if (v === undefined || v === '') query.delete(k);
    else query.set(k, v);
  }
  pending = {};
  const text = query.toString();
  const next = `#${path}${text ? `?${text}` : ''}`;
  if (next !== window.location.hash) window.history.replaceState(null, '', next);
}
export function writeParams(updates: Record<string, string | undefined>) {
  Object.assign(pending, updates);
  if (timer) window.clearTimeout(timer);
  timer = window.setTimeout(flush, 180);
}

/** Navigate to a view, optionally with a fresh query (keeps `fixture=1`). */
export function go(view: string, params: Record<string, string | undefined> = {}) {
  const q = new URLSearchParams();
  if (readParam('fixture') === '1') q.set('fixture', '1');
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== '') q.set(k, v);
  const text = q.toString();
  window.location.hash = `${view}${text ? `?${text}` : ''}`;
}
export function href(view: string, params: Record<string, string | undefined> = {}) {
  const q = new URLSearchParams();
  if (typeof window !== 'undefined' && readParam('fixture') === '1') q.set('fixture', '1');
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== '') q.set(k, v);
  const text = q.toString();
  return `#${view}${text ? `?${text}` : ''}`;
}

/** String state mirrored into the URL. Re-reads when the hash changes (Back/links). */
export function useUrlState(key: string, fallback = '') {
  const [value, setValue] = useState(() => readParam(key) ?? fallback);
  useEffect(() => {
    const on = () => setValue(readParam(key) ?? fallback);
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, [key, fallback]);
  const set = useCallback(
    (next: string) => {
      setValue(next);
      writeParams({ [key]: next === fallback ? undefined : next });
    },
    [key, fallback],
  );
  return [value, set] as const;
}

export function useHashView(defaultView: string) {
  const [view, setView] = useState(() => currentView() || defaultView);
  useEffect(() => {
    const on = () => setView(currentView() || defaultView);
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, [defaultView]);
  return view;
}
