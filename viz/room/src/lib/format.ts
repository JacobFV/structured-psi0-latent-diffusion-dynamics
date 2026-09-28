/** Formatting, tolerant field access and small statistics. Pure functions. */

export type Row = Record<string, unknown>;

export function isObj(v: unknown): v is Row {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}
export function arr(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}
export function rows(v: unknown): Row[] {
  if (Array.isArray(v)) return v.filter(isObj);
  if (isObj(v)) {
    // {key: {...}} maps become rows with a `key` column
    return Object.entries(v).map(([key, val]) => (isObj(val) ? { key, ...val } : { key, value: val }));
  }
  return [];
}
/** First present field among `keys`. */
export function pick(o: unknown, ...keys: string[]): unknown {
  if (!isObj(o)) return undefined;
  for (const k of keys) if (o[k] !== undefined && o[k] !== null) return o[k];
  return undefined;
}
export function num(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) return Number(v);
  return null;
}
export function str(v: unknown): string {
  if (v === null || v === undefined) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  return JSON.stringify(v);
}
/** Text of an item that may be a string or an object with a text-like field. */
export function text(o: unknown): string {
  if (typeof o === 'string') return o;
  return str(pick(o, 'text', 'claim', 'title', 'summary', 'name', 'item', 'label', 'description', 'body'));
}

export function fmtNum(v: unknown, digits = 3): string {
  const n = num(v);
  if (n === null) return v === null || v === undefined ? '—' : str(v);
  const a = Math.abs(n);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return n.toExponential(2);
  if (Number.isInteger(n)) return n.toLocaleString('en-US');
  return Number(n.toPrecision(digits)).toString();
}
export function pct(v: unknown, digits = 0): string {
  const n = num(v);
  return n === null ? '—' : `${(n * 100).toFixed(digits)}%`;
}
export function fmtBytes(v: unknown): string {
  const n = num(v);
  if (n === null) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let x = n, i = 0;
  while (Math.abs(x) >= 1024 && i < units.length - 1) { x /= 1024; i++; }
  return `${x.toFixed(x >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}
export function ago(iso: unknown, now = Date.now()): string {
  const t = typeof iso === 'number' ? (iso < 1e12 ? iso * 1000 : iso) : Date.parse(str(iso));
  if (!Number.isFinite(t)) return '—';
  const s = Math.round((now - t) / 1000);
  if (s < 0) return 'in the future';
  if (s < 90) return `${s} s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 172800) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}
export function ageSeconds(iso: unknown): number | null {
  const t = typeof iso === 'number' ? (iso < 1e12 ? iso * 1000 : iso) : Date.parse(str(iso));
  return Number.isFinite(t) ? (Date.now() - t) / 1000 : null;
}
export function timeOf(v: unknown): number | null {
  if (typeof v === 'number') return v < 1e12 ? v * 1000 : v;
  const t = Date.parse(str(v));
  return Number.isFinite(t) ? t : null;
}
export function fmtTime(ms: number | null): string {
  if (ms === null) return '—';
  const d = new Date(ms);
  return d.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
}
export function shortSha(v: unknown) {
  const s = str(v);
  return s.length > 10 ? s.slice(0, 10) : s || '—';
}

/** Wilson score interval for k/n (95%). */
export function wilson(k: number, n: number, z = 1.959964): [number, number] {
  if (n <= 0) return [0, 1];
  const p = k / n, d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}
/** Newcombe hybrid-score 95% CI for p2 − p1 (independent proportions). */
export function newcombe(k1: number, n1: number, k2: number, n2: number): [number, number] {
  const p1 = k1 / n1, p2 = k2 / n2;
  const [l1, u1] = wilson(k1, n1), [l2, u2] = wilson(k2, n2);
  const d = p2 - p1;
  return [d - Math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + Math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)];
}

export function uniq<T>(xs: T[]): T[] {
  return Array.from(new Set(xs));
}
export function sortNatural(a: string, b: string) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}
export function clamp(x: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, x));
}
/** Keep every k-th point so that at most `max` remain (plus the last). */
export function thin<T>(xs: T[], max = 1500): T[] {
  if (xs.length <= max) return xs;
  const k = Math.ceil(xs.length / max);
  const out = xs.filter((_, i) => i % k === 0);
  if (out[out.length - 1] !== xs[xs.length - 1]) out.push(xs[xs.length - 1]);
  return out;
}
export function decisionIds(s: string): string[] {
  return uniq(s.match(/\b[DP]-\d{2,4}\b/g) || []);
}
