/**
 * Source labels (AGENTS.md: teacher, scripted, privileged, random, mock and learned sources are marked unmistakably).
 * Colour identifies provenance only, never performance, and is always paired with the spelled-out label.
 */
export type SourceKind = 'teacher' | 'oracle' | 'learned' | 'bc' | 'tracker' | 'privileged' | 'random' | 'mock' | 'fixture' | 'other';

export type SourceStyle = { kind: SourceKind; label: string; detail: string; tone: string };

export function sourceStyle(raw: unknown): SourceStyle {
  const s = typeof raw === 'string' ? raw : raw == null ? '' : String(raw);
  const low = s.toLowerCase();
  const after = s.includes(':') ? s.split(':').slice(1).join(':') : '';
  if (low.startsWith('fixture')) return { kind: 'fixture', label: 'FIXTURE (synthetic)', detail: s, tone: 'fixture' };
  if (low.includes('scripted') || low.includes('teacher'))
    return { kind: 'teacher', label: 'SCRIPTED TEACHER · privileged', detail: s, tone: 'teacher' };
  if (low.includes('oracle')) return { kind: 'oracle', label: 'ORACLE DIAGNOSTIC · not deployable', detail: s, tone: 'oracle' };
  if (low.startsWith('learned_tracker')) return { kind: 'tracker', label: `LEARNED TRACKER ${after}`, detail: s, tone: 'tracker' };
  if (low.startsWith('bc')) return { kind: 'bc', label: `BC baseline ${after}`.trim(), detail: s, tone: 'bc' };
  if (low.startsWith('learned')) return { kind: 'learned', label: `LEARNED ${after}`.trim(), detail: s, tone: 'learned' };
  if (low.includes('privileged')) return { kind: 'privileged', label: 'PRIVILEGED', detail: s, tone: 'oracle' };
  if (low.includes('random')) return { kind: 'random', label: 'RANDOM', detail: s, tone: 'mock' };
  if (low.includes('mock')) return { kind: 'mock', label: 'MOCK', detail: s, tone: 'mock' };
  return { kind: 'other', label: s || 'unlabelled source', detail: s, tone: 'other' };
}

/** DAG / job node states -> status tone (reserved status colours, always with text). */
export function stateTone(state: unknown): 'good' | 'warning' | 'serious' | 'critical' | 'neutral' | 'active' {
  const s = String(state ?? '').toLowerCase();
  if (/(complete|done|succeed|verified|ok|pass)/.test(s)) return 'good';
  if (/(run|active|implement|lease|progress)/.test(s)) return 'active';
  if (/(fail|error|kill|oom|critical|test_failed)/.test(s)) return 'critical';
  if (/(block|budget|throttl|hold|stale|warn)/.test(s)) return 'serious';
  if (/(wait|pending|queued|planned|ready)/.test(s)) return 'neutral';
  return 'neutral';
}

export const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)', 'var(--s7)', 'var(--s8)'];
/** Stable colour per entity name (hash into the fixed categorical order) so filters never repaint survivors. */
export function seriesColor(name: string, order?: string[]) {
  if (order) {
    const i = order.indexOf(name);
    if (i >= 0) return SERIES[i % SERIES.length];
  }
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return SERIES[h % SERIES.length];
}
