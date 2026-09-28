import type { ReactNode } from 'react';
import { Card, DataTable, KV, RawDoc } from './ui';
import Markdown from './Markdown';
import { isObj, rows, str, type Row } from '../lib/format';

export type SectionSpec = { keys: string[]; title: string; hint?: string; render?: (rows: Row[], raw: unknown) => ReactNode };

/** Known sections first (in order, with titles), then every other top-level field so nothing in the document is hidden. */
export default function Sections({ d, specs }: { d: Row; specs: SectionSpec[] }) {
  const used = new Set(['schema', 'generated_at', 'git_sha', 'sources', 'stale', 'fixture']);
  const blocks: ReactNode[] = [];
  for (const s of specs) {
    const key = s.keys.find((k) => d[k] !== undefined);
    s.keys.forEach((k) => used.add(k));
    if (!key) {
      blocks.push(<Card key={s.title} title={s.title} hint={s.hint}><p className="muted small">Not in the document (looked for <code>{s.keys.join(' / ')}</code>).</p></Card>);
      continue;
    }
    const raw = d[key];
    const rs = rows(raw);
    blocks.push(
      <Card key={s.title} title={s.title} hint={s.hint ? `${s.hint} · ${key}` : key}>
        {s.render ? s.render(rs, raw) : typeof raw === 'string' ? <Markdown source={raw} /> : rs.length ? <DataTable rows={rs} tall /> : isObj(raw) ? <KV data={raw} /> : <code>{str(raw)}</code>}
      </Card>,
    );
  }
  const rest = Object.fromEntries(Object.entries(d).filter(([k]) => !used.has(k)));
  return (
    <div className="stack">
      {blocks}
      {Object.keys(rest).length > 0 && (
        <Card title="Other fields in the document"><RawDoc data={rest} /></Card>
      )}
    </div>
  );
}
