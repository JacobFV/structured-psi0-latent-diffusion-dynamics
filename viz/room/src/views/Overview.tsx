import { Card, DataTable, Did, Gate, ModeBanner, PageHead, Provenance, SourceBadge, Caveat } from '../components/ui';
import Markdown from '../components/Markdown';
import { useDoc, type Envelope } from '../lib/api';
import { arr, decisionIds, fmtNum, isObj, pick, rows, str, text, type Row } from '../lib/format';
import { href } from '../lib/url';

function refsOf(o: unknown): string[] {
  const direct = pick(o, 'decision', 'decisions', 'd', 'ref', 'refs', 'decision_refs', 'roadmap');
  const list = Array.isArray(direct) ? direct.map(str) : direct ? [str(direct)] : [];
  return list.length ? list : decisionIds(text(o));
}

function Claims({ items, empty }: { items: unknown[]; empty: string }) {
  if (!items.length) return <p className="muted small">{empty}</p>;
  return (
    <ul style={{ margin: 0, paddingLeft: 18, display: 'grid', gap: 6 }}>
      {items.map((c, i) => (
        <li key={i}>
          <span>{text(c)}</span>{' '}
          {refsOf(c).map((r) => /^[DP]-/.test(r)
            ? <a key={r} className="did" href={href('knowledge', { tab: 'decisions', d: r })}>{r}</a>
            : <span key={r} className="did">{/^\d+$/.test(r) ? `roadmap #${r}` : r}</span>)}
          {isObj(c) && c.source_label ? <> <SourceBadge label={c.source_label} /></> : null}
          {isObj(c) && c.caveat ? <> <Caveat text={c.caveat} /></> : null}
        </li>
      ))}
    </ul>
  );
}

export default function Overview() {
  const { result, reload, busy } = useDoc<Envelope>('overview');
  return (
    <>
      <PageHead
        title="Overview"
        sub="What is established (with its decision), what is open (roadmap), the caveats that qualify it, key numbers and the latest decisions."
      />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="overview (/api/overview)">
        {(d) => {
          const claims = isObj(d.claims) ? d.claims : {};
          const established = arr(pick(d, 'established') ?? pick(claims, 'established'));
          const open = arr(pick(d, 'open') ?? pick(claims, 'open'));
          const caveats = arr(pick(d, 'caveats') ?? pick(claims, 'caveats'));
          const numbers = rows(pick(d, 'key_numbers', 'numbers', 'headline_numbers'));
          const decisions = rows(pick(d, 'decisions', 'latest_decisions', 'recent_decisions'));
          const summary = str(pick(d, 'summary', 'status_markdown', 'headline'));
          return (
            <div className="stack">
              {summary && <Card title="Status"><Markdown source={summary} /></Card>}
              <div className="grid g3">
                <Card title="Established" hint={`${established.length}`}><Claims items={established} empty="No established claims in the document." /></Card>
                <Card title="Open" hint={`${open.length}`}><Claims items={open} empty="No open items in the document." /></Card>
                <Card title="Caveats" hint={`${caveats.length} shown, never hidden`}><Claims items={caveats} empty="No caveats in the document." /></Card>
              </div>
              <Card title="Key numbers" hint="each with its source label, file and decision">
                <KeyNumbers rows={numbers} />
              </Card>
              <Card title="Latest decisions" right={<a href={href('knowledge', { tab: 'decisions' })}>all decisions →</a>}>
                {decisions.length ? (
                  <div className="dlist">
                    {decisions.slice(0, 15).map((r, i) => (
                      <a key={i} className="d" href={href('knowledge', { tab: 'decisions', d: str(pick(r, 'id')) })} style={{ color: 'inherit' }}>
                        <span className="date">{str(pick(r, 'date'))}</span>
                        <span><Did id={pick(r, 'id')} /></span>
                        <span>{str(pick(r, 'title'))}</span>
                      </a>
                    ))}
                  </div>
                ) : <p className="muted small">No decisions in the document.</p>}
              </Card>
            </div>
          );
        }}
      </Gate>
      <Provenance result={result} />
    </>
  );
}

function KeyNumbers({ rows: rs }: { rows: Row[] }) {
  if (!rs.length) return <p className="muted small">No key numbers in the document.</p>;
  const shaped = rs.map((r) => {
    const k = pick(r, 'k'), n = pick(r, 'n');
    const value = pick(r, 'value', 'rate', 'effect');
    const lo = pick(r, 'ci_lo'), hi = pick(r, 'ci_hi');
    const ci = pick(r, 'ci');
    return {
      metric: str(pick(r, 'label', 'name', 'metric', 'key')),
      value: typeof value === 'number' ? fmtNum(value) : str(value),
      'k/n': k != null && n != null ? `${k}/${n}` : '',
      ci: lo != null && hi != null ? `[${fmtNum(lo)}, ${fmtNum(hi)}]` : Array.isArray(ci) ? `[${fmtNum(ci[0])}, ${fmtNum(ci[1])}]` : '',
      source_label: pick(r, 'source_label'),
      decision: pick(r, 'decision'),
      interim: pick(r, 'interim'),
      caveat: pick(r, 'caveat'),
      source_file: pick(r, 'source_file'),
    } as Row;
  });
  return <DataTable rows={shaped} />;
}
