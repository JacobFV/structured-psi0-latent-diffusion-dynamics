/** Curated heatmap: rate colour + CI whisker per cell, or a diverging Δ with ▲▼. Hover shows k/n and the source. */
import { divColor, seqColor, seqInk } from './charts';
import { newcombe, wilson } from '../lib/format';

export type KN = { k: number; n: number; tip?: string; link?: string };
export function Heat({ rows, cols, cell, base, colLabel, rowLabel, showK }: {
  rows: string[]; cols: string[]; cell: (r: string, c: string) => KN | null; base?: (r: string, c: string) => KN | null;
  colLabel?: (c: string) => string; rowLabel?: (r: string) => string; showK?: boolean;
}) {
  return (
    <div className="hm" style={{ gridTemplateColumns: `minmax(84px, max-content) repeat(${cols.length}, minmax(52px, 1fr))` }}>
      <div />
      {cols.map((c) => <div key={c} className="hm-h" title={c}>{colLabel ? colLabel(c) : c}</div>)}
      {rows.map((r) => (
        <Row key={r} r={r} cols={cols} cell={cell} base={base} label={rowLabel ? rowLabel(r) : r} showK={showK} />
      ))}
    </div>
  );
}
function Row({ r, cols, cell, base, label, showK }: { r: string; cols: string[]; cell: (r: string, c: string) => KN | null; base?: (r: string, c: string) => KN | null; label: string; showK?: boolean }) {
  return (
    <>
      <div className="hm-r" title={r}>{label}</div>
      {cols.map((c) => {
        const x = cell(r, c);
        if (!x || !x.n) return <div key={c} className="hm-c e" />;
        const b = base?.(r, c);
        if (base) {
          if (!b || !b.n) return <div key={c} className="hm-c e" title="no matched baseline" />;
          const d = x.k / x.n - b.k / b.n;
          const [lo, hi] = newcombe(b.k, b.n, x.k, x.n);
          const sig = lo > 0 || hi < 0;
          return (
            <a key={c} className="hm-c" href={x.link} style={{ background: divColor(d), color: Math.abs(d) > 0.3 ? '#fff' : 'var(--ink)' }}
              title={`${r} × ${c}\nΔ ${(d * 100).toFixed(1)} pts [${(lo * 100).toFixed(0)}, ${(hi * 100).toFixed(0)}] (Newcombe)\n${b.k}/${b.n} → ${x.k}/${x.n}${x.tip ? `\n${x.tip}` : ''}`}>
              <b>{d > 0 ? '▲' : d < 0 ? '▼' : '■'}{Math.abs(d * 100).toFixed(0)}{sig ? '' : '˙'}</b>
            </a>
          );
        }
        const p = x.k / x.n;
        const [lo, hi] = wilson(x.k, x.n);
        return (
          <a key={c} className="hm-c" href={x.link} style={{ background: seqColor(p), color: seqInk(p) }}
            title={`${r} × ${c}\n${x.k}/${x.n} = ${(p * 100).toFixed(1)}% [${(lo * 100).toFixed(0)}, ${(hi * 100).toFixed(0)}] (Wilson)${x.tip ? `\n${x.tip}` : ''}`}>
            <b>{showK ? `${x.k}` : Math.round(p * 100)}</b>
            <span className="wh"><i style={{ left: `${lo * 100}%`, width: `${Math.max(1, (hi - lo) * 100)}%` }} /><u style={{ left: `${p * 100}%` }} /></span>
          </a>
        );
      })}
    </>
  );
}
