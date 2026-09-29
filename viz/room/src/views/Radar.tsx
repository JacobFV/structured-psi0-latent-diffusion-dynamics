/** Evaluations → Route radar: the full declared radar (viz/radar_axes.json → /api/radar) and its evidence table. */
import { Cap, ModeBadge } from '../components/board';
import RadarChart, { SERIES_COLOR, type RadarDoc } from '../components/RadarChart';
import { Gate } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { fmtNum } from '../lib/format';

export default function Radar() {
  const { result } = useDoc<Envelope>('radar');
  return (
    <Gate result={result} what="route radar (/api/radar)">
      {(d) => {
        const radar = d as unknown as RadarDoc & { missing?: string[] };
        if (!radar.axes?.length) return <p className="side-note">no axes: {(radar.missing || []).join('; ')}</p>;
        return (
          <div className="ev-grid" style={{ gridTemplateColumns: 'minmax(460px, 560px) minmax(0, 1fr)' }}>
            <section className="ev-panel">
              <header>Route radar<ModeBadge result={result} /><span className="meta" title={radar.normalization.rule}>{radar.axes.length} declared axes · 0 = floor · 1 = reference</span></header>
              <RadarChart radar={radar} size={520} />
              <Cap>spokes: declared axes from viz/radar_axes.json (metric, direction, protocol and decision on hover) · radius r: 0 = the axis floor, ring 1 = the reference route's value (BC or teacher, or a declared constant), clamped to 1.5 · lines: routes; a missing value is a gap (no point, no edge) · whiskers: seed range or 95% CI</Cap>
            </section>
            <section className="ev-panel">
              <header>Evidence<span className="meta">value · r (normalized) · spread · source; empty = gap</span></header>
              <div className="table-wrap" style={{ maxHeight: 620 }}>
                <table className="bt">
                  <thead><tr><th>axis</th><th>ref</th>{radar.series.map((s) => <th key={s.id} className="n"><i className="sw" style={{ background: SERIES_COLOR[s.id] }} />{s.id}</th>)}<th>evidence</th></tr></thead>
                  <tbody>
                    {radar.axes.map((a) => {
                      const files = [...new Set(Object.values(a.series).flatMap((v) => v.evidence || []))];
                      return (
                        <tr key={a.id}>
                          <td title={`${a.metric}\n${a.protocol}`}>{a.label} <span className="did">{a.decision}</span></td>
                          <td className="mono" title={a.reference.meaning}>{a.reference.series ?? fmtNum(a.reference.value)}</td>
                          {radar.series.map((s) => {
                            const v = a.series[s.id];
                            if (!v || v.value == null) return <td key={s.id} className="n muted" title={v?.missing}>—</td>;
                            return (
                              <td key={s.id} className="n" title={`${v.spread ? `${v.spread.meaning}: ${v.spread.x.map((x) => fmtNum(x)).join('–')}\n` : ''}${(v.evidence || []).join('\n')}${v.missing_r ? `\n${v.missing_r}` : ''}`}>
                                {fmtNum(v.value)}{v.k != null ? <span className="muted"> {v.k}/{v.n}</span> : null}<br /><span className="muted">r {fmtNum(v.r)}</span>
                              </td>
                            );
                          })}
                          <td className="mono" style={{ fontSize: 10, maxWidth: 220, whiteSpace: 'normal' }}>{files.map((f) => f.split('/').slice(-2).join('/')).join(' · ')}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <Cap>rows: radar axes · cols: routes · cell: raw value (k/n where counted) and r (normalized) · — = gap (hover: reason) · last column: evidence files</Cap>
            </section>
          </div>
        );
      }}
    </Gate>
  );
}
