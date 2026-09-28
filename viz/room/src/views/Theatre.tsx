import { useEffect, useMemo, useRef, useState } from 'react';
import Stage, { type StageOptions } from '../components/Stage';
import Timelines, { type Side } from '../components/Timelines';
import { Caveat, Did, Gate, Loading, ModeBanner, NoData, ErrorState, PageHead, Provenance, Select, SourceBadge, Tabs } from '../components/ui';
import { useDoc, useReplay, type DocResult, type Envelope } from '../lib/api';
import { fmtNum, pick, rows, shortSha, sortNatural, str, uniq, type Row } from '../lib/format';
import { Clock, relTimes, useClock, type Replay } from '../lib/replay';
import { readParam, useUrlState, writeParams } from '../lib/url';

const SPEEDS = ['0.25', '0.5', '1', '2', '4'];

export default function Theatre() {
  const [tab, setTab] = useUrlState('tab', 'theatre');
  const index = useDoc<Envelope>('replays');
  const videos = useDoc<Envelope>('videos');
  return (
    <>
      <PageHead
        title="Episode theatre"
        sub="Recorded replays in 3D with synchronized signal timelines, side-by-side comparison on a shared clock, and the labelled video fallback. Everything drawn comes from the replay file; missing signals are omitted."
      />
      <Tabs value={tab as 'theatre' | 'videos'} onChange={setTab} options={[{ id: 'theatre', label: 'Theatre' }, { id: 'videos', label: 'Video library' }]} />
      {tab === 'videos' ? (
        <>
          <ModeBanner result={videos.result} reload={videos.reload} busy={videos.busy} />
          <Gate result={videos.result} what="videos (/api/videos)">{(d) => <VideoLibrary d={d} />}</Gate>
          <Provenance result={videos.result} />
        </>
      ) : (
        <>
          <ModeBanner result={index.result} reload={index.reload} busy={index.busy} />
          <Gate result={index.result} what="replay index (/api/replays)">
            {(d) => <TheatreBody entries={rows(pick(d, 'replays', 'rows', 'entries'))} videos={videos.result} />}
          </Gate>
          <Provenance result={index.result} />
        </>
      )}
    </>
  );
}

function TheatreBody({ entries, videos }: { entries: Row[]; videos: DocResult<Envelope> }) {
  const [a, setA] = useUrlState('a', '');
  const [b, setB] = useUrlState('b', '');
  const [speed, setSpeed] = useUrlState('speed', '1');
  const [loop, setLoop] = useUrlState('loop', '1');
  const [follow, setFollow] = useUrlState('follow', '0');
  const [contacts, setContacts] = useUrlState('contacts', '1');
  const [trails, setTrails] = useUrlState('trails', '1');
  const idA = a || str(entries[0]?.id);
  const ra = useReplay<Replay>(idA || undefined);
  const rb = useReplay<Replay>(b || undefined);
  const clock = useMemo(() => new Clock(), []);
  useEffect(() => () => clock.dispose(), [clock]);
  const snap = useClock(clock);
  const replayA = ra?.status === 'ok' ? ra.data : null;
  const replayB = rb?.status === 'ok' ? rb.data : null;
  const sides: Side[] = useMemo(() => {
    const out: Side[] = [];
    if (replayA) out.push({ replay: replayA, times: relTimes(replayA), color: 'var(--s1)', tag: 'A' });
    if (replayB) out.push({ replay: replayB, times: relTimes(replayB), color: 'var(--s2)', tag: 'B' });
    return out;
  }, [replayA, replayB]);
  useEffect(() => {
    const d = Math.max(0, ...sides.map((s) => s.times[s.times.length - 1] || 0));
    clock.setDuration(d);
    const t0 = Number(readParam('t'));
    if (Number.isFinite(t0) && t0 > 0) clock.set(t0);
  }, [sides, clock]);
  useEffect(() => { clock.speed = Number(speed) || 1; }, [speed, clock]);
  useEffect(() => { clock.loop = loop === '1'; }, [loop, clock]);
  useEffect(() => {
    if (!snap.playing) writeParams({ t: snap.t > 0 ? snap.t.toFixed(2) : undefined });
  }, [snap.t, snap.playing]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'SELECT') return;
      const dt = 1 / Math.max(1, sides[0]?.replay.fps || 30);
      if (e.key === ' ') { e.preventDefault(); clock.toggle(); }
      else if (e.key === 'ArrowRight') clock.set(clock.t + (e.shiftKey ? 1 : dt));
      else if (e.key === 'ArrowLeft') clock.set(clock.t - (e.shiftKey ? 1 : dt));
      else if (e.key === 'Home') clock.set(0);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [clock, sides]);

  const opts: StageOptions = { follow: follow === '1', contacts: contacts === '1', trails: trails === '1', ghost: null };
  const entryA = entries.find((e) => str(e.id) === idA);
  const entryB = entries.find((e) => str(e.id) === b);
  const stageH = b ? 380 : 480;
  if (!entries.length) return <NoData expected="~/work/rrp-data/viz/replays/index.json (via /api/replays)" detail="The replay index is empty: no episodes have been recorded yet." />;
  return (
    <div className="theatre">
      <div style={{ minWidth: 0 }}>
        {sides.map((s) => {
          const res = s.tag === 'A' ? ra : rb;
          return <LabelBanner key={s.tag} r={s.replay} tag={b ? s.tag : undefined} color={s.color} fixture={res?.status === 'ok' && res.mode === 'fixture'} />;
        })}
        <div className={b ? 'stage-pair' : ''}>
          <StageSlot result={ra} clock={clock} opts={opts} height={stageH} tag={b ? 'A' : undefined} color="var(--s1)" entry={entryA} videos={videos} />
          {b && <StageSlot result={rb} clock={clock} opts={opts} height={stageH} tag="B" color="var(--s2)" entry={entryB} videos={videos} />}
        </div>
        <div className="transport" role="group" aria-label="Playback">
          <button className="primary" onClick={() => clock.toggle()} disabled={!sides.length} aria-label={snap.playing ? 'Pause' : 'Play'} style={{ minWidth: 64 }}>
            {snap.playing ? 'Pause' : 'Play'}
          </button>
          <button onClick={() => clock.set(clock.t - 1 / Math.max(1, sides[0]?.replay.fps || 30))} aria-label="Previous frame">◀︎</button>
          <button onClick={() => clock.set(clock.t + 1 / Math.max(1, sides[0]?.replay.fps || 30))} aria-label="Next frame">▶︎</button>
          <input type="range" min={0} max={snap.duration || 0} step={0.001} value={snap.t} onChange={(e) => clock.set(Number(e.target.value))} aria-label="Scrub" />
          <span className="time">{snap.t.toFixed(2)} / {snap.duration.toFixed(2)} s{sides[0] ? ` · f${frameLabel(sides[0], snap.t)}` : ''}</span>
          <label className="small">speed <select value={speed} onChange={(e) => setSpeed(e.target.value)}>{SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}</select></label>
          <label className="small"><input type="checkbox" checked={loop === '1'} onChange={(e) => setLoop(e.target.checked ? '1' : '0')} /> loop</label>
          <label className="small"><input type="checkbox" checked={follow === '1'} onChange={(e) => setFollow(e.target.checked ? '1' : '0')} /> follow base</label>
          <label className="small"><input type="checkbox" checked={contacts === '1'} onChange={(e) => setContacts(e.target.checked ? '1' : '0')} /> contacts</label>
          <label className="small"><input type="checkbox" checked={trails === '1'} onChange={(e) => setTrails(e.target.checked ? '1' : '0')} /> trails</label>
        </div>
        <p className="muted small" style={{ margin: '0 0 8px' }}>
          Space play/pause · ←/→ frame · shift+←/→ 1 s · drag to orbit, scroll to zoom. Trails: green = object, violet = base (last 1.5 s bright, whole episode faint).
          {b ? ' Compare: both stages share one clock (seconds from each episode start).' : ''}
        </p>
        {sides.length > 0 && <Timelines sides={sides} clock={clock} t={snap.t} duration={snap.duration} />}
      </div>
      <aside className="stack" style={{ alignContent: 'start' }}>
        <Picker entries={entries} a={idA} b={b} setA={(v) => { setA(v); clock.set(0); }} setB={(v) => { setB(v); clock.set(0); }} />
      </aside>
    </div>
  );
}

function frameLabel(s: Side, t: number) {
  let i = 0;
  while (i + 1 < s.times.length && s.times[i + 1] <= t) i++;
  return `${i + 1}/${s.times.length}`;
}

function StageSlot({ result, clock, opts, height, tag, color, entry, videos }: {
  result: DocResult<Replay> | null; clock: Clock; opts: StageOptions; height: number; tag?: string; color: string; entry?: Row; videos: DocResult<Envelope>;
}) {
  const [showVideo, setShowVideo] = useState(false);
  const vids = videoCandidates(entry, videos);
  const hud = (
    <div className="hud">
      {tag && <span className="badge" style={{ borderColor: color, color }}>{tag}</span>}
      {entry && <SourceBadge label={entry.source_label} />}
      {vids.length > 0 && <button className="small" onClick={() => setShowVideo(!showVideo)}>{showVideo ? '3D replay' : 'video'}</button>}
    </div>
  );
  if (showVideo || (result && result.status !== 'ok' && result.status !== 'loading' && vids.length)) {
    return (
      <div className="stage" style={{ minHeight: height }}>
        <VideoPanel vids={vids} reason={result && result.status !== 'ok' && result.status !== 'loading' ? 'replay unavailable: showing the video fallback' : undefined} />
        {hud}
      </div>
    );
  }
  if (!result) return <div className="stage" style={{ height }}><div className="state">Choose a replay.</div></div>;
  if (result.status === 'loading') return <div className="stage" style={{ height, padding: 12 }}><Loading what="replay" /></div>;
  if (result.status === 'missing') return <div style={{ minHeight: height }}><NoData expected={result.expected} detail={result.detail} what="this replay" /></div>;
  if (result.status === 'error') return <ErrorState message={result.message} />;
  return (
    <div className="stage" style={{ height }}>
      <Stage replay={result.data} clock={clock} options={opts} height={height} />
      {hud}
      <StageClock clock={clock} r={result.data} />
    </div>
  );
}

function StageClock({ clock, r }: { clock: Clock; r: Replay }) {
  const snap = useClock(clock, 10);
  const times = useMemo(() => relTimes(r), [r]);
  const end = times[times.length - 1] || 0;
  return <div className="hud-br">{Math.min(snap.t, end).toFixed(2)} s{snap.t > end + 1e-6 ? ' (episode ended)' : ''} · {fmtNum(r.fps)} fps · {r.n_frames} frames</div>;
}

function LabelBanner({ r, tag, color, fixture }: { r: Replay; tag?: string; color: string; fixture?: boolean }) {
  const m = r.meta;
  const phys = m.physics || {};
  const outcome = m.success === true ? 'success' : m.success === false ? `failure${m.failure_stage ? ` (${m.failure_stage})` : ''}` : 'outcome undefined';
  return (
    <div className={`label-banner ${fixture ? 'fixture' : ''}`} style={tag ? { borderLeft: `4px solid ${color}` } : undefined}>
      <div className="row">
        {tag && <b style={{ color }}>{tag}</b>}
        {fixture && <span className="badge src t-fixture">FIXTURE · synthetic motion, not a recorded episode</span>}
        <SourceBadge label={m.source_label} />
        <b>{[m.family, m.task, m.body, m.route].filter(Boolean).join(' · ')}</b>
        <span className={`status ${m.success ? 'good' : m.success === false ? 'critical' : 'neutral'}`}><i />{outcome}</span>
        {(m.decision_refs || []).map((d) => <Did key={d} id={d} />)}
        {m.caveat && <Caveat text={m.caveat} />}
      </div>
      <div className="kv">
        <span>variant <b>{str(m.variant) || '—'}</b></span>
        <span>seed <b className="num">{str(m.seed) || '—'}</b></span>
        <span>condition <b>{str(m.condition) || '—'}</b></span>
        <span>checkpoint <b className="mono">{m.ckpt_sha ? shortSha(m.ckpt_sha) : '— (no checkpoint)'}</b></span>
        {Object.entries(phys).map(([k, v]) => <span key={k}>{k.replace(/_version$/, '')} <b>{str(v)}</b></span>)}
        <span className="muted">id <code>{r.id}</code></span>
      </div>
    </div>
  );
}

function Picker({ entries, a, b, setA, setB }: { entries: Row[]; a: string; b: string; setA: (v: string) => void; setB: (v: string) => void }) {
  const dims = ['family', 'task', 'body', 'route', 'source_label', 'variant', 'condition', 'seed'];
  const [f, setF] = useState<Record<string, string>>(() => Object.fromEntries(dims.map((d) => [d, readParam(d) || ''])));
  const [q, setQ] = useState('');
  const shown = entries.filter((e) => dims.every((d) => !f[d] || str(e[d]) === f[d]) && (!q || JSON.stringify(e).toLowerCase().includes(q.toLowerCase())));
  const opt = (d: string) => uniq(entries.map((e) => str(e[d]))).filter(Boolean).sort(sortNatural);
  // quick pairs: same task/body/seed with a different condition or variant
  const cur = entries.find((e) => str(e.id) === a);
  const partners = cur ? entries.filter((e) => str(e.id) !== a && str(e.task) === str(cur.task) && str(e.body) === str(cur.body) && str(e.seed) === str(cur.seed)) : [];
  return (
    <>
      <section className="card">
        <header><h2>Replays</h2><span className="hint">{shown.length} of {entries.length}</span></header>
        <div className="body">
          <div className="filters" style={{ marginBottom: 8 }}>
            {dims.map((d) => <Select key={d} label={d} value={f[d]} options={opt(d)} onChange={(v) => { setF({ ...f, [d]: v }); writeParams({ [d]: v || undefined }); }} width={150} />)}
            <label>search<input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="id, seed…" /></label>
          </div>
          <div className="picker">
            {shown.slice(0, 300).map((e) => {
              const id = str(e.id);
              return (
                <div key={id} className="row" style={{ alignItems: 'stretch', gap: 4 }}>
                  <button aria-pressed={id === a} onClick={() => setA(id)} style={{ flex: 1, minWidth: 0 }} title={id}>
                    <span className="row" style={{ gap: 6 }}>
                      <b style={{ fontWeight: 600 }}>{[e.task, e.body].map(str).filter(Boolean).join(' · ')}</b>
                      <span className={`status ${e.success === true ? 'good' : e.success === false ? 'critical' : 'neutral'}`}><i />{e.success === true ? 'success' : e.success === false ? 'failure' : '—'}</span>
                    </span>
                    <span className="small muted">{[e.route, e.variant, e.condition, `seed ${str(e.seed)}`].map(str).filter(Boolean).join(' · ')}</span>
                    <SourceBadge label={e.source_label} />
                  </button>
                  <button aria-pressed={id === b} onClick={() => setB(id === b ? '' : id)} title="Compare as B" className="small">B</button>
                </div>
              );
            })}
          </div>
        </div>
      </section>
      {cur && (
        <section className="card">
          <header><h2>Compare with</h2><span className="hint">same task, body and seed</span></header>
          <div className="body">
            {partners.length ? (
              <div className="picker">
                {partners.map((e) => (
                  <button key={str(e.id)} aria-pressed={str(e.id) === b} onClick={() => setB(str(e.id) === b ? '' : str(e.id))}>
                    <b style={{ fontWeight: 600 }}>{[e.route, e.variant, e.condition].map(str).filter(Boolean).join(' · ')}</b>
                    <SourceBadge label={e.source_label} />
                  </button>
                ))}
              </div>
            ) : <p className="muted small">No other replay with the same task, body and seed.</p>}
            {b && <button className="ghost small" onClick={() => setB('')} style={{ marginTop: 6 }}>clear B</button>}
          </div>
        </section>
      )}
    </>
  );
}

function videoRows(videos: DocResult<Envelope>): Row[] {
  return videos.status === 'ok' ? rows(pick(videos.data, 'videos', 'entries', 'rows')) : [];
}
function videoName(v: Row) {
  return str(pick(v, 'file', 'name', 'path')).split('/').pop() || '';
}
function videoCandidates(entry: Row | undefined, videos: DocResult<Envelope>): { name: string; label: string; exact: boolean }[] {
  if (!entry) return [];
  const all = videoRows(videos);
  const direct = str(entry.video);
  if (direct) {
    const name = direct.split('/').pop()!;
    const v = all.find((x) => videoName(x) === name);
    return [{ name, label: v ? str(pick(v, 'label', 'text', 'description')) : 'linked from the replay index', exact: true }];
  }
  const seed = str(entry.seed), body = str(entry.body), task = str(entry.task);
  return all
    .filter((v) => { const n = videoName(v); return seed && n.includes(`s${seed}`) && (!body || n.includes(body)) && (!task || n.includes(task) || /ladder|triptych/.test(n)); })
    .slice(0, 6)
    .map((v) => ({ name: videoName(v), label: str(pick(v, 'label', 'text', 'description')), exact: false }));
}

function VideoPanel({ vids, reason }: { vids: { name: string; label: string; exact: boolean }[]; reason?: string }) {
  const [i, setI] = useState(0);
  const v = vids[Math.min(i, vids.length - 1)];
  const ref = useRef<HTMLVideoElement>(null);
  if (!v) return null;
  return (
    <div style={{ padding: 10, display: 'grid', gap: 6, paddingTop: 40 }}>
      {reason && <div className="badge caveat">{reason}</div>}
      {!v.exact && <div className="small ink2">Matched by file name (seed/body/task), not linked by the recorder: check the label.</div>}
      <video ref={ref} className="media" src={`/media/${encodeURIComponent(v.name)}`} controls loop playsInline preload="metadata" />
      {vids.length > 1 && <select value={i} onChange={(e) => setI(Number(e.target.value))}>{vids.map((x, k) => <option key={x.name} value={k}>{x.name}</option>)}</select>}
      <div className="small">{v.label || <span className="muted">no label in the video index</span>}</div>
    </div>
  );
}

function VideoLibrary({ d }: { d: Envelope }) {
  const vids = rows(pick(d, 'videos', 'entries', 'rows'));
  const [sel, setSel] = useUrlState('video', '');
  const [q, setQ] = useState('');
  const shown = vids.filter((v) => !q || JSON.stringify(v).toLowerCase().includes(q.toLowerCase()));
  const cur = vids.find((v) => videoName(v) === sel) || shown[0];
  if (!vids.length) return <NoData expected="artifacts/video/INDEX.md (via /api/videos)" />;
  return (
    <div className="theatre">
      <div>
        {cur && (
          <section className="card">
            <header><h2 style={{ wordBreak: 'break-all' }}>{videoName(cur)}</h2></header>
            <div className="body">
              <video className="media" key={videoName(cur)} src={`/media/${encodeURIComponent(videoName(cur))}`} controls loop playsInline preload="metadata" />
              <p>{str(pick(cur, 'label', 'text', 'description'))}</p>
              <div className="row">{pick(cur, 'source_label') ? <SourceBadge label={pick(cur, 'source_label')} /> : null}</div>
            </div>
          </section>
        )}
      </div>
      <aside className="card">
        <header><h2>Videos</h2><span className="hint">{shown.length} of {vids.length}</span></header>
        <div className="body">
          <input type="search" placeholder="filter" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: '100%', marginBottom: 8 }} />
          <div className="picker" style={{ maxHeight: 640 }}>
            {shown.map((v) => (
              <button key={videoName(v)} aria-pressed={cur === v} onClick={() => setSel(videoName(v))}>
                <span className="small" style={{ wordBreak: 'break-all' }}>{videoName(v)}</span>
                {pick(v, 'source_label') ? <SourceBadge label={pick(v, 'source_label')} /> : null}
              </button>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
