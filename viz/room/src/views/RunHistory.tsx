/**
 * Run history (v3, after IBM-2's environment run history): the sidebar lists the recorded runs with filters and a compare
 * picker; the workspace is one vertical stack of linked visualizations for the selected run, sharing one time cursor,
 * rendered lazily as they scroll into view. Only panels whose inputs the replay records are shown.
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { ModeBadge } from '../components/board';
import { EditPanel, EventGantt, GaitDiagram, hasEdit, hasMap, hasProbeTruth, JointHeatmap, PacketHeatmap, pcaSpeed, PhaseLanes, PhasePortrait, ProbeTruth, TopDownMap } from '../components/runpanels';
import { SideGroup, SidebarControls } from '../components/shell';
import { hasMorphology, hasPacketStructure, hasPipeline, Morphology, PacketStructure, PipelineFlow } from '../components/diagrams';
import Stage, { PcaPlot, type StageOptions } from '../components/Stage';
import { CategoryTrack, ProbeTracks, RasterTrack, ScalarTrack, type Side, type Val } from '../components/Timelines';
import { Caveat, Did, ErrorState, Loading, NoData, SourceBadge } from '../components/ui';
import { useDoc, useReplay, type DocResult, type Envelope } from '../lib/api';
import { arr, fmtNum, isObj, pick, rows, shortSha, sortNatural, str, uniq, type Row } from '../lib/format';
import { Clock, relTimes, useClock, type Replay } from '../lib/replay';
import { readParam, useUrlState, writeParams } from '../lib/url';
import { useFixtureIndex, VideoLibrary, VideoPanel, videoCandidates } from './Theatre';

const SPEEDS = ['0.25', '0.5', '1', '2', '4'];
declare global { var __RRP_EAGER__: boolean | undefined }

/** Renders children once scrolled into view (IBM-2 WhenVisible); eager under the SSR render check. */
function WhenVisible({ children, minHeight = 120 }: { children: ReactNode; minHeight?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(!!globalThis.__RRP_EAGER__);
  useEffect(() => {
    const el = ref.current;
    if (!el || shown) return;
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) setShown(true); }, { rootMargin: '200px' });
    io.observe(el);
    return () => io.disconnect();
  }, [shown]);
  return <div ref={ref} style={shown ? undefined : { minHeight }}>{shown ? children : <p className="lazy">loads when visible</p>}</div>;
}
function Pane({ title, meta, children, lazy = true, minHeight }: { title: string; meta?: ReactNode; children: ReactNode; lazy?: boolean; minHeight?: number }) {
  return (
    <section className="rh-pane">
      <header>{title}{meta !== undefined && <span className="meta">{meta}</span>}</header>
      <div className="pb">{lazy ? <WhenVisible minHeight={minHeight}>{children}</WhenVisible> : children}</div>
    </section>
  );
}

const ENVS: [string, string][] = [['arm', 'arm (pick and place)'], ['legged', 'legged (waypoints, edits)'], ['dual', 'dual arm'], ['physics', 'physics (tracker, grasp rig)']];
function envOf(e: Row) {
  return /^(grasp_rig|tracker_validation)$/.test(str(e.task)) ? 'physics' : str(e.family);
}
function taskOf(e: Row) {
  return `${str(e.task)} · ${str(e.route) || '—'}`;
}
/** Search tokens: key:value on body/route/variant/condition/seed/grasp/contact/id, anything else as free text. */
function matches(e: Row, q: string) {
  const ph = isObj(e.physics) ? e.physics : {};
  const fields: Record<string, string> = {
    body: str(e.body), route: str(e.route), variant: str(e.variant), condition: str(e.condition), seed: str(e.seed), id: str(e.id),
    grasp: str(ph.grasp_contact_version), contact: str(ph.contact_version), task: str(e.task),
  };
  return q.toLowerCase().split(/\s+/).filter(Boolean).every((tok) => {
    const m = /^(\w+):(.*)$/.exec(tok);
    if (m && m[1] in fields) return fields[m[1]].toLowerCase().includes(m[2]);
    return JSON.stringify(e).toLowerCase().includes(tok);
  });
}

export default function RunHistory() {
  const index = useDoc<Envelope>('replays');
  const videos = useDoc<Envelope>('videos');
  const [list, setList] = useUrlState('list', 'runs');
  const [demo, setDemo] = useUrlState('demo', '0');
  const real = index.result.status === 'ok' ? rows(pick(index.result.data, 'replays', 'rows')) : [];
  const fx = useFixtureIndex(demo === '1' && index.result.status === 'ok' && !real.length);
  const entries = real.length ? real : demo === '1' ? fx || [] : [];
  const header = list === 'videos' ? (
    <SidebarControls>
      <SideGroup title="Environment" right={<ModeBadge result={videos.result} />}>
        <select value="videos" onChange={(e) => { if (e.target.value !== 'videos') { setList('runs'); window.location.hash = `#runs?env=${e.target.value}`; } }} aria-label="environment">
          {ENVS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          <option value="videos">video library</option>
        </select>
      </SideGroup>
    </SidebarControls>
  ) : null;
  if (list === 'videos') {
    return <>{header}{videos.result.status === 'ok' ? <VideoLibrary d={videos.result.data} /> : <Status r={videos.result} what="videos (/api/videos)" />}</>;
  }
  if (index.result.status !== 'ok') return <Status r={index.result} what="replay index (/api/replays)" />;
  if (!entries.length) {
    return (
      <>
        <div className="state">
          <h3>No replays recorded yet</h3>
          <div>Expected <code>~/work/rrp-data/viz/replays/index.json</code> (recorder: <code>python -m rrp.viz.record</code>, peer only).</div>
          <button onClick={() => setDemo('1')} style={{ marginTop: 6 }}>open a synthetic FIXTURE demo</button>
        </div>
      </>
    );
  }
  return <Runs entries={entries} videos={videos.result} fixture={!real.length} indexResult={index.result} onVideos={() => setList('videos')} />;
}
function Status({ r, what }: { r: DocResult<Envelope>; what: string }) {
  if (r.status === 'loading') return <Loading what={what} />;
  if (r.status === 'missing') return <NoData expected={r.expected} detail={r.detail} what={what} />;
  if (r.status === 'error') return <ErrorState message={r.message} />;
  return null;
}

function Runs({ entries, videos, fixture, indexResult, onVideos }: { entries: Row[]; videos: DocResult<Envelope>; fixture: boolean; indexResult: DocResult<Envelope>; onVideos: () => void }) {
  const [a, setA] = useUrlState('a', '');
  const [b, setB] = useUrlState('b', '');
  const [q, setQ] = useUrlState('q', '');
  const [speed, setSpeed] = useUrlState('speed', '1');
  const [loop, setLoop] = useUrlState('loop', '1');
  const [follow, setFollow] = useUrlState('follow', '0');
  const envs = ENVS.filter(([v]) => entries.some((e) => envOf(e) === v));
  const [env, setEnv] = useUrlState('env', envs[0]?.[0] ?? 'arm');
  const [task, setTask] = useUrlState('task', '');
  const [res, setRes] = useUrlState('res', '');
  const inEnv = entries.filter((e) => envOf(e) === env);
  const tasks = uniq(inEnv.map(taskOf)).sort(sortNatural);
  const shown = inEnv.filter((e) => (!task || taskOf(e) === task) && (!res || (res === 'success' ? e.success === true : e.success !== true)) && (!q || matches(e, q)));
  const idA = a || str(shown[0]?.id ?? entries[0]?.id);
  const ra = useReplay<Replay>(idA || undefined);
  const rb = useReplay<Replay>(b || undefined);
  const clock = useMemo(() => new Clock(), []);
  useEffect(() => () => clock.dispose(), [clock]);
  const snap = useClock(clock);
  const A = ra?.status === 'ok' ? ra.data : null, B = rb?.status === 'ok' ? rb.data : null;
  const sides: Side[] = useMemo(() => {
    const out: Side[] = [];
    if (A) out.push({ replay: A, times: relTimes(A), color: 'var(--s1)', tag: 'A' });
    if (B) out.push({ replay: B, times: relTimes(B), color: 'var(--s2)', tag: 'B' });
    return out;
  }, [A, B]);
  useEffect(() => {
    clock.setDuration(Math.max(0, ...sides.map((s) => s.times[s.times.length - 1] || 0)));
    const t0 = Number(readParam('t'));
    if (Number.isFinite(t0) && t0 > 0) clock.set(t0);
  }, [sides, clock]);
  useEffect(() => { clock.speed = Number(speed) || 1; }, [speed, clock]);
  useEffect(() => { clock.loop = loop === '1'; }, [loop, clock]);
  useEffect(() => { if (!snap.playing) writeParams({ t: snap.t > 0 ? snap.t.toFixed(2) : undefined }); }, [snap.t, snap.playing]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(tag) || (tag === 'BUTTON' && e.key === ' ')) return;
      if (e.key === '[' || e.key === ']') {
        const ids = shown.map((x) => str(x.id));
        const i = ids.indexOf(idA);
        const next = ids[(i + (e.key === ']' ? 1 : -1) + ids.length) % ids.length];
        if (next) { setA(next); clock.set(0); }
      } else if (e.key === ' ') { e.preventDefault(); clock.toggle(); }
      else if (e.key === 'ArrowRight') clock.set(clock.t + (e.shiftKey ? 1 : 1 / Math.max(1, sides[0]?.replay.fps || 20)));
      else if (e.key === 'ArrowLeft') clock.set(clock.t - (e.shiftKey ? 1 : 1 / Math.max(1, sides[0]?.replay.fps || 20)));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [clock, shown, idA, setA, sides]);
  const cur = entries.find((e) => str(e.id) === idA);
  const partners = cur ? entries.filter((e) => str(e.id) !== idA && str(e.body) === str(cur.body) && str(e.seed) === str(cur.seed)) : [];
  const others = entries.filter((e) => str(e.id) !== idA && !partners.includes(e));
  const p = { sides, t: snap.t, duration: snap.duration, onSeek: (x: number) => clock.set(x) };
  const opts: StageOptions = { follow: follow === '1', contacts: true, trails: true, ghost: null };

  return (
    <>
      <SidebarControls>
        <SideGroup title="Environment" right={<ModeBadge result={indexResult} />}>
          <select value={env} onChange={(e) => { if (e.target.value === 'videos') { onVideos(); return; } setEnv(e.target.value); setTask(''); }} aria-label="environment">
            {envs.map(([v, l]) => <option key={v} value={v}>{l} ({entries.filter((e) => envOf(e) === v).length})</option>)}
            <option value="videos">video library</option>
          </select>
        </SideGroup>
        <SideGroup title="Task" right={<span title="bar = share of runs that succeeded (green) or failed (red)"><i className="sw" style={{ background: 'var(--good)' }} />✓ <i className="sw" style={{ background: 'var(--critical)' }} />✗</span>}>
          <div className="side-list" role="listbox" aria-label="task">
            <button aria-pressed={!task} onClick={() => setTask('')}><span className="row1"><b>all tasks</b><small style={{ marginLeft: 'auto' }}>{inEnv.length}</small></span></button>
            {tasks.map((t) => {
              const xs = inEnv.filter((e) => taskOf(e) === t);
              const ok = xs.filter((e) => e.success === true).length;
              return (
                <button key={t} aria-pressed={task === t} onClick={() => setTask(t)} title={`${ok} of ${xs.length} succeeded`}>
                  <span className="row1"><b>{t.split(' · ')[1]}</b><small style={{ marginLeft: 'auto' }}>{xs.length}</small></span>
                  <span style={{ display: 'flex', height: 3 }}><i style={{ flex: ok, background: 'var(--good)' }} /><i style={{ flex: xs.length - ok, background: 'var(--critical)' }} /></span>
                </button>
              );
            })}
          </div>
        </SideGroup>
        <SideGroup title="Result">
          <select value={res} onChange={(e) => setRes(e.target.value)} aria-label="result"><option value="">all</option><option value="success">success</option><option value="failure">failure</option></select>
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search runs… (body:go2 variant:semfix)" aria-label="search runs" />
        </SideGroup>
        <SideGroup title={`Runs · ${shown.length}`} right={<span>[ ] step</span>}>
          <div className="side-list">
            {shown.slice(0, 400).map((e) => {
              const id = str(e.id);
              return (
                <button key={id} aria-pressed={id === idA} onClick={() => { setA(id); clock.set(0); }} title={`${id}${e.reproduced === false ? ' · NOT reproduced' : ''}`}>
                  <span className="row1">
                    <b>{[e.body, e.variant || e.route, e.condition].map(str).filter(Boolean).join(' · ')}</b>
                    <span style={{ marginLeft: 'auto', color: e.success === true ? 'var(--good)' : e.success === false ? 'var(--critical)' : 'var(--muted)' }}>{e.success === true ? '✓' : e.success === false ? '✗' : '–'}</span>
                  </span>
                </button>
              );
            })}
          </div>
        </SideGroup>
        <SideGroup title="Compare with…">
          <select value={b} onChange={(e) => { setB(e.target.value); clock.set(0); }} aria-label="compare with">
            <option value="">nothing</option>
            {partners.length > 0 && <optgroup label="same body and seed">{partners.map((e) => <option key={str(e.id)} value={str(e.id)}>{[e.route, e.variant, e.condition].map(str).filter(Boolean).join(' · ')}</option>)}</optgroup>}
            <optgroup label="other runs">{others.slice(0, 300).map((e) => <option key={str(e.id)} value={str(e.id)}>{str(e.id)}</option>)}</optgroup>
          </select>
        </SideGroup>
      </SidebarControls>

      {fixture && <div className="mode-banner fixture"><span className="badge src t-fixture">FIXTURE DEMO</span>No replays have been recorded; these synthetic episodes only demonstrate the view. Not results.</div>}
      {sides.map((s) => <Banner key={s.tag} r={s.replay} tag={B ? s.tag : undefined} color={s.color} entry={entries.find((e) => str(e.id) === s.replay.id)} />)}
      {ra && ra.status !== 'ok' && <Status r={ra as DocResult<Envelope>} what="replay A" />}
      {rb && rb.status !== 'ok' && <Status r={rb as DocResult<Envelope>} what="replay B" />}

      {sides.length > 0 && (
        <>
          <div className="rh-top" role="group" aria-label="Playback">
            <button className="primary" onClick={() => clock.toggle()} style={{ minWidth: 52 }}>{snap.playing ? 'PAUSE' : 'PLAY'}</button>
            <button onClick={() => clock.set(clock.t - 1 / Math.max(1, sides[0].replay.fps))} aria-label="previous frame">◀︎</button>
            <button onClick={() => clock.set(clock.t + 1 / Math.max(1, sides[0].replay.fps))} aria-label="next frame">▶︎</button>
            <input type="range" min={0} max={snap.duration || 0} step={0.001} value={snap.t} onChange={(e) => clock.set(Number(e.target.value))} aria-label="time cursor" />
            <span className="time">{snap.t.toFixed(2)} / {snap.duration.toFixed(2)} s</span>
            <select value={speed} onChange={(e) => setSpeed(e.target.value)} aria-label="speed">{SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}</select>
            <label className="small"><input type="checkbox" checked={loop === '1'} onChange={(e) => setLoop(e.target.checked ? '1' : '0')} /> loop</label>
            <label className="small"><input type="checkbox" checked={follow === '1'} onChange={(e) => setFollow(e.target.checked ? '1' : '0')} /> follow</label>
          </div>
          <div className="rh-stack">
            {panes(sides, p, clock, opts, cur, videos)}
          </div>
        </>
      )}
    </>
  );
}

function Banner({ r, tag, color, entry }: { r: Replay; tag?: string; color: string; entry?: Row }) {
  const m = r.meta;
  const ph = m.physics || {};
  return (
    <div className="label-banner" style={{ borderLeft: `3px solid ${color}`, marginBottom: -1 }}>
      <div className="row">
        {tag && <b style={{ color }}>{tag}</b>}
        <SourceBadge label={m.source_label} />
        <b>{[m.family, m.task, m.body, m.route, m.variant, m.condition].map(str).filter(Boolean).join(' · ')}</b>
        <span className="mono">seed {str(m.seed)}</span>
        <span className={`status ${m.success ? 'good' : m.success === false ? 'critical' : 'neutral'}`}><i />{m.success === true ? 'success' : m.success === false ? `failure${m.failure_stage ? ` (${m.failure_stage})` : ''}` : 'no outcome'}</span>
        {m.reproduced === true ? <span className="status good" title="the re-run matches the original eval row"><i />reproduced</span> : m.reproduced === false ? <span className="status critical"><i />NOT reproduced</span> : null}
        {(m.decision_refs || []).map((d) => <Did key={d} id={d} />)}
        {m.caveat ? <Caveat text={m.caveat} /> : null}
        <span className="muted small" style={{ marginLeft: 'auto' }}>
          {Object.entries(ph).filter(([, v]) => v).map(([k, v]) => `${k.replace(/_version$/, '')} ${str(v)}`).join(' · ')}
          {m.ckpt_sha && typeof m.ckpt_sha === 'object' ? ` · ${Object.entries(m.ckpt_sha).map(([k, v]) => `${k} ${shortSha(v)}`).join(' ')}` : ''}
          {entry?.video ? ' · video' : ''}
        </span>
      </div>
    </div>
  );
}

function panes(sides: Side[], p: { sides: Side[]; t: number; duration: number; onSeek: (t: number) => void }, clock: Clock, opts: StageOptions, entry: Row | undefined, videos: DocResult<Envelope>) {
  const A = sides[0].replay;
  const any = (f: (r: Replay) => unknown) => sides.some((s) => { const v = f(s.replay); return Array.isArray(v) ? v.some((x) => x !== null && x !== undefined) : !!v; });
  const out: ReactNode[] = [];
  const stageH = sides.length > 1 ? 380 : 480;
  out.push(
    <Pane key="stage" title="3D replay" meta={`${A.geoms.length} geoms · ${A.bodies.length} bodies · drag to orbit`} lazy={false}>
      <div className="rh-pair">
        {sides.map((s) => <div key={s.tag} className="stage" style={{ height: stageH }}><Stage replay={s.replay} clock={clock} options={opts} height={stageH} />{sides.length > 1 && <div className="hud"><span className="badge" style={{ color: s.color, borderColor: s.color }}>{s.tag}</span></div>}</div>)}
      </div>
    </Pane>,
  );
  out.push(...sides.filter((s) => hasPipeline(s.replay)).map((s) => (
    <Pane key={`flow-${s.tag}`} title={`Pipeline${sides.length > 1 ? ` · ${s.tag}` : ''}`} meta="task context → system i → packet → system 0 → targets → tracker → robot, at the cursor" lazy={false}>
      <PipelineFlow r={s.replay} times={s.times} t={p.t} />
    </Pane>
  )));
  if (hasMorphology(A)) out.push(<Pane key="morph" title="Morphology" meta="recorded body positions, contacts, torque"><Morphology r={A} times={sides[0].times} t={p.t} /></Pane>);
  if (hasPacketStructure(A)) out.push(<Pane key="pstruct" title="Packet structure" meta="knots × assemblies with probe heads"><PacketStructure r={A} times={sides[0].times} t={p.t} /></Pane>);
  if (sides.some((s) => hasMap(s.replay))) out.push(<Pane key="map" title="Top-down trajectory" meta="base and object paths, waypoints, edit onset, fall"><TopDownMap {...p} /></Pane>);
  if (any((r) => r.signals.phase)) out.push(<Pane key="phase" title="Phase" meta={str((A.meta.signal_notes as Record<string, string> | undefined)?.phase)}><PhaseLanes {...p} /></Pane>);
  if (any((r) => r.signals.task_events)) out.push(<Pane key="events" title="Task events" meta="status of each task event over time"><EventGantt {...p} /></Pane>);
  if (any((r) => r.signals.contacts)) out.push(<Pane key="gait" title={A.meta.family === 'legged' ? 'Gait diagram' : 'Contact diagram'} meta={str((A.meta.signal_notes as Record<string, string> | undefined)?.contacts)}><GaitDiagram {...p} /></Pane>);
  if (any((r) => r.signals.forward_progress) || any((r) => r.signals.object_pose)) out.push(
    <Pane key="progress" title="Progress" meta="forward progress · object height">
      {any((r) => r.signals.forward_progress) && <ScalarTrack title="Forward progress" unit="m" noteKey="forward_progress" get={(r) => r.signals.forward_progress as Val[] | undefined} {...p} />}
      {any((r) => r.signals.object_pose) && <ScalarTrack title="Object height (z)" unit="m" noteKey="object_pose" get={(r) => r.signals.object_pose?.map((v) => (v && v.length >= 3 ? v[2] : null))} {...p} />}
    </Pane>,
  );
  if (sides.some((s) => hasEdit(s.replay)) || sides.length > 1) out.push(<Pane key="edit" title={sides.length > 1 ? 'Edit and difference (B − A)' : 'Edit'} meta="edit onset, recorded eval row, difference traces"><EditPanel {...p} /></Pane>);
  if (any((r) => r.signals.joint_pos) || any((r) => r.signals.joint_target)) {
    out.push(<Pane key="jheat" title="Joint heatmap" meta="joints × time"><JointHeatmap {...p} /></Pane>);
    out.push(<Pane key="phaseportrait" title="Phase portrait" meta="position vs velocity for one joint"><PhasePortrait {...p} /></Pane>);
  }
  if (any((r) => r.signals.slip) || any((r) => r.signals.penetration_mm) || any((r) => r.signals.contact_force) || any((r) => r.signals.joint_torque)) out.push(
    <Pane key="force" title="Forces, slip, penetration" meta="privileged diagnostics (display only)">
      {any((r) => r.signals.contact_force) && <ScalarTrack title="Contact normal force" unit="N" noteKey="contact_force" names={(r) => r.meta.contact_bodies} get={(r) => r.signals.contact_force as Val[] | undefined} {...p} />}
      {any((r) => r.signals.contact_force_tangential) && <ScalarTrack title="Contact tangential force" unit="N" noteKey="contact_force_tangential" names={(r) => r.meta.contact_bodies} get={(r) => r.signals.contact_force_tangential as Val[] | undefined} {...p} />}
      {any((r) => r.signals.slip) && <ScalarTrack title="Slip" noteKey="slip" names={(r) => r.meta.contact_bodies} get={(r) => r.signals.slip as Val[] | undefined} {...p} />}
      {any((r) => r.signals.penetration_mm) && <ScalarTrack title="Penetration" unit="mm" noteKey="penetration_mm" get={(r) => r.signals.penetration_mm as Val[] | undefined} {...p} />}
      {any((r) => r.signals.joint_torque) && <ScalarTrack title="Joint torque" unit="N·m" noteKey="joint_torque" names={(r) => r.meta.joint_torque_names as string[] | undefined} get={(r) => r.signals.joint_torque as Val[] | undefined} {...p} />}
    </Pane>,
  );
  if (any((r) => r.signals.base_vel) || any((r) => r.signals.object_vel) || any((r) => r.signals.base_ang_vel)) out.push(
    <Pane key="vel" title="Velocities" meta="world frame · privileged">
      {any((r) => r.signals.base_vel) && <ScalarTrack title="Base velocity" unit="m/s" noteKey="base_vel" names={() => ['vx', 'vy', 'vz']} get={(r) => r.signals.base_vel as Val[] | undefined} {...p} />}
      {any((r) => r.signals.base_ang_vel) && <ScalarTrack title="Base angular velocity" unit="rad/s" noteKey="base_ang_vel" names={() => ['wx', 'wy', 'wz']} get={(r) => r.signals.base_ang_vel as Val[] | undefined} {...p} />}
      {any((r) => r.signals.object_vel) && <ScalarTrack title="Object velocity" unit="m/s" noteKey="object_vel" names={() => ['vx', 'vy', 'vz']} get={(r) => r.signals.object_vel as Val[] | undefined} {...p} />}
    </Pane>,
  );
  if (any((r) => r.signals.gripper_aperture) || any((r) => r.signals.grasp_state) || any((r) => r.signals.grip_drift) || any((r) => r.signals.hand_contact)) out.push(
    <Pane key="grasp" title="Grasp" meta="aperture, grasp state, hand contact, grip drift">
      {any((r) => r.signals.grasp_state) && <CategoryTrack title="Grasp state (privileged)" noteKey="grasp_state" get={(r) => r.signals.grasp_state as unknown[] | undefined} {...p} />}
      {any((r) => r.signals.gripper_aperture) && <ScalarTrack title="Gripper aperture" unit="m" noteKey="gripper_aperture" get={(r) => r.signals.gripper_aperture as Val[] | undefined} {...p} />}
      {any((r) => r.signals.hand_contact) && <RasterTrack title="Hand contact (public touch)" noteKey="hand_contact" names={(r) => r.meta.hands as string[] | undefined} get={(r) => r.signals.hand_contact as unknown[] | undefined} {...p} />}
      {any((r) => r.signals.grip_drift) && <ScalarTrack title="Grip drift since grasp (position)" unit="mm" noteKey="grip_drift" names={(r) => r.meta.hands as string[] | undefined}
        get={(r) => (r.signals.grip_drift as (unknown[] | null)[] | undefined)?.map((f) => (Array.isArray(f) ? f.map((h) => (Array.isArray(h) && typeof h[0] === 'number' ? h[0] : null)) : null))} {...p} />}
    </Pane>,
  );
  if (any((r) => r.signals.energy) || any((r) => r.signals.power) || any((r) => r.signals.cot)) out.push(
    <Pane key="energy" title="Energy and cost of transport" meta={A.meta.energy_total_j != null ? `total ${fmtNum(A.meta.energy_total_j)} J (recorded)` : 'per step, as recorded'}>
      {any((r) => r.signals.energy) && <ScalarTrack title="Energy per step" unit="J" noteKey="energy" get={(r) => r.signals.energy as Val[] | undefined} {...p} />}
      {any((r) => r.signals.power) && <ScalarTrack title="Power" unit="W" noteKey="power" get={(r) => r.signals.power as Val[] | undefined} {...p} />}
      {any((r) => r.signals.cot) && <ScalarTrack title="Cost of transport" noteKey="cot" get={(r) => r.signals.cot as Val[] | undefined} {...p} />}
    </Pane>,
  );
  if (any((r) => r.signals.packet_pca) || any((r) => r.signals.packet_z)) {
    const basis = (r: Replay) => JSON.stringify((r.meta.packet_pca as Record<string, unknown> | undefined)?.explained_variance ?? r.meta.packet_pca_basis?.explained_variance_ratio ?? null) + str(r.meta.packet_pca_basis?.n_fit);
    const withP = sides.filter((s) => s.replay.signals.packet_pca);
    const same = withP.length < 2 || basis(withP[0].replay) === basis(withP[1].replay);
    const series = (same ? withP : withP.slice(0, 1)).map((s) => ({ label: s.tag, color: s.color, pts: s.replay.signals.packet_pca!, times: s.times, edit: s.replay.signals.edit_active }));
    const fitOn = str((A.meta.packet_pca as Record<string, unknown> | undefined)?.fit_on);
    out.push(
      <Pane key="packet" title="Packet" meta={fitOn || 'packet executed by system 0'}>
        {series.length > 0 && <PcaPlot series={series} clock={clock} height={260} />}
        {series.length > 0 && <div className="legend small">{series.map((s) => <span key={s.label}><i className="sw" style={{ background: s.color }} />{s.label} PCA path</span>)}<span><i className="sw" style={{ background: 'var(--s2)' }} />edit-active frames</span>{!same && <span>bases differ: only A drawn</span>}</div>}
        {withP.length > 0 && <ScalarTrack title="Packet change rate ‖Δ PCA‖/Δt (3 of D dims; computed here)" get={(r) => pcaSpeed(r, relTimes(r))} {...p} />}
        <PacketHeatmap {...p} />
        {sides[0].replay.signals.packet_norm ? <PacketHeatmap {...p} signal="packet_norm" label="full-dz ‖·‖ per knot × assembly" /> : null}
      </Pane>,
    );
  }
  if (any((r) => r.signals.probe)) out.push(
    <Pane key="probe" title="Probe readouts" meta={hasProbeTruth(A) ? 'predicted vs privileged truth' : 'diagnostics; truth not recorded in this replay'}>
      {hasProbeTruth(A) ? <ProbeTruth {...p} /> : <ProbeTracks {...p} />}
    </Pane>,
  );
  const vids = videoCandidates(entry, videos);
  if (vids.length) out.push(<Pane key="video" title="Video" meta={vids[0].exact ? 'linked by the recorder' : 'matched by file name: check the label'}><VideoPanel vids={vids} /></Pane>);
  out.push(<Pane key="evidence" title="Evidence" meta="the original eval row, the reproduction check and provenance" lazy={false}><Evidence sides={sides} /></Pane>);
  return out;
}

const SKIP = new Set(['packet_pca_basis', 'signal_notes', 'geoms']);
function Evidence({ sides }: { sides: Side[] }) {
  return (
    <div className="rh-pair">
      {sides.map((s) => {
        const m = s.replay.meta;
        const rep = isObj(m.reproduction) ? m.reproduction : null;
        const compared = rep && isObj(rep.compared) ? rep.compared : null;
        return (
          <div key={s.tag} style={{ padding: '0 6px' }}>
            {sides.length > 1 && <b style={{ color: s.color }}>{s.tag}</b>}
            {compared && (
              <table className="bt" style={{ marginBottom: 6 }}>
                <thead><tr><th>reproduction check</th><th>recorded</th><th>re-run</th><th /></tr></thead>
                <tbody>
                  {Object.entries(compared).map(([k, v]) => {
                    const rec = isObj(v) ? v.recorded : undefined, rr = isObj(v) ? v.rerun : undefined;
                    const same = JSON.stringify(rec) === JSON.stringify(rr);
                    return <tr key={k}><td>{k}</td><td className="mono">{JSON.stringify(rec)}</td><td className="mono">{JSON.stringify(rr)}</td><td style={{ color: same ? 'var(--good)' : 'var(--critical)' }}>{same ? '=' : '≠'}</td></tr>;
                  })}
                </tbody>
              </table>
            )}
            <dl className="evid">
              {Object.entries(m).filter(([k]) => !SKIP.has(k)).map(([k, v]) => (
                <FragmentDl key={k} k={k} v={k === 'reproduction' && compared ? { ...(v as Row), compared: '(table above)' } : v} />
              ))}
              <dt>replay</dt><dd>{s.replay.id} · {s.replay.n_frames} frames @ {fmtNum(s.replay.fps)} fps · signals: {Object.keys(s.replay.signals).join(', ')}</dd>
              <dt>signal notes</dt><dd>{Object.entries((m.signal_notes as Record<string, string>) || {}).map(([k, v]) => `${k}: ${v}`).join(' | ') || '—'}</dd>
            </dl>
          </div>
        );
      })}
    </div>
  );
}
function FragmentDl({ k, v }: { k: string; v: unknown }) {
  const text = typeof v === 'string' ? v : JSON.stringify(v);
  return <><dt>{k}</dt><dd title={text.length > 400 ? text : undefined}>{text.length > 400 ? `${text.slice(0, 400)}…` : text}{arr(v).length > 20 ? ` (${arr(v).length} items)` : ''}</dd></>;
}
