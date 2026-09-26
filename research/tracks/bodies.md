# track: sprint_bodies (demo sprint 2026-09-25)

Goal: honest, labelled video evidence of the NON-ARM bodies the stack supports (plus arm breadth), for the demo page.
Everything here is **scripted_teacher (privileged) driving a frozen body tracker**. No learned legged/humanoid
high-level model exists (D-040); nothing here is evidence about the latent-packet architecture.

## SPRINT BODIES RESULT

state: **completed** (code + clips + 20-seed teacher references; verified by rendering and by the eval harness).

### teacher reference, waypoint_contact (walk to A, walk to B, halt), dev seeds 10000-10019, privileged evaluator
| body | kind | tracker | teacher | success / 20 | fell | fail reason (non-fall) | raw |
|---|---|---|---|---|---|---|---|
| t1 | humanoid | frozen learned t1:iter2999 | default | **20** | 0 | - | `artifacts/runs/sprint_bodies_teacher_ref/eval_dev_t1_default.jsonl` |
| g1 | humanoid | frozen learned g1:iter3599 | arc_only (declared variant; the tracker only supports forward + arc turns) | **18** | 0 | 2 halt failed (drifts off waypoint B while halting) | `.../eval_dev_g1_arc_only.jsonl` |
| g1 | humanoid | same | default (in-place turns, outside the tracker's validated use) | 17 | **2** | 1 halt failed | `.../eval_dev_g1_default.jsonl` |
| h1 | humanoid | frozen learned h1:iter2999 | default | **14** | 0 | 4 halt failed, 2 timeouts (60 s) | `.../eval_dev_h1_default.jsonl` |
| h1 | humanoid | same | arc_only | 13 | 0 | 7 halt failed | `.../eval_dev_h1_arc_only.jsonl` |
| anymal_c | quadruped | frozen learned anymal_c:iter1274 | default | **20** | 0 | - | `.../eval_dev_anymal_c_default.jsonl` |
| go2, pquad4, hexapod6, sprawl4, sprawl8, hexapod6_long | legged | go2 learned; others CPG gait tracker (scripted) | default | 20 each | 0 | - | `artifacts/runs/legged_vlm_teacher_ref/eval_dev.jsonl` (legged_vlm track) |

Summaries for the demo builder (format `{"per_body": {body: {success, n, fell, note}}}`):
`artifacts/runs/sprint_bodies_teacher_ref/humanoids_anymal_default.summary.json`,
`.../humanoids_anymal_arc_only.summary.json`, `.../legged_prior.summary.json` (copy of the six-body legged_vlm reference).

Honest reading: t1 and anymal_c are solid under the scripted teacher. g1 walks but its limited tracker cannot hold a
stable halt in 2/20 (arc_only) and falls on 2/20 when the default teacher asks for in-place turns. h1 is the weakest
humanoid (14/20): it reaches both waypoints but drifts during the halt phase or times out. These are teacher +
tracker limits, not learned-policy results. Previous train-seed (0-19) numbers from the archive agree
(`artifacts/assets/legged_teacher/*.json`: g1 16/20 default, 19/20 arc_only; h1 11/20 / 12/20; t1 and anymal_c 20/20).

### clips (artifacts/video/, all `scripted_teacher`, one INDEX.md line each)
Legged / humanoid, waypoint_contact, tracking camera, caption = source + tracker, body, task, seed, outcome,
playback speed (10 Hz frames; 1.5x for short episodes, 3x for the long humanoid ones; every clip <= 20 s, <= 0.4 MB):
- go2: `2026-09-25_scripted_teacher_legged_go2_waypoint_contact_s1000{0,1}_success.mp4`
- anymal_c: `2026-09-25_scripted_teacher_legged_anymal_c_waypoint_contact_s1000{0,1}_success.mp4`
- pquad4: `2026-09-25_scripted_teacher_legged_pquad4_waypoint_contact_s1000{0,1}_success.mp4`
- hexapod6: `2026-09-25_scripted_teacher_legged_hexapod6_waypoint_contact_s1000{0,1}_success.mp4`
- hexapod6_long: `2026-09-25_scripted_teacher_legged_hexapod6_long_waypoint_contact_s1000{0,1}_success.mp4`
- sprawl4: `2026-09-25_scripted_teacher_legged_sprawl4_waypoint_contact_s1000{0,1}_success.mp4`
- sprawl8: `2026-09-25_scripted_teacher_legged_sprawl8_waypoint_contact_s1000{0,1}_success.mp4`
- t1 (humanoid): `2026-09-25_scripted_teacher_legged_t1_waypoint_contact_s1000{0,1}_success.mp4`
- g1 (humanoid, arc_only teacher): `2026-09-25_scripted_teacher_legged_g1_arconly_waypoint_contact_s10001_success.mp4`;
  FAILURE `..._g1_arconly_waypoint_contact_s10000_failure.mp4` (reaches A and B, halt fails: drifts off B)
- g1 (default teacher) FALL: `2026-09-25_scripted_teacher_legged_g1_waypoint_contact_s10002_fell.mp4` (falls at 8.8 s
  while turning toward B; the same seed falls in the 20-seed eval, not staged)
- h1 (humanoid): `2026-09-25_scripted_teacher_legged_h1_waypoint_contact_s1000{0,1}_success.mp4`;
  FAILURE `..._h1_waypoint_contact_s10005_failure.mp4` (halt fails)

Montage (next item picked after the list: one clip showing morphology breadth at a glance):
`2026-09-25_scripted_teacher_legged_montage_10bodies.mp4` (1280x760, 19 s, 0.7 MB): 10 tiles, all scripted_teacher
successes, header says "NOT a learned policy". The name has no single body key, so the demo builder must reference it by name.
Built from the per-body clips with a small PIL/imageio tiling script (not committed; one-off).

Arm breadth, pick_place, `scripts/render_episode.py --source scripted_teacher --tag bodies`, dev seeds 3000001-3000002:
- ur5e_pg2 2/2, sawyer_tf3 2/2, parm5s_tf3 (procedural) 2/2 success:
  `2026-09-25_scripted_teacher_bodies_{ur5e_pg2,sawyer_tf3,parm5s_tf3}_pick_place_s300000{1,2}_success.mp4`
- xarm7_pg2 (held-out TARGET body; teacher only, no learned policy was run on it): 2/2 success,
  `2026-09-25_scripted_teacher_bodies_xarm7_pg2_pick_place_s300000{1,2}_success.mp4`
- parm7_pg2 (procedural): 0/2, `..._bodies_parm7_pg2_pick_place_s300000{1,2}_failure.mp4`: the teacher grasps the cube
  but never completes the place (hovers over the zone); a scripted-teacher weakness on this body (2 seeds only).

Page guidance for sprint_demo: label every one of these "scripted teacher (privileged) + frozen tracker"; do not put
them next to learned numbers as if comparable; say explicitly that no learned legged/humanoid policy exists.

## code
- `scripts/render_legged_episode.py`: renders the teacher route of `rrp.evaluation.legged_latent_eval.run_episode`
  (same scene builder, WaypointTeacher, frozen tracker), tracking camera, clip <= 20 s.
- `src/rrp/evaluation/legged_latent_eval.py`: `run_episode` now uses the frozen learned tracker ("auto") for
  go2/t1/g1/h1/anymal_c (previously only go2/t1; g1/h1/anymal_c would have raised with "cpg"); `--arc-only` flag for
  the declared teacher variant (source recorded as `scripted_teacher:arc_only`); rows record the tracker version.
  Behaviour for the six previously evaluated bodies is unchanged.

## commands (peer, RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/bodies)
```bash
O=/dev/shm/rrp-brandonin/repo/artifacts/runs/bodies_teacher_ref   # copied to artifacts/runs/sprint_bodies_teacher_ref
scripts/peer_run.sh --cpu 1 --mem 3G --label bodies_ref_<b>_<variant> --max-seconds 7200 --detach -- env OMP_NUM_THREADS=1 \
  PY -m rrp.evaluation.legged_latent_eval --bodies <b> --seeds 10000-10019 --arc-only <none|b> --out $O/eval_dev_<b>_<variant>.jsonl
scripts/peer_run.sh --gpu --gpu-mem 2G --cpu 2 --mem 6G --label bodies_render_legged --max-seconds 3600 -- PY scripts/render_legged_episode.py \
  --bodies go2,pquad4,hexapod6,sprawl8,sprawl4,hexapod6_long,t1,g1,h1,anymal_c --seeds 10000,10001 --arc-only g1 \
  --out /dev/shm/rrp-brandonin/bodies_video --rows /dev/shm/rrp-brandonin/bodies_video/render_rows.jsonl
#   failures: --bodies g1 --seeds 10002 ; --bodies h1 --seeds 10005
scripts/peer_run.sh --gpu --gpu-mem 2G --cpu 2 --mem 6G --label bodies_render_arms --max-seconds 3600 -- bash -c 'for r in ur5e_pg2 sawyer_tf3 parm7_pg2 parm5s_tf3 xarm7_pg2; do PY scripts/render_episode.py --robot $r --seeds 3000001,3000002 --source scripted_teacher --tag bodies --max-steps 400 --out /dev/shm/rrp-brandonin/bodies_video_arms; done'
```
Leases (all finished rc=0): refs 1790399672_965e54 (t1), 1790399672_dd314b (g1), 1790399673_08d5b9 (g1 arc),
1790399674_c750c6 (h1), 1790399674_48c264 (h1 arc), 1790399675_cd4a2a (anymal_c); renders 1790399728_bb190d,
1790399738_90f00d, plus two smoke/failure render leases. Rendered episodes reproduce the eval rows exactly
(deterministic; e.g. g1 s10002 falls at 8.8 s in both).

Side note: `artifacts/trackers/{go2,hexapod6}/actor.pt` (and go2/rejected_iter1999/actor.pt) are tracked in git on
main from before this track, contrary to the no-weights rule. Not changed here; flagged to the lead.
