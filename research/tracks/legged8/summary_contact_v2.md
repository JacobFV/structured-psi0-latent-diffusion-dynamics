# W8: legged semantic supervision on contact_v2 (anymal_c, go2, t1)

Protocol D-090 (R2 deployable route; edits on snap_s4000, dev seeds 10000-10019, window t=2-5 s; 3 training seeds per variant; exact one-sided permutation over the 3 vs 3 training-seed means). Sources: learned:<flow ckpt>; commands scripted_teacher; gait learned_tracker (per body below).

| body | tracker | D-112 dataset gate | ctx halt semfix s0/s1/s2 → pooled | ctx halt nosem s0/s1/s2 → pooled | pooled semfix−nosem | every seed ordered (p one-sided) | inactive control semfix / nosem | goal steering semfix / nosem (ordered, p) | R2 success final: semfix / nosem | teacher / BC |
|---|---|---|---|---|---|---|---|---|---|---|
| anymal_c | anymal_c v2c iter2499 (2a16532b) | PASS (slip<0.15 100.0%, falls@0 0) | -0.32 / -0.10 / -0.16 → -0.19 [-0.26, -0.13] | +0.08 / +0.43 / +0.12 → +0.21 [+0.14, +0.28] | -0.40 [-0.50, -0.30] | True (0.05) | +0.002 / -0.003 | +0.22 / +0.12 (True, 0.05 fixsem>nosem) | 78/90 (0 fell) / 83/90 (0 fell) | 30/30 / 30/30 |
| go2 | go2 clearance-floor cf2 iter799 (af3f06f4) | PASS (slip<0.15 100.0%, falls@0 0) | -0.34 / -0.22 / -0.30 → -0.29 [-0.34, -0.24] | +0.54 / +0.21 / +0.34 → +0.37 [+0.32, +0.42] | -0.66 [-0.72, -0.59] | True (0.05) | -0.019 / -0.012 | +0.43 / +0.58 (False, 0.15 fixsem<nosem) | 87/90 (0 fell) / 87/90 (2 fell) | 30/30 / 30/30 |
| t1sl | t1 w8d sourced limits iter799 (36e91467) | FAIL (slip<0.15 86.2%, falls@0 0) | -0.43 / -0.10 / -0.04 → -0.31 [-0.37, -0.24] | – / – / +0.14 → +0.14 [+0.06, +0.22] (n=14) | -0.45 [-0.55, -0.34] | n/a (too few seeds with usable pairs) (–) | -0.003 / +0.004 | +0.07 / +0.07 (n/a (too few seeds with usable pairs), – ) | 30/90 (60 fell) / 33/90 (57 fell) | 30/30 / 0/30 |

Caveats:
- **t1sl: t1 dataset fails D-112 slip gate (86.2% < 95%), tracker w8d fails lab forward 0.72; t1 tracker w8d exceeds joint limits (margin -0.053) and peak foot force 4.46 BW (W6 gate backfill); trained as a recorded exception (D-113)**
