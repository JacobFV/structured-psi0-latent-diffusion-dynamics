"""EXPO-FT-INSPIRED online adaptation (see research/methods/expo-port.md for ingredients/departures).

Actor (on-the-fly policy): sample N base chunks from the flow policy (ODE sampler), draw one bounded
tanh-Gaussian edit per base chunk, and execute argmax over the 2N candidates of the target-critic
min-of-2 score. Learner per update call: `utd` critic minibatches (min-of-2 target ensemble, chunk-level
n-step backup with the same on-the-fly rule at s'), one edit-policy + temperature update, one base-policy
flow-matching update on windows from SUCCESSFUL replay episodes. Codec and controller frozen.
"""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from rrp.learning.critics import QEnsemble, EditPolicy, make_target, min_of_random_pair, soft_update, td_target
from rrp.learning.grpo import set_trainable
from rrp.learning.replay_buffer import ReplayBuffer, ReplayRecord
from rrp.learning.rollout import PolicyAdapter, EpisodeState, drive, finalize
from rrp.model.batch import collate_inputs


@dataclass
class ExpoConfig:
    n_base: int = 8                  # N base samples per decision (EXPO-FT: 8)
    beta: float = 0.05               # edit bound (EXPO-FT: 0.05 or 0.2)
    ensemble: int = 10
    hidden: int = 256
    gamma: float = 0.99
    tau_q: float = 5e-3
    lr_critic: float = 3e-4
    lr_edit: float = 3e-4
    lr_alpha: float = 3e-4
    alpha0: float = 1.0
    lr_base: float = 1e-5
    utd: int = 20                    # critic minibatches per update call
    batch: int = 64
    env_steps_per_update: int = 40  # ~1 update call per 40 env steps (EXPO-FT paper)
    base_update: bool = True         # False = frozen base (separately named "frozen-base edit" baseline)
    base_success_only: bool = True   # EXPO-FT code: actor_success_only
    replay_capacity: int = 200000
    nfe: int = 8
    trainable: str = "action_expert"


class ExpoAgent(PolicyAdapter):
    def __init__(self, model, codec, device, cfg: ExpoConfig, *, execute_prefix=8, action_nodes: int, seed=0,
                 version_prefix="expo"):
        super().__init__(model, codec, device, execute_prefix=execute_prefix, version=f"{version_prefix}@0")
        self.cfg = cfg
        self.C = execute_prefix
        self.n = action_nodes
        self.A = self.C * self.n
        self.encoder = copy.deepcopy(model.context).eval()        # frozen critic/edit observation encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        D = model.cfg.D
        self.q = QEnsemble(D, self.A, cfg.ensemble, cfg.hidden).to(device)
        self.q_t = make_target(self.q)
        self.edit = EditPolicy(D + 0, self.A, beta=cfg.beta, hidden=cfg.hidden).to(device)
        self.log_alpha = torch.tensor(math.log(cfg.alpha0), device=device, requires_grad=True)
        self.target_entropy = -self.A / 2.0
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_critic)
        self.opt_e = torch.optim.Adam(self.edit.parameters(), lr=cfg.lr_edit)
        self.opt_a = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.frozen = set_trainable(model, cfg.trainable)
        self.base_params = [p for p in model.parameters() if p.requires_grad]
        self.opt_b = torch.optim.AdamW(self.base_params, lr=cfg.lr_base, weight_decay=1e-4)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.cpu_gen = torch.Generator().manual_seed(seed)
        self.rng = random.Random(seed)
        self.deterministic = False
        self.last_records: list[dict] = []
        self.base_iter = 0
        self.edit_iter = 0
        self.stats = dict(critic_steps=0, edit_steps=0, base_steps=0, update_calls=0, update_s=0.0,
                          base_velocity_evals_rollout=0, base_velocity_evals_update=0, selected_edited=0,
                          selected_base=0)

    # ---------------------------------------------------------------- versions
    @property
    def policy_version(self):
        return f"expo-base@{self.base_iter}"

    @property
    def edit_version(self):
        return f"expo-edit@{self.edit_iter}"

    # ---------------------------------------------------------------- observation embedding
    @torch.no_grad()
    def embed(self, batch):
        h, mask, _ = self.encoder(batch)
        m = mask.to(h.dtype)[..., None]
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def _score(self, s, a):
        """target-critic min-of-2 score used by the on-the-fly policy."""
        return min_of_random_pair(self.q_t(s, a), self.cpu_gen)

    # ---------------------------------------------------------------- actor
    @torch.no_grad()
    def chunks(self, sessions):
        t0 = time.perf_counter()
        self.model.eval()
        feats, obs = self.featurize(sessions)
        B, N, H, C = len(sessions), self.cfg.n_base, self.model.cfg.horizon, self.C
        batch = collate_inputs(feats).to(self.device)
        s_emb = self.embed(batch)                                            # [B, D]
        rep = collate_inputs([f for f in feats for _ in range(N)]).to(self.device)
        cache = self.model.prepare(rep)
        z = self.model.sample(cache, H, nfe=self.cfg.nfe, generator=self.gen)
        a = self.decode(z, rep)[:, :C, :self.n].float()                      # [B*N, C, n]
        a = a.clamp(-6, 6)
        self.stats["base_velocity_evals_rollout"] += B * N * self.cfg.nfe
        base = a.reshape(B, N, C * self.n)
        s_rep = s_emb.repeat_interleave(N, 0)
        e, _ = self.edit.sample(s_rep, base.reshape(B * N, -1), deterministic=self.deterministic, gen=self.gen)
        edited = (base.reshape(B * N, -1) + e).reshape(B, N, -1)
        cand = torch.cat([base, edited], 1)                                  # [B, 2N, A]
        sc = self._score(s_emb.repeat_interleave(2 * N, 0), cand.reshape(B * 2 * N, -1)).reshape(B, 2 * N)
        best = sc.argmax(1)
        out, self.last_records = [], []
        for i, (s, pi, o) in enumerate(zip(sessions, feats, obs)):
            k = int(best[i])
            bi = k % N
            chosen = cand[i, k].reshape(C, self.n).cpu().numpy()
            ch, rows = self.make_chunk(s, pi, o, chosen, policy_version=f"{self.policy_version}+{self.edit_version}")
            f = self.featurizer(s)
            exec_norm = f.aspace.normalize(rows, pi.q0)
            self.prev[id(s)] = np.clip(chosen[-1], -6, 6)
            self.stats["selected_edited" if k >= N else "selected_base"] += 1
            out.append(ch)
            self.last_records.append(dict(
                pi=pi, state_emb=s_emb[i].cpu().numpy(), base_proposal=base[i, bi].reshape(C, self.n).cpu().numpy(),
                edit=(cand[i, k] - base[i, bi]).reshape(C, self.n).cpu().numpy(), executed_norm=exec_norm,
                executed_native=rows, base_candidates=base[i].reshape(N, C, self.n).cpu().numpy(),
                selected="edited" if k >= N else "base", graph_version=ch.graph_version,
                runtime_version=ch.runtime_version, robot_spec_hash=ch.robot_spec_hash,
                controller_version=ch.controller_version, observation_id=o.observation_id))
        self.calls += B
        self.batched_calls += 1
        self.infer_s += time.perf_counter() - t0
        return out

    # ---------------------------------------------------------------- learner
    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def critic_step(self, recs: list[ReplayRecord]) -> dict:
        cfg = self.cfg
        s = self._t(np.stack([r.state_emb for r in recs]))
        a = self._t(np.stack([r.executed_norm.reshape(-1) for r in recs]))
        R = self._t([r.reward for r in recs])
        n = self._t([r.n_steps for r in recs])
        done = self._t([float(r.done) for r in recs])
        with torch.no_grad():
            nb = [r for r in recs]
            B, N = len(recs), recs[0].base_candidates.shape[0]
            s2 = self._t(np.stack([r.next_state_emb if r.next_state_emb is not None else r.state_emb for r in nb]))
            c2 = self._t(np.stack([r.next_base_candidates if r.next_base_candidates is not None
                                   else r.base_candidates for r in nb])).reshape(B, N, -1)
            s2r = s2.repeat_interleave(N, 0)
            e2, _ = self.edit.sample(s2r, c2.reshape(B * N, -1), gen=self.gen)
            cand = torch.cat([c2, (c2.reshape(B * N, -1) + e2).reshape(B, N, -1)], 1)
            qt = self.q_t(s2.repeat_interleave(2 * N, 0), cand.reshape(B * 2 * N, -1))     # [E, B*2N]
            q2 = min_of_random_pair(qt, self.cpu_gen).reshape(B, 2 * N)
            y = td_target(R, n, done, cfg.gamma, q2.max(1).values)
        q = self.q(s, a)                                                                 # [E, B]
        loss = ((q - y[None]) ** 2).mean()
        self.opt_q.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_q.step()
        soft_update(self.q_t, self.q, cfg.tau_q)
        self.stats["critic_steps"] += 1
        return dict(q_loss=loss.item(), q_mean=q.mean().item(), y_mean=y.mean().item())

    def edit_step(self, recs: list[ReplayRecord]) -> dict:
        s = self._t(np.stack([r.state_emb for r in recs]))
        a = self._t(np.stack([r.base_candidates[self.rng.randrange(r.base_candidates.shape[0])].reshape(-1)
                              for r in recs]))
        e, logp = self.edit.sample(s, a, gen=self.gen)
        q = self.q(s, a + e).mean(0)                        # mean over online members (EXPO-FT code)
        alpha = self.log_alpha.exp().detach()
        loss = (alpha * logp - q).mean()
        self.opt_e.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_e.step()
        a_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_a.zero_grad(set_to_none=True)
        a_loss.backward()
        self.opt_a.step()
        self.stats["edit_steps"] += 1
        self.edit_iter += 1
        return dict(edit_loss=loss.item(), alpha=alpha.item(), entropy=-logp.mean().item(),
                    edit_abs=e.abs().mean().item())

    def base_step(self, buf: ReplayBuffer) -> dict:
        samples = buf.sample_bc(self.cfg.batch, self.rng)
        if not samples:
            return {}
        self.model.train()
        batch = collate_inputs([x["pi"] for x in samples]).to(self.device)
        Bn, Nn = batch.node_mask.shape
        H = self.model.cfg.horizon
        a = torch.zeros(Bn, H, Nn, device=self.device)
        v = torch.zeros(Bn, H, Nn, dtype=torch.bool, device=self.device)
        for i, x in enumerate(samples):
            k = x["a"].shape[1]
            a[i, :, :k] = self._t(x["a"])
            v[i, :, :k] = torch.as_tensor(x["valid"], device=self.device)
        if self.codec is not None:
            with torch.no_grad():
                _, target, _ = self.codec.encode(a, batch.node_feats, batch.node_mask)
        else:
            target = a[..., None]
        loss, logs = self.model.loss(batch, target, v, None, generator=None)
        self.opt_b.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.base_params, 1.0)
        self.opt_b.step()
        self.model.eval()
        self.stats["base_steps"] += 1
        self.stats["base_velocity_evals_update"] += Bn
        self.base_iter += 1
        return dict(base_flow_loss=loss.item())

    def update(self, buf: ReplayBuffer, calls: int) -> dict:
        t0 = time.perf_counter()
        logs = {}
        for _ in range(calls):
            for _ in range(self.cfg.utd):
                recs = buf.sample(self.cfg.batch, self.rng)
                if not recs:
                    return logs
                logs.update(self.critic_step(recs))
            logs.update(self.edit_step(buf.sample(self.cfg.batch, self.rng)))
            if self.cfg.base_update:
                logs.update(self.base_step(buf))
            self.stats["update_calls"] += 1
        self.stats["update_s"] += time.perf_counter() - t0
        return logs


# -------------------------------------------------------------------- episode -> replay
def collect_expo_episodes(agent: ExpoAgent, make_scenario, seeds: list[int], max_steps: int, buf: ReplayBuffer,
                          H: int) -> list[dict]:
    from rrp.sim.native import Session
    states = []
    for sd in seeds:
        s = Session(make_scenario(sd), seed=sd)
        states.append(EpisodeState(s, sd, max_steps, tag=dict(seed=sd, rows=[])))

    def on_step(st, r):
        st.tag["rows"].append(r.command)

    agent.deterministic = False
    drive(agent, states, on_step=on_step)
    out = []
    for st in states:
        fin = finalize(st)
        success = fin["privileged_success"]
        terminal = st.outcome in ("success", "failure") and (success or st.fell)
        recs = []
        chunks = st.chunks
        final_emb = None
        if not terminal and chunks:
            feats, _ = agent.featurize([st.session])
            final_emb = agent.embed(collate_inputs(feats).to(agent.device))[0].cpu().numpy()
        for j, c in enumerate(chunks):
            last = j == len(chunks) - 1
            n_steps = (st.steps if last else chunks[j + 1]["step"]) - c["step"]
            if n_steps <= 0:
                continue
            rew = (agent.cfg.gamma ** (n_steps - 1)) * float(success) if last else 0.0
            recs.append(ReplayRecord(
                episode_id=f"expo_s{st.seed}_{id(st)}", step=c["step"], state_emb=c["state_emb"],
                base_proposal=c["base_proposal"], edit=c["edit"], executed_norm=c["executed_norm"],
                executed_native=c["executed_native"], base_candidates=c["base_candidates"], reward=rew,
                n_steps=n_steps, done=bool(last and terminal), truncated=bool(last and not terminal),
                next_state_emb=(final_emb if last else chunks[j + 1]["state_emb"]) if not (last and terminal) else None,
                next_base_candidates=None if last else chunks[j + 1]["base_candidates"], selected=c["selected"],
                robot_spec_hash=c["robot_spec_hash"], controller_version=c["controller_version"],
                graph_version=c["graph_version"], runtime_version=c["runtime_version"],
                policy_version=agent.policy_version, edit_version=agent.edit_version,
                codec_version=agent.codec.cfg.version if agent.codec else None))
        # truncated last record without next candidates: reuse own candidates (documented approximation)
        buf.extend_episode(recs)
        if success or not agent.cfg.base_success_only:
            f = agent.featurizer(st.session)
            rows = st.tag["rows"]
            for c in chunks:
                t = c["step"]
                seq = rows[t:t + H]
                if not seq or seq[0] is None:
                    continue
                valid_rows = [r is not None for r in seq]
                filled, last_ok = [], seq[0]
                for r in seq:
                    last_ok = r if r is not None else last_ok
                    filled.append(last_ok)
                nv = len(filled)
                filled = filled + [filled[-1]] * (H - nv)
                an = f.aspace.normalize(filled, c["pi"].q0)
                valid = np.zeros_like(an, bool)
                valid[:nv] = np.array(valid_rows)[:, None]
                buf.add_bc(dict(pi=c["pi"], a=an.astype(np.float32), valid=valid))
        out.append(dict(seed=st.seed, steps=st.steps, **fin, chunks=len(chunks),
                        edited=sum(c["selected"] == "edited" for c in chunks)))
    agent.forget([st.session for st in states])
    return out
