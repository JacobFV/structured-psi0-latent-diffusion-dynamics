"""System 0 interface: the packet-acceptance protocol every realizer runtime shares (W11 stable core API).

A system 0 holds at most one received LatentActionChunk and turns it, every control tick, into native commands of its
body from fresh local state (proprio, declared local sensors, phase since valid_from). What is common to all bodies,
and therefore lives here, is the acceptance protocol:

- `receive(packet, now=, graph_version=)` checks the packet against this system 0's compatibility IDs
  (latent_space_version, realizer_compat_version), the robot spec hash, validity window and graph version
  (`check_packet`); a rejection is counted, logged and re-raised (never silently reinterpreted);
- `invalidate(reason, now)` drops the packet (graph edit, expiry) so the declared fallback applies;
- `stats` / `log` record packets, rejections, ticks and fallback holds.

Subclasses implement `robot_spec_hash` and `tick(...)` (the body-specific realization). The arm implementation is
`rrp.policies.system0.LatentSystem0`; an external body (e.g. psi1z's G1 through the Psi0 action
interface) subclasses `System0Base` the same way. numpy/pydantic only (contracts layer).
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import ControllerRejection, StaleActionError
from .latent_action import LatentActionChunk, check_packet


@dataclass
class System0Stats:
    ticks: int = 0
    packets: int = 0
    rejected: int = 0
    fallback_holds: int = 0


class System0Base:
    """Holds the current packet; `receive` / `invalidate` implement the shared acceptance protocol."""

    lsv: str        # latent_space_version this system 0 accepts
    rcv: str        # realizer_compat_version this system 0 accepts

    def __init__(self, *, latent_space_version: str, realizer_compat_version: str, fallback: str = "hold_measured"):
        self.lsv, self.rcv = latent_space_version, realizer_compat_version
        self.packet: LatentActionChunk | None = None
        self.fallback = fallback
        self.stats = System0Stats()
        self.log: list[dict] = []

    @property
    def robot_spec_hash(self) -> str:
        raise NotImplementedError

    def receive(self, packet: LatentActionChunk, *, now: float, graph_version: int | None = None):
        try:
            check_packet(packet, latent_space_version=self.lsv, realizer_compat_version=self.rcv,
                         robot_spec_hash=self.robot_spec_hash, now=now, graph_version=graph_version)
        except (ControllerRejection, StaleActionError) as e:
            self.stats.rejected += 1
            self.log.append(dict(t=now, event="packet_rejected", code=e.code))
            raise
        self.packet = packet
        self.stats.packets += 1
        self.log.append(dict(t=now, event="packet_accepted", obs=packet.observation_id))

    def invalidate(self, reason: str, now: float):
        if self.packet is not None:
            self.log.append(dict(t=now, event="packet_invalidated", reason=reason))
        self.packet = None

    def tick(self, *args, **kwargs):
        """Body-specific: realize the held packet from fresh local state; None = declared fallback."""
        raise NotImplementedError
