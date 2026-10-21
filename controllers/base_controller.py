"""
controllers/base_controller.py
================================
Abstract base class every controller implements. Keeping a common
interface lets `main.py` run every retained strategy through the
exact same simulation loop, guaranteeing a fair, apples-to-apples
comparison.
"""

import time
from abc import ABC, abstractmethod

import config
from utils.sumo_utils import get_graph


class BaseController(ABC):
    name = "Base"

    def __init__(self, conn):
        self.conn = conn
        self.graph = get_graph(conn)
        self.tls_ids = self.graph.tls_ids
        # per-TL bookkeeping used by several controllers
        self.current_phase = {tl: conn.trafficlight.getPhase(tl) for tl in self.tls_ids}
        self.time_in_phase = {tl: 0 for tl in self.tls_ids}
        self.last_decision_time = {tl: -config.DECISION_INTERVAL for tl in self.tls_ids}
        self.pending_phase = {tl: None for tl in self.tls_ids}

    @abstractmethod
    def decide(self, tl: str, sim_time: float):
        """Called every simulation step for every TL; controller decides
        internally (based on DECISION_INTERVAL / MIN_GREEN / MAX_GREEN)
        whether to act. Must be non-blocking / fast."""
        raise NotImplementedError

    def step(self, sim_time: float) -> float:
        """Run one control decision pass over all TLs; returns the wall
        clock compute time spent (used for computational-performance
        metrics)."""
        t0 = time.perf_counter()
        for tl in self.tls_ids:
            self.time_in_phase[tl] += config.STEP_LENGTH
            self._apply_pending_after_clearance(tl)
            self.decide(tl, sim_time)
        return time.perf_counter() - t0

    # ------------------------------------------------------------------- #
    def _safe_set_phase(self, tl: str, phase_idx: int):
        n = self.graph.num_phases[tl]
        phase_idx = int(phase_idx) % n
        current = self.conn.trafficlight.getPhase(tl)
        if phase_idx == current:
            return
        state = self.conn.trafficlight.getRedYellowGreenState(tl)
        self.pending_phase[tl] = phase_idx
        if "y" in state:
            return
        # RESCO grid programs place a three-second clearance immediately
        # after every selectable green stage.  Enter that clearance before
        # applying an arbitrary Max-Pressure target; direct green-to-green
        # jumps would give the baseline an unsafe and unfair zero-lost-time
        # advantage.
        transition = (current + 1) % n
        transition_state = self.graph.phase_states[tl][transition]
        if "y" in transition_state:
            self.conn.trafficlight.setPhase(tl, transition)
            self.time_in_phase[tl] = 0
        else:
            self.conn.trafficlight.setPhase(tl, phase_idx)
            self.pending_phase[tl] = None
            self.time_in_phase[tl] = 0

    def _apply_pending_after_clearance(self, tl: str):
        target = self.pending_phase[tl]
        if target is None:
            return
        state = self.conn.trafficlight.getRedYellowGreenState(tl)
        if "y" not in state:
            self.conn.trafficlight.setPhase(tl, target)
            self.pending_phase[tl] = None
            self.time_in_phase[tl] = 0
