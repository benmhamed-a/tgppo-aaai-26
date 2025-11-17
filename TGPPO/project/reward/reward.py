import math
import numpy as np


def _safe_tanh(x, s=1.0):
    return float(np.tanh(s * x))


def _ratio(num, den, cap=None):
    den = max(float(den), 1e-12)
    val = float(num) / den
    if cap is not None:
        return min(val, cap)
    return val


class RewardH1:
    """Baseline-normalized node efficiency + terminal bonus.

    Simple, scale-robust: penalizes nodes per step normalized by baseline nodes
    and bonuses at terminal based on speedup vs. baseline and problem status.
    """

    def __init__(self, logger=None, alpha=1.0, bonus_cap=3.0):
        self.logger = logger
        self.alpha = alpha
        self.bonus_cap = bonus_cap
        self.reset(1.0, 0.0, 1.0, "timelimit", 3600.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=3600.0, logger=None):
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)
        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 3600.0), 1.0)
        self.logger = logger or self.logger
        self.prev_nodes = 0

    def compute(self, model, done):
        nodes = int(model.getNNodes())
        dn = max(nodes - self.prev_nodes, 0)
        self.prev_nodes = nodes

        # Step penalty normalized by baseline nodes; bounded by tanh
        step_penalty = - _safe_tanh(dn / (self.B * 0.02 + 1.0), s=self.alpha)  # ~2% of baseline gives ~0.76 penalty
        if not done:
            if self.logger:
                self.logger.info(f"H1 step: nodes={nodes}, dn={dn}, penalty={step_penalty:.4f}")
            return step_penalty

        # Terminal bonus
        status = model.getStatus()
        gap = float(model.getGap())
        pdi = float(model.getPrimalDualIntegral())

        speedup = _ratio(self.B, max(nodes, 1), cap=self.bonus_cap)  # >1 if better than baseline
        if status == "optimal":
            bonus = 1.0 + 2.0 * speedup
        elif status in ("infeasible", "unbounded"):
            bonus = 0.5 + 1.5 * speedup
        elif status == "timelimit":
            # progress vs baseline
            gap_gain = _safe_tanh((self.baseline_gap - gap))
            pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
            bonus = 0.2 * speedup + 0.6 * gap_gain + 0.2 * pdi_gain
        else:
            bonus = 0.2 * speedup

        if self.logger:
            self.logger.info(f"H1 terminal: status={status}, nodes={nodes}, speedup={speedup:.3f}, bonus={bonus:.4f}")
        return float(bonus)


class RewardH2:
    """Log-scaled node efficiency + progress shaping.

    Adds: (1) log scaling to further damp huge baselines, (2) pace term comparing
    current nodes to a power-law expected curve, (3) gap/PDI improvement shaping.
    """

    def __init__(self, logger=None, scale=1.5, pace_power=0.7, w_nodes=0.5, w_pace=0.2, w_gap=0.2, w_pdi=0.1):
        self.logger = logger
        self.scale = scale
        self.pace_power = pace_power
        self.w_nodes = w_nodes
        self.w_pace = w_pace
        self.w_gap = w_gap
        self.w_pdi = w_pdi
        self.reset(1.0, 0.0, 1.0, "timelimit", 3600.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=3600.0, logger=None):
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        self.logB = math.log1p(self.B)
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)
        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 3600.0), 1.0)
        self.logger = logger or self.logger
        self.prev_nodes = 0
        self.prev_gap = float('inf')
        self.prev_pdi = 0.0

    def _node_efficiency(self, nodes):
        # 1 - log(nodes+1)/log(B+1) in [-inf,1]; map via tanh
        eff = 1.0 - (math.log1p(nodes) / self.logB)
        return _safe_tanh(eff, s=self.scale)

    def _pace_term(self, nodes, t):
        expected = self.B * (t ** self.pace_power)
        # positive when under expected nodes
        pace = (expected - nodes) / (expected + 1.0)
        return _safe_tanh(pace, s=self.scale)

    def compute(self, model, done):
        nodes = int(model.getNNodes())
        t = float(model.getSolvingTime()) / self.time_limit
        t = min(max(t, 0.0), 1.0)

        gap = float(model.getGap())
        if math.isinf(gap):
            gap = 1e6
        pdi = float(model.getPrimalDualIntegral())

        # Components
        r_nodes = self._node_efficiency(nodes)
        r_pace = self._pace_term(nodes, t)
        gap_impr = _safe_tanh((self.prev_gap - gap) / max(abs(self.prev_gap), 1e-9)) if self.prev_gap < float('inf') else 0.0
        pdi_impr = _safe_tanh((self.prev_pdi - pdi) / self.baseline_pdi) if self.prev_pdi > 0 else 0.0

        self.prev_nodes = nodes
        self.prev_gap = gap
        self.prev_pdi = pdi

        if not done:
            reward = (self.w_nodes * r_nodes + self.w_pace * r_pace +
                      self.w_gap * gap_impr + self.w_pdi * pdi_impr)
            reward = float(np.clip(reward, -1.0, 1.0))
            if self.logger:
                self.logger.info(f"H2 step: nodes={nodes}, r_nodes={r_nodes:.3f}, r_pace={r_pace:.3f}, gap_impr={gap_impr:.3f}, pdi_impr={pdi_impr:.3f}, R={reward:.3f}")
            return reward

        # Terminal bonus: blend speedup with final quality
        status = model.getStatus()
        speedup = _ratio(self.B, max(nodes, 1), cap=4.0)
        gap_gain = _safe_tanh((self.baseline_gap - gap))
        pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)

        if status == "optimal":
            bonus = 1.0 + 2.5 * speedup
        elif status in ("infeasible", "unbounded"):
            bonus = 0.7 + 2.0 * speedup
        else:  # timelimit or other
            bonus = 0.4 * speedup + 0.4 * gap_gain + 0.2 * pdi_gain
        if self.logger:
            self.logger.info(f"H2 terminal: status={status}, speedup={speedup:.3f}, gap_gain={gap_gain:.3f}, pdi_gain={pdi_gain:.3f}, bonus={bonus:.3f}")
        return float(bonus)


class RewardH3:
    """Adaptive difficulty-aware reward.

    Uses a smooth mapping from baseline difficulty (log baseline nodes) to
    weights over components (nodes efficiency, progress, gap, PDI, time pace).
    Encourages anytime improvement on hard instances while still rewarding
    beating baseline on easy ones.
    """

    def __init__(self, logger=None, scale=2.0):
        self.logger = logger
        self.scale = scale
        self.reset(1.0, 0.0, 1.0, "timelimit", 3600.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=3600.0, logger=None):
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        self.logB = math.log1p(self.B)
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)
        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 3600.0), 1.0)
        self.logger = logger or self.logger
        self.prev_nodes = 0
        self.prev_gap = float('inf')
        self.prev_pdi = 0.0
        self.prev_pb = None
        self.prev_db = None

        # Difficulty weight schedule via sigmoid over log baseline nodes
        # easy -> emphasize speedup; hard -> emphasize gap/PDI progress
        # map logB in [log(1), log(1e6)] roughly to [0,1]
        d = min(max((self.logB - math.log1p(1.0)) / (math.log1p(1e6) - math.log1p(1.0)), 0.0), 1.0)
        self.w_nodes = 0.55 * (1 - d) + 0.25 * d    # 0.55 -> 0.25
        self.w_gap = 0.10 * (1 - d) + 0.30 * d      # 0.10 -> 0.30
        self.w_pdi = 0.05 * (1 - d) + 0.20 * d      # 0.05 -> 0.20
        self.w_progress = 0.15 * (1 - d) + 0.15 * d # 0.15 -> 0.15
        self.w_pace = 0.15 * (1 - d) + 0.10 * d     # 0.15 -> 0.10
        s = self.w_nodes + self.w_gap + self.w_pdi + self.w_progress + self.w_pace
        self.w_nodes /= s; self.w_gap /= s; self.w_pdi /= s; self.w_progress /= s; self.w_pace /= s

    def _node_eff(self, nodes):
        eff = 1.0 - (math.log1p(nodes) / self.logB)
        return _safe_tanh(eff, s=self.scale)

    def _pace(self, nodes, t):
        # piecewise expected curve: early aggressive, then conservative
        # exponent varies with difficulty; harder -> smaller exponent
        exponent = 0.9 - 0.5 * min(max(self.logB / math.log1p(1e6), 0.0), 1.0)
        expected = self.B * (t ** exponent)
        return _safe_tanh((expected - nodes) / (expected + 1.0), s=self.scale)

    def compute(self, model, done):
        nodes = int(model.getNNodes())
        tfrac = min(max(model.getSolvingTime() / self.time_limit, 0.0), 1.0)
        gap = float(model.getGap())
        if math.isinf(gap):
            gap = 1e6
        pdi = float(model.getPrimalDualIntegral())
        pb = model.getPrimalbound()
        db = model.getDualbound()

        r_nodes = self._node_eff(nodes)
        r_pace = self._pace(nodes, tfrac)
        r_progress = 0.0

        # gap improvement
        if self.prev_gap < float('inf'):
            r_progress += _safe_tanh((self.prev_gap - gap) / max(abs(self.prev_gap), 1e-9), s=self.scale)
        # bound improvements
        if self.prev_pb is not None and pb < self.prev_pb:
            r_progress += _safe_tanh((self.prev_pb - pb) / (abs(self.prev_pb) + 1e-9), s=self.scale)
        if self.prev_db is not None and db > self.prev_db:
            r_progress += _safe_tanh((db - self.prev_db) / (abs(self.prev_db) + 1e-9), s=self.scale)
        # pdi decrease
        if self.prev_pdi > 0:
            r_progress += _safe_tanh((self.prev_pdi - pdi) / self.baseline_pdi, s=self.scale)

        # update prevs
        self.prev_gap = gap
        self.prev_pdi = pdi
        self.prev_pb = pb
        self.prev_db = db

        if not done:
            reward = (self.w_nodes * r_nodes + self.w_pace * r_pace +
                      self.w_gap * ( - _safe_tanh(gap / (self.baseline_gap + 1e-9), s=0.5) ) +
                      self.w_pdi * _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi) +
                      self.w_progress * r_progress)
            reward = float(np.clip(reward, -1.0, 1.0))
            if self.logger:
                self.logger.info(f"H3 step: nodes={nodes}, r_nodes={r_nodes:.3f}, r_pace={r_pace:.3f}, r_prog={r_progress:.3f}, R={reward:.3f}")
            return reward

        # Terminal: blend speedup and quality with difficulty-aware weights
        status = model.getStatus()
        speedup = _ratio(self.B, max(nodes, 1), cap=5.0)
        gap_gain = _safe_tanh((self.baseline_gap - gap))
        pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
        if status == "optimal":
            bonus = 1.0 + 3.0 * speedup
        elif status in ("infeasible", "unbounded"):
            bonus = 0.8 + 2.0 * speedup
        elif status == "timelimit":
            bonus = 0.5 * speedup + 0.3 * gap_gain + 0.2 * pdi_gain
        else:
            bonus = 0.3 * speedup
        if self.logger:
            self.logger.info(f"H3 terminal: status={status}, speedup={speedup:.3f}, gap_gain={gap_gain:.3f}, pdi_gain={pdi_gain:.3f}, bonus={bonus:.3f}")
        return float(bonus)

