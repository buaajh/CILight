"""Robust Lyapunov-budget Stage-5 CI-MPC with an optional PPO residual policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from controllers.external_cooperative_ci_mpc import (
    ExternalCooperativeCIMPCController,
    lyapunov_budget_limit,
)


@dataclass(frozen=True)
class RobustLyapunovBudget:
    """Auditable components of the L-CIMPC-V2 one-step budget."""

    limit: float
    nominal_limit: float
    prediction_error: float
    prediction_error_ewma: float
    robust_reserve: float


def robust_lyapunov_budget_limit(
    current_value,
    nominal_next_value,
    previous_applied_prediction=np.nan,
    previous_error_ewma=0.0,
    error_alpha=0.20,
    reserve_gain=1.0,
    descent_retention=0.50,
    slack=1e-6,
):
    """Tighten the nominal descent budget using causal positive model error.

    The reserve never makes the physically executable nominal CI-MPC action
    inadmissible: the tightened limit is lower-bounded by ``nominal_next_value``.
    Only positive under-prediction is accumulated because an over-prediction of
    congestion is already conservative for this certificate.
    """

    alpha = float(error_alpha)
    gain = float(reserve_gain)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("error_alpha must be in [0, 1]")
    if gain < 0.0:
        raise ValueError("reserve_gain must be non-negative")
    current = float(current_value)
    nominal = float(nominal_next_value)
    prior_ewma = max(float(previous_error_ewma), 0.0)
    if not np.isfinite(current) or not np.isfinite(nominal):
        raise ValueError("Lyapunov values must be finite")

    if np.isfinite(float(previous_applied_prediction)):
        prediction_error = max(
            current - float(previous_applied_prediction), 0.0
        )
        error_ewma = (1.0 - alpha) * prior_ewma + alpha * prediction_error
    else:
        prediction_error = 0.0
        error_ewma = prior_ewma

    nominal_limit = lyapunov_budget_limit(
        current,
        nominal,
        descent_retention=descent_retention,
        slack=slack,
    )
    reserve = gain * error_ewma
    limit = max(nominal, nominal_limit - reserve)
    return RobustLyapunovBudget(
        limit=float(limit),
        nominal_limit=float(nominal_limit),
        prediction_error=float(prediction_error),
        prediction_error_ewma=float(error_ewma),
        robust_reserve=float(nominal_limit - limit),
    )


class RobustLyapunovBudgetMixin:
    """Share the V2 robust budget with residual and proposal PPO front ends."""

    def _compute_lyapunov_limit(self, current_value, nominal_next_value):
        budget = robust_lyapunov_budget_limit(
            current_value=current_value,
            nominal_next_value=nominal_next_value,
            previous_applied_prediction=self.previous_lyapunov_prediction,
            previous_error_ewma=self.last_lyapunov_prediction_error_ewma,
            error_alpha=config.L_CIMPC_V2_ROBUST_ERROR_ALPHA,
            reserve_gain=config.L_CIMPC_V2_ROBUST_RESERVE_GAIN,
            descent_retention=config.EXTERNAL_CIMPC_LYAPUNOV_DESCENT_RETENTION,
            slack=config.EXTERNAL_CIMPC_LYAPUNOV_SLACK,
        )
        self.last_lyapunov_nominal_limit = budget.nominal_limit
        self.last_lyapunov_robust_reserve = budget.robust_reserve
        self.last_lyapunov_prediction_error = budget.prediction_error
        self.last_lyapunov_prediction_error_ewma = (
            budget.prediction_error_ewma
        )
        return budget.limit


class LCIMPCV2Controller(
    RobustLyapunovBudgetMixin,
    ExternalCooperativeCIMPCController,
):
    """L-CIMPC-V2: robust Lyapunov projection around nominal Stage-5 CI-MPC.

    A PPO policy may score all first-phase candidates, but the applied action
    must pass signal timing, overflow, receiving-space, and robust network
    Lyapunov checks.  With no model, the zero residual retains nominal CI-MPC.
    """

    name = "L_CIMPC_V2"
    diagnostic_prefix = "l_cimpc_v2_solver"

    def __init__(self, conn, model_path=None, external_actions=False):
        super().__init__(
            conn,
            model_path=model_path or config.L_CIMPC_V2_MODEL_PATH,
            external_actions=external_actions,
            budget_mode="lyapunov",
        )
