"""Configuration for CI-MPC, distributed MPC, and MP baselines."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUMOCFG = None
NET_FILE = None
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

SUMO_BINARY = "sumo"
STEP_LENGTH = 1.0
SIM_SECONDS = 3600
RANDOM_SEED = 42
COLLECT_EMISSIONS = True
REPORT_ONLY_METRICS = False

DECISION_INTERVAL = 10
MIN_GREEN = 10
MAX_GREEN = 60

KF_PROCESS_VAR = 1.0
KF_MEASUREMENT_VAR = 4.0

CONTROLLERS = ["CI_MPC"]
APPROACHES = ["N", "S", "E", "W"]

# Distributed stochastic MPC (Pham & Ahn, IEEE T-ITS 2023)
DSMPC_CYCLE = 60.0
DSMPC_HORIZON = 4
DSMPC_EPSILON = 0.2
DSMPC_RHO = 1.0
# The paper reports 1e-6 and roughly 2100 iterations.  The default below is
# the explicitly documented grid4x4 test profile; use 1e-6 with a much larger
# iteration budget for an offline paper-accuracy run.
DSMPC_ADMM_TOL = 2e-2
DSMPC_ADMM_MAX_ITER = 100
DSMPC_PARALLEL_WORKERS = 4
DSMPC_SOLVER = "CLARABEL"
DSMPC_SOLVER_MAX_ITER = 100

# grid4x4 has eight green stages and an actual three-second yellow after
# each stage.  Hence 60 - 8*3 = 36 seconds of effective green per cycle.
DSMPC_YELLOW = 3.0
DSMPC_AVAILABLE_GREEN = 36.0
DSMPC_MIN_GREEN = 1.0
DSMPC_MAX_GREEN = 29.0

DSMPC_SATURATION_PER_LANE = 0.55  # veh/s/lane (paper: 0.55--0.68)
DSMPC_EFFECTIVE_VEHICLE_LENGTH = 7.5  # m/vehicle, for link storage
DSMPC_TURN_STD = 0.2 / (12.0 ** 0.5)  # U[-0.1, 0.1]
DSMPC_TURN_PRIOR_STRENGTH = 3.0  # one equal pseudo-observation per movement
DSMPC_SOURCE_UNCERTAINTY_RANGE_VPH = 400.0  # U[-50, 350] veh/h
DSMPC_LINEAR_STATE_WEIGHT = 0.3
DSMPC_FLOW_REWARD = 0.3
DSMPC_SLACK_PENALTY = 1_000.0

# Independent deterministic lane-level DMPC (not derived from the paper model)
DMPC_INTERVAL = 10.0
DMPC_HORIZON = 3
DMPC_YELLOW = 3.0
DMPC_MIN_GREEN = 10.0
DMPC_SATURATION_PER_LANE = 0.55
DMPC_EFFECTIVE_VEHICLE_LENGTH = 7.5
DMPC_INITIAL_BOUNDARY_RATE = 0.02
DMPC_DEMAND_EWMA_ALPHA = 0.35
DMPC_COORDINATION_ITERATIONS = 3
DMPC_COORDINATION_DAMPING = 0.7
DMPC_COORDINATION_TOL = 0.25
DMPC_PARALLEL_WORKERS = 4
DMPC_LOCAL_TIME_LIMIT = 1.0
DMPC_MIP_GAP = 0.02
DMPC_MAX_QUEUE_WEIGHT = 0.25
DMPC_TERMINAL_QUEUE_WEIGHT = 2.0
DMPC_FLOW_REWARD = 0.05
DMPC_SWITCH_PENALTY = 0.6
DMPC_SPILL_PENALTY = 100.0

# Complementarity-Implicit MPC (independent implementation)
CIMPC_INTERVAL = 10.0
CIMPC_HORIZON = 3
CIMPC_YELLOW = 3.0
CIMPC_MIN_GREEN = 10.0
CIMPC_SATURATION_PER_LANE = 0.55
CIMPC_EFFECTIVE_VEHICLE_LENGTH = 7.5
CIMPC_INITIAL_BOUNDARY_RATE = 0.02
CIMPC_DEMAND_EWMA_ALPHA = 0.35
CIMPC_COORDINATION_ITERATIONS = 2
CIMPC_COORDINATION_DAMPING = 0.7
CIMPC_COORDINATION_TOL = 0.25
CIMPC_PARALLEL_WORKERS = 4

# Reproducible distributed implementation stages:
# 0 = legacy two-round synchronous threaded Jacobi
# 1 = one-round cross-cycle asynchronous messages
# 2 = stage 1 with eight persistent fixed-shard processes
# 3 = stage 2 plus event-triggered one-hop correction
# 4 = stage 3 plus strict fixed-phase specialised complementarity flow solves
# 5 = stage 4 plus exact enumeration of complete horizon phase sequences
CIMPC_IMPLEMENTATION_STAGE = 4
CIMPC_SEQUENCE_ENUMERATION_LIMIT = 100_000
CIMPC_ASYNC_STORAGE_MARGIN = 1.0
CIMPC_PROCESS_WORKERS = 8
CIMPC_PROCESS_TIMEOUT = 20.0
CIMPC_EVENT_RESIDUAL_THRESHOLD = 1.0
CIMPC_EVENT_RECEIVING_THRESHOLD = 5.0
CIMPC_EVENT_OCCUPANCY_THRESHOLD = 0.80
CIMPC_EVENT_MAX_SEEDS = 2
CIMPC_EVENT_MAX_AGENTS = 8
CIMPC_PROX_FLOW_WEIGHT = 0.05
CIMPC_PROX_PHASE_WEIGHT = 0.10

# The inner allocation maximizes w(q)^T f - eps/2 ||f||^2.  Its KKT system
# is embedded in the outer signal MPC using a Scholtes product relaxation.
CIMPC_INNER_REGULARIZATION = 0.03
CIMPC_INNER_PRESSURE_WEIGHT = 1.5
CIMPC_SCHOLTES_RHO = 0.02
CIMPC_DUAL_UPPER_BOUND = 1_000.0
CIMPC_WARM_PHASE_MEMORY = 0.0

CIMPC_QUEUE_WEIGHT = 4.0
CIMPC_BALANCE_WEIGHT = 0.15
CIMPC_TERMINAL_QUEUE_WEIGHT = 8.0
CIMPC_FLOW_REWARD = 0.08
CIMPC_SWITCH_PENALTY = 0.2
CIMPC_ONE_HOT_PENALTY = 1.0
CIMPC_SPILL_PENALTY = 500.0
CIMPC_PROJECTION_LINEAR_QUEUE_WEIGHT = 0.6
CIMPC_PROJECTION_SWITCH_PENALTY = 0.5
CIMPC_RELAXED_PHASE_PROJECTION_WEIGHT = 0.05

CIMPC_LOCAL_TIME_LIMIT = 1.5
CIMPC_IPOPT_MAX_ITER = 180
CIMPC_IPOPT_TOL = 1e-5
CIMPC_IPOPT_ACCEPTABLE_TOL = 5e-4
CIMPC_ACCEPTED_PRIMAL_VIOLATION = 5e-3
CIMPC_ACCEPTED_STATIONARITY_RESIDUAL = 5e-3

# RL-parameterized CI-MPC. The real-time stage-5 solver has exact stagewise
# complementarity, so the deployed policy adapts only parameters that remain
# active in that formulation. Zero normalized action reproduces the nominal
# stage-5 controller exactly.
RLCIMPC_MODEL_PATH = None
RLCIMPC_DETERMINISTIC = True
RLCIMPC_PARAMETER_NAMES = (
    "alpha_q",
    "alpha_f",
    "alpha_spill",
    "alpha_delta_phase",
    "lambda_safe",
)
RLCIMPC_PARAMETER_LOWER = (1.0, 0.01, 0.0, 0.10, 0.0)
RLCIMPC_PARAMETER_NOMINAL = (
    CIMPC_QUEUE_WEIGHT,
    CIMPC_FLOW_REWARD,
    0.0,
    CIMPC_PROJECTION_SWITCH_PENALTY,
    0.0,
)
RLCIMPC_PARAMETER_UPPER = (12.0, 0.30, 2.0, 2.0, 50.0)
RLCIMPC_SAFE_OCCUPANCY = 0.80

# Reward coefficients used by the Gymnasium training environment. Terms are
# normalized in rl/train_rl_cimpc.py before these weights are applied.
RLCIMPC_REWARD_QUEUE = 1.0
RLCIMPC_REWARD_DELAY = 0.25
RLCIMPC_REWARD_FLOW = 0.50
RLCIMPC_REWARD_SPILL = 2.0
RLCIMPC_REWARD_SWITCH = 0.05
RLCIMPC_REWARD_VIOLATION = 10.0

# External-cooperative residual RL + CI-MPC.  Stage-5 first solves the nominal
# MPC and exposes the best horizon objective conditional on every possible
# first phase.  The deployed shield uses a network-level Lyapunov budget: RL
# may rerank phases only while preserving a configurable fraction of the
# nominal CI-MPC one-step decrease.  The legacy cost-budget constants remain
# available for old checkpoints/tests, but are no longer used by the external
# cooperative controllers.
EXTERNAL_CIMPC_MODEL_PATH = None
EXTERNAL_CIMPC_DETERMINISTIC = True
PPO_PROPOSAL_MODEL_PATH = None
PPO_PROPOSAL_GATE_MODEL_PATH = None
PPO_PROPOSAL_GATE_DETERMINISTIC = True
EXTERNAL_CIMPC_MAX_GREEN = 60.0
EXTERNAL_CIMPC_RELATIVE_COST_BUDGET = 0.05
EXTERNAL_CIMPC_ABSOLUTE_COST_BUDGET = 1.0
# The published/migrated main result uses the original 5% MPC-cost budget.
# Lyapunov projection remains available as an explicit CLI ablation.
EXTERNAL_CIMPC_BUDGET_MODE = "cost"
EXTERNAL_CIMPC_LYAPUNOV_DESCENT_RETENTION = 0.50
EXTERNAL_CIMPC_LYAPUNOV_SLACK = 1e-6
EXTERNAL_CIMPC_LYAPUNOV_OVERFLOW_WEIGHT = 4.0
EXTERNAL_CIMPC_RESIDUAL_GAIN = 1.0
EXTERNAL_CIMPC_OVERFLOW_TOLERANCE = 1e-6
EXTERNAL_CIMPC_RECEIVING_TOLERANCE = 1e-6
EXTERNAL_CIMPC_REWARD_CORRECTION = 0.01

# L-CIMPC-V2 always uses the network-level Lyapunov certificate.  Its robust
# reserve is an EWMA of positive one-step prediction error: if the measured
# network Lyapunov value exceeds the previous applied prediction, the next
# policy budget is tightened while the nominal CI-MPC action remains feasible.
L_CIMPC_V2_MODEL_PATH = None
L_CIMPC_V2_ROBUST_ERROR_ALPHA = 0.20
L_CIMPC_V2_ROBUST_RESERVE_GAIN = 1.0

# Independent PPO direct-signal baseline. Each junction owns an independent
# actor-critic and sees only local/downstream/neighbour summaries.
IPPO_MODEL_PATH = None
IPPO_HIDDEN_SIZE = 128
IPPO_GAMMA = 0.99
IPPO_GAE_LAMBDA = 0.95
IPPO_CLIP_RANGE = 0.20
IPPO_VALUE_WEIGHT = 0.50
IPPO_ENTROPY_WEIGHT = 0.01
IPPO_LEARNING_RATE = 3e-4
IPPO_UPDATE_EPOCHS = 10
IPPO_MINIBATCH_SIZE = 60
IPPO_REWARD_QUEUE = 1.0
IPPO_REWARD_WAIT = 0.2
IPPO_REWARD_DOWNSTREAM = 0.5
IPPO_REWARD_DISCHARGE = 0.5
IPPO_REWARD_SWITCH = 0.05
IPPO_REWARD_GLOBAL_QUEUE = 0.1

# SUMO-RL 1.4.5 shared-policy PPO baseline. Environment and optimiser values
# follow the repository's experiments/sb3_grid4x4.py example. The upstream
# defaults define the MDP timing and diff-waiting-time reward.
SUMORL_PPO_MODEL_PATH = None
SUMORL_PPO_DETERMINISTIC = True
SUMORL_DELTA_TIME = 5
SUMORL_YELLOW_TIME = 2
SUMORL_MIN_GREEN = 5
SUMORL_MAX_GREEN = 50
SUMORL_ENFORCE_MAX_GREEN = False
SUMORL_REWARD_FN = "diff-waiting-time"
SUMORL_PPO_TOTAL_TIMESTEPS = 50_000
SUMORL_PPO_GAMMA = 0.95
SUMORL_PPO_N_STEPS = 256
SUMORL_PPO_ENTROPY_COEF = 0.0905168
SUMORL_PPO_LEARNING_RATE = 0.00062211
SUMORL_PPO_VALUE_COEF = 0.042202
SUMORL_PPO_MAX_GRAD_NORM = 0.9
SUMORL_PPO_GAE_LAMBDA = 0.99
SUMORL_PPO_EPOCHS = 5
SUMORL_PPO_CLIP_RANGE = 0.3
SUMORL_PPO_BATCH_SIZE = 256


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def configure_scenario(sumocfg: str, output_dir: str | None = None):
    """Select a SUMO config and resolve its network file."""
    global SUMOCFG, NET_FILE, RESULTS_DIR, FIGURES_DIR

    SUMOCFG = os.path.abspath(sumocfg)
    if not os.path.isfile(SUMOCFG):
        raise FileNotFoundError(f"SUMO config not found: {SUMOCFG}")

    root = ET.parse(SUMOCFG).getroot()
    net_value = None
    for elem in root.iter():
        if _local_name(elem.tag) == "net-file":
            net_value = elem.get("value")
            break
    if not net_value:
        raise ValueError(f"No <net-file> entry found in {SUMOCFG}")

    NET_FILE = os.path.abspath(os.path.join(os.path.dirname(SUMOCFG), net_value))
    if not os.path.isfile(NET_FILE):
        raise FileNotFoundError(f"Network referenced by config was not found: {NET_FILE}")

    if output_dir is None:
        scenario_name = f"{os.path.basename(os.path.dirname(SUMOCFG))}_{os.path.splitext(os.path.basename(SUMOCFG))[0]}"
        output_dir = os.path.join(PROJECT_ROOT, "results", scenario_name)
    RESULTS_DIR = os.path.abspath(output_dir)
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(FIGURES_DIR, exist_ok=True)
