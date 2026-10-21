# CILight

CILight augments complementarity-implicit model predictive control (CI-MPC) with a learned terminal cost for traffic signal control. This repository contains **CILight** and its **CI-MPC** ablation, with configurations for Grid16, Ingolstadt21, Nanshan55, and New York196.

The two methods share the traffic model, phase constraints, and execution logic. CILight adds a shared local terminal critic to candidate-sequence evaluation; CI-MPC uses no learned critic. The critic is a 12–32–32–1 multilayer perceptron. The released CILight path does not require a PPO actor or a graph neural network.

## Contents

```text
configs/             Four network profiles, seed panels, input/model SHA-256
controllers/         CI prediction, phase search, coordination and execution
rl/terminal_value.py  Terminal features, critic, NumPy inference and learning
cilight/             Portable runner, metrics, validation selection and reporting
checkpoints/         Four tensor-exact, validation-selected CILight weight archives
provenance/          Original source and checkpoint identities
tests/               Simulator-free unit tests
run.py               check / evaluate / train / summarize
```

The numerical controller files retain their historical module names to preserve traceability. Only the terminal-value branch is exported from the former A/B experiment modules; its functions and classes are AST-identical to the originals. Historical launchers, monitoring scripts, runtime environments, raw run directories, and other comparison methods are excluded.

## Installation

Use Python 3.12 and [Eclipse SUMO 1.27.1](https://sumo.dlr.de/docs/Downloads.php). Set `SUMO_HOME` to the installation directory containing `bin/` and `tools/`. Matching TraCI and sumolib modules are taken from that installation.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The packaged code was checked with Python 3.12.14, NumPy 1.26.4, SciPy 1.14.1 and PyTorch 2.7.1. A CPU PyTorch installation is sufficient for the small critic. SUMO and the local MPC workers run on CPU. The original Windows execution environment used a CUDA-enabled PyTorch build; this release does not claim cross-platform bitwise equivalence.

Example environment variables:

```powershell
$env:SUMO_HOME = 'D:/software/sumo-1.27.1'
$env:CILIGHT_DATA_ROOT = 'D:/datasets/cilight/scenarios'
```

```bash
export SUMO_HOME=/opt/sumo-1.27.1
export CILIGHT_DATA_ROOT=/path/to/cilight/scenarios
```

## Road networks and data

| CLI network | Signals | Original test seeds | Selected episode | Yellow (s) | Solver workers |
|---|---:|---|---:|---:|---:|
| `grid16` | 16 | 962001–962004 | 30 | 3 | 1 |
| `ingolstadt21` | 21 | 972001–972004 | 40 | 3 | 1 |
| `nanshan55` | 55 | 982001–982004 | 90 | 3 | 1 |
| `newyork196` | 196 | 992001–992004 | 10 | 5 | 8 |

Place the scenario files under the following directory layout, or pass `--data-root` explicitly:

```text
scenarios/
  grid16/grid4x4.sumocfg
  ingolstadt21/ingolstadt21.sumocfg
  nanshan55/osm.sumocfg
  newyork196/newyork.sumocfg
```

Each directory also needs the network, route and additional files listed in `configs/<network>.json`. These manifests record the exact experimental inputs, rather than silently substituting another scenario with the same name.

**Scenario data are distributed separately from this code repository.** Grid16 and Ingolstadt21 were imported through RESCO. The New York profile is the 196-signal SUMO conversion of the `roadnet_28_7` / `anon_28_7_newyork_real_double` scenario, not a 165-signal network. See [DATASETS.md](DATASETS.md) for sources and redistribution status. The prepared local companion directory contains all four exact inputs; it is not part of the public-code archive.

## Check the installation inputs

```bash
python -B run.py check --network grid16
python -B run.py check --network ingolstadt21
python -B run.py check --network nanshan55
python -B run.py check --network newyork196
```

These checks verify source, checkpoint, scenario hashes and signal counts. They do not start SUMO or deserialize models.

## Evaluate CI-MPC and CILight

```bash
python -B run.py evaluate --network grid16 --method both --output runs/grid16
python -B run.py evaluate --network ingolstadt21 --method both --output runs/ingolstadt21
python -B run.py evaluate --network nanshan55 --method both --output runs/nanshan55
python -B run.py evaluate --network newyork196 --method both --output runs/newyork196
```

Run the commands sequentially. Each command uses the four original test seeds, 3,600 simulated seconds per seed, and the supplied validation-selected checkpoint. Use `--method ci-mpc` or `--method cilight` for one method. CI-MPC does not load a model. Shorter `--seconds` values are smoke tests, not paper results.

The shared controller profile is a 10 s decision interval, a three-step prediction horizon, a 10 s minimum green and a 60 s maximum green. CILight's evaluation terminal weight is 0.5. The preserved SUMO launcher explicitly uses `--time-to-teleport 300`, including for New York; this command-line value overrides the setting in its source configuration.

Every run writes metrics, action decisions, observed phases, native SUMO trip/statistics XML, lane queue samples and a hash-bound completion marker. A failed or interrupted run is retained without a completion marker. Existing output directories are never overwritten or automatically resumed. Create an `output/STOP` file to request shutdown at a control boundary. The serial launcher requires at least 12 GiB available physical RAM and 10 GiB free output-disk space; full training needs substantially more storage. It refuses to start alongside unrelated SUMO processes. It does not manage the machine's other applications.

The entry-point wrapper is new packaging code. Source/checkpoint identity and simulator-free tests have been checked; the packaged wrapper has **not** been validated by rerunning the full benchmark. See [VALIDATION.md](VALIDATION.md).

## Train a fresh CILight critic

```bash
python -B run.py train --network grid16 --output runs/train_grid16
```

Replace the network name for the other three profiles. Each invocation starts from a fresh initialization and uses the original disjoint training and validation seed panels. Training uses Adam (learning rate 0.0003), 25 fitting epochs per episode and a discount factor of 0.98. The terminal weight is 0 for episodes 1–2, then 0.05, 0.10 and 0.15, and 0.20 from episode 6 onward.

Validation uses four fixed seeds every 10 episodes at terminal weight 0.5. Selection requires at least a 1% mean ATT improvement and improvement on at least three seeds, with the original AWT, ADT and completion-count guards. Early stopping begins at episode 60, with patience of six validation points and a 300-episode cap. Teleports are recorded but do not enter checkpoint selection. No test outcomes are used for selection.

Training writes `selected.pt` and `selection.json` and then stops. Evaluate that checkpoint explicitly, using its recorded SHA-256:

```bash
python -B run.py evaluate --network grid16 --method both --output runs/grid16_fresh_test \
  --checkpoint runs/train_grid16/selected.pt --checkpoint-sha256 SHA256_FROM_SELECTION_JSON
```

Join this command onto one line in PowerShell. The supplied `.npz` files contain only six numeric weight/bias arrays and are loaded with `allow_pickle=False`. Their tensors exactly match the original selected checkpoints; export receipts are in `provenance/weight_export.json`. Training-generated `.pt` snapshots also contain optimizer/RNG state and use pickle-based deserialization: load only trusted files. A hash verifies identity, not the safety of an unknown file.

## Metrics and summaries

`summary.csv` contains means and **sample standard deviations** (`ddof=1`) across the completed seed panel. A separate summary can be generated with:

```bash
python -B run.py summarize --input runs/nanshan55 --output runs/nanshan55_summary.csv
```

- **ATT**: native SUMO `totalTravelTime / count`.
- **AWT** and **ADT**: native `waitingTime` and `timeLoss` averages.
- Trip output includes vehicles that have departed but have not finished at the horizon. These averages do not silently become completed-trip-only metrics, and uninserted vehicles are not assigned fabricated travel times.
- **AQL**: time-averaged halting count per unique controlled incoming lane, including empty lanes.
- Decision latency is measured over whole-network controller updates. Simulation wall time is reported separately. These timings depend on hardware, concurrency and instrumentation.

Four traffic seeds quantify traffic-seed variability for one trained model; they are not four independent training repetitions. The main-table checkpoint identities are retained. Previously observed Nanshan baseline discrepancies remain unresolved; this packaging work neither investigates them nor certifies exact reproduction of published aggregates. The RQ3 rerun baselines are not substituted for the original main-table baseline.

## Tests

```bash
python -B -m unittest discover -s tests -v
```

Tests exercise the terminal critic, metric accounting, source identities, seed separation, output isolation and aggregation without running SUMO or loading the four pretrained checkpoints.

## License and attribution

Original CILight code is released under the [MIT License](LICENSE). Third-party software and datasets retain their own terms; they are not relicensed by this repository. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Repository: [buaajh/CILight](https://github.com/buaajh/CILight). Paper citation metadata can be added when the final title and publication details are available.
