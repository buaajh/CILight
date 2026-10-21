# CILight

Evaluation code for **CILight** and **CI-MPC** on four road networks. CILight augments complementarity-implicit model predictive control with a learned terminal cost; CI-MPC uses the same controller without that cost.

The repository includes the scenario inputs and **final weights** needed to run both methods.

## Installation

Use Python 3.12 and [Eclipse SUMO 1.27.1](https://sumo.dlr.de/docs/Downloads.php).

```bash
git clone https://github.com/buaajh/CILight.git
cd CILight
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set `SUMO_HOME` to your SUMO installation, which contains `bin/` and `tools/`:

```powershell
$env:SUMO_HOME = 'D:/software/sumo-1.27.1'  # Windows: adjust this path
```

```bash
export SUMO_HOME=/opt/sumo-1.27.1  # Linux/macOS: adjust this path
```

CPU execution is supported. Run evaluations serially with at least 12 GiB available RAM and 10 GiB free output-disk space.

## Networks and final weights

| Network argument | Signals | Test seeds |
|---|---:|---|
| `grid16` | 16 | 962001–962004 |
| `ingolstadt21` | 21 | 972001–972004 |
| `nanshan55` | 55 | 982001–982004 |
| `newyork196` | 196 | 992001–992004 |

- `scenarios/`: SUMO network, demand and signal files.
- `final_weights/`: one numeric `.npz` weight file per network, loaded automatically for CILight.
- `configs/`: evaluation settings, test seeds and file hashes.
- `controllers/` and `rl/`: traffic control and terminal-value inference.

Clone the repository or use **Code → Download ZIP** for the complete package. `CILight_public_code.zip` contains code and final weights only. Dataset sources and terms are listed in [DATASETS.md](DATASETS.md).

## Run evaluation

From the repository root, check the inputs and then evaluate both methods:

```bash
python -B run.py check --network grid16
python -B run.py evaluate --network grid16 --method both --output runs/grid16
python -B run.py evaluate --network ingolstadt21 --method both --output runs/ingolstadt21
python -B run.py evaluate --network nanshan55 --method both --output runs/nanshan55
python -B run.py evaluate --network newyork196 --method both --output runs/newyork196
```

Each evaluation uses the four listed seeds and 3,600 simulation seconds per method and seed. Use `--method cilight` or `--method ci-mpc` to run one method. Inputs default to `scenarios/`; `--data-root` can specify another location. Use a fresh output directory for each run.

## Results

Each output directory contains per-seed metrics and traces, plus `summary.csv` with means and **sample standard deviations** across the four traffic seeds using the fixed final weights.

ATT is SUMO's `totalTravelTime / count`; AWT and ADT are its average `waitingTime` and `timeLoss`. Trip output includes unfinished trips at the simulation horizon. AQL is the time-averaged halting count per controlled incoming lane. Decision latency and simulation wall time are reported separately and depend on the hardware.

To regenerate a summary:

```bash
python -B run.py summarize --input runs/nanshan55 --output runs/nanshan55_summary.csv
```

## Validation and license

Run the simulator-free checks with `python -B -m unittest discover -s tests -v`. See [VALIDATION.md](VALIDATION.md) for verification scope.

Original code is released under the [MIT License](LICENSE). Third-party software and datasets retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
