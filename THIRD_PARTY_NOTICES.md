# Third-party notices

This repository does not redistribute SUMO, Python, PyTorch, NumPy, SciPy or psutil. Install those dependencies separately under their respective licenses. TraCI and sumolib are imported from the selected SUMO installation.

- [Eclipse SUMO](https://sumo.dlr.de/docs/) supplies the traffic simulation and native statistics. The locally checked SUMO 1.27.1 binary reports `EPL-2.0 OR GPL-2.0-or-later`.
- [PyTorch](https://pytorch.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [psutil](https://github.com/giampaolo/psutil) supply the numerical/runtime dependencies.
- All four scenarios are included under `scenarios/`; their sources, attribution and recorded license status are in [DATASETS.md](DATASETS.md). Grid16 and Ingolstadt21 retain RESCO GPL-3.0 notices. OpenStreetMap-derived map data are credited to OpenStreetMap contributors under the ODbL. The code's MIT license does not relicense these datasets.

Some preserved internal modules include optional legacy branches that mention CasADi or Stable-Baselines3. The published Stage-5 CI-MPC / terminal-critic path does not enable those branches. They are not dependencies of the documented commands.

Original source/module names are retained in `provenance/sources.json`. No third-party license notice has been removed from the copied files. If additional upstream authorship applies to a source file, preserve its notice before public release.
