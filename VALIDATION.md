# Release validation

Packaging date: 2026-10-21.

## Checked

- Eleven controller/configuration/metric helper files copied byte-for-byte from the mainline frozen source manifest.
- `TerminalValueCIMPC` and six terminal-critic definitions extracted without changing their ASTs; unused A/B-policy imports excluded.
- AQL and validation-selection policy copied from the mainline sources.
- Four selected original checkpoint hashes match the original selection receipts (episodes 30, 40, 90 and 10).
- All six tensors in each exported `.npz` are exactly equal to the corresponding original checkpoint tensors. No controller/model was instantiated during this conversion, and no inference was run. Original `.pt` files remain in the local companion directory.
- All four scenario manifests match the exact local input files; controlled signal counts are 16, 21, 55 and 196.
- Core controller import and SUMO 1.27.1 binary version check succeed.
- All 15 simulator-free tests pass. They cover source identities, syntax, seed panels, tensor archives, terminal-zero behavior, NumPy/PyTorch critic consistency, synthetic critic fitting, the beta schedule, AQL, native metric cross-checks, output isolation, demand counting, sample-SD aggregation and a mocked full episode/close lifecycle. The lifecycle test uses a TraCI substitute, not a real simulation.

## Scope of these checks

No SUMO simulation, full training run or formal baseline rerun was launched for packaging. The portable orchestration/IO wrapper is newly written and has not been certified against the historical full-seed action traces. Exact core-source or weight identity is not a claim of full benchmark equivalence. Timing instrumentation is also not identical to the old campaign launcher.

No Nanshan baseline-discrepancy diagnosis was performed. The existing unresolved discrepancy is retained as a reporting limitation, not relabeled as a passed reproducibility check.

The public archive intentionally excludes scenario data pending a separate data-release review. All four exact local scenarios are available in the private companion directory for local use.
