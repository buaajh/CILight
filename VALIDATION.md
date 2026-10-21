# Release validation

## Evaluation-only release

- The four final-weight archives and all scenario inputs are unchanged.
- Core CI-MPC source files, terminal features and terminal-value inference retain their reference definitions.
- The runner supports input checks, fixed-weight evaluation and result summaries. Training, model selection and training-only trajectory outputs are excluded.
- The terminal-value controller retains its control computation; only training-sample collection has been removed.

## Checks

Simulator-free tests cover file identities, numeric weight loading, fixed test seeds, terminal-value inference, metric accounting, output isolation, sample-standard-deviation aggregation and a mocked TraCI run with native cleanup.

`run.py check` verifies the source, final-weight and scenario hashes and the controlled-signal counts for all four networks. It does not run a simulation.

These checks validate the release structure and evaluation components, not full benchmark equivalence. The packaged runner has not been compared against complete reference action traces. Reported runtimes depend on the execution environment and instrumentation. The previously observed Nanshan baseline discrepancy remains unresolved; no further diagnosis was performed.
