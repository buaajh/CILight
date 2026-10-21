# Scenario provenance and data layout

The repository includes all four scenarios under `scenarios/`, using the exact inputs referenced by the selected SUMO configurations. The data were added at the repository owner's request on 2026-10-21. Existing notices and input bytes are preserved. Detector-output XML, historical evaluation outputs and runtime binaries are excluded. The separately retained `CILight_public_code.zip` remains a code-and-weights-only archive; clone the repository or download its source ZIP to obtain the scenarios.

| Profile | Recorded source | Redistribution status in this release |
|---|---|---|
| Grid16 | RESCO `resco_envs/grid4x4` | Included in `scenarios/grid16/`; RESCO GPL-3.0 notice included |
| Ingolstadt21 | RESCO `resco_envs/ingolstadt21` | Included in `scenarios/ingolstadt21/`; original local GPL-3.0 `LICENSE` preserved |
| Nanshan55 | Imported `nanshan/osm.sumocfg`; network header records conversion from OpenStreetMap | Included in `scenarios/nanshan55/`; OpenStreetMap attribution below; a separate license for the complete scenario was not recorded in the local import |
| New York196 | SUMO conversion of `roadnet_28_7.json` and `anon_28_7_newyork_real_double.json` | Included in `scenarios/newyork196/`; source family recorded below; dataset-specific terms were not recorded in the local conversion |

Sources:

- [RESCO](https://github.com/Pi-Star-Lab/RESCO), [RESCO license](https://github.com/Pi-Star-Lab/RESCO/blob/main/LICENSE).
- [LibSignal](https://github.com/DaRL-LibSignal/LibSignal) is the recorded source family for the New York scenario. Do not assume that a project's software terms automatically resolve every dataset's terms.
- OpenStreetMap-derived map data: **© OpenStreetMap contributors**, available under the [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/). See [OpenStreetMap copyright and license](https://www.openstreetmap.org/copyright). This attribution applies to the underlying OSM map data, not automatically to every demand or simulation file.

For a local setup, point `CILIGHT_DATA_ROOT` or `--data-root` at this repository's `scenarios/` directory. For example, from the repository root: `python -B run.py check --network grid16 --data-root ./scenarios`. The code verifies exact SHA-256 digests and controlled-signal counts. The four configurations bind the required network, detector, signal, conversion and demand files.

The supplied New York SUMO conversion has 196 controlled intersections, 70 virtual intersections, 854 directed roads and 11,044 vehicles. The conversion report records 14 invalid input flow records omitted from 11,058 source records. Eight source green phases per signal are selectable, with separate 5 s yellow transitions. These are scenario characteristics, not new conversions performed for this release.

The MIT license applies to original code, not to these external data. Publication does not establish missing dataset-specific license terms; the provenance gaps above remain recorded. Preserve upstream attribution and applicable terms when redistributing the scenarios.
