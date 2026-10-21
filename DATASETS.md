# Scenario provenance and data layout

The public code archive contains four profiles and exact input hashes, but no third-party scenario payloads. The local companion `local_assets/scenarios/` contains only the inputs referenced by the four selected SUMO configurations, with existing notices preserved. Detector-output XML, historical evaluation outputs and runtime binaries are excluded.

| Profile | Recorded source | Redistribution status in this release |
|---|---|---|
| Grid16 | RESCO `resco_envs/grid4x4` | Separate data; RESCO repository declares GPL-3.0 |
| Ingolstadt21 | RESCO `resco_envs/ingolstadt21` | Separate data; original local `LICENSE` preserved (GPL-3.0) |
| Nanshan55 | Imported `nanshan/osm.sumocfg` scenario | Separate data; confirm dataset provenance and redistribution permission before publication |
| New York196 | Converted `roadnet_28_7.json` and `anon_28_7_newyork_real_double.json` | Separate data; confirm the precise upstream data terms before publication |

Sources:

- [RESCO](https://github.com/Pi-Star-Lab/RESCO), [RESCO license](https://github.com/Pi-Star-Lab/RESCO/blob/main/LICENSE).
- [LibSignal](https://github.com/DaRL-LibSignal/LibSignal) is the recorded source family for the New York scenario. Do not assume that a project's software terms automatically resolve every dataset's terms.
- Where road data originate from OpenStreetMap, preserve the relevant attribution and data-license obligations: [OpenStreetMap copyright and license](https://www.openstreetmap.org/copyright).

For a local setup, point `CILIGHT_DATA_ROOT` or `--data-root` at the companion `scenarios/` directory. The code verifies the exact SHA-256 digests. Downloading an upstream scenario with a matching name is not enough: detector additions, signal plans, conversion and demand files must also match the manifests.

The supplied New York SUMO conversion has 196 controlled intersections, 70 virtual intersections, 854 directed roads and 11,044 vehicles. The conversion report records 14 invalid input flow records omitted from 11,058 source records. Eight source green phases per signal are selectable, with separate 5 s yellow transitions. These are scenario characteristics, not new conversions performed for this release.

The MIT license applies to original code, not to these external data. Publish an approved scenario archive separately and add its verified download link here when its redistribution terms have been confirmed.
