"""Aggregate complete, hash-verified rollouts; ddof=1, never population SD."""
import csv
from pathlib import Path
import statistics
from cilight.io import verify_result

FIELDS = ('average_trip_time_s', 'average_waiting_time_s', 'average_delay_time_s',
          'aql_vehicles_per_lane', 'decision_mean_ms', 'wall_time_s')


def summarize(directory, output):
    groups = {}
    for candidate in Path(directory).glob('*/seed_*'):
        if candidate.is_dir() and not (candidate/'complete.json').is_file():
            raise ValueError('Incomplete seed directory: ' + str(candidate))
    for marker in sorted(Path(directory).glob('*/seed_*/complete.json')):
        row = verify_result(marker.parent)
        key = row['network'], row['method'], row['seconds']
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ValueError('No complete evaluation rollouts found')
    rows = []
    panels = {}
    for (network, method, seconds), values in sorted(groups.items()):
        seeds = [r['seed'] for r in values]
        if len(set(seeds)) != len(seeds):
            raise ValueError('Duplicate seed')
        panel = panels.setdefault((network, seconds), set(seeds))
        if panel != set(seeds):
            raise ValueError('Methods must share the same seed panel')
        out = dict(network=network, method=method, seconds=seconds, n=len(values), seeds=';'.join(map(str, sorted(seeds))))
        for field in FIELDS:
            numbers = [r[field] for r in values]
            if any(x is None for x in numbers):
                raise ValueError('Metric missing: ' + field)
            out[field+'_mean'] = statistics.mean(numbers)
            out[field+'_sample_std'] = statistics.stdev(numbers) if len(numbers) > 1 else ''
        rows.append(out)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(str(output))
    return rows
