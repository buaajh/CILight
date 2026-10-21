"""Read SUMO's native vehicleTripStatistics, preserving reported denominators."""
from __future__ import annotations
import math
import xml.etree.ElementTree as ET

PROFILE = 'sumo_vehicleTripStatistics_write_unfinished_v1'


def summarize_statistics(path):
    root = ET.parse(path).getroot()
    node = root.find('vehicleTripStatistics')
    if node is None:
        raise ValueError('vehicleTripStatistics missing: enable --duration-log.statistics')
    required = ('count', 'totalTravelTime', 'waitingTime', 'timeLoss')
    values = {k: float(node.attrib[k]) for k in required}
    if not all(math.isfinite(v) and v >= 0 for v in values.values()):
        raise ValueError('Invalid vehicleTripStatistics values')
    count = values['count']
    if not count.is_integer():
        raise ValueError('Noninteger vehicle count')
    vehicles = root.find('vehicles')
    result = {
        'metric_profile': PROFILE,
        'statistics_vehicle_count': int(count),
        'average_trip_time_s': values['totalTravelTime']/count if count else None,
        'average_waiting_time_s': values['waitingTime'] if count else None,
        'average_delay_time_s': values['timeLoss'] if count else None,
        'total_travel_time_s': values['totalTravelTime'],
        'raw_vehicleTripStatistics': dict(node.attrib),
        'raw_vehicles': dict(vehicles.attrib) if vehicles is not None else {},
    }
    for source, target in [('departDelay', 'average_depart_delay_s'),
                           ('departDelayWaiting', 'average_uninserted_depart_delay_s'),
                           ('totalDepartDelay', 'total_depart_delay_s')]:
        if source in node.attrib:
            value = float(node.attrib[source])
            # SUMO can report -1 when a category contains no vehicles.
            result[target] = value if math.isfinite(value) and value >= 0 else None
    return result


def verify_statistics_against_trips(stats, tripinfo_path):
    """Native XML rounds averages to 0.01 s; cross-check rather than replace."""
    trips = ET.parse(tripinfo_path).getroot().findall('tripinfo')
    if stats['statistics_vehicle_count'] != len(trips):
        raise ValueError('Native statistics count disagrees with emitted tripinfo records')
    if not trips:
        return
    duration = sum(float(t.attrib['duration']) for t in trips)/len(trips)
    waiting = sum(float(t.attrib['waitingTime']) for t in trips)/len(trips)
    loss = sum(float(t.attrib['timeLoss']) for t in trips)/len(trips)
    for key, value in [('average_trip_time_s', duration), ('average_waiting_time_s', waiting),
                       ('average_delay_time_s', loss)]:
        if abs(stats[key]-value) > .021:
            raise ValueError(f'Native statistics differs from tripinfo: {key}')
