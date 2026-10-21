"""Read-only lane AQL diagnostic; not imported by the active MP runner."""
import hashlib
import json
import math
import xml.etree.ElementTree as ET


def controlled_incoming_lanes(net_path):
    root=ET.parse(net_path).getroot()
    edges={e.get('id'):e for e in root.findall('edge')}
    result=set()
    for c in root.findall('connection'):
        if not c.get('tl'):continue
        edge=edges[c.get('from')]
        if edge.get('function')=='internal':continue
        candidates=[l.get('id') for l in edge.findall('lane') if l.get('index')==c.get('fromLane')]
        if len(candidates)!=1:raise ValueError('Ambiguous controlled source lane')
        result.add(candidates[0])
    if not result:raise ValueError('No controlled incoming lanes')
    return tuple(sorted(result))


class AQL:
    """Time-weighted mean halting count per unique controlled incoming lane.

    SUMO getLastStepHaltingNumber uses speed < 0.1 m/s. Full physical lane,
    not a 200 m observation window; no weighting by vehicles or intersections.
    Missing samples are rejected, never filled with zeros.
    """
    def __init__(self,lanes,begin=0.,step=1.):
        if not lanes or len(lanes)!=len(set(lanes)):raise ValueError('Empty or duplicate lanes')
        if not math.isfinite(begin) or not math.isfinite(step) or step<=0:raise ValueError('Invalid time grid')
        self.lanes=tuple(sorted(lanes));self.begin=float(begin);self.step=float(step)
        self.samples=0;self.vehicle_seconds=0.;self.peak=0.

    def add(self,time,counts):
        expected=self.begin+(self.samples+1)*self.step
        if not math.isfinite(time) or not math.isclose(time,expected,abs_tol=1e-7,rel_tol=0):
            raise ValueError('Missing, repeated or out-of-order sample')
        if set(counts)!=set(self.lanes):raise ValueError('Lane panel changed or missing')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
               or v<0 or int(v)!=v for v in counts.values()):raise ValueError('Invalid queue count')
        total=sum(counts.values())
        self.vehicle_seconds+=total*self.step;self.samples+=1
        self.peak=max(self.peak,total/len(self.lanes))

    def sample(self,connection):
        time=connection.simulation.getTime()
        counts={lane:connection.lane.getLastStepHaltingNumber(lane) for lane in self.lanes}
        self.add(time,counts)
        return {'time':time,'halting_by_lane':counts}

    def result(self,end):
        if not self.samples or not math.isclose(end,self.begin+self.samples*self.step,abs_tol=1e-7,rel_tol=0):
            raise ValueError('Incomplete requested horizon')
        return {'aql_vehicles_per_lane':self.vehicle_seconds/(len(self.lanes)*(end-self.begin)),
                'peak_network_mean_vehicles_per_lane':self.peak,'lane_count':len(self.lanes),
                'samples':self.samples,'begin':self.begin,'end':end,'sample_step_s':self.step,
                'lane_ids_sha256':hashlib.sha256(json.dumps(self.lanes,separators=(',',':')).encode()).hexdigest(),
                'scope':'unique controlled incoming external lanes, full length, empty lanes included',
                'queue_definition':'SUMO last-step halting count; speed < 0.1 m/s'}
