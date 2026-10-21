"""
utils/sumo_utils.py
====================
TraCI helper layer shared by every controller.

Responsibilities
-----------------
1. Start / stop SUMO through TraCI with a reproducible seed.
2. Auto-discover every signalized intersection (traffic light) in the
   network -- nothing is hard-coded to "J1/J2/J5/J6" so the framework
   is reusable on any SUMO network.
3. Build and CACHE the intersection graph (nodes = traffic lights,
   edges = connecting roads) once per process for Max-Pressure's
   upstream/downstream lane lookup and result metadata.
4. Cache the mapping   traffic_light -> approach (N/S/E/W) -> [lane_area_detector_ids]
   once per process and thereafter only issue *aggregated* TraCI calls
   (one dict lookup instead of per-lane queries every step) to keep the
   framework fast on a CPU-only laptop.
"""

from __future__ import annotations
import os
import sys
import math
import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Make sure SUMO_HOME/tools is importable so we can use traci & sumolib
# --------------------------------------------------------------------------- #
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        # Prefer the TraCI client shipped with the installed SUMO binary.
        # Mixing a newer pip client with an older SUMO server corrupts some
        # lane-area detector responses before the version check can help.
        sys.path.insert(0, tools)

import traci  # noqa: E402
import sumolib  # noqa: E402
from traci import main as traci_main  # noqa: E402

import config


# --------------------------------------------------------------------------- #
# SUMO process control
# --------------------------------------------------------------------------- #
def start_sumo(gui: bool = False, seed: int = config.RANDOM_SEED,
               sim_seconds: int = config.SIM_SECONDS, label: str = "default",
               tripinfo_output: str | None = None,
               tripinfo_write_unfinished: bool = False,
               statistics_output: str | None = None):
    """Start (or restart) a SUMO instance connected via TraCI.

    `label` allows independent controller runs to use named connections.
    """
    binary = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    cmd = [
        binary,
        "-c", config.SUMOCFG,
        "--step-length", str(config.STEP_LENGTH),
        "--seed", str(seed),
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--no-warnings", "true",
        "--time-to-teleport", "300",
        # Benchmark metrics are collected through TraCI. Disable bulky
        # config-declared FCD output (notably in the Fenglin scenario).
        "--fcd-output", os.devnull,
    ]
    if tripinfo_output:
        tripinfo_output = os.path.abspath(tripinfo_output)
        os.makedirs(os.path.dirname(tripinfo_output), exist_ok=True)
        cmd.extend(["--tripinfo-output", tripinfo_output])
        if tripinfo_write_unfinished:
            cmd.append("--tripinfo-output.write-unfinished")
    if statistics_output:
        statistics_output = os.path.abspath(statistics_output)
        os.makedirs(os.path.dirname(statistics_output), exist_ok=True)
        cmd.extend(["--duration-log.statistics", "true",
                    "--statistic-output", statistics_output])
    conn_label = f"{label}"
    traci.start(cmd, label=conn_label)
    random.seed(seed)
    np.random.seed(seed)
    return traci.getConnection(conn_label)


def close_sumo(conn=None):
    try:
        if conn is not None:
            # Use the public connection API and wait for the SUMO process so
            # output files (especially tripinfo.xml) are complete on return.
            # SUMO 1.26 moved its private named-connection registry away from
            # traci.main, so consulting that registry before closing can skip
            # the close entirely on newer releases.
            try:
                conn.close(wait=True)
            except TypeError:  # compatibility with older TraCI clients
                conn.close()

            # Older clients may leave a closed named connection registered.
            # Clean either historical registry location without relying on it
            # for the close itself.
            registry_owners = (
                traci_main,
                getattr(traci, "connection", None),
            )
            for owner in registry_owners:
                connections = getattr(owner, "_connections", None)
                if not isinstance(connections, dict):
                    continue
                for label, registered in list(connections.items()):
                    if registered is conn:
                        connections.pop(label, None)
        else:
            traci.close()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Network / graph discovery  (cached once per process)
# --------------------------------------------------------------------------- #
class NetworkGraph:
    """Caches the traffic-light graph & detector groupings so we never
    re-parse the network or re-query static topology every simulation step.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._built = False
        return cls._instance

    def build(self, conn):
        if self._built:
            return
        self.conn = conn
        self.net = sumolib.net.readNet(config.NET_FILE, withInternal=True)
        self.tls_ids: List[str] = sorted(conn.trafficlight.getIDList())
        version_text = conn.getVersion()[1]
        match = re.search(r"(\d+)\.(\d+)", version_text)
        self.use_lane_detector_fallback = bool(
            match and (int(match.group(1)), int(match.group(2))) < (1, 19)
        )

        # --- controlled lanes per TL (ordered as in the traffic light logic) ---
        self.tl_lanes: Dict[str, List[str]] = {}
        self.tl_controlled_links = {}
        self.tl_nodes = {}
        for tl in self.tls_ids:
            links = conn.trafficlight.getControlledLinks(tl)
            self.tl_controlled_links[tl] = links
            in_lanes = []
            nodes = set()
            for link_group in links:
                if link_group:
                    in_lanes.append(link_group[0][0])
                    for link in link_group:
                        try:
                            lane = self.net.getLane(link[0])
                            nodes.add(lane.getEdge().getToNode())
                        except (KeyError, IndexError):
                            continue
                else:
                    in_lanes.append(None)
            self.tl_lanes[tl] = in_lanes
            if not nodes:
                try:
                    nodes.add(self.net.getNode(tl))
                except KeyError:
                    pass
            self.tl_nodes[tl] = sorted(nodes, key=lambda node: node.getID())

        # --- approach classification (N/S/E/W) using incoming edge heading ---
        self.tl_approach_lanes: Dict[str, Dict[str, List[str]]] = {}
        for tl in self.tls_ids:
            approach_lanes = defaultdict(set)
            for link_group in self.tl_controlled_links[tl]:
                for link in link_group:
                    lane_id = link[0]
                    try:
                        edge = self.net.getLane(lane_id).getEdge()
                    except KeyError:
                        continue
                    fx, fy = edge.getFromNode().getCoord()
                    x0, y0 = edge.getToNode().getCoord()
                    dx, dy = x0 - fx, y0 - fy
                    heading = math.degrees(math.atan2(dx, dy)) % 360
                    # incoming traffic that travels roughly south->north approaches
                    # the junction "from the south" -> label it S, etc.
                    if 45 <= heading < 135:
                        label = "E"
                    elif 135 <= heading < 225:
                        label = "N"
                    elif 225 <= heading < 315:
                        label = "W"
                    else:
                        label = "S"
                    approach_lanes[label].add(lane_id)
            self.tl_approach_lanes[tl] = {k: sorted(v) for k, v in approach_lanes.items()}

        # --- detector -> lane map & lane -> detector(s) map (from additional file) ---
        self.detectors_by_lane: Dict[str, List[str]] = defaultdict(list)
        try:
            for det_id in conn.lanearea.getIDList():
                lane = conn.lanearea.getLaneID(det_id)
                self.detectors_by_lane[lane].append(det_id)
        except Exception:
            pass

        # --- detectors grouped per TL per approach (cached, built ONCE) ---
        self.tl_approach_detectors: Dict[str, Dict[str, List[str]]] = {}
        for tl in self.tls_ids:
            self.tl_approach_detectors[tl] = {}
            for appr, lanes in self.tl_approach_lanes[tl].items():
                dets = []
                for lane in lanes:
                    dets.extend(self.detectors_by_lane.get(lane, []))
                self.tl_approach_detectors[tl][appr] = dets

        # --- downstream (outgoing) lanes per TL, used by Max-Pressure ---
        self.tl_outgoing_lanes: Dict[str, List[str]] = {}
        for tl in self.tls_ids:
            out_lanes = set()
            for link_group in self.tl_controlled_links[tl]:
                for link in link_group:
                    if len(link) > 1 and link[1]:
                        out_lanes.add(link[1])
            self.tl_outgoing_lanes[tl] = sorted(out_lanes)

        # --- graph adjacency: TL <-> TL connected via a road ---
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        node_to_tls = defaultdict(set)
        for tl, nodes in self.tl_nodes.items():
            for node in nodes:
                node_to_tls[node.getID()].add(tl)
        for tl in self.tls_ids:
            neighbours = set()
            for node in self.tl_nodes[tl]:
                adjacent_nodes = [edge.getToNode() for edge in node.getOutgoing()]
                adjacent_nodes += [edge.getFromNode() for edge in node.getIncoming()]
                for adjacent_node in adjacent_nodes:
                    neighbours.update(node_to_tls.get(adjacent_node.getID(), set()))
            neighbours.discard(tl)
            self.adjacency[tl] = sorted(neighbours)

        # --- phase count per TL (from the loaded program) ---
        self.num_phases: Dict[str, int] = {}
        self.phase_states: Dict[str, List[str]] = {}
        for tl in self.tls_ids:
            logic = conn.trafficlight.getAllProgramLogics(tl)[0]
            self.num_phases[tl] = len(logic.phases)
            self.phase_states[tl] = [p.state for p in logic.phases]

        # --- lane -> approach reverse lookup (per TL) ---
        self.lane_to_approach: Dict[str, Dict[str, str]] = {}
        for tl in self.tls_ids:
            rev = {}
            for appr, lanes in self.tl_approach_lanes[tl].items():
                for lane in lanes:
                    rev[lane] = appr
            self.lane_to_approach[tl] = rev

        # --- phase -> set of approaches served with green (cached) ---
        self.phase_green_approaches: Dict[str, Dict[int, set]] = {}
        for tl in self.tls_ids:
            phase_map = {}
            for idx, state in enumerate(self.phase_states[tl]):
                served = set()
                for link_idx, ch in enumerate(state):
                    if ch in ("G", "g") and link_idx < len(self.tl_lanes[tl]):
                        lane = self.tl_lanes[tl][link_idx]
                        appr = self.lane_to_approach[tl].get(lane)
                        if appr:
                            served.add(appr)
                phase_map[idx] = served
            self.phase_green_approaches[tl] = phase_map

        # --- identify "green" phases (i.e. exclude pure yellow/all-red) ---
        self.green_phase_indices: Dict[str, List[int]] = {
            # A staggered clearance can still contain a protected green for a
            # different movement while another movement is yellow.  Such a
            # mixed state is a transition, not a selectable steady green.
            tl: [
                i
                for i, s in enumerate(self.phase_states[tl])
                if ("G" in s or "g" in s) and "y" not in s and "Y" not in s
            ]
            for tl in self.tls_ids
        }

        self._built = True

    # ------------------------------------------------------------------- #
    def approach_state(self, tl: str) -> Dict[str, dict]:
        """Aggregate detector data for one TL, grouped by approach.
        ONE call per approach group (few TraCI calls) instead of per-lane."""
        result = {}
        for appr, dets in self.tl_approach_detectors[tl].items():
            if self.use_lane_detector_fallback or not dets:
                lanes = self.tl_approach_lanes[tl].get(appr, [])
                queue = occ = spd = wait = 0.0
                count = 0
                for lane in lanes:
                    queue += self.conn.lane.getLastStepHaltingNumber(lane)
                    occ += self.conn.lane.getLastStepOccupancy(lane)
                    count += self.conn.lane.getLastStepVehicleNumber(lane)
                    spd += self.conn.lane.getLastStepMeanSpeed(lane)
                    for vehicle_id in self.conn.lane.getLastStepVehicleIDs(lane):
                        wait += self.conn.vehicle.getWaitingTime(vehicle_id)
                n = max(len(lanes), 1)
                result[appr] = dict(queue=queue, occupancy=occ / n, count=count,
                                    speed=spd / n, waiting_time=wait)
                continue
            queue = occ = spd = wait = 0.0
            count = 0
            for d in dets:
                try:
                    queue += self.conn.lanearea.getJamLengthVehicle(d)
                    occ += self.conn.lanearea.getLastStepOccupancy(d)
                    count += self.conn.lanearea.getLastStepVehicleNumber(d)
                    spd += self.conn.lanearea.getLastStepMeanSpeed(d)
                    wait += self.conn.lanearea.getLastIntervalVehicleHaltingNumber(d) \
                        if hasattr(self.conn.lanearea, "getLastIntervalVehicleHaltingNumber") else 0.0
                except Exception:
                    continue
            n = max(len(dets), 1)
            result[appr] = dict(queue=queue, occupancy=occ / n, count=count,
                                 speed=spd / n, waiting_time=wait)
        return result


def get_graph(conn) -> NetworkGraph:
    g = NetworkGraph()
    g.build(conn)
    return g


def reset_graph_cache():
    NetworkGraph._instance = None
