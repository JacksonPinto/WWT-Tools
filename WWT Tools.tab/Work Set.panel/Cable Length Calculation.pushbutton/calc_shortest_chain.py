#! python3
# calc_shortest_chain.py
# Version: 5.0.0-chain (mirrors calc_shortest.py 3.9.2 logic; adds daisy-chain legs)
#
# Logic:
#   1. Read topologic.JSON
#   2. Build vertices + edges (INFRA + JUMPER)
#   3. Cluster.ByTopologies -> SelfMerge -> Graph.ByTopology
#   4. Map start_points to nearest graph vertices in listed order.
#   5. Legs: (Start0->Start1), (Start1->Start2), ..., (Start{n-2}->Start{n-1}), (Start{n-1}->End)
#   6. For each leg run Graph.ShortestPath (edgeKey="cost").
#   7. Path vertices taken from Wire.Vertices (simplified) OR (if EXPAND_EDGES=True) expanded via edge sequence.
#
# Output: topologic_results.json with one result per leg.
#
# NOTE:
#   DEVICE_EDGE_PENALTY_FACTOR < 1 makes jumper edges cheaper; > 1 penalizes them.
#   EXPAND_EDGES=False reproduces behavior similar to calc_shortest.py (may simplify bends).
#   Set EXPAND_EDGES=True to expand each path via traversed edge sequence (preserve bends).
#
# Dependencies: topologicpy must be installed in the external Python interpreter used by script.py.

import os, sys, json, math, time
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Graph import Graph
from topologicpy.Wire import Wire
from topologicpy.Dictionary import Dictionary

# ------------- CONFIG -------------
TOLERANCE = 5e-4
MAP_TOLERANCE = 1.0
DEVICE_EDGE_PENALTY_FACTOR = 1e-4
EXPAND_EDGES = True  # Set True to preserve all bends via edge sequence
LOG_PREFIX = "[CALC]"
# ----------------------------------

def log(msg):
    print("{} {}".format(LOG_PREFIX, msg))

def dist3(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def nearest_vertex_index(pt, coords):
    md = float('inf'); mi = None
    for i, c in enumerate(coords):
        d = dist3(pt, c)
        if d < md:
            md = d; mi = i
    return mi, md

def make_edge(i, j, cat, topo_vertices, vertices):
    v1 = topo_vertices[i]; v2 = topo_vertices[j]
    e = Edge.ByVertices(v1, v2)
    length = dist3(vertices[i], vertices[j])
    cost = length * (DEVICE_EDGE_PENALTY_FACTOR if cat == "JUMPER" else 1.0)
    d = Dictionary.ByKeysValues(["category", "length", "cost"], [cat, length, cost])
    try:
        e.SetDictionary(d)
    except:
        pass
    return e, length

def expand_wire_edges(wire):
    """Return (coords_list, length) by traversing edge sequence (bend-preserving)."""
    try:
        edges = Wire.Edges(wire)
    except:
        edges = []
    if not edges:
        # fallback to simplified vertices
        try:
            vs = Wire.Vertices(wire)
            coords = [v.Coordinates() for v in vs]
            length = sum(dist3(coords[i], coords[i+1]) for i in range(len(coords)-1))
            return coords, length
        except:
            return [], None

    coords = []
    total_len = 0.0
    for e in edges:
        try:
            evs = Edge.Vertices(e)
        except:
            evs = []
        if len(evs) < 2:
            continue
        c0 = evs[0].Coordinates()
        c1 = evs[-1].Coordinates()
        if not coords:
            coords.append(c0)
        elif coords[-1] != c0:
            coords.append(c0)
        coords.append(c1)
        total_len += dist3(c0, c1)

    # Deduplicate consecutive
    cleaned = []
    for c in coords:
        if not cleaned or cleaned[-1] != c:
            cleaned.append(c)
    return cleaned, total_len

def simplified_wire_vertices(wire):
    """Return (coords_list, length) using Wire.Vertices (may skip intermediate colinear bends)."""
    if not wire:
        return [], None
    try:
        vs = Wire.Vertices(wire)
        coords = [v.Coordinates() for v in vs]
        length = sum(dist3(coords[i], coords[i+1]) for i in range(len(coords)-1))
        return coords, length
    except:
        return [], None

def main():
    t0 = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "topologic.JSON")
    if not os.path.isfile(json_path):
        log("ERROR missing topologic.JSON")
        sys.exit(1)

    data = json.load(open(json_path, "r"))

    vertices = data.get("vertices", [])
    infra_edges = data.get("infra_edges", [])
    device_edges = data.get("device_edges", [])
    combined = data.get("edges", [])
    starts = data.get("start_points", [])
    end_pt = data.get("end_point", None)

    if not vertices or not combined:
        log("ERROR vertices or edges missing in JSON")
        sys.exit(1)

    edges_source = (infra_edges, device_edges) if (infra_edges and device_edges) else (combined, [])
    log("Vertices:{} InfraEdges:{} DeviceEdges:{} TotalEdges:{} ChainStarts:{} EndPoint:{}"
        .format(len(vertices), len(infra_edges), len(device_edges), len(combined), len(starts),
                "Yes" if isinstance(end_pt, list) and len(end_pt)==3 else "No"))

    # Build topologic vertices
    topo_vertices = [Vertex.ByCoordinates(*v) for v in vertices]

    # Build edges with cost dictionary
    device_set = set(tuple(sorted(e)) for e in device_edges)
    edge_objs = []
    for (i, j) in edges_source[0]:
        e,_ = make_edge(i, j, "INFRA", topo_vertices, vertices)
        edge_objs.append(e)
    for (i, j) in edges_source[1]:
        e,_ = make_edge(i, j, "JUMPER", topo_vertices, vertices)
        edge_objs.append(e)

    # Cluster + SelfMerge + Graph
    cluster = Cluster.ByTopologies(*edge_objs)
    merged = Topology.SelfMerge(cluster, tolerance=TOLERANCE)
    graph = Graph.ByTopology(merged, tolerance=TOLERANCE)
    graph_vertices = Graph.Vertices(graph)
    coords = [v.Coordinates() for v in graph_vertices]
    log("Graph vertices: {}".format(len(coords)))

    # Map start points (ordered)
    mapped = []
    for sp in starts:
        coord = sp.get("point")
        eid = sp.get("element_id")
        if not (isinstance(coord, list) and len(coord) == 3):
            continue
        idx, d = nearest_vertex_index(coord, coords)
        if d > MAP_TOLERANCE:
            log("WARN start {} mapping distance {:.6f}".format(eid, d))
        mapped.append({
            "element_id": eid,
            "graph_vertex": graph_vertices[idx],
            "coord": coord,
            "mapped_distance": d
        })

    if not mapped:
        log("No valid start points mapped.")
        out = {
            "meta": {
                "version": "5.0.0-chain",
                "tolerance": TOLERANCE,
                "map_tolerance": MAP_TOLERANCE,
                "device_edge_penalty_factor": DEVICE_EDGE_PENALTY_FACTOR,
                "vertices_input": len(vertices),
                "graph_vertices": len(coords),
                "chain_points": 0,
                "paths_success": 0,
                "paths_failed": 0,
                "duration_sec": time.time()-t0
            },
            "results": []
        }
        json.dump(out, open(os.path.join(script_dir,"topologic_results.json"),"w"), indent=2)
        sys.exit(0)

    # Map end point
    end_mapped = None
    end_d = None
    if isinstance(end_pt, list) and len(end_pt) == 3:
        end_idx, end_d = nearest_vertex_index(end_pt, coords)
        if end_d > MAP_TOLERANCE:
            log("WARN end mapping distance {:.6f}".format(end_d))
        end_mapped = graph_vertices[end_idx]
    else:
        log("WARN missing/invalid end point; last leg will be skipped.")

    # Build chain legs: consecutive starts + (last start -> end)
    legs = []
    for i in range(len(mapped)-1):
        legs.append((i, mapped[i], mapped[i+1], False))  # (index_of_origin, origin_obj, dest_obj, is_final_to_end=False)
    if end_mapped:
        legs.append((len(mapped)-1, mapped[-1], {
            "element_id": None,
            "graph_vertex": end_mapped,
            "coord": end_pt,
            "mapped_distance": end_d
        }, True))  # final leg to end

    results = []
    success = 0

    for leg_idx, origin, dest, final_leg in legs:
        eid = origin["element_id"]
        sv = origin["graph_vertex"]
        dv = dest["graph_vertex"]
        mdist = origin["mapped_distance"]
        if not sv or not dv:
            results.append({
                "start_index": leg_idx,
                "element_id": eid,
                "length": None,
                "vertex_path_xyz": [],
                "mapped_distance": mdist
            })
            continue
        try:
            wire = Graph.ShortestPath(graph, sv, dv, edgeKey="cost", tolerance=TOLERANCE)
        except Exception as ex:
            log("ERROR leg {} ({}) path: {}".format(leg_idx, eid, ex))
            wire = None

        if not wire:
            results.append({
                "start_index": leg_idx,
                "element_id": eid,
                "length": None,
                "vertex_path_xyz": [],
                "mapped_distance": mdist
            })
            continue

        if EXPAND_EDGES:
            coords_path, path_len = expand_wire_edges(wire)
        else:
            coords_path, path_len = simplified_wire_vertices(wire)

        results.append({
            "start_index": leg_idx,
            "element_id": eid,
            "length": path_len,
            "vertex_path_xyz": coords_path,
            "mapped_distance": mdist
        })
        if path_len is not None:
            success += 1

    out = {
        "meta": {
            "version": "5.0.0-chain",
            "tolerance": TOLERANCE,
            "map_tolerance": MAP_TOLERANCE,
            "device_edge_penalty_factor": DEVICE_EDGE_PENALTY_FACTOR,
            "vertices_input": len(vertices),
            "infra_edges": len(infra_edges),
            "device_edges": len(device_edges),
            "combined_edges": len(combined),
            "graph_vertices": len(coords),
            "chain_points": len(mapped) + (1 if end_mapped else 0),
            "start_points": len(mapped),
            "end_point_present": bool(end_mapped),
            "legs_total": len(legs),
            "paths_success": success,
            "paths_failed": len(legs) - success,
            "expand_edges": EXPAND_EDGES,
            "duration_sec": time.time() - t0
        },
        "results": results
    }

    out_path = os.path.join(script_dir, "topologic_results.json")
    json.dump(out, open(out_path, "w"), indent=2)
    log("Results written: {}".format(out_path))
    log("Summary: {} success / {} legs".format(success, len(legs)))

if __name__ == "__main__":
    main()