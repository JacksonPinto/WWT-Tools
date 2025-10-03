#! python3
# calc_shortest_chain.py  (Reverted style: "version 5" simplified workflow)
#
# Workflow:
#  1. Read topologic.JSON (produced by main script).
#  2. Build Vertex objects for every JSON vertex.
#  3. Build Edge objects for every JSON edge (combined list) tagging INFRA vs JUMPER.
#  4. Flatten: vertices + edges -> Topology.ByGeometry(*geoms)
#  5. Cluster.ByTopologies(...)
#  6. Topology.SelfMerge(...)
#  7. Graph.ByTopology(...)
#  8. Map ordered start_points (JSON order) and end_point to graph vertices by coordinate.
#  9. For each consecutive pair (A->B, B->C, ..., lastStart->End) run Graph.ShortestPath.
#
# Output schema (reverted to earlier "working" form):
# {
#   "meta": {...},
#   "results": [
#       {
#         "start_index": i,
#         "element_id": <origin start point element id or None for final leg>,
#         "length": <path length or None>,
#         "vertex_path_xyz": [ [x,y,z], ... ],   # expanded from edge sequence (no chord underestimation)
#         "mapped_distance": <euclidean distance between original start point coordinate and mapped graph vertex>
#       }, ...
#   ]
# }
#
# Notes:
# - Uses only the vertices & edges present in topologic.JSON (no device-edge exclusion).
# - Edge cost = geometric length * DEVICE_EDGE_PENALTY_FACTOR if JUMPER else length.
# - "vertex_path_xyz" is derived from traversed edge sequence to preserve bends.
# - If Graph.ShortestPath fails (returns None), length stays None for that leg.
#
# Adjust CONFIG values below if needed.

import os, sys, json, math, time
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Graph import Graph
from topologicpy.Wire import Wire
from topologicpy.Dictionary import Dictionary

# ---------------- CONFIG ----------------
TOLERANCE = 5e-4                     # Graph / merge tolerance (match main script MERGE_TOL order of magnitude)
MAP_TOLERANCE = 1.0                  # Max distance allowed when mapping a start/end point to a graph vertex (warning if exceeded)
DEVICE_EDGE_PENALTY_FACTOR = 1e-2    # Cost multiplier for JUMPER (device) edges ( <1 => cheaper, >1 => penalize )
LOG_PREFIX = "[CALC]"
# ----------------------------------------

def log(msg):
    print("{} {}".format(LOG_PREFIX, msg))

def dist3(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def nearest_vertex_index(pt, coords):
    """Return (index, distance) of nearest coordinate in coords to pt."""
    md = float('inf'); mi = None
    for i,c in enumerate(coords):
        d = dist3(pt,c)
        if d < md:
            md = d; mi = i
    return mi, md

def expand_wire_vertices(wire):
    """
    Returns (path_vertices_xyz, length) by traversing the wire's edge sequence so that
    bends are preserved (no chord simplification).
    """
    try:
        edges = Wire.Edges(wire)
    except:
        edges = []
    if not edges:
        # fallback: attempt vertex list only
        try:
            vs = Wire.Vertices(wire)
            coords = [v.Coordinates() for v in vs]
            length = sum(dist3(coords[i], coords[i+1]) for i in range(len(coords)-1))
            return coords, length
        except:
            return [], None

    path_coords = []
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
        # Append first endpoint if starting or different from previous
        if not path_coords:
            path_coords.append(c0)
        else:
            if path_coords[-1] != c0:
                path_coords.append(c0)
        path_coords.append(c1)
        total_len += dist3(c0, c1)

    # Deduplicate consecutive duplicates
    cleaned = []
    for c in path_coords:
        if not cleaned or cleaned[-1] != c:
            cleaned.append(c)

    return cleaned, total_len

def main():
    t0 = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "topologic.JSON")
    if not os.path.isfile(json_path):
        log("ERROR missing topologic.JSON")
        sys.exit(1)

    data = json.load(open(json_path,"r"))

    vertices_raw   = data.get("vertices", [])
    infra_edges    = data.get("infra_edges", [])
    device_edges   = data.get("device_edges", [])
    combined_edges = data.get("edges", [])  # main script stores combined (infra first then device)
    starts         = data.get("start_points", [])
    end_point      = data.get("end_point", None)
    meta_in        = data.get("meta", {})

    if not vertices_raw or not combined_edges:
        log("ERROR empty vertices or edges in JSON.")
        sys.exit(1)

    log("Input counts: vertices={} infra_edges={} device_edges={} combined_edges={} start_points={}".format(
        len(vertices_raw), len(infra_edges), len(device_edges), len(combined_edges), len(starts)
    ))

    # 2+3: Build Vertex + Edge objects
    topo_vertices = [Vertex.ByCoordinates(*v) for v in vertices_raw]

    device_set = set(tuple(sorted(e)) for e in device_edges)
    edge_objects = []

    for (i,j) in combined_edges:
        i = int(i); j = int(j)
        v1 = topo_vertices[i]; v2 = topo_vertices[j]
        length = dist3(vertices_raw[i], vertices_raw[j])
        cat = "JUMPER" if tuple(sorted((i,j))) in device_set else "INFRA"
        cost = length * (DEVICE_EDGE_PENALTY_FACTOR if cat == "JUMPER" else 1.0)
        d = Dictionary.ByKeysValues(["category","length","cost"], [cat,length,cost])
        e = Edge.ByVertices(v1, v2)
        try: e.SetDictionary(d)
        except: pass
        edge_objects.append(e)

    # 4: Topology.ByGeometry (all vertices and edges)
    # (We include vertices explicitly so all are present even if isolated.)
    geometry_items = topo_vertices + edge_objects
    topology_geom = Topology.ByGeometry(*geometry_items)

    # 5: Cluster.ByTopologies
    cluster = Cluster.ByTopologies(topology_geom)

    # 6: SelfMerge (to clean duplicates but keep real corners if tolerance small)
    merged = Topology.SelfMerge(cluster, tolerance=TOLERANCE)

    # 7: Graph.ByTopology
    graph = Graph.ByTopology(merged, tolerance=TOLERANCE)
    graph_vertices = Graph.Vertices(graph)
    graph_coords = [v.Coordinates() for v in graph_vertices]
    log("Graph vertex count after merge: {}".format(len(graph_coords)))

    # Helper: map coordinate to nearest graph vertex (within MAP_TOLERANCE)
    def map_point_to_graph_vertex(pt):
        idx, d = nearest_vertex_index(pt, graph_coords)
        gv = graph_vertices[idx]
        return gv, d, idx

    # 8: Map start points (preserving order)
    mapped_chain = []
    for sp in starts:
        coord = sp.get("point")
        eid   = sp.get("element_id")
        if not (isinstance(coord, list) and len(coord) == 3):
            continue
        gv, d, _ = map_point_to_graph_vertex(coord)
        if d > MAP_TOLERANCE:
            log("WARN start element {} mapping distance {:.6f}".format(eid, d))
        mapped_chain.append({
            "element_id": eid,
            "coord": coord,
            "graph_vertex": gv,
            "mapped_distance": d
        })

    if not (isinstance(end_point, list) and len(end_point) == 3):
        log("ERROR invalid end point in JSON.")
        sys.exit(1)

    end_gv, end_d, _ = map_point_to_graph_vertex(end_point)
    if end_d > MAP_TOLERANCE:
        log("WARN end point mapping distance {:.6f}".format(end_d))

    # Build full chain list including end as terminal
    chain_with_end = mapped_chain + [{
        "element_id": None,
        "coord": end_point,
        "graph_vertex": end_gv,
        "mapped_distance": end_d
    }]

    # 9: Shortest paths for each consecutive pair
    results = []
    success = 0
    total_legs = len(chain_with_end) - 1

    for i in range(total_legs):
        orig = chain_with_end[i]
        dest = chain_with_end[i+1]
        eid  = orig["element_id"]
        sv   = orig["graph_vertex"]
        dv   = dest["graph_vertex"]
        mdist = orig["mapped_distance"]

        try:
            wire = Graph.ShortestPath(graph, sv, dv, edgeKey="cost", tolerance=TOLERANCE)
        except Exception as ex:
            log("ERROR path element {} leg {}: {}".format(eid, i, ex))
            wire = None

        if not wire:
            results.append({
                "start_index": i,
                "element_id": eid,
                "length": None,
                "vertex_path_xyz": [],
                "mapped_distance": mdist
            })
            continue

        path_coords, path_len = expand_wire_vertices(wire)
        if path_coords and path_len is not None:
            success += 1
            results.append({
                "start_index": i,
                "element_id": eid,
                "length": path_len,
                "vertex_path_xyz": path_coords,
                "mapped_distance": mdist
            })
        else:
            results.append({
                "start_index": i,
                "element_id": eid,
                "length": None,
                "vertex_path_xyz": [],
                "mapped_distance": mdist
            })

    out = {
        "meta": {
            "version": "CHAIN-1.0.0",
            "tolerance": TOLERANCE,
            "map_tolerance": MAP_TOLERANCE,
            "device_edge_penalty_factor": DEVICE_EDGE_PENALTY_FACTOR,
            "vertices_input": len(vertices_raw),
            "combined_edges": len(combined_edges),
            "infra_edges": len(infra_edges),
            "device_edges": len(device_edges),
            "graph_vertices": len(graph_coords),
            "chain_points": len(chain_with_end),
            "mapped_chain": len(mapped_chain),
            "paths_success": success,
            "paths_failed": total_legs - success,
            "duration_sec": time.time() - t0
        },
        "results": results
    }

    out_path = os.path.join(script_dir, "topologic_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    log("Results written: {}".format(out_path))
    log("Summary: {} success / {} total legs".format(success, total_legs))

if __name__ == "__main__":
    main()