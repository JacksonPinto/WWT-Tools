#! python3
# calc_shortest_chain.py
# Calcula menor caminho para cada par consecutivo de elementos selecionados (A->B, B->C, ..., Z->EndPoint)
# e gera a saída compatível com update_cable_lengths.py

import os, sys, json, math, time
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Graph import Graph
from topologicpy.Wire import Wire
from topologicpy.Dictionary import Dictionary

# Updated tolerances to match script.py
TOLERANCE = 5e-4
MAP_TOLERANCE = 5
DEVICE_EDGE_PENALTY_FACTOR = 1e-4
LOG_PREFIX = "[CALC]"

def log(msg):
    print("{} {}".format(LOG_PREFIX, msg))

def dist3(a, b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def nearest_vertex_index(pt, coords):
    md = float('inf'); mi = None
    for i, c in enumerate(coords):
        d = dist3(pt, c)
        if d < md: md = d; mi = i
    return mi, md

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

    if infra_edges and device_edges:
        edges_source = (infra_edges, device_edges)
    else:
        edges_source = (combined, [])

    log("Vertices:{} InfraEdges:{} DeviceEdges:{} TotalEdges:{} ChainPoints:{}".format(
        len(vertices), len(infra_edges), len(device_edges), len(combined), len(starts)
    ))

    topo_vertices = [Vertex.ByCoordinates(*v) for v in vertices]

    def make_edge(i, j, cat):
        v1 = topo_vertices[i]; v2 = topo_vertices[j]
        e = Edge.ByVertices(v1, v2)
        length = dist3(vertices[i], vertices[j])
        cost = length * (DEVICE_EDGE_PENALTY_FACTOR if cat == "JUMPER" else 1.0)
        d = Dictionary.ByKeysValues(["category", "length", "cost"], [cat, length, cost])
        try: e.SetDictionary(d)
        except: pass
        return e

    edge_objs = []
    for i, j in edges_source[0]:
        edge_objs.append(make_edge(i, j, "INFRA"))
    for i, j in edges_source[1]:
        edge_objs.append(make_edge(i, j, "JUMPER"))

    cluster = Cluster.ByTopologies(*edge_objs)
    merged = Topology.SelfMerge(cluster, tolerance=TOLERANCE)
    graph = Graph.ByTopology(merged, tolerance=TOLERANCE)

    graph_vertices = Graph.Vertices(graph)
    coords = [v.Coordinates() for v in graph_vertices]
    log("Graph vertices: {}".format(len(coords)))

    # ---------------- PATCH: Preserve and log manual sequence (seq_index) ----------------
    mapped_points = []
    for sp in starts:
        coord = sp.get("point")
        eid = sp.get("element_id")
        seq_index = sp.get("seq_index")  # added
        if not (isinstance(coord, list) and len(coord) == 3): continue
        idx, d = nearest_vertex_index(coord, coords)
        if d > MAP_TOLERANCE:
            log("WARN mapping {} distance {:.6f}".format(eid, d))
        mapped_points.append({
            "element_id": eid,
            "vertex": graph_vertices[idx],
            "coord": coord,
            "mapped_distance": d,
            "seq_index": seq_index
        })
    log("Start point sequence (seq_index -> element_id): {}".format(
        ["{}->{}".format(mp.get("seq_index"), mp.get("element_id")) for mp in mapped_points]
    ))
    # ---------------- END PATCH ----------------

    if not (isinstance(end_pt, list) and len(end_pt) == 3):
        log("ERROR invalid end point"); sys.exit(1)
    end_idx, end_d = nearest_vertex_index(end_pt, coords)
    if end_d > MAP_TOLERANCE:
        log("WARN end mapping distance {:.6f}".format(end_d))
    end_v = graph_vertices[end_idx]

    # Monta pares: [A,B], [B,C], ..., [Z,End]
    results = []
    success = 0
    chain_list = mapped_points + [{
        "element_id": None,
        "vertex": end_v,
        "coord": end_pt,
        "mapped_distance": end_d,
        "seq_index": None
    }]
    for i in range(len(chain_list)-1):
        orig = chain_list[i]
        dest = chain_list[i+1]
        eid = orig["element_id"]
        sv = orig["vertex"]
        dv = dest["vertex"]
        mdist = orig["mapped_distance"]
        seq_idx = orig.get("seq_index")  # added
        try:
            wire = Graph.ShortestPath(graph, sv, dv, edgeKey="cost", tolerance=TOLERANCE)
            if not wire:
                results.append({
                    "start_index": i,
                    "element_id": eid,
                    "seq_index": seq_idx,        # added
                    "length": None,
                    "vertex_path_xyz": [],
                    "mapped_distance": mdist
                })
                continue
            wv = Wire.Vertices(wire)
            path = [x.Coordinates() for x in wv]
            length = sum(dist3(path[j], path[j+1]) for j in range(len(path)-1))
            results.append({
                "start_index": i,
                "element_id": eid,
                "seq_index": seq_idx,            # added
                "length": length,
                "vertex_path_xyz": path,
                "mapped_distance": mdist
            })
            success += 1
        except Exception as ex:
            log("ERROR path element {}: {}".format(eid, ex))
            results.append({
                "start_index": i,
                "element_id": eid,
                "seq_index": seq_idx,            # added
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
            "vertices_input": len(vertices),
            "infra_edges": len(infra_edges),
            "device_edges": len(device_edges),
            "graph_vertices": len(coords),
            "chain_points": len(chain_list),
            "mapped_chain": len(mapped_points),
            "paths_success": success,
            "paths_failed": len(chain_list)-1-success,
            "duration_sec": time.time() - t0
        },
        "results": results
    }
    out_path = os.path.join(script_dir, "topologic_results.json")
    json.dump(out, open(out_path, "w"), indent=2)
    log("Results written: {}".format(out_path))
    log("Summary: {} success / {} total".format(success, len(chain_list)-1))

if __name__ == "__main__":
    main()