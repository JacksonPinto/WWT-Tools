#! python3
# calc_shortest_chain.py
# Daisy-chain shortest path (sequência ordenada pelo usuário) usando TopologicPy

import os, sys, json, math, time
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Graph import Graph
from topologicpy.Wire import Wire
from topologicpy.Dictionary import Dictionary

TOLERANCE = 1e-4
MAP_TOLERANCE = 1e-3
DEVICE_EDGE_PENALTY_FACTOR = 1.0
LOG_PREFIX = "[CHAIN]"

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

    # Seleção de edges (igual ao main)
    if infra_edges and device_edges:
        edges_source = (infra_edges, device_edges)
    else:
        edges_source = (combined, [])

    log("Vertices:{} InfraEdges:{} DeviceEdges:{} TotalEdges:{} OrderedPoints:{}".format(
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

    # Mapear pontos ordenados da sequência do usuário
    ordered_points = []
    for sp in starts:
        coord = sp.get("point")
        eid = sp.get("element_id")
        if not (isinstance(coord, list) and len(coord) == 3): continue
        idx, d = nearest_vertex_index(coord, coords)
        if d > MAP_TOLERANCE:
            log("WARN mapping {} distance {:.6f}".format(eid, d))
        ordered_points.append((eid, graph_vertices[idx], d))
    log("Ordered points mapped: {}".format(len(ordered_points)))

    # Mapear end_point
    if not (isinstance(end_pt, list) and len(end_pt) == 3):
        log("ERROR invalid end point"); sys.exit(1)
    end_idx, end_d = nearest_vertex_index(end_pt, coords)
    if end_d > MAP_TOLERANCE:
        log("WARN end mapping distance {:.6f}".format(end_d))
    end_v = graph_vertices[end_idx]

    # Montar sequência: todos start_points (na ordem), depois end_v
    chain_vertices = [v for (_, v, _) in ordered_points] + [end_v]

    # Calcular caminho daisy-chain
    full_path_vertices = []
    success = True
    path_lengths = []
    mapped_distances = [d for (_, _, d) in ordered_points]
    try:
        for i in range(len(chain_vertices) - 1):
            seg_wire = Graph.ShortestPath(graph, chain_vertices[i], chain_vertices[i+1], edgeKey="cost", tolerance=TOLERANCE)
            if not seg_wire:
                log("ERROR: No path between chain points {} and {}".format(i, i+1))
                success = False
                break
            wv = Wire.Vertices(seg_wire)
            seg_coords = [x.Coordinates() for x in wv]
            if i > 0 and len(seg_coords) > 0:
                seg_coords = seg_coords[1:] # Não repetir o nó de conexão
            full_path_vertices.extend(seg_coords)
            if len(seg_coords) > 1:
                seg_length = sum(dist3(seg_coords[j], seg_coords[j+1]) for j in range(len(seg_coords)-1))
                path_lengths.append(seg_length)
    except Exception as ex:
        log("ERROR during chain path: {}".format(ex))
        success = False

    total_length = sum(path_lengths) if path_lengths else None

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
            "chain_points": len(chain_vertices),
            "mapped_chain": len(ordered_points),
            "success": success,
            "total_length": total_length,
            "path_lengths": path_lengths,
            "mapped_distances": mapped_distances,
            "duration_sec": time.time() - t0
        },
        "results": [{
            "element_ids": [sp.get("element_id") for sp in starts],
            "length": total_length,
            "vertex_path_xyz": full_path_vertices,
            "mapped_distances": mapped_distances
        }]
    }
    out_path = os.path.join(script_dir, "topologic_results.json")
    json.dump(out, open(out_path, "w"), indent=2)
    log("Results written: {}".format(out_path))
    log("Summary: success={}, total_length={}".format(success, total_length))

if __name__ == "__main__":
    main()