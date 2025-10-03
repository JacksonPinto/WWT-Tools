#! python3
# calc_shortest_chain.py
# Strict Daisy Chain (INFRA-ONLY interior) Version 6.0.0
#
# Enforces that inter-device routing follows ONLY infrastructure edges.
# Device (jumper) edges are used only to measure the local connection from each
# device to the nearest infrastructure vertex.
#
# PATH LENGTH (per leg i -> i+1):
#   len = device_i_to_infra + shortest_path_along_infra + infra_to_device_i+1
#
# OUTPUT vertex_path_xyz:
#   [device_i_point] + [infra path vertex coords in order (including mapped start infra vertex first)] + [device_i+1_point]
#
# If end point exists the last leg is (last_device -> end_point) using same logic (end counts as "device" without element_id).
#
# OPTIONAL: If USE_DEVICE_EDGE_GEOMETRY = True attempts to find actual device edge(s) in device_edges
# connecting a device vertex to its mapped infra vertex; if found, uses the precise length (sum of
# those edge lengths) instead of direct straight-line distance.
#
# REQUIREMENT: topologicpy must be installed in external CPython interpreter.

import os, sys, json, math, time
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology
from topologicpy.Graph import Graph
from topologicpy.Wire import Wire

VERSION = "6.0.0-INFRA_ONLY"

# ---------------- CONFIG ----------------
INFRA_TOLERANCE = 5e-4
MAP_TOLERANCE = 1.0
USE_DEVICE_EDGE_GEOMETRY = True
SEARCH_RADIUS_FOR_DEVICE_EDGE = 1.0  # feet: max distance allowed between a device point and a vertex coord to consider as same
LOG_PREFIX = "[CALC]"
# ----------------------------------------

def log(msg):
    print("{} {}".format(LOG_PREFIX, msg))

def dist3(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def nearest_vertex_index(pt, coords):
    md=float('inf'); mi=None
    for i,c in enumerate(coords):
        d=dist3(pt,c)
        if d<md:
            md=d; mi=i
    return mi, md

def build_infra_graph(vertices, infra_edges):
    """Return (graph, topo_vertices, vertex_coords_map, infra_vertex_objects)."""
    topo_vertices = [Vertex.ByCoordinates(*v) for v in vertices]

    edge_objs=[]
    for (i,j) in infra_edges:
        v1=topo_vertices[i]; v2=topo_vertices[j]
        try:
            e=Edge.ByVertices(v1,v2)
            edge_objs.append(e)
        except:
            pass

    cluster = Cluster.ByTopologies(*edge_objs)
    merged = Topology.SelfMerge(cluster, tolerance=INFRA_TOLERANCE)
    graph = Graph.ByTopology(merged, tolerance=INFRA_TOLERANCE)
    gvs = Graph.Vertices(graph)
    gcoords=[v.Coordinates() for v in gvs]
    return graph, topo_vertices, gcoords, gvs

def dijkstra_infra(graph, sv, dv):
    """Use Topologic Graph.ShortestPath (it internally uses costs=length)."""
    try:
        wire = Graph.ShortestPath(graph, sv, dv, tolerance=INFRA_TOLERANCE)
        return wire
    except Exception as ex:
        log("ERROR shortest path: {}".format(ex))
        return None

def expand_wire_vertices(wire):
    """Expand path with all intermediate wire vertices (not edges)."""
    if not wire:
        return [], None
    try:
        wv = Wire.Vertices(wire)
        coords=[v.Coordinates() for v in wv]
        length=sum(dist3(coords[i], coords[i+1]) for i in range(len(coords)-1))
        return coords, length
    except:
        return [], None

def try_device_local_length(device_point, infra_coord, vertices, device_edges, device_vertex_index_cache, device_vertex_xyz_cache):
    """
    Attempts to find actual device edge chain from a device point to the infra vertex
    by searching for a vertex in 'vertices' matching device_point, then following
    device_edges to an infra vertex whose coordinate matches infra_coord (within small tolerance).
    Returns (length, chain_coords) or (None, None) if not found.
    """
    tol = 1e-6
    # Build a mapping once
    if not device_vertex_index_cache:
        # Accept any vertex from the export whose coordinate is exactly device coordinates
        for idx, v in enumerate(vertices):
            device_vertex_xyz_cache[idx] = tuple(v)
    # Find closest vertex to device_point
    best_idx=None; best_d=1e9
    for idx, c in device_vertex_xyz_cache.items():
        d=dist3(device_point, c)
        if d<best_d:
            best_d=d; best_idx=idx
    if best_d>SEARCH_RADIUS_FOR_DEVICE_EDGE:
        return None, None  # device point not near any existing vertex (maybe main script didn't create explicit device vertex)
    # BFS limited to device_edges
    dev_adj={}
    for a,b in device_edges:
        dev_adj.setdefault(a,[]).append(b)
        dev_adj.setdefault(b,[]).append(a)
    # We also need to know candidate infra target vertex index (nearest to infra_coord)
    # If multiple infra vertices share same coord within tol, any is fine.
    target_candidate=None; target_d=1e9
    for idx,c in device_vertex_xyz_cache.items():
        d=dist3(infra_coord,c)
        if d<target_d:
            target_d=d; target_candidate=idx
    if target_candidate is None or target_d>SEARCH_RADIUS_FOR_DEVICE_EDGE:
        return None, None

    from collections import deque
    q=deque([(best_idx, [best_idx])])
    visited=set([best_idx])
    found_path=None
    while q:
        cur, path = q.popleft()
        if cur==target_candidate:
            found_path=path
            break
        for nb in dev_adj.get(cur,[]):
            if nb not in visited:
                visited.add(nb)
                q.append((nb, path+[nb]))
    if not found_path:
        return None, None
    coords=[device_vertex_xyz_cache[i] for i in found_path]
    length=sum(dist3(coords[i], coords[i+1]) for i in range(len(coords)-1))
    return length, coords

def main():
    t0=time.time()
    script_dir=os.path.dirname(os.path.abspath(__file__))
    json_path=os.path.join(script_dir,"topologic.JSON")
    if not os.path.isfile(json_path):
        log("ERROR missing topologic.JSON"); sys.exit(1)

    data=json.load(open(json_path,"r"))

    vertices      = data.get("vertices",[])
    infra_edges   = data.get("infra_edges",[])
    device_edges  = data.get("device_edges",[])
    combined      = data.get("edges",[])
    starts        = data.get("start_points",[])
    end_pt        = data.get("end_point",None)

    log("INPUT vertices={} infra_edges={} device_edges={} combined_edges={} start_points={}".format(
        len(vertices), len(infra_edges), len(device_edges), len(combined), len(starts)
    ))

    if not vertices or not infra_edges:
        log("ERROR: Missing essential infra graph data.")
        sys.exit(1)

    graph, topo_vertices, gcoords, gverts = build_infra_graph(vertices, infra_edges)
    log("Infra graph vertices: {}".format(len(gcoords)))

    # Map each start point to nearest infra graph vertex
    mapped_starts=[]
    for sp in starts:
        coord=sp.get("point")
        eid=sp.get("element_id")
        if not (isinstance(coord,list) and len(coord)==3):
            continue
        gi, md = nearest_vertex_index(coord, gcoords)
        if md>MAP_TOLERANCE:
            log("WARN start {} mapping distance {:.6f}".format(eid, md))
        mapped_starts.append({
            "element_id": eid,
            "device_coord": coord,
            "graph_vertex": gverts[gi],
            "graph_coord": gcoords[gi],
            "map_dist": md
        })

    # Map end point
    end_mapped=None
    if isinstance(end_pt,list) and len(end_pt)==3:
        ei, ed = nearest_vertex_index(end_pt, gcoords)
        if ed>MAP_TOLERANCE:
            log("WARN end mapping distance {:.6f}".format(ed))
        end_mapped={
            "element_id": None,
            "device_coord": end_pt,
            "graph_vertex": gverts[ei],
            "graph_coord": gcoords[ei],
            "map_dist": ed
        }
    else:
        log("WARN No valid end point present; last leg to end omitted.")

    # Build chain (A->B->... last -> End)
    chain_nodes = mapped_starts + ([end_mapped] if end_mapped else [])
    legs=[]
    for i in range(len(chain_nodes)-1):
        legs.append((i, chain_nodes[i], chain_nodes[i+1]))

    log("Leg count: {}".format(len(legs)))

    # Pre-cache for device edge search (if enabled)
    device_vertex_index_cache=True
    device_vertex_xyz_cache={}
    results=[]
    success=0

    for leg_index, start_node, dest_node in legs:
        start_eid = start_node["element_id"]
        s_dev = start_node["device_coord"]
        s_inf = start_node["graph_coord"]
        d_dev = dest_node["device_coord"]
        d_inf = dest_node["graph_coord"]

        # 1. local connection lengths (device->infra) at start & at dest
        if USE_DEVICE_EDGE_GEOMETRY:
            s_local_len, s_local_coords = try_device_local_length(
                tuple(s_dev), tuple(s_inf), vertices, device_edges,
                device_vertex_index_cache, device_vertex_xyz_cache
            )
            d_local_len, d_local_coords = try_device_local_length(
                tuple(d_dev), tuple(d_inf), vertices, device_edges,
                device_vertex_index_cache, device_vertex_xyz_cache
            )
        else:
            s_local_len, s_local_coords = None, None
            d_local_len, d_local_coords = None, None

        if s_local_len is None:
            s_local_len = dist3(s_dev, s_inf)
            s_local_coords = [tuple(s_dev), tuple(s_inf)]
        if d_local_len is None:
            d_local_len = dist3(d_inf, d_dev)
            d_local_coords = [tuple(d_inf), tuple(d_dev)]

        # 2. shortest path along infra graph between s_inf vertex and d_inf vertex
        wire = dijkstra_infra(graph, start_node["graph_vertex"], dest_node["graph_vertex"])
        if not wire:
            log("WARN no infra path leg {} (start eid={})".format(leg_index, start_eid))
            results.append({
                "start_index": leg_index,
                "element_id": start_eid,
                "length": None,
                "vertex_path_xyz": [],
                "mapped_distance": start_node["map_dist"]
            })
            continue

        infra_coords, infra_len = expand_wire_vertices(wire)
        if infra_len is None:
            log("WARN empty infra path leg {} (start eid={})".format(leg_index, start_eid))
            results.append({
                "start_index": leg_index,
                "element_id": start_eid,
                "length": None,
                "vertex_path_xyz": [],
                "mapped_distance": start_node["map_dist"]
            })
            continue

        # 3. build final coordinate path:
        # device start -> (device->infra local chain without repeating infra first point) -> infra path (without duplicating first) -> (infra->device local) -> device end
        full_coords=[]

        # start device
        full_coords.append(tuple(s_dev))

        # remove first duplicate in s_local_coords if it equals s_dev
        sc_seq = s_local_coords[:]
        if sc_seq and sc_seq[0] == tuple(s_dev):
            sc_seq = sc_seq[1:]
        # also drop last if equals first infra coord in infra path (to avoid duplicates)
        if sc_seq and infra_coords and sc_seq[-1] == infra_coords[0]:
            pass
        full_coords.extend(sc_seq)

        # add infra path (drop first if already in list)
        if full_coords and infra_coords and full_coords[-1] == infra_coords[0]:
            full_coords.extend(infra_coords[1:])
        else:
            full_coords.extend(infra_coords)

        # dest local connection (from infra to device end)
        dc_seq = d_local_coords[:]
        # remove first if it equals last infra coord
        if dc_seq and full_coords and dc_seq[0] == full_coords[-1]:
            dc_seq = dc_seq[1:]
        # ensure final device point present
        # last element of dc_seq should be device dest
        full_coords.extend(dc_seq)

        # 4. compute total length (local + infra)
        total_len = s_local_len + infra_len + d_local_len

        results.append({
            "start_index": leg_index,
            "element_id": start_eid,
            "length": total_len,
            "vertex_path_xyz": [list(c) for c in full_coords],
            "mapped_distance": start_node["map_dist"],
            "device_local_start_len": s_local_len,
            "device_local_end_len": d_local_len,
            "infra_len": infra_len
        })
        success += 1

    out = {
        "meta":{
            "version": VERSION,
            "infra_tolerance": INFRA_TOLERANCE,
            "map_tolerance": MAP_TOLERANCE,
            "use_device_edge_geometry": USE_DEVICE_EDGE_GEOMETRY,
            "vertices_input": len(vertices),
            "infra_edges": len(infra_edges),
            "device_edges": len(device_edges),
            "graph_vertices": len(gcoords),
            "chain_points": len(chain_nodes),
            "legs_total": len(legs),
            "paths_success": success,
            "paths_failed": len(legs)-success,
            "duration_sec": time.time()-t0
        },
        "results": results
    }

    out_path = os.path.join(script_dir,"topologic_results.json")
    with open(out_path,"w") as f:
        json.dump(out,f,indent=2)
    log("Results written: {}".format(out_path))
    log("Summary: {} success / {} legs".format(success, len(legs)))

if __name__ == "__main__":
    main()