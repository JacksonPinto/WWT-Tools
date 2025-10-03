#! python3
# calc_shortest_chain.py
# Version: 9.0.0 (Pure geometric shortest path – NO dictionaries)
#
# Description:
#   Daisy‑chain wiring calculation using ONLY the geometric length of edges.
#   No edge or vertex dictionaries are created. Graph.ShortestPath is called
#   WITHOUT edgeKey / vertexKey so Topologic uses raw geometric edge lengths.
#
# Chain Legs:
#   (Start0 -> Start1), (Start1 -> Start2), ..., (Start{n-2} -> Start{n-1}), (Start{n-1} -> End)
#
# Path Output:
#   vertex_path_xyz = list of coordinates (expanded along the chosen wire’s edge sequence).
#   Length = sum of geometric distances of each traversed edge (no chord shortcut).
#
# Assumptions:
#   - topologicpy is installed in the external CPython used by the main pyRevit script.
#   - topologic.JSON contains keys: vertices, edges (combined infra + device), start_points, end_point.
#   - You want the path strictly along the given edges (no extra points, no dictionaries).
#
# Configurable parameters below.

import os, sys, json, math, time
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology
from topologicpy.Graph import Graph
from topologicpy.Wire import Wire

VERSION = "9.0.0"

# ---------------- CONFIG ----------------
TOLERANCE       = 5e-4   # Small merge tolerance (match exporter MERGE_TOL scale)
SELF_MERGE      = True  # Set True if you need coincident vertices unified (can shorten paths)
MAP_TOLERANCE   = 1.0    # Warn if device/end maps farther than this to a graph vertex
INCLUDE_DEVICE_COORDS = False  # If True, first/last coords in path replaced by original device/end coords
LOG_PREFIX = "[CALC]"
# ----------------------------------------

def log(msg):
    print("{} {}".format(LOG_PREFIX, msg))

def dist3(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def nearest_vertex_index(pt, coords):
    best=None; bd=1e9
    for i,c in enumerate(coords):
        d=dist3(pt,c)
        if d<bd:
            bd=d; best=i
    return best, bd

def build_graph(vertices_raw, edges_raw):
    """Build a Topologic graph from raw vertices + edges without dictionaries."""
    topo_vertices = [Vertex.ByCoordinates(*v) for v in vertices_raw]
    edge_objs=[]
    for (i,j) in edges_raw:
        v1=topo_vertices[i]; v2=topo_vertices[j]
        try:
            e=Edge.ByVertices(v1,v2)
            edge_objs.append(e)
        except:
            pass
    cluster = Cluster.ByTopologies(*edge_objs)
    topo = Topology.SelfMerge(cluster, tolerance=TOLERANCE) if SELF_MERGE else cluster
    graph = Graph.ByTopology(topo, tolerance=TOLERANCE)
    gverts = Graph.Vertices(graph)
    gcoords = [v.Coordinates() for v in gverts]
    return graph, gverts, gcoords

def expand_wire_edges_geometric(wire):
    """
    Expand a wire to a coordinate list following its edge sequence
    and compute geometric length (sum of edge segment distances).
    """
    if not wire:
        return [], None
    try:
        edges = Wire.Edges(wire)
    except:
        edges=[]
    if edges:
        coords=[]
        total=0.0
        for e in edges:
            try:
                evs = Edge.Vertices(e)
            except:
                evs=[]
            if len(evs)<2:
                continue
            c0=evs[0].Coordinates(); c1=evs[-1].Coordinates()
            if not coords:
                coords.append(c0)
            elif coords[-1]!=c0:
                coords.append(c0)
            coords.append(c1)
            total += dist3(c0,c1)
        # Deduplicate consecutive duplicates
        clean=[]
        for c in coords:
            if not clean or clean[-1]!=c:
                clean.append(c)
        return clean, total
    # Fallback to vertices if edges list empty
    try:
        vs=Wire.Vertices(wire)
        coords=[v.Coordinates() for v in vs]
        length=sum(dist3(coords[i],coords[i+1]) for i in range(len(coords)-1))
        return coords,length
    except:
        return [], None

def main():
    t0=time.time()
    script_dir=os.path.dirname(os.path.abspath(__file__))
    json_path=os.path.join(script_dir,"topologic.JSON")
    if not os.path.isfile(json_path):
        log("ERROR: topologic.JSON not found.")
        sys.exit(1)

    data=json.load(open(json_path,"r"))
    vertices = data.get("vertices", [])
    combined  = data.get("edges", [])
    starts    = data.get("start_points", [])
    end_pt    = data.get("end_point", None)

    if not vertices or not combined:
        log("ERROR: Missing vertices or edges.")
        sys.exit(1)

    if len(starts)<2 and not end_pt:
        log("ERROR: Need chain of at least two start points or an end point.")
        sys.exit(1)

    log("Vertices:{} Edges:{} StartPoints:{} EndPoint:{}".format(
        len(vertices), len(combined), len(starts),
        "Yes" if isinstance(end_pt,list) and len(end_pt)==3 else "No"
    ))

    graph, gverts, gcoords = build_graph(vertices, combined)
    log("Graph vertices: {} (SELF_MERGE={})".format(len(gcoords), SELF_MERGE))

    # Build device chain: all starts + final end point
    chain = starts[:]
    if isinstance(end_pt,list) and len(end_pt)==3:
        chain.append({"element_id": None, "point": end_pt})

    results=[]
    success=0
    total_legs=len(chain)-1

    for i in range(total_legs):
        A = chain[i]
        B = chain[i+1]
        A_pt = A.get("point")
        B_pt = B.get("point")
        A_id = A.get("element_id")

        ai, ad = nearest_vertex_index(A_pt, gcoords)
        bi, bd = nearest_vertex_index(B_pt, gcoords)

        if ad>MAP_TOLERANCE:
            log("WARN leg {} start element {} mapping distance {:.6f}".format(i, A_id, ad))
        if bd>MAP_TOLERANCE and B.get("element_id") is not None:
            log("WARN leg {} dest element {} mapping distance {:.6f}".format(i, B.get("element_id"), bd))

        vA=gverts[ai]; vB=gverts[bi]

        # Pure geometric shortest path (no edgeKey / vertexKey)
        wire=None
        try:
            wire = Graph.ShortestPath(graph, vA, vB, tolerance=TOLERANCE)
        except Exception as ex:
            log("ERROR leg {} (eid {}) ShortestPath: {}".format(i, A_id, ex))
            wire=None

        if not wire:
            results.append({
                "start_index": i,
                "element_id": A_id,
                "length": None,
                "vertex_path_xyz": [],
                "start_map_distance": ad,
                "end_map_distance": bd
            })
            continue

        path_coords, path_len = expand_wire_edges_geometric(wire)

        if INCLUDE_DEVICE_COORDS and path_coords:
            # Replace endpoints with original device coordinates for reference only
            path_coords[0] = tuple(A_pt)
            path_coords[-1]= tuple(B_pt)

        results.append({
            "start_index": i,
            "element_id": A_id,
            "length": path_len,
            "vertex_path_xyz": [list(c) for c in path_coords],
            "start_map_distance": ad,
            "end_map_distance": bd
        })
        if path_len is not None:
            success+=1

    out={
        "meta":{
            "version": VERSION,
            "tolerance": TOLERANCE,
            "self_merge": SELF_MERGE,
            "map_tolerance": MAP_TOLERANCE,
            "include_device_coords": INCLUDE_DEVICE_COORDS,
            "vertices_input": len(vertices),
            "edges_input": len(combined),
            "graph_vertices": len(gcoords),
            "start_points": len(starts),
            "chain_points": len(chain),
            "legs_total": total_legs,
            "paths_success": success,
            "paths_failed": total_legs-success,
            "duration_sec": time.time()-t0
        },
        "results": results
    }

    out_path=os.path.join(script_dir,"topologic_results.json")
    with open(out_path,"w") as f:
        json.dump(out,f,indent=2)

    log("Results written: {}".format(out_path))
    log("Summary: {} success / {} legs".format(success, total_legs))

if __name__ == "__main__":
    main()