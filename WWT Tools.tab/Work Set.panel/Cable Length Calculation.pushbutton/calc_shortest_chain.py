#! python3
# calc_shortest_chain.py
# Version: 8.0.0 (Minimal chain: pure Topologic Graph.ShortestPath with edgeKey='length')
#
# Steps:
#   1. Read topologic.JSON (expects "vertices", "edges", "start_points", "end_point")
#   2. Create Vertex objects in original order.
#   3. Create Edge objects for EVERY combined edge; assign dictionary with key 'length'.
#   4. (Optional) SelfMerge OFF by default to prevent corner loss.
#   5. Graph.ByTopology
#   6. For each consecutive start pair + last start to end: map each device/end to nearest graph vertex.
#   7. Compute Graph.ShortestPath(graph, vA, vB, edgeKey='length').
#   8. Expand wire edges => ordered coordinates, sum edge 'length' dictionary values.
#
# No device jumper penalties, no candidate expansions, no added synthetic points.
# Output lengths should align with the exact sum of the edges Topologic selects.
#
# If a device coordinate is not exactly a graph vertex we map to the nearest one;
# only that graph vertex appears in the path (as per your instruction to "use only the 2 points start / end").
#
# CONFIG toggles below allow minor adjustments.

import os, sys, json, math, time
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology
from topologicpy.Graph import Graph
from topologicpy.Wire import Wire
from topologicpy.Dictionary import Dictionary

VERSION = "8.0.0"

# ---------------- CONFIG ----------------
TOLERANCE          = 1e-9     # Very small to avoid merging if SELF_MERGE=False
SELF_MERGE         = True    # Set True if you must collapse coincident duplicates (may shorten paths)
MAP_TOLERANCE      = 1.0      # Warn if device->nearest vertex distance exceeds this
INCLUDE_DEVICE_COORDS = False # If True prepend/append original device/end coords to path list (for reference)
LOG_PREFIX         = "[CALC]"
# ----------------------------------------

def log(msg):
    print("{} {}".format(LOG_PREFIX, msg))

def dist3(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def nearest_vertex_index(pt, coords):
    bd = float('inf'); bi = None
    for i,c in enumerate(coords):
        d = dist3(pt,c)
        if d < bd:
            bd = d; bi = i
    return bi, bd

def build_graph(vertices_raw, edges_raw):
    verts = [Vertex.ByCoordinates(*v) for v in vertices_raw]
    edge_objs=[]
    for (i,j) in edges_raw:
        v1=verts[i]; v2=verts[j]
        length = dist3(vertices_raw[i], vertices_raw[j])
        e = Edge.ByVertices(v1,v2)
        d = Dictionary.ByKeysValues(["length"], [length])  # ONLY the 'length' key
        try: e.SetDictionary(d)
        except: pass
        edge_objs.append(e)
    topo = Cluster.ByTopologies(*edge_objs)
    if SELF_MERGE:
        topo = Topology.SelfMerge(topo, tolerance=TOLERANCE)
    graph = Graph.ByTopology(topo, tolerance=TOLERANCE)
    gverts = Graph.Vertices(graph)
    gcoords = [v.Coordinates() for v in gverts]
    return graph, gverts, gcoords

def expand_wire_edge_sequence(wire):
    """Return (coords_list, sum_length_from_edge_dicts) using the 'length' key."""
    if not wire:
        return [], None
    try:
        edges = Wire.Edges(wire)
    except:
        edges = []
    if not edges:
        # fallback: vertices only
        try:
            vs = Wire.Vertices(wire)
            coords = [v.Coordinates() for v in vs]
            length = sum(dist3(coords[i], coords[i+1]) for i in range(len(coords)-1))
            return coords, length
        except:
            return [], None
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
        # Dictionary length
        length_val=None
        try:
            dct=e.Dictionary()
            if dct:
                keys=dct.Keys(); vals=dct.Values()
                if "length" in keys:
                    length_val = vals[keys.index("length")]
        except:
            pass
        if length_val is None:
            # fallback geometric
            length_val = dist3(c0,c1)
        total += length_val
    # Deduplicate consecutives
    clean=[]
    for c in coords:
        if not clean or clean[-1]!=c:
            clean.append(c)
    return clean,total

def main():
    t0=time.time()
    script_dir=os.path.dirname(os.path.abspath(__file__))
    json_path=os.path.join(script_dir,"topologic.JSON")
    if not os.path.isfile(json_path):
        log("ERROR: topologic.JSON not found.")
        sys.exit(1)

    data=json.load(open(json_path,"r"))
    vertices = data.get("vertices",[])
    combined  = data.get("edges",[])
    starts    = data.get("start_points",[])
    end_pt    = data.get("end_point",None)

    if not vertices or not combined:
        log("ERROR: vertices or edges missing.")
        sys.exit(1)

    if len(starts)<2 and not end_pt:
        log("ERROR: Need at least 2 start points or end point for a leg.")
        sys.exit(1)

    log("Vertices:{} Edges:{} Starts:{} EndPoint:{}".format(
        len(vertices), len(combined), len(starts),
        "Yes" if isinstance(end_pt,list) and len(end_pt)==3 else "No"
    ))

    graph, gverts, gcoords = build_graph(vertices, combined)
    log("Graph vertices after build (SELF_MERGE={}): {}".format(SELF_MERGE, len(gcoords)))

    # Build chain device list: start points in order + end
    chain_devices = starts[:]
    if isinstance(end_pt,list) and len(end_pt)==3:
        chain_devices.append({"element_id": None, "point": end_pt})

    results=[]
    success=0
    total_legs=len(chain_devices)-1

    for i in range(total_legs):
        A = chain_devices[i]
        B = chain_devices[i+1]
        A_pt = A.get("point")
        B_pt = B.get("point")
        A_id = A.get("element_id")

        # Map each to nearest graph vertex
        ai, ad = nearest_vertex_index(A_pt, gcoords)
        bi, bd = nearest_vertex_index(B_pt, gcoords)

        if ad>MAP_TOLERANCE:
            log("WARN start element {} leg {} mapping distance {:.6f}".format(A_id, i, ad))
        if bd>MAP_TOLERANCE and B.get("element_id") is not None:
            log("WARN dest element {} leg {} mapping distance {:.6f}".format(B.get("element_id"), i, bd))

        vA = gverts[ai]
        vB = gverts[bi]

        # Run Graph.ShortestPath using edgeKey='length'
        wire=None
        try:
            wire = Graph.ShortestPath(graph, vA, vB, edgeKey="length", tolerance=TOLERANCE)
        except Exception as ex:
            log("ERROR Graph.ShortestPath leg {}: {}".format(i, ex))
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

        path_coords, path_len = expand_wire_edge_sequence(wire)

        # If requested, include original device coords at ends (even if not identical to snapped graph verts)
        if INCLUDE_DEVICE_COORDS and path_coords:
            # Replace first & last with original device coordinates for reference
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

    out = {
        "meta":{
            "version": VERSION,
            "tolerance_used": TOLERANCE,
            "self_merge": SELF_MERGE,
            "map_tolerance": MAP_TOLERANCE,
            "include_device_coords": INCLUDE_DEVICE_COORDS,
            "vertices_input": len(vertices),
            "edges_input": len(combined),
            "graph_vertices": len(gcoords),
            "start_points": len(starts),
            "chain_points": len(chain_devices),
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
    log("Summary: {} success / {} legs".format(success,total_legs))

if __name__ == "__main__":
    main()