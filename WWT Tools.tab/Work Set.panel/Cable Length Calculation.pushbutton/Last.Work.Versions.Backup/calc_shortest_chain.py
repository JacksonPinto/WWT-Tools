#! python3
# calc_shortest_chain.py
# Version: 7.0.0 (Infrastructure-only, candidate mapping, detour elimination)
#
# Daisy-chain: (Start0->Start1) (Start1->Start2) ... (Start{n-2}->Start{n-1}) (Start{n-1}->End)
#
# GOALS:
#   * Use only infrastructure edges (infra_edges) for interior path.
#   * Do not introduce artificial intermediate vertices that cause detours.
#   * Enumerate multiple candidate infra vertices around each device and choose the pair
#     that minimizes the infra shortest path length for the leg.
#   * Optionally snap device coordinates directly to the chosen infra vertex (zero local segment).
#   * Output either full infra path (expanded) or just [device_start, device_end].
#
# OUTPUT:
#   topologic_results.json with:
#     - For each leg: chosen start/end infra vertex indices, candidate counts, path length.
#     - vertex_path_xyz: start device, full infra path (if OUTPUT_ONLY_DEVICE_AND_END=False), end device.
#
# DEPENDENCY: topologicpy installed in external Python interpreter invoked by main script.

import os, sys, json, math, time
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology
from topologicpy.Graph import Graph
from topologicpy.Wire import Wire

VERSION = "7.0.0"

# ---------------- CONFIG ----------------
INFRA_TOLERANCE = 5e-4              # Merge tolerance for infra graph
MAP_TOLERANCE   = 1.0               # Warn if mapping distance > this
CANDIDATE_SEARCH_RADIUS_FT = 1.0    # Radius to collect candidate infra vertices around each device
MAX_START_CANDIDATES = 8            # Safety cap
MAX_END_CANDIDATES   = 8
SNAP_DEVICE_TO_VERTEX = True        # If True: vertex_path_xyz uses exact infra vertex coords for device points
OUTPUT_ONLY_DEVICE_AND_END = False  # If True: only [device_start, device_end] in vertex_path_xyz
STRICT_INFRA_LENGTH = True          # Compute length from actual edge segments in Wire (if possible)
LOG_VERBOSE = True
LOG_PREFIX = "[CALC]"
# ----------------------------------------

def log(msg, verbose=False):
    if verbose and not LOG_VERBOSE:
        return
    print("{} {}".format(LOG_PREFIX, msg))

def dist3(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def nearest_vertex_index(pt, coords):
    best=None; bd=1e9
    for i,c in enumerate(coords):
        d=dist3(pt,c)
        if d<bd:
            bd=d; best=i
    return best, bd

def find_candidates(pt, coords, radius, limit):
    """Return list of (index, distance) for coords within radius, sorted by distance, truncated by limit.
       Always include the nearest even if outside radius (fallback)."""
    idx, d0 = nearest_vertex_index(pt, coords)
    cand=[]
    for i,c in enumerate(coords):
        d=dist3(pt,c)
        if d<=radius: cand.append((i,d))
    if not cand:  # no one inside radius, use nearest
        cand=[(idx,d0)]
    cand.sort(key=lambda x:x[1])
    if len(cand)>limit: cand=cand[:limit]
    return cand

def build_infra_graph(vertices, infra_edges):
    topo_vertices=[Vertex.ByCoordinates(*v) for v in vertices]
    edge_objs=[]
    for (i,j) in infra_edges:
        v1=topo_vertices[i]; v2=topo_vertices[j]
        try:
            e=Edge.ByVertices(v1,v2)
            edge_objs.append(e)
        except:
            pass
    cluster=Cluster.ByTopologies(*edge_objs)
    merged=Topology.SelfMerge(cluster,tolerance=INFRA_TOLERANCE)
    graph=Graph.ByTopology(merged,tolerance=INFRA_TOLERANCE)
    graph_vertices=Graph.Vertices(graph)
    coords=[gv.Coordinates() for gv in graph_vertices]
    return graph, graph_vertices, coords

def wire_length_and_coords(wire, strict=True):
    if not wire:
        return None, []
    if strict:
        # Use edges so we sum true segment lengths
        try:
            edges=Wire.Edges(wire)
        except:
            edges=[]
        if edges:
            coords=[]
            total=0.0
            for e in edges:
                try:
                    ev=Edge.Vertices(e)
                except:
                    ev=[]
                if len(ev)<2: continue
                c0=ev[0].Coordinates(); c1=ev[-1].Coordinates()
                if not coords:
                    coords.append(c0)
                elif coords[-1]!=c0:
                    coords.append(c0)
                coords.append(c1)
                total+=dist3(c0,c1)
            # Deduplicate consecutives
            clean=[]
            for c in coords:
                if not clean or clean[-1]!=c:
                    clean.append(c)
            return total, clean
    # Fallback simplified
    try:
        vs=Wire.Vertices(wire)
        coords=[v.Coordinates() for v in vs]
        total=sum(dist3(coords[i],coords[i+1]) for i in range(len(coords)-1))
        return total, coords
    except:
        return None, []

def shortest_path(graph, gv_start, gv_end):
    try:
        return Graph.ShortestPath(graph, gv_start, gv_end, tolerance=INFRA_TOLERANCE)
    except:
        return None

def main():
    t0=time.time()
    script_dir=os.path.dirname(os.path.abspath(__file__))
    json_path=os.path.join(script_dir,"topologic.JSON")
    if not os.path.isfile(json_path):
        log("ERROR missing topologic.JSON"); sys.exit(1)
    data=json.load(open(json_path,"r"))

    vertices=data.get("vertices",[])
    infra_edges=data.get("infra_edges",[])
    device_edges=data.get("device_edges",[])  # not used interior
    starts=data.get("start_points",[])
    end_pt=data.get("end_point",None)

    log("Input: vertices={} infra_edges={} device_edges={} starts={}".format(
        len(vertices), len(infra_edges), len(device_edges), len(starts)
    ))

    if not vertices or not infra_edges or len(starts)<2:
        log("ERROR need at least 2 start points and infra graph.")
        sys.exit(1)

    graph, gverts, gcoords = build_infra_graph(vertices, infra_edges)
    log("Infra graph vertices: {}".format(len(gcoords)))

    # Map device start points to candidate infra vertices
    device_chain = starts[:]  # preserve order
    if isinstance(end_pt,list) and len(end_pt)==3:
        device_chain = device_chain + [{"element_id": None, "point": end_pt}]
    else:
        log("WARN no end point; chain ends with last start.")

    # Precompute candidate lists per device
    device_candidates=[]
    for idx, sp in enumerate(device_chain):
        coord=sp.get("point")
        eid=sp.get("element_id")
        if not (isinstance(coord,list) and len(coord)==3):
            log("WARN invalid coord at chain index {}".format(idx))
            device_candidates.append([])
            continue
        cand=find_candidates(coord, gcoords, CANDIDATE_SEARCH_RADIUS_FT,
                             MAX_START_CANDIDATES if idx < len(device_chain)-1 else MAX_END_CANDIDATES)
        # Always sort by distance
        device_candidates.append(cand)
        if LOG_VERBOSE:
            log("Device {} (eid={}): {} candidates".format(idx, eid, len(cand)), verbose=True)

    results=[]
    success=0

    # Iterate through legs
    for leg_index in range(len(device_chain)-1):
        origin=device_chain[leg_index]
        dest  =device_chain[leg_index+1]
        origin_coord=origin["point"]; dest_coord=dest["point"]
        origin_eid=origin["element_id"]

        origin_cands=device_candidates[leg_index]
        dest_cands  =device_candidates[leg_index+1]

        best_len=None
        best_wire=None
        best_pair=None
        best_coords=None

        # Try all candidate pairs
        for (oi, odist) in origin_cands:
            gv_start=gverts[oi]
            for (di, ddist) in dest_cands:
                gv_end=gverts[di]
                wire=shortest_path(graph, gv_start, gv_end)
                if not wire:
                    continue
                wlen, wcoords = wire_length_and_coords(wire, strict=STRICT_INFRA_LENGTH)
                if wlen is None:
                    continue
                total_leg_len=wlen  # (No local device->vertex segments unless SNAP_DEVICE_TO_VERTEX False)
                # Add local offsets if not snapping:
                if not SNAP_DEVICE_TO_VERTEX:
                    total_leg_len += dist3(origin_coord, wcoords[0]) + dist3(wcoords[-1], dest_coord)

                if (best_len is None) or (total_leg_len < best_len):
                    best_len=total_leg_len
                    best_wire=wire
                    best_pair=(oi, di, odist, ddist)
                    best_coords=wcoords

        if best_len is None:
            log("WARN leg {} (eid {}) no path found among candidates".format(leg_index, origin_eid))
            results.append({
                "start_index": leg_index,
                "element_id": origin_eid,
                "length": None,
                "vertex_path_xyz": [],
                "mapped_start_candidates": len(origin_cands),
                "mapped_end_candidates": len(dest_cands)
            })
            continue

        # Build output coordinate path
        if OUTPUT_ONLY_DEVICE_AND_END:
            if SNAP_DEVICE_TO_VERTEX:
                # Snap to start/end of chosen graph path
                start_snap = best_coords[0]
                end_snap   = best_coords[-1]
                path_xyz=[list(start_snap), list(end_snap)]
            else:
                path_xyz=[origin_coord, dest_coord]
        else:
            # Full infra path
            path_xyz=[]
            if SNAP_DEVICE_TO_VERTEX:
                # Use origin snapped to first infra coord
                path_xyz.append(list(best_coords[0]))
            else:
                path_xyz.append(origin_coord)
                # ensure not duplicating
                if best_coords and origin_coord!=best_coords[0]:
                    path_xyz.append(list(best_coords[0]))
            # interior
            for c in best_coords[1:-1]:
                path_xyz.append(list(c))
            # destination
            if SNAP_DEVICE_TO_VERTEX:
                path_xyz.append(list(best_coords[-1]))
            else:
                if best_coords and dest_coord!=best_coords[-1]:
                    path_xyz.append(list(best_coords[-1]))
                path_xyz.append(dest_coord)

        if LOG_VERBOSE:
            oi, di, od, dd = best_pair
            log("Leg {} eid {}: chosen startV={} endV={} infraLen={:.6f} totalLen={:.6f} startMapDist={:.4f} endMapDist={:.4f}"
                .format(leg_index, origin_eid, oi, di, best_len if SNAP_DEVICE_TO_VERTEX else best_len - (dist3(origin_coord,best_coords[0])+dist3(best_coords[-1],dest_coord)),
                        best_len, od, dd), verbose=True)

        results.append({
            "start_index": leg_index,
            "element_id": origin_eid,
            "length": best_len,
            "vertex_path_xyz": path_xyz,
            "mapped_start_candidates": len(origin_cands),
            "mapped_end_candidates": len(dest_cands),
            "chosen_start_vertex_index": best_pair[0],
            "chosen_end_vertex_index": best_pair[1],
            "start_map_distance": best_pair[2],
            "end_map_distance": best_pair[3],
            "snapped": SNAP_DEVICE_TO_VERTEX,
            "output_only_devices": OUTPUT_ONLY_DEVICE_AND_END
        })
        success+=1

    out={
        "meta":{
            "version": VERSION,
            "infra_tolerance": INFRA_TOLERANCE,
            "map_tolerance": MAP_TOLERANCE,
            "candidate_radius": CANDIDATE_SEARCH_RADIUS_FT,
            "max_start_candidates": MAX_START_CANDIDATES,
            "max_end_candidates": MAX_END_CANDIDATES,
            "snap_device_to_vertex": SNAP_DEVICE_TO_VERTEX,
            "output_only_device_and_end": OUTPUT_ONLY_DEVICE_AND_END,
            "strict_infra_length": STRICT_INFRA_LENGTH,
            "vertices_input": len(vertices),
            "infra_edges": len(infra_edges),
            "device_edges": len(device_edges),
            "graph_vertices": len(gcoords),
            "start_points": len(starts),
            "chain_points": len(device_chain),
            "legs_total": len(results),
            "paths_success": success,
            "paths_failed": len(results)-success,
            "duration_sec": time.time()-t0
        },
        "results": results
    }

    out_path=os.path.join(script_dir,"topologic_results.json")
    with open(out_path,"w") as f:
        json.dump(out,f,indent=2)
    log("Results written: {}".format(out_path))
    log("Summary: {} success / {} legs".format(success, len(results)))

if __name__ == "__main__":
    main()