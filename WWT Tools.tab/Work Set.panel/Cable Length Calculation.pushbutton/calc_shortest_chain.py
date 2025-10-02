#! python3
# calc_shortest_chain.py (with robust fallback Dijkstra + full original path retention)

import os, sys, json, math, time
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Graph import Graph
from topologicpy.Wire import Wire
from topologicpy.Dictionary import Dictionary

# ---------------- CONFIG ----------------
TOLERANCE = 1e-6            # Small tolerance
MAP_TOLERANCE = 5
DEVICE_EDGE_PENALTY_FACTOR = 1e-4  # NOTE: <1 makes device edges CHEAPER (reward). Set >1 to penalize.
PRESERVE_ALL_EDGES = True          # If True skip SelfMerge (may fragment Topologic graph)
INDEX_MATCH_TOL = 1e-8
LOG_PREFIX = "[CALC]"
USE_FALLBACK_IF_TOPOLOGIC_FAILS = True
# ----------------------------------------

def log(msg): print("{} {}".format(LOG_PREFIX, msg))

def dist3(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def nearest_vertex_index(pt, coords):
    md=float('inf'); mi=None
    for i,c in enumerate(coords):
        d=dist3(pt,c)
        if d<md: md=d; mi=i
    return mi,md

def map_coord_to_index(pt, original_vertices, tol=INDEX_MATCH_TOL):
    px,py,pz=pt
    for i,(x,y,z) in enumerate(original_vertices):
        if abs(x-px)<=tol and abs(y-py)<=tol and abs(z-pz)<=tol:
            return i
    idx,_=nearest_vertex_index(pt, original_vertices)
    return idx

# ---------------- Fallback Dijkstra ----------------
def dijkstra_path(start_idx, end_idx, vertices, infra_edges, device_edges, penalty_factor):
    # Build adjacency
    infra_set = set(tuple(sorted(e)) for e in infra_edges)
    device_set = set(tuple(sorted(e)) for e in device_edges)
    adj = [[] for _ in range(len(vertices))]
    for a,b in infra_set.union(device_set):
        va = vertices[a]; vb = vertices[b]
        length = dist3(va, vb)
        if (a,b) in device_set or (b,a) in device_set:
            cost = length * penalty_factor
        else:
            cost = length
        adj[a].append((b, cost, length))
        adj[b].append((a, cost, length))

    import heapq
    heap = [(0.0, start_idx, -1)]  # (cost, vertex, parent)
    visited = {}
    parent = {}
    edge_len_acc = {}

    while heap:
        c, v, p = heapq.heappop(heap)
        if v in visited: continue
        visited[v] = c
        parent[v] = p
        if v == end_idx: break
        for nb, wcost, wlen in adj[v]:
            if nb in visited: continue
            nc = c + wcost
            heapq.heappush(heap, (nc, nb, v))

    if end_idx not in visited:
        return None, None, None  # no path

    # Reconstruct indices path (original vertex indices)
    path_indices = []
    cur = end_idx
    while cur != -1:
        path_indices.append(cur)
        cur = parent.get(cur, -1)
    path_indices.reverse()

    # Compute true length (sum of geometric lengths)
    true_len = 0.0
    for i in range(len(path_indices)-1):
        a = vertices[path_indices[i]]
        b = vertices[path_indices[i+1]]
        true_len += dist3(a,b)

    # Also produce raw cost length if needed (not strictly required)
    total_cost = visited[end_idx]

    return path_indices, true_len, total_cost

def main():
    t0=time.time()
    script_dir=os.path.dirname(os.path.abspath(__file__))
    json_path=os.path.join(script_dir,"topologic.JSON")
    if not os.path.isfile(json_path):
        log("ERROR missing topologic.JSON"); sys.exit(1)
    data=json.load(open(json_path,"r"))

    vertices     = data.get("vertices",[])
    infra_edges  = data.get("infra_edges",[])
    device_edges = data.get("device_edges",[])
    combined     = data.get("edges",[])
    starts       = data.get("start_points",[])
    end_pt       = data.get("end_point",None)

    if infra_edges and device_edges:
        edges_source = (infra_edges, device_edges)
    else:
        edges_source = (combined, [])

    log("Vertices:{} InfraEdges:{} DeviceEdges:{} TotalEdges:{} ChainPoints:{}".format(
        len(vertices), len(infra_edges), len(device_edges), len(combined), len(starts)
    ))

    # ---------- Build Topologic graph (may be fragmented if PRESERVE_ALL_EDGES=True) ----------
    topo_vertices = [Vertex.ByCoordinates(*v) for v in vertices]

    def make_edge(i,j,cat):
        v1=topo_vertices[i]; v2=topo_vertices[j]
        e=Edge.ByVertices(v1,v2)
        length=dist3(vertices[i],vertices[j])
        # cost dictionary
        cost=length*(DEVICE_EDGE_PENALTY_FACTOR if cat=="JUMPER" else 1.0)
        d=Dictionary.ByKeysValues(["category","length","cost"],[cat,length,cost])
        try: e.SetDictionary(d)
        except: pass
        return e

    edge_objs=[]
    for i,j in edges_source[0]: edge_objs.append(make_edge(i,j,"INFRA"))
    for i,j in edges_source[1]: edge_objs.append(make_edge(i,j,"JUMPER"))

    cluster = Cluster.ByTopologies(*edge_objs)
    if PRESERVE_ALL_EDGES:
        merged = cluster
    else:
        merged = Topology.SelfMerge(cluster, tolerance=TOLERANCE)

    graph = Graph.ByTopology(merged, tolerance=TOLERANCE)
    graph_vertices = Graph.Vertices(graph)
    topo_coords = [v.Coordinates() for v in graph_vertices]
    log("Topologic graph vertices: {} (PRESERVE_ALL_EDGES={})".format(len(topo_coords), PRESERVE_ALL_EDGES))

    # Map starts to Topologic graph (for topologic attempt)
    mapped_points=[]
    for sp in starts:
        coord=sp.get("point"); eid=sp.get("element_id"); seq=sp.get("seq_index")
        if not (isinstance(coord,list) and len(coord)==3): continue
        gidx, d = nearest_vertex_index(coord, topo_coords)
        if d>MAP_TOLERANCE:
            log("WARN start mapping element {} distance {:.6f}".format(eid,d))
        mapped_points.append({
            "element_id":eid,
            "seq_index":seq,
            "coord":coord,
            "graph_vertex": graph_vertices[gidx],
            "graph_map_distance": d
        })

    if not (isinstance(end_pt,list) and len(end_pt)==3):
        log("ERROR invalid end point"); sys.exit(1)
    end_gidx, end_d = nearest_vertex_index(end_pt, topo_coords)
    if end_d>MAP_TOLERANCE:
        log("WARN end mapping distance {:.6f}".format(end_d))
    end_gv = graph_vertices[end_gidx]

    # Also map starts & end to ORIGINAL (unmerged) vertex list for fallback Dijkstra
    def map_to_original(pt):
        oi, od = nearest_vertex_index(pt, vertices)
        return oi, od
    original_mapped = []
    for mp in mapped_points:
        oi, od = map_to_original(mp["coord"])
        mp["orig_vertex_index"] = oi
        mp["orig_map_distance"] = od
        original_mapped.append(mp)
    end_orig_idx, end_orig_d = map_to_original(end_pt)

    chain_list = mapped_points + [{
        "element_id": None,
        "seq_index": None,
        "coord": end_pt,
        "graph_vertex": end_gv,
        "graph_map_distance": end_d,
        "orig_vertex_index": end_orig_idx,
        "orig_map_distance": end_orig_d
    }]

    results=[]
    success=0

    for i in range(len(chain_list)-1):
        start_node = chain_list[i]
        dest_node  = chain_list[i+1]
        eid = start_node["element_id"]
        seq = start_node["seq_index"]

        # ---------- Attempt Topologic shortest path ----------
        path_method = None
        topologic_key_xyz = []
        topologic_full_xyz = []
        length_key = None
        length_full = None
        used_costs_from = None

        wire = None
        try:
            wire = Graph.ShortestPath(graph,
                                      start_node["graph_vertex"],
                                      dest_node["graph_vertex"],
                                      edgeKey="cost",
                                      tolerance=TOLERANCE)
            if wire is None:
                log("INFO leg {} element {}: Topologic returned None".format(i, eid))
        except Exception as ex:
            log("INFO leg {} element {}: Topologic exception {}".format(i,eid,ex))
            wire = None

        if wire:
            path_method = "topologic"
            # Simplified vertices
            try:
                key_vertices = Wire.Vertices(wire)
                topologic_key_xyz = [v.Coordinates() for v in key_vertices]
            except:
                topologic_key_xyz = []

            # Edge sequence
            try:
                edges_seq = Wire.Edges(wire)
            except:
                edges_seq = []

            # Build full coords from edges
            full_coords=[]
            for e in edges_seq:
                try:
                    ev = Edge.Vertices(e)
                except:
                    ev=[]
                if not ev: continue
                c0=ev[0].Coordinates(); c1=ev[-1].Coordinates()
                if not full_coords: full_coords.append(c0)
                if full_coords[-1] != c0:
                    full_coords.append(c0)
                full_coords.append(c1)
            # Deduplicate consecutive
            cleaned=[]
            for c in full_coords:
                if not cleaned or cleaned[-1]!=c:
                    cleaned.append(c)
            topologic_full_xyz = cleaned

            # Compute lengths
            if topologic_key_xyz:
                lk=0.0
                for jj in range(len(topologic_key_xyz)-1):
                    lk += dist3(topologic_key_xyz[jj], topologic_key_xyz[jj+1])
                length_key = lk
            if edges_seq:
                lf=0.0
                for e in edges_seq:
                    # Try dictionary length
                    try:
                        dct = e.Dictionary()
                        if dct:
                            keys=dct.Keys(); vals=dct.Values()
                            if "length" in keys:
                                lf+=vals[keys.index("length")]
                                continue
                    except: pass
                    try:
                        ev = Edge.Vertices(e)
                        if len(ev)==2:
                            cA=ev[0].Coordinates(); cB=ev[1].Coordinates()
                            lf+=dist3(cA,cB)
                    except: pass
                length_full = lf
            used_costs_from="topologic"

        # ---------- Fallback Dijkstra if needed ----------
        final_indices = []
        final_coords  = []
        final_length_exact = None
        fallback_used = False

        if (wire is None or length_full is None) and USE_FALLBACK_IF_TOPOLOGIC_FAILS:
            s_orig = start_node["orig_vertex_index"]
            d_orig = dest_node["orig_vertex_index"]
            path_indices, true_len, total_cost = dijkstra_path(
                s_orig, d_orig, vertices, infra_edges, device_edges, DEVICE_EDGE_PENALTY_FACTOR
            )
            if path_indices:
                path_method = path_method or "fallback_dijkstra"
                fallback_used = True
                final_indices = path_indices
                final_coords  = [vertices[idx] for idx in path_indices]
                final_length_exact = true_len
                used_costs_from = used_costs_from or "dijkstra"
            else:
                path_method = path_method or "none"

        # If topologic succeeded AND we did not fallback, set final path to topologic (but map coords to original indices)
        if wire and not fallback_used:
            final_coords = topologic_full_xyz if topologic_full_xyz else topologic_key_xyz
            final_indices = [map_coord_to_index(c, vertices) for c in final_coords]
            final_length_exact = length_full if length_full is not None else length_key

        succeeded = final_length_exact is not None
        if succeeded: success += 1

        results.append({
            "start_index": i,
            "element_id": eid,
            "seq_index": seq,
            "path_method": path_method,
            "used_costs_from": used_costs_from,
            # Topologic (may be empty if fallback only)
            "topologic_length_key": length_key,
            "topologic_length_full": length_full,
            "topologic_key_vertices_path_xyz": topologic_key_xyz,
            "topologic_full_edge_path_xyz": topologic_full_xyz,
            # Final chosen path (ALWAYS prefer fallback or topologic full)
            "final_vertex_indices_path": final_indices,
            "final_vertex_coords_path": final_coords,
            "final_length_exact": final_length_exact,
            # Legacy fields (kept for compatibility – if topologic failed they remain null)
            "length": length_key,
            "length_key": length_key,
            "length_full": length_full,
            "mapped_distance_start_graph": start_node.get("graph_map_distance"),
            "mapped_distance_start_original": start_node.get("orig_map_distance"),
            "mapped_distance_end_graph": dest_node.get("graph_map_distance"),
            "mapped_distance_end_original": dest_node.get("orig_map_distance"),
            "fallback_used": fallback_used
        })

    out={
        "meta":{
            "version":"CHAIN-1.0.0",
            "tolerance":TOLERANCE,
            "map_tolerance":MAP_TOLERANCE,
            "device_edge_penalty_factor":DEVICE_EDGE_PENALTY_FACTOR,
            "vertices_input":len(vertices),
            "infra_edges":len(infra_edges),
            "device_edges":len(device_edges),
            "graph_vertices":len(topo_coords),
            "chain_points":len(chain_list),
            "mapped_chain":len(mapped_points),
            "paths_success":success,
            "paths_failed":len(chain_list)-1-success,
            "duration_sec":time.time()-t0,
            "preserve_all_edges":PRESERVE_ALL_EDGES,
            "fallback_used_any": any(r["fallback_used"] for r in results)
        },
        "results":results
    }
    out_path=os.path.join(script_dir,"topologic_results.json")
    json.dump(out, open(out_path,"w"), indent=2)
    log("Results written: {}".format(out_path))
    log("Summary: {} success / {} total (fallback_used_any={})".format(
        success, len(chain_list)-1, out["meta"]["fallback_used_any"])
    )

if __name__ == "__main__":
    main()