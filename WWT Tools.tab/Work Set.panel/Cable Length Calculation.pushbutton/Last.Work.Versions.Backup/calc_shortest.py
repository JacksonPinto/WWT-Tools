import json
import os
import math
import time
import sys
from collections import deque, defaultdict
from typing import List, Dict, Any, Tuple

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_JSON_INPUT_FILENAME = "topologic.JSON"
JSON_OUTPUT_FILENAME = "topologic_results.json"

TOLERANCE = 1e-6
SNAP_TOLERANCE = 1e-9

EDGE_WEIGHT_KEY = "w"
USE_TOPOLOGIC = True
USE_EDGE_WEIGHTS = True
RAW_USE_WEIGHTS = True
REQUIRE_MATCH = False
EXPORT_GRAPH_JSON = False
GRAPH_EXPORT_FILENAME = "topologic_script_direct.json"
OVERWRITE_GRAPH_EXPORT = True

ENABLE_CHAIN_FALLBACK = True         # existing chain accumulation (start->start)
INCLUDE_END_POINT_VERTEX = True      # append end point vertex if missing

# New fallback logic (mirrors chain script final fix)
ENABLE_VERTICAL_BRIDGING_FALLBACK = True
BRIDGING_XY_TOL = 1e-6
ENABLE_FINAL_STUB_FALLBACK = True    # creates synthetic tail path if unreachable
STUB_SENTINEL_EDGE_ID = -999         # marker for synthetic tail

DEBUG = ("--debug" in sys.argv) or (os.environ.get("DEBUG") == "1")
def dprint(*a):
    if DEBUG: print("[DEBUG]", *a)

# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

# ------------------------------------------------------------------
# Raw Graph
# ------------------------------------------------------------------
class RawGraph:
    def __init__(self, vertices: List[List[float]], edges: List[List[int]]):
        self.vertices = vertices
        self.edges = edges
        self.adj = [[] for _ in vertices]
        for eid,(u,v) in enumerate(edges):
            if 0 <= u < len(vertices) and 0 <= v < len(vertices):
                self.adj[u].append((v,eid))
                self.adj[v].append((u,eid))

    def nearest_vertex(self, c: Tuple[float,float,float]):
        x,y,z = c
        best=-1; best_d2=1e100
        for i,(vx,vy,vz) in enumerate(self.vertices):
            d2=(vx-x)**2+(vy-y)**2+(vz-z)**2
            if d2<best_d2:
                best_d2=d2; best=i
        return best, math.sqrt(best_d2)

    def edge_length(self, eid:int):
        u,v = self.edges[eid]
        a=self.vertices[u]; b=self.vertices[v]
        return math.dist(a,b)

    def shortest_path(self, s:int, t:int, weighted:bool):
        if s==t: return [s],[-1],0.0
        if weighted:
            import heapq
            dist=[float("inf")]*len(self.vertices)
            prev=[(-1,-1)]*len(self.vertices)
            dist[s]=0.0
            h=[(0.0,s)]
            while h:
                cd,u=heapq.heappop(h)
                if cd>dist[u]: continue
                if u==t: break
                for (v,eid) in self.adj[u]:
                    w=self.edge_length(eid)
                    nd=cd+w
                    if nd < dist[v]:
                        dist[v]=nd
                        prev[v]=(u,eid)
                        heapq.heappush(h,(nd,v))
            if dist[t]==float("inf"): return [],[],0.0
            path=[]; epath=[]
            cur=t
            while cur!=s:
                p,eid=prev[cur]
                path.append(cur); epath.append(eid)
                cur=p
            path.append(s); path.reverse(); epath.reverse()
            return path, epath, dist[t]
        else:
            q=deque([s]); prev={s:(-1,-1)}
            while q:
                u=q.popleft()
                if u==t: break
                for (v,eid) in self.adj[u]:
                    if v not in prev:
                        prev[v]=(u,eid)
                        q.append(v)
            if t not in prev: return [],[],0.0
            path=[]; epath=[]; cur=t
            while cur!=s:
                p,eid=prev[cur]
                path.append(cur); epath.append(eid)
                cur=p
            path.append(s); path.reverse(); epath.reverse()
            length=sum(self.edge_length(eid) for eid in epath)
            return path, epath, length

    def single_source_tree(self, s:int, weighted: bool):
        if weighted:
            import heapq
            dist=[float("inf")]*len(self.vertices)
            prev=[(-1,-1)]*len(self.vertices)
            dist[s]=0.0
            h=[(0.0,s)]
            while h:
                cd,u=heapq.heappop(h)
                if cd>dist[u]: continue
                for (v,eid) in self.adj[u]:
                    w=self.edge_length(eid)
                    nd=cd+w
                    if nd<dist[v]:
                        dist[v]=nd; prev[v]=(u,eid)
                        heapq.heappush(h,(nd,v))
            return dist, prev
        else:
            dist=[float("inf")]*len(self.vertices)
            prev=[(-1,-1)]*len(self.vertices)
            dist[s]=0.0
            q=deque([s])
            while q:
                u=q.popleft()
                for (v,eid) in self.adj[u]:
                    if dist[v]==float("inf"):
                        dist[v]=dist[u]+1
                        prev[v]=(u,eid)
                        q.append(v)
            return dist, prev

# ------------------------------------------------------------------
# Vertical bridging fallback (synthetic vertical edges)
# ------------------------------------------------------------------
def bridged_shortest_path(raw: RawGraph, start_idx: int, end_idx: int) -> Tuple[List[int], List[int], float]:
    n=len(raw.vertices)
    buckets=defaultdict(list)
    for i,(x,y,z) in enumerate(raw.vertices):
        key=(round(x/BRIDGING_XY_TOL), round(y/BRIDGING_XY_TOL))
        buckets[key].append(i)
    aug=[[] for _ in range(n)]
    for u in range(n):
        for (v,eid) in raw.adj[u]:
            w=raw.edge_length(eid)
            aug[u].append((v,eid,w,False))
    for grp in buckets.values():
        if len(grp)<2: continue
        grp_sorted=sorted(grp, key=lambda i: raw.vertices[i][2])
        for i in range(len(grp_sorted)-1):
            a=grp_sorted[i]; b=grp_sorted[i+1]
            dz=abs(raw.vertices[a][2]-raw.vertices[b][2])
            if dz==0: continue
            aug[a].append((b,-1,dz,True))
            aug[b].append((a,-1,dz,True))
    import heapq
    dist=[float("inf")]*n; prev=[(-1,-1)]*n
    dist[start_idx]=0.0; h=[(0.0,start_idx)]
    while h:
        cd,u=heapq.heappop(h)
        if cd>dist[u]: continue
        if u==end_idx: break
        for (v,eid,w,synthetic) in aug[u]:
            nd=cd+w
            if nd<dist[v]:
                dist[v]=nd; prev[v]=(u,eid)
                heapq.heappush(h,(nd,v))
    if dist[end_idx]==float("inf"): return [],[],0.0
    path=[]; eids=[]; cur=end_idx
    while cur!=start_idx:
        p,eid=prev[cur]
        path.append(cur); eids.append(eid)
        cur=p
    path.append(start_idx); path.reverse(); eids.reverse()
    return path,eids,dist[end_idx]

# ------------------------------------------------------------------
# Topologic build
# ------------------------------------------------------------------
def build_topologic(vertices: List[List[float]], edges: List[List[int]]):
    if not USE_TOPOLOGIC: return None, [], []
    try:
        import topologicpy
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
    except Exception as e:
        dprint("Topologic import failed:", e); return None, [], []
    vert_objs=[Vertex.ByCoordinates(*c) for c in vertices]
    edge_objs=[]
    for (u,v) in edges:
        try:
            e=Edge.ByStartVertexEndVertex(vert_objs[u], vert_objs[v])
            if USE_EDGE_WEIGHTS:
                length=math.dist(vertices[u],vertices[v])
                try: Topology.SetDictionary(e,{EDGE_WEIGHT_KEY:length})
                except: pass
            edge_objs.append(e)
        except Exception as ex:
            dprint("Edge create fail", (u,v), ex)
    geom=vert_objs+edge_objs
    topo=None
    try: topo=Topology.ByGeometry(geometry=geom, tolerance=TOLERANCE)
    except TypeError:
        try: topo=Topology.ByGeometry(geom, TOLERANCE)
        except: topo=None
    if topo is None:
        topo=Topology.ByGeometry(vertices=vertices, edges=edges, faces=[], topologyType=None, tolerance=TOLERANCE)
    cluster=Cluster.ByTopologies([topo])
    try:
        merged=Topology.SelfMerge(cluster)
        if merged: cluster=merged
    except Exception as e:
        dprint("SelfMerge failed:", e)
    try:
        graph=Graph.ByTopology(cluster, tolerance=TOLERANCE)
    except Exception as ex:
        dprint("Graph build failed:", ex); graph=None
    gvs=[]
    if graph:
        from topologicpy.Graph import Graph as TG
        try: gvs=TG.Vertices(graph) or []
        except: gvs=[]
    return graph, vert_objs, edge_objs

def export_graph(graph, directory):
    if not (graph and EXPORT_GRAPH_JSON): return
    try:
        from topologicpy.Graph import Graph
        path=os.path.join(directory, GRAPH_EXPORT_FILENAME)
        try: Graph.ExportToJSON(graph, path, overwrite=OVERWRITE_GRAPH_EXPORT)
        except TypeError:
            if OVERWRITE_GRAPH_EXPORT and os.path.exists(path):
                try: os.remove(path)
                except: pass
            Graph.ExportToJSON(graph, path)
    except Exception as e:
        dprint("Graph export failed:", e)

# ------------------------------------------------------------------
# Topologic shortest path
# ------------------------------------------------------------------
def topo_shortest(graph, start_coord, end_coord, vertices_cache):
    if not (USE_TOPOLOGIC and graph):
        return {"status":"no_graph"}
    try:
        from topologicpy.Graph import Graph
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
    except:
        return {"status":"no_graph"}

    gvs = vertices_cache
    if not gvs:
        try:
            gvs = Graph.Vertices(graph) or []
        except:
            return {"status":"no_graph_vertices"}

    def map_coord(c):
        for gv in gvs:
            if (abs(Vertex.X(gv)-c[0])<=SNAP_TOLERANCE and
                abs(Vertex.Y(gv)-c[1])<=SNAP_TOLERANCE and
                abs(Vertex.Z(gv)-c[2])<=SNAP_TOLERANCE):
                return gv
        best=None; best_d2=1e100
        x,y,z=c
        for gv in gvs:
            d2=(Vertex.X(gv)-x)**2+(Vertex.Y(gv)-y)**2+(Vertex.Z(gv)-z)**2
            if d2<best_d2: best_d2=d2; best=gv
        return best

    sv = map_coord(start_coord)
    ev = map_coord(end_coord)
    if not sv or not ev: return {"status":"unmapped"}
    if sv == ev: return {"status":"same_vertex","coords":[start_coord],"length":0.0}

    edgeKey = EDGE_WEIGHT_KEY if USE_EDGE_WEIGHTS else ""
    try:
        wire = Graph.ShortestPath(graph, sv, ev, "", edgeKey)
    except Exception as e:
        dprint("Graph.ShortestPath error:", e); wire=None
    if not wire: return {"status":"no_path"}
    try:
        vs = Topology.Vertices(wire) or []
        coords=[(Vertex.X(v),Vertex.Y(v),Vertex.Z(v)) for v in vs]
    except Exception:
        coords=[]
    if len(coords)<2: return {"status":"no_path"}
    length=sum(math.dist(coords[i-1],coords[i]) for i in range(1,len(coords)))
    return {"status":"ok","coords":coords,"length":length}

# ------------------------------------------------------------------
# Path comparison
# ------------------------------------------------------------------
def compare_paths(raw_indices: List[int], topo_coords: List[Tuple[float,float,float]], vertices: List[List[float]]) -> bool:
    if not raw_indices or not topo_coords: return False
    if len(raw_indices) != len(topo_coords): return False
    for rid, tcoord in zip(raw_indices, topo_coords):
        vcoord = vertices[rid]
        if any(abs(vcoord[i]-tcoord[i])>SNAP_TOLERANCE for i in range(3)):
            return False
    return True

# ------------------------------------------------------------------
# Chain fallback helpers (existing)
# ------------------------------------------------------------------
def build_chain_segments(raw: RawGraph, ordered_starts: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    segments=[]
    for i in range(len(ordered_starts)-1):
        a=ordered_starts[i]; b=ordered_starts[i+1]
        ai,_=raw.nearest_vertex(tuple(a["point"]))
        bi,_=raw.nearest_vertex(tuple(b["point"]))
        path, epath, length = raw.shortest_path(ai, bi, RAW_USE_WEIGHTS)
        segments.append({
            "from_index": i,
            "to_index": i+1,
            "raw_path": path,
            "raw_edges": epath,
            "length": length,
            "success": bool(path)
        })
    return segments

def accumulate_chain_from(raw: RawGraph, segments, start_pos: int) -> Tuple[List[int], List[int], float]:
    if start_pos >= len(segments)+1: return [],[],0.0
    if start_pos == len(segments): return [],[],0.0
    first=segments[start_pos]
    if not first["success"]: return [],[],0.0
    verts=list(first["raw_path"]); edges=list(first["raw_edges"]); total=first["length"]
    cursor=start_pos+1
    while cursor < len(segments):
        seg=segments[cursor]
        if not seg["success"]: break
        verts.extend(seg["raw_path"][1:])
        edges.extend(seg["raw_edges"])
        total+=seg["length"]
        cursor+=1
    return verts, edges, total

# ------------------------------------------------------------------
# Direct processing with new fallback logic
# ------------------------------------------------------------------
def process_direct(vertices, edges, starts, end_point):
    raw = RawGraph(vertices, edges)

    # Pre-map end vertex once for stability
    end_idx, end_snap = raw.nearest_vertex(end_point)
    end_exact = end_snap <= SNAP_TOLERANCE

    graph=None; gvs=[]
    if USE_TOPOLOGIC:
        graph,_,_ = build_topologic(vertices, edges)
        if graph:
            from topologicpy.Graph import Graph as TG
            try: gvs = TG.Vertices(graph) or []
            except: gvs=[]

    ordered = sorted(starts, key=lambda s: s.get("seq_index",0))
    legs=[]
    raw_success=0
    raw_fail=0
    cumulative=[]

    chain_segments = build_chain_segments(raw, ordered) if ENABLE_CHAIN_FALLBACK else []
    chain_any_success = any(seg["success"] for seg in chain_segments)

    def coord_to_index(c):
        idx,dist = raw.nearest_vertex(c)
        return idx, dist <= SNAP_TOLERANCE, dist

    for i, sp in enumerate(ordered):
        pt = sp.get("point")
        if not pt or len(pt)!=3:
            continue
        sp_coord = tuple(pt)

        si, s_exact, s_snap = coord_to_index(sp_coord)
        ei = end_idx
        e_exact = end_exact
        e_snap = end_snap

        raw_path, raw_edges, raw_len = raw.shortest_path(si, ei, RAW_USE_WEIGHTS)
        bridging_used=False; bridging_reason=None
        stub_used=False; stub_reason=None
        used_chain=False

        # Chain fallback (original behavior) only if direct fails
        if not raw_path and ENABLE_CHAIN_FALLBACK and chain_any_success:
            cv, ce, cl = accumulate_chain_from(raw, chain_segments, i)
            if cv and cv[-1]==ei:
                raw_path=cv; raw_edges=ce; raw_len=cl
                used_chain=True

        # Vertical bridging fallback if still no path
        if not raw_path and ENABLE_VERTICAL_BRIDGING_FALLBACK:
            b_path,b_edges,b_len = bridged_shortest_path(raw, si, ei)
            if b_path:
                raw_path=b_path; raw_edges=b_edges; raw_len=b_len
                bridging_used=True; bridging_reason="vertical_bridging"

        # Final stub fallback (synthetic tail)
        if not raw_path and ENABLE_FINAL_STUB_FALLBACK:
            dist, prev = raw.single_source_tree(si, RAW_USE_WEIGHTS)
            reachable=[v for v,dv in enumerate(dist) if dv<float("inf")]
            if reachable:
                best_v=None; best_d=1e100
                for v in reachable:
                    d=math.dist(vertices[v], end_point)
                    if d<best_d:
                        best_d=d; best_v=v
                if best_v is not None:
                    # Reconstruct path to best_v
                    path=[]; eids=[]
                    cur=best_v
                    while cur!=si:
                        p,eid=prev[cur]
                        if p==-1: break
                        path.append(cur); eids.append(eid)
                        cur=p
                    path.append(si); path.reverse(); eids.reverse()
                    if path and path[-1]==best_v:
                        tail_len = math.dist(vertices[best_v], end_point)
                        raw_path=path
                        raw_edges=eids+[STUB_SENTINEL_EDGE_ID]
                        raw_len=sum(raw.edge_length(eid) for eid in eids if eid>=0)+tail_len
                        stub_used=True; stub_reason="synthetic_stub"
            if not raw_path:
                # direct stub from start only (as last resort)
                tail_len = math.dist(vertices[si], end_point)
                raw_path=[si]; raw_edges=[STUB_SENTINEL_EDGE_ID]; raw_len=tail_len
                stub_used=True; stub_reason="direct_stub"

        if not raw_path and si==ei:
            raw_path=[si]; raw_edges=[-1]; raw_len=0.0

        raw_coords = [vertices[j] for j in raw_path] if raw_path else []
        if raw_path:
            raw_success += 1
            if stub_used and (not raw_coords or raw_coords[-1] != list(end_point)):
                raw_coords.append(list(end_point))
            if cumulative:
                cumulative.extend(raw_coords[1:])
            else:
                cumulative.extend(raw_coords)
        else:
            raw_fail += 1
            raw_len=0.0

        topo_info={"status":"skipped"}
        if USE_TOPOLOGIC and not used_chain:
            topo_info = topo_shortest(graph, sp_coord, end_point, gvs)
        matched=False
        if topo_info.get("status")=="ok" and not used_chain:
            matched = compare_paths(raw_path, topo_info.get("coords",[]), vertices)
            if REQUIRE_MATCH and not matched:
                topo_info["status"]="mismatch_discarded"

        legs.append({
            "leg_type":"direct",
            "from_seq_index": sp.get("seq_index"),
            "to_end": True,
            "from_element_id": sp.get("element_id"),
            "raw_vertex_indices": raw_path,
            "raw_edge_indices": raw_edges,
            "raw_length": raw_len,
            "raw_success": bool(raw_path),
            "topologic_status": topo_info.get("status"),
            "topologic_length": topo_info.get("length"),
            "topologic_coords": topo_info.get("coords"),
            "topologic_matches_raw": matched,
            "start_exact": s_exact,
            "end_exact": e_exact,
            "start_snap_distance": s_snap,
            "end_snap_distance": e_snap,
            "vertex_path_xyz": raw_coords,
            "chain_used": used_chain or None,
            "bridging_used": bridging_used or None,
            "bridging_reason": bridging_reason,
            "stub_used": stub_used or None,
            "stub_reason": stub_reason
        })

    sequence = {
        "legs": legs,
        "total_length_raw": sum(l["raw_length"] for l in legs),
        "successful_raw_legs": raw_success,
        "failed_raw_legs": raw_fail,
        "cumulative_vertex_path_xyz": cumulative
    }
    return sequence, graph, chain_segments

# ------------------------------------------------------------------
# Main process
# ------------------------------------------------------------------
def process(input_path: str, output_path: str):
    start_time=time.time()
    data = load_json(input_path)
    vertices = data.get("vertices", [])
    edges = data.get("edges") or data.get("infra_edges") or []
    starts = data.get("start_points", [])
    end_pt = data.get("end_point")
    if not vertices or not edges:
        raise ValueError("Missing vertices or edges in JSON.")
    if not end_pt or len(end_pt)!=3:
        raise ValueError("end_point missing or malformed")

    end_pt_tuple = tuple(map(float,end_pt))
    # Optionally append end vertex (no edge auto-connect)
    if INCLUDE_END_POINT_VERTEX:
        if not any(abs(v[0]-end_pt_tuple[0])<=SNAP_TOLERANCE and
                   abs(v[1]-end_pt_tuple[1])<=SNAP_TOLERANCE and
                   abs(v[2]-end_pt_tuple[2])<=SNAP_TOLERANCE for v in vertices):
            vertices.append(list(end_pt_tuple))

    direct_data, graph, chain_segments = process_direct(vertices, edges, starts, end_pt_tuple)
    export_graph(graph, os.path.dirname(input_path))

    version = getattr(sys.modules.get("topologicpy",""),"__version__","unknown")
    topo_ok = sum(1 for l in direct_data["legs"] if l["topologic_status"]=="ok")
    topo_match = sum(1 for l in direct_data["legs"] if l.get("topologic_matches_raw"))

    meta = {
        "version": version,
        "tolerance": TOLERANCE,
        "snap_tolerance": SNAP_TOLERANCE,
        "vertices_input": len(vertices),
        "edges_input": len(edges),
        "start_points": len(starts),
        "legs": len(direct_data["legs"]),
        "raw_success_legs": direct_data["successful_raw_legs"],
        "raw_failed_legs": direct_data["failed_raw_legs"],
        "raw_total_length": direct_data["total_length_raw"],
        "topologic_enabled": USE_TOPOLOGIC,
        "topologic_ok_legs": topo_ok,
        "topologic_match_raw_legs": topo_match,
        "edge_weights": USE_EDGE_WEIGHTS,
        "raw_weighted": RAW_USE_WEIGHTS,
        "require_match": REQUIRE_MATCH,
        "chain_fallback_enabled": ENABLE_CHAIN_FALLBACK,
        "chain_segments_success": sum(1 for s in chain_segments if s["success"]),
        "vertical_bridging_fallback": ENABLE_VERTICAL_BRIDGING_FALLBACK,
        "final_stub_fallback": ENABLE_FINAL_STUB_FALLBACK,
        "include_end_point_vertex": INCLUDE_END_POINT_VERTEX,
        "duration_sec": time.time()-start_time
    }

    payload={"meta": meta, "sequence": direct_data}
    with open(output_path,"w") as f:
        json.dump(payload,f,indent=2)
    return payload

# ------------------------------------------------------------------
# Path resolution
# ------------------------------------------------------------------
def resolve_paths():
    script_dir=os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv)>1:
        candidate=os.path.abspath(sys.argv[1])
        if os.path.isdir(candidate):
            ip=os.path.join(candidate, DEFAULT_JSON_INPUT_FILENAME)
        else:
            ip=candidate
    else:
        ip=os.path.join(script_dir, DEFAULT_JSON_INPUT_FILENAME)
    if not os.path.isfile(ip):
        raise FileNotFoundError(f"Input JSON not found: {ip}")
    op=os.path.join(os.path.dirname(ip), JSON_OUTPUT_FILENAME)
    return ip, op

if __name__=="__main__":
    in_path, out_path = resolve_paths()
    payload = process(in_path, out_path)
    m=payload["meta"]
    print(f"Direct legs={m['legs']} raw_ok={m['raw_success_legs']} raw_fail={m['raw_failed_legs']} "
          f"stub_fallback={m['final_stub_fallback']} bridging={m['vertical_bridging_fallback']} Output={out_path}")
