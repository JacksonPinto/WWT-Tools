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

PREFER_INFRA_EDGES = True
RAW_WEIGHTED = True
USE_TOPOLOGIC = True
USE_EDGE_WEIGHTS = True
EDGE_WEIGHT_KEY = "w"
REQUIRE_MATCH = False
EXPORT_GRAPH_JSON = False
GRAPH_EXPORT_FILENAME = "topologic_script.json"
OVERWRITE_GRAPH_EXPORT = True
INCLUDE_END_POINT_VERTEX = True

# Fallback flags
ENABLE_VERTICAL_BRIDGING_FALLBACK = True          # Existing vertical bridging
BRIDGING_XY_TOL = 1e-6
BRIDGING_ALLOW_ALL_LEGS = False
ENABLE_FINAL_STUB_FALLBACK = True                 # New: create synthetic path to end if unreachable
STUB_SENTINEL_EDGE_ID = -999                      # Edge id marker for stub tail segment

DEBUG = ("--debug" in sys.argv) or (os.environ.get("DEBUG") == "1")
def dprint(*a):
    if DEBUG: print("[DEBUG]", *a)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def euclid(a, b): return math.dist(a, b)
def same_coord(a, b, tol): return abs(a[0]-b[0])<=tol and abs(a[1]-b[1])<=tol and abs(a[2]-b[2])<=tol

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
        return euclid(self.vertices[u], self.vertices[v])

    def shortest_path(self, s:int, t:int, weighted: bool):
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
                    if nd<dist[v]:
                        dist[v]=nd; prev[v]=(u,eid)
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
                        prev[v]=(u,eid); q.append(v)
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
        """
        Returns dist, prev arrays for all reachable vertices from s.
        prev[v] = (parentVertex, edgeId)
        """
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

# Bridging fallback (unchanged from earlier version posting)
def bridged_shortest_path(raw: RawGraph, start_idx: int, end_idx: int) -> Tuple[List[int], List[int], float]:
    n = len(raw.vertices)
    buckets = defaultdict(list)
    for i,(x,y,z) in enumerate(raw.vertices):
        key=(round(x/BRIDGING_XY_TOL), round(y/BRIDGING_XY_TOL))
        buckets[key].append(i)
    aug=[[] for _ in range(n)]
    for u in range(n):
        for (v,eid) in raw.adj[u]:
            w=raw.edge_length(eid)
            aug[u].append((v,eid,w,False))
    for verts in buckets.values():
        if len(verts)<2: continue
        vs=sorted(verts, key=lambda i: raw.vertices[i][2])
        for i in range(len(vs)-1):
            a=vs[i]; b=vs[i+1]
            w=abs(raw.vertices[a][2]-raw.vertices[b][2])
            if w==0: continue
            aug[a].append((b,-1,w,True))
            aug[b].append((a,-1,w,True))
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
    if dist[end_idx]==float("inf"):
        return [],[],0.0
    path=[]; eids=[]; cur=end_idx
    while cur!=start_idx:
        p,eid=prev[cur]
        path.append(cur); eids.append(eid)
        cur=p
    path.append(start_idx); path.reverse(); eids.reverse()
    return path, eids, dist[end_idx]

# Topologic builder
def build_topologic_graph(vertices: List[List[float]], edges: List[List[int]]):
    if not USE_TOPOLOGIC: return None, []
    try:
        import topologicpy
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
    except Exception as e:
        dprint("Topologic import failed:", e); return None, []
    vertex_objs=[Vertex.ByCoordinates(*c) for c in vertices]
    edge_objs=[]
    for (u,v) in edges:
        if 0 <= u < len(vertex_objs) and 0 <= v < len(vertex_objs):
            try:
                e=Edge.ByStartVertexEndVertex(vertex_objs[u], vertex_objs[v])
                if USE_EDGE_WEIGHTS:
                    length=euclid(vertices[u], vertices[v])
                    try: Topology.SetDictionary(e,{EDGE_WEIGHT_KEY:length})
                    except: pass
                edge_objs.append(e)
            except Exception as ex:
                dprint("Edge create fail", (u,v), ex)
    geom=vertex_objs+edge_objs
    topo=None
    try: topo = Topology.ByGeometry(geometry=geom, tolerance=TOLERANCE)
    except TypeError:
        try: topo = Topology.ByGeometry(geom, TOLERANCE)
        except: topo=None
    if topo is None:
        topo = Topology.ByGeometry(vertices=vertices, edges=edges, faces=[], topologyType=None, tolerance=TOLERANCE)
    cluster = Cluster.ByTopologies([topo])
    try:
        merged = Topology.SelfMerge(cluster)
        if merged: cluster=merged
    except Exception as e:
        dprint("SelfMerge failed:", e)
    try:
        graph = Graph.ByTopology(cluster, tolerance=TOLERANCE)
    except Exception as ex:
        dprint("Graph build failed:", ex); graph=None
    gvs=[]
    if graph:
        try: gvs = Graph.Vertices(graph) or []
        except: gvs=[]
    return graph, gvs

def topo_shortest(graph, start_coord, end_coord, gvs):
    if not (USE_TOPOLOGIC and graph): return {"status":"no_graph"}
    try:
        from topologicpy.Graph import Graph
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
    except:
        return {"status":"no_graph"}
    def map_coord(c):
        for gv in gvs:
            if same_coord((Vertex.X(gv),Vertex.Y(gv),Vertex.Z(gv)), c, SNAP_TOLERANCE):
                return gv
        best=None; best_d2=1e100
        x,y,z=c
        for gv in gvs:
            d2=(Vertex.X(gv)-x)**2+(Vertex.Y(gv)-y)**2+(Vertex.Z(gv)-z)**2
            if d2<best_d2: best_d2=d2; best=gv
        return best
    sv=map_coord(start_coord); ev=map_coord(end_coord)
    if not sv or not ev: return {"status":"unmapped"}
    if sv==ev: return {"status":"same_vertex","coords":[start_coord],"length":0.0}
    edgeKey = EDGE_WEIGHT_KEY if USE_EDGE_WEIGHTS else ""
    try: wire = Graph.ShortestPath(graph, sv, ev, "", edgeKey)
    except Exception as e:
        dprint("Graph.ShortestPath error:", e); wire=None
    if not wire: return {"status":"no_path"}
    try:
        vs = Topology.Vertices(wire) or []
        coords=[(Vertex.X(v),Vertex.Y(v),Vertex.Z(v)) for v in vs]
    except: coords=[]
    if len(coords)<2: return {"status":"no_path"}
    length=sum(euclid(coords[i-1],coords[i]) for i in range(1,len(coords)))
    return {"status":"ok","coords":coords,"length":length}

def compare_paths(raw_indices, topo_coords, vertices):
    if not raw_indices or not topo_coords: return False
    if len(raw_indices)!=len(topo_coords): return False
    for rid,c in zip(raw_indices, topo_coords):
        vx,vy,vz = vertices[rid]
        if (abs(vx-c[0])>SNAP_TOLERANCE or abs(vy-c[1])>SNAP_TOLERANCE or abs(vz-c[2])>SNAP_TOLERANCE):
            return False
    return True

def process_sequence(vertices, edges, start_points, end_point):
    raw = RawGraph(vertices, edges)

    mapped_starts=[]
    for sp in start_points:
        idx,snap=raw.nearest_vertex(sp["point"])
        mapped_starts.append({
            "seq_index": sp["seq_index"],
            "element_id": sp["element_id"],
            "point": sp["point"],
            "raw_index": idx,
            "snap_dist": snap,
            "exact": snap<=SNAP_TOLERANCE
        })

    end_idx, end_snap = raw.nearest_vertex(end_point)
    end_exact = end_snap <= SNAP_TOLERANCE

    graph=None; gvs=[]
    if USE_TOPOLOGIC:
        graph,gvs = build_topologic_graph(vertices, edges)

    legs=[]; success=0; failed=0; cumulative=[]

    def register(path_indices):
        coords=[vertices[i] for i in path_indices]
        if cumulative:
            cumulative.extend(coords[1:])
        else:
            cumulative.extend(coords)
        return coords

    def reconstruct(prev, s, t):
        path=[]; epath=[]
        cur=t
        while cur!=s:
            p,eid=prev[cur]
            if p==-1: return [],[]
            path.append(cur); epath.append(eid)
            cur=p
        path.append(s); path.reverse(); epath.reverse()
        return path, epath

    def build_leg(a, tgt_idx, tgt_coord, to_seq=None, final=False):
        nonlocal success, failed
        si=a["raw_index"]; ei=tgt_idx
        raw_path, raw_edges, raw_len = raw.shortest_path(si, ei, RAW_WEIGHTED)
        bridging_used=False; bridging_reason=None
        stub_used=False; stub_reason=None

        # Vertical bridging if enabled
        if not raw_path and ENABLE_VERTICAL_BRIDGING_FALLBACK and (final or BRIDGING_ALLOW_ALL_LEGS):
            b_path,b_edges,b_len = bridged_shortest_path(raw, si, ei)
            if b_path:
                raw_path=b_path; raw_edges=b_edges; raw_len=b_len
                bridging_used=True; bridging_reason="vertical_bridging"

        # Final stub fallback
        if final and not raw_path and ENABLE_FINAL_STUB_FALLBACK:
            # Multi-source Dijkstra from si
            dist, prev = raw.single_source_tree(si, RAW_WEIGHTED)
            # Collect reachable vertices
            reachable=[i for i,dv in enumerate(dist) if dv<float("inf")]
            if reachable:
                # Pick reachable vertex closest to end point in straight line
                best_v=None; best_d=1e100
                for v in reachable:
                    d=euclid(vertices[v], end_point)
                    if d<best_d:
                        best_d=d; best_v=v
                if best_v is not None:
                    path_r, edges_r = reconstruct(prev, si, best_v)
                    if path_r:
                        tail_len = euclid(vertices[best_v], end_point)
                        # Accept even if tail_len==0 (snap)
                        raw_path=path_r
                        raw_edges=edges_r
                        raw_len=sum(raw.edge_length(eid) for eid in raw_edges if eid>=0)+tail_len
                        # Append a synthetic marker for tail (not a graph edge)
                        raw_edges.append(STUB_SENTINEL_EDGE_ID)
                        # Add synthetic coordinate (end point) to vertex path xyz only (below)
                        stub_used=True; stub_reason="synthetic_stub"
            # If still nothing, last resort: direct stub from start
            if not raw_path:
                tail_len = euclid(vertices[si], end_point)
                raw_path=[si]
                raw_edges=[STUB_SENTINEL_EDGE_ID]
                raw_len=tail_len
                stub_used=True; stub_reason="direct_stub"

        if not raw_path and si==ei:
            raw_path=[si]; raw_edges=[-1]; raw_len=0.0

        raw_coords=[]
        if raw_path:
            success+=1
            raw_coords=register(raw_path)
            # If stub tail added and last coordinate != end_point add it
            if (stub_used and (not raw_coords or not same_coord(raw_coords[-1], end_point, SNAP_TOLERANCE))):
                raw_coords.append(list(end_point))
                cumulative.append(list(end_point))
        else:
            failed+=1
            raw_len=0.0

        topo_info={"status":"skipped"}
        if USE_TOPOLOGIC:
            topo_info = topo_shortest(graph, a["point"], tgt_coord, gvs)
        matched=False
        if topo_info.get("status")=="ok":
            matched=compare_paths(raw_path, topo_info.get("coords",[]), vertices)
            if REQUIRE_MATCH and not matched:
                topo_info["status"]="mismatch_discarded"

        legs.append({
            "leg_type":"final_to_end" if final else "sequence",
            "from_seq_index": a["seq_index"],
            "to_seq_index": None if final else to_seq,
            "to_end": True if final else None,
            "from_element_id": a["element_id"],
            "to_element_id": None,
            "raw_vertex_indices": raw_path,
            "raw_edge_indices": raw_edges,
            "raw_length": raw_len,
            "raw_success": bool(raw_path),
            "topologic_status": topo_info.get("status"),
            "topologic_length": topo_info.get("length"),
            "topologic_coords": topo_info.get("coords"),
            "topologic_matches_raw": matched,
            "start_exact": a["exact"],
            "end_exact": end_exact if final else None,
            "start_snap_distance": a["snap_dist"],
            "end_snap_distance": end_snap if final else None,
            "vertex_path_xyz": raw_coords,
            "bridging_used": bridging_used or None,
            "bridging_reason": bridging_reason,
            "stub_used": stub_used or None,
            "stub_reason": stub_reason
        })

    for i in range(len(mapped_starts)-1):
        a=mapped_starts[i]; b=mapped_starts[i+1]
        build_leg(a, b["raw_index"], b["point"], to_seq=b["seq_index"], final=False)

    if mapped_starts:
        last=mapped_starts[-1]
        build_leg(last, end_idx, end_point, final=True)

    return {
        "legs": legs,
        "total_length_raw": sum(l["raw_length"] for l in legs),
        "successful_raw_legs": success,
        "failed_raw_legs": failed,
        "cumulative_vertex_path_xyz": cumulative
    }

def process(input_path: str, output_path: str):
    start_time=time.time()
    data = load_json(input_path)

    base_vertices = list(data.get("vertices", []))
    if not base_vertices: raise ValueError("No vertices in JSON.")

    infra_edges = data.get("infra_edges") or []
    primary_edges = data.get("edges") or []
    device_edges = data.get("device_edges") or []

    if PREFER_INFRA_EDGES and infra_edges:
        merged = infra_edges + device_edges
    else:
        merged = primary_edges + device_edges

    seen=set(); base_edges=[]
    for e in merged:
        if not isinstance(e,(list,tuple)) or len(e)!=2: continue
        a,b=e; key=(a,b) if a<b else (b,a)
        if key in seen: continue
        seen.add(key)
        base_edges.append([a,b])

    if not base_edges:
        raise ValueError("No edges after merge.")

    end_point = data.get("end_point")
    if not end_point or len(end_point)!=3: raise ValueError("end_point missing or malformed")
    end_point = tuple(map(float,end_point))

    if INCLUDE_END_POINT_VERTEX and not any(same_coord(tuple(v), end_point, SNAP_TOLERANCE) for v in base_vertices):
        base_vertices.append(list(end_point))
        dprint("End point appended index={}".format(len(base_vertices)-1))

    start_points=[]
    for sp in data.get("start_points", []):
        pt=sp.get("point")
        if not pt or len(pt)!=3: continue
        start_points.append({
            "seq_index": sp.get("seq_index"),
            "element_id": sp.get("element_id"),
            "point": tuple(map(float, pt))
        })
    if start_points and all(p["seq_index"] is not None for p in start_points):
        start_points.sort(key=lambda s: s["seq_index"])

    seq_data = process_sequence(base_vertices, base_edges, start_points, end_point) if start_points else {
        "legs":[], "total_length_raw":0.0,"successful_raw_legs":0,"failed_raw_legs":0,"cumulative_vertex_path_xyz":[]
    }

    version = getattr(sys.modules.get("topologicpy",""),"__version__","unknown") if USE_TOPOLOGIC else "raw_only"
    meta = {
        "version": version,
        "tolerance": TOLERANCE,
        "snap_tolerance": SNAP_TOLERANCE,
        "vertices_input": len(base_vertices),
        "edges_input": len(base_edges),
        "start_points": len(start_points),
        "legs": len(seq_data["legs"]),
        "raw_success_legs": seq_data["successful_raw_legs"],
        "raw_failed_legs": seq_data["failed_raw_legs"],
        "raw_total_length": seq_data["total_length_raw"],
        "topologic_enabled": USE_TOPOLOGIC,
        "edge_weights": USE_EDGE_WEIGHTS,
        "raw_weighted": RAW_WEIGHTED,
        "prefer_infra_edges": PREFER_INFRA_EDGES,
        "require_match": REQUIRE_MATCH,
        "include_end_point_vertex": INCLUDE_END_POINT_VERTEX,
        "vertical_bridging_fallback": ENABLE_VERTICAL_BRIDGING_FALLBACK,
        "final_stub_fallback": ENABLE_FINAL_STUB_FALLBACK,
        "duration_sec": time.time()-start_time
    }

    payload={"meta": meta, "sequence": seq_data}
    with open(output_path,"w") as f:
        json.dump(payload,f,indent=2)
    return payload

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
        raise FileNotFoundError("Input JSON not found: {}".format(ip))
    op=os.path.join(os.path.dirname(ip), JSON_OUTPUT_FILENAME)
    return ip, op

if __name__=="__main__":
    in_path, out_path = resolve_paths()
    payload = process(in_path, out_path)
    m=payload["meta"]
    print("Sequence legs={} raw_ok={} raw_fail={} stub_fallback={} Output={}".format(
        m["legs"], m["raw_success_legs"], m["raw_failed_legs"], m.get("final_stub_fallback"), out_path))
