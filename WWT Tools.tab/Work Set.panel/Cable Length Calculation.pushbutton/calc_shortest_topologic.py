import json
import os
import math
import time
import sys
from collections import deque, defaultdict
from typing import List, Dict, Any, Tuple

# ------------------------------------------------------------------
# Configuration (shared)
# ------------------------------------------------------------------
DEFAULT_JSON_INPUT_FILENAME = "topologic.JSON"
JSON_OUTPUT_FILENAME = "topologic_results.json"

TOLERANCE = 1e-6
SNAP_TOLERANCE = 1e-9

PREFER_INFRA_EDGES = True          # Prefer infra_edges over edges when present
INCLUDE_DEVICE_EDGES = True        # Merge device_edges to ensure full connectivity

RAW_WEIGHTED = True
USE_TOPOLOGIC = True
USE_EDGE_WEIGHTS = True
EDGE_WEIGHT_KEY = "w"
REQUIRE_MATCH = False

INCLUDE_END_POINT_VERTEX = True    # Append end point vertex if not present (no auto-edge)

# Direct mode extras
ENABLE_CHAIN_FALLBACK = True       # Accumulate sequential start->start legs for direct mode if direct start->end fails

# Bridging & stub fallbacks (both modes)
ENABLE_VERTICAL_BRIDGING_FALLBACK = True
BRIDGING_XY_TOL = 1e-6
ENABLE_FINAL_STUB_FALLBACK = True  # Create synthetic stub tail if unreachable
STUB_SENTINEL_EDGE_ID = -999       # Edge id marker for synthetic stub tail

EXPORT_GRAPH_JSON = False
GRAPH_EXPORT_FILENAME = "topologic_script.json"
OVERWRITE_GRAPH_EXPORT = True

DEBUG = ("--debug" in sys.argv) or (os.environ.get("DEBUG") == "1")
def dprint(*a):
    if DEBUG: print("[DEBUG]", *a)

# ------------------------------------------------------------------
# JSON Load
# ------------------------------------------------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def same_coord(a,b,tol): return abs(a[0]-b[0])<=tol and abs(a[1]-b[1])<=tol and abs(a[2]-b[2])<=tol

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
                    if nd<dist[v]:
                        dist[v]=nd; prev[v]=(u,eid)
                        heapq.heappush(h,(nd,v))
            if dist[t]==float("inf"): return [],[],0.0
            path=[]; epath=[]; cur=t
            while cur!=s:
                p,eid=prev[cur]
                path.append(cur); epath.append(eid)
                cur=p
            path.append(s); path.reverse(); epath.reverse()
            return path,epath,dist[t]
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
                p,eid=prev[cur]; path.append(cur); epath.append(eid); cur=p
            path.append(s); path.reverse(); epath.reverse()
            return path, epath, sum(self.edge_length(eid) for eid in epath)

    def single_source_tree(self, s:int, weighted: bool):
        if weighted:
            import heapq
            dist=[float("inf")]*len(self.vertices)
            prev=[(-1,-1)]*len(self.vertices)
            dist[s]=0.0; h=[(0.0,s)]
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
            dist[s]=0.0; q=deque([s])
            while q:
                u=q.popleft()
                for (v,eid) in self.adj[u]:
                    if dist[v]==float("inf"):
                        dist[v]=dist[u]+1; prev[v]=(u,eid); q.append(v)
            return dist, prev

# ------------------------------------------------------------------
# Vertical Bridging Fallback
# ------------------------------------------------------------------
def bridged_shortest_path(raw: RawGraph, start_idx: int, end_idx: int, xy_tol: float) -> Tuple[List[int], List[int], float]:
    n=len(raw.vertices)
    buckets=defaultdict(list)
    for i,(x,y,z) in enumerate(raw.vertices):
        key=(round(x/xy_tol), round(y/xy_tol))
        buckets[key].append(i)
    aug=[[] for _ in range(n)]
    for u in range(n):
        for (v,eid) in raw.adj[u]:
            w=raw.edge_length(eid)
            aug[u].append((v,eid,w,False))
    for grp in buckets.values():
        if len(grp)<2: continue
        sgrp=sorted(grp, key=lambda i: raw.vertices[i][2])
        for i in range(len(sgrp)-1):
            a=sgrp[i]; b=sgrp[i+1]
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
# Topologic Graph Build
# ------------------------------------------------------------------
def build_topologic(vertices: List[List[float]], edges: List[List[int]]):
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
    vert_objs=[Vertex.ByCoordinates(*c) for c in vertices]
    edge_objs=[]
    for (u,v) in edges:
        if not (0<=u<len(vert_objs) and 0<=v<len(vert_objs)): continue
        try:
            e=Edge.ByStartVertexEndVertex(vert_objs[u], vert_objs[v])
            if USE_EDGE_WEIGHTS:
                length=math.dist(vertices[u], vertices[v])
                try: Topology.SetDictionary(e,{EDGE_WEIGHT_KEY:length})
                except: pass
            edge_objs.append(e)
        except Exception as ex:
            dprint("Edge create fail", (u,v), ex)
    geom=vert_objs+edge_objs
    topo=None
    from topologicpy.Topology import Topology
    try: topo=Topology.ByGeometry(geometry=geom, tolerance=TOLERANCE)
    except TypeError:
        try: topo=Topology.ByGeometry(geom, TOLERANCE)
        except: topo=None
    if topo is None:
        topo=Topology.ByGeometry(vertices=vertices, edges=edges, faces=[], topologyType=None, tolerance=TOLERANCE)
    from topologicpy.Cluster import Cluster
    cluster=Cluster.ByTopologies([topo])
    try:
        merged=Topology.SelfMerge(cluster)
        if merged: cluster=merged
    except Exception as e:
        dprint("SelfMerge failed:", e)
    from topologicpy.Graph import Graph
    try:
        graph=Graph.ByTopology(cluster, tolerance=TOLERANCE)
    except Exception as ex:
        dprint("Graph build failed:", ex); graph=None
    gvs=[]
    if graph:
        try: gvs=Graph.Vertices(graph) or []
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
    sv=map_coord(start_coord); ev=map_coord(end_coord)
    if not sv or not ev: return {"status":"unmapped"}
    if sv==ev: return {"status":"same_vertex","coords":[start_coord],"length":0.0}
    edgeKey = EDGE_WEIGHT_KEY if USE_EDGE_WEIGHTS else ""
    try:
        wire=Graph.ShortestPath(graph, sv, ev, "", edgeKey)
    except Exception as e:
        dprint("Graph.ShortestPath error:", e); wire=None
    if not wire: return {"status":"no_path"}
    try:
        vs=Topology.Vertices(wire) or []
        coords=[(Vertex.X(v),Vertex.Y(v),Vertex.Z(v)) for v in vs]
    except:
        coords=[]
    if len(coords)<2: return {"status":"no_path"}
    length=sum(math.dist(coords[i-1],coords[i]) for i in range(1,len(coords)))
    return {"status":"ok","coords":coords,"length":length}

def compare_paths(raw_indices, topo_coords, vertices):
    if not raw_indices or not topo_coords: return False
    if len(raw_indices)!=len(topo_coords): return False
    for rid,c in zip(raw_indices, topo_coords):
        vx,vy,vz = vertices[rid]
        if (abs(vx-c[0])>SNAP_TOLERANCE or abs(vy-c[1])>SNAP_TOLERANCE or abs(vz-c[2])>SNAP_TOLERANCE):
            return False
    return True

# ------------------------------------------------------------------
# Direct Mode
# ------------------------------------------------------------------
def build_chain_segments(raw: RawGraph, ordered_starts: List[Dict[str,Any]]):
    segments=[]
    for i in range(len(ordered_starts)-1):
        a=ordered_starts[i]; b=ordered_starts[i+1]
        ai,_=raw.nearest_vertex(a["point"])
        bi,_=raw.nearest_vertex(b["point"])
        path,epath,length=raw.shortest_path(ai,bi,RAW_WEIGHTED)
        segments.append({"from":i,"to":i+1,"raw_path":path,"raw_edges":epath,"length":length,"success":bool(path)})
    return segments

def accumulate_chain_from(raw: RawGraph, segments, start_pos: int):
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

def process_direct(vertices, edges, starts, end_point) -> Dict[str,Any]:
    raw=RawGraph(vertices, edges)
    end_idx, end_snap = raw.nearest_vertex(end_point)
    end_exact = end_snap <= SNAP_TOLERANCE

    graph=None; gvs=[]
    if USE_TOPOLOGIC:
        graph,gvs = build_topologic(vertices, edges)

    ordered=sorted(starts, key=lambda s: s.get("seq_index",0))
    chain_segments = build_chain_segments(raw, ordered) if ENABLE_CHAIN_FALLBACK else []
    chain_any = any(seg["success"] for seg in chain_segments)

    legs=[]; success=0; failed=0; cumulative=[]

    def coord_to_index(c):
        i,d = raw.nearest_vertex(c)
        return i, d<=SNAP_TOLERANCE, d

    for i, sp in enumerate(ordered):
        pt=sp["point"]
        si,s_exact,s_snap = coord_to_index(pt)
        ei=end_idx; e_exact=end_exact; e_snap=end_snap

        raw_path, raw_edges, raw_len = raw.shortest_path(si, ei, RAW_WEIGHTED)
        chain_used=False; bridging_used=False; bridging_reason=None
        stub_used=False; stub_reason=None

        # Chain fallback (only if reaches same end index)
        if not raw_path and ENABLE_CHAIN_FALLBACK and chain_any:
            cv,ce,cl = accumulate_chain_from(raw, chain_segments, i)
            if cv and cv[-1]==ei:
                raw_path=cv; raw_edges=ce; raw_len=cl; chain_used=True

        # Vertical bridging
        if not raw_path and ENABLE_VERTICAL_BRIDGING_FALLBACK:
            b_path,b_edges,b_len = bridged_shortest_path(raw, si, ei, BRIDGING_XY_TOL)
            if b_path:
                raw_path=b_path; raw_edges=b_edges; raw_len=b_len
                bridging_used=True; bridging_reason="vertical_bridging"

        # Stub fallback
        if not raw_path and ENABLE_FINAL_STUB_FALLBACK:
            dist, prev = raw.single_source_tree(si, RAW_WEIGHTED)
            reachable=[v for v,dv in enumerate(dist) if dv<float("inf")]
            if reachable:
                best_v=None; best_d=1e100
                for v in reachable:
                    d=math.dist(vertices[v], end_point)
                    if d<best_d:
                        best_d=d; best_v=v
                # reconstruct to best_v
                path=[]; eids=[]
                if best_v is not None:
                    cur=best_v
                    while cur!=si:
                        p,eid=prev[cur]
                        if p==-1: break
                        path.append(cur); eids.append(eid)
                        cur=p
                    path.append(si); path.reverse(); eids.reverse()
                if path:
                    tail_len=math.dist(vertices[best_v], end_point)
                    raw_path=path; raw_edges=eids+[STUB_SENTINEL_EDGE_ID]
                    raw_len=sum(raw.edge_length(eid) for eid in eids if eid>=0)+tail_len
                    stub_used=True; stub_reason="synthetic_stub"
            if not raw_path:
                tail_len=math.dist(vertices[si], end_point)
                raw_path=[si]; raw_edges=[STUB_SENTINEL_EDGE_ID]; raw_len=tail_len
                stub_used=True; stub_reason="direct_stub"

        if not raw_path and si==ei:
            raw_path=[si]; raw_edges=[-1]; raw_len=0.0

        raw_coords=[vertices[j] for j in raw_path] if raw_path else []
        if raw_path:
            success+=1
            if stub_used and (not raw_coords or raw_coords[-1]!=list(end_point)):
                raw_coords.append(list(end_point))
            if cumulative: cumulative.extend(raw_coords[1:])
            else: cumulative.extend(raw_coords)
        else:
            failed+=1; raw_len=0.0

        topo_info={"status":"skipped"}
        if USE_TOPOLOGIC and not chain_used:
            topo_info = topo_shortest(graph, pt, end_point, gvs)
        matched=False
        if topo_info.get("status")=="ok" and not chain_used:
            matched=compare_paths(raw_path, topo_info.get("coords",[]), vertices)
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
            "chain_used": chain_used or None,
            "bridging_used": bridging_used or None,
            "bridging_reason": bridging_reason,
            "stub_used": stub_used or None,
            "stub_reason": stub_reason
        })

    return {
        "legs": legs,
        "total_length_raw": sum(l["raw_length"] for l in legs),
        "successful_raw_legs": success,
        "failed_raw_legs": failed,
        "cumulative_vertex_path_xyz": cumulative
    }, graph, chain_segments

# ------------------------------------------------------------------
# Chain Mode
# ------------------------------------------------------------------
def process_chain(vertices, edges, starts, end_point) -> Dict[str,Any]:
    raw=RawGraph(vertices, edges)
    mapped=[]
    for sp in starts:
        idx, snap = raw.nearest_vertex(sp["point"])
        mapped.append({
            "seq_index": sp["seq_index"],
            "element_id": sp["element_id"],
            "point": sp["point"],
            "idx": idx,
            "snap": snap,
            "exact": snap<=SNAP_TOLERANCE
        })
    end_idx, end_snap = raw.nearest_vertex(end_point)
    end_exact = end_snap <= SNAP_TOLERANCE

    graph=None; gvs=[]
    if USE_TOPOLOGIC:
        graph,gvs = build_topologic(vertices, edges)

    legs=[]; success=0; failed=0; cumulative=[]

    def register(path):
        coords=[vertices[i] for i in path]
        if cumulative: cumulative.extend(coords[1:])
        else: cumulative.extend(coords)
        return coords

    def reconstruct(prev, s, t):
        path=[]; eids=[]
        cur=t
        while cur!=s:
            p,eid=prev[cur]
            if p==-1: return [],[]
            path.append(cur); eids.append(eid)
            cur=p
        path.append(s); path.reverse(); eids.reverse()
        return path,eids

    def build_leg(a, target_idx, target_coord, final=False, to_seq=None):
        nonlocal success, failed
        si=a["idx"]; ei=target_idx
        raw_path, raw_edges, raw_len = raw.shortest_path(si, ei, RAW_WEIGHTED)
        bridging_used=False; bridging_reason=None
        stub_used=False; stub_reason=None

        if not raw_path and ENABLE_VERTICAL_BRIDGING_FALLBACK and final:
            bpath,bedges,blen = bridged_shortest_path(raw, si, ei, BRIDGING_XY_TOL)
            if bpath:
                raw_path=bpath; raw_edges=bedges; raw_len=blen
                bridging_used=True; bridging_reason="vertical_bridging"

        if final and not raw_path and ENABLE_FINAL_STUB_FALLBACK:
            dist, prev = raw.single_source_tree(si, RAW_WEIGHTED)
            reachable=[v for v,dv in enumerate(dist) if dv<float("inf")]
            if reachable:
                best_v=None; best_d=1e100
                for v in reachable:
                    d=math.dist(vertices[v], end_point)
                    if d<best_d: best_d=d; best_v=v
                if best_v is not None:
                    pth,eids=reconstruct(prev, si, best_v)
                    if pth:
                        tail_len=math.dist(vertices[best_v], end_point)
                        raw_path=pth; raw_edges=eids+[STUB_SENTINEL_EDGE_ID]
                        raw_len=sum(raw.edge_length(eid) for eid in eids if eid>=0)+tail_len
                        stub_used=True; stub_reason="synthetic_stub"
            if not raw_path:
                tail_len=math.dist(vertices[si], end_point)
                raw_path=[si]; raw_edges=[STUB_SENTINEL_EDGE_ID]; raw_len=tail_len
                stub_used=True; stub_reason="direct_stub"

        if not raw_path and si==ei:
            raw_path=[si]; raw_edges=[-1]; raw_len=0.0

        raw_coords=[]
        if raw_path:
            success+=1
            raw_coords=register(raw_path)
            if stub_used and (not raw_coords or raw_coords[-1]!=list(end_point)):
                raw_coords.append(list(end_point)); cumulative.append(list(end_point))
        else:
            failed+=1; raw_len=0.0

        topo_info={"status":"skipped"}
        if USE_TOPOLOGIC:
            topo_info = topo_shortest(graph, a["point"], target_coord, gvs)
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
            "start_snap_distance": a["snap"],
            "end_snap_distance": end_snap if final else None,
            "vertex_path_xyz": raw_coords,
            "bridging_used": bridging_used or None,
            "bridging_reason": bridging_reason,
            "stub_used": stub_used or None,
            "stub_reason": stub_reason
        })

    for i in range(len(mapped)-1):
        a=mapped[i]; b=mapped[i+1]
        build_leg(a, b["idx"], b["point"], final=False, to_seq=b["seq_index"])
    if mapped:
        build_leg(mapped[-1], end_idx, end_point, final=True)

    return {
        "legs": legs,
        "total_length_raw": sum(l["raw_length"] for l in legs),
        "successful_raw_legs": success,
        "failed_raw_legs": failed,
        "cumulative_vertex_path_xyz": cumulative
    }, graph

# ------------------------------------------------------------------
# Shared process
# ------------------------------------------------------------------
def integrate_edges(data: Dict[str,Any]) -> List[List[int]]:
    infra = data.get("infra_edges") or []
    prim  = data.get("edges") or []
    devices = data.get("device_edges") or []
    if PREFER_INFRA_EDGES and infra:
        merged = infra + (devices if INCLUDE_DEVICE_EDGES else [])
    else:
        merged = prim + (devices if INCLUDE_DEVICE_EDGES else [])
    seen=set(); out=[]
    for e in merged:
        if not isinstance(e,(list,tuple)) or len(e)!=2: continue
        a,b=e; key=(a,b) if a<b else (b,a)
        if key in seen: continue
        seen.add(key)
        out.append([a,b])
    return out

def process(input_path: str, output_path: str, mode: str):
    start_time=time.time()
    data = load_json(input_path)

    vertices = list(data.get("vertices", []))
    if not vertices: raise ValueError("No vertices in JSON.")
    edges = integrate_edges(data)
    if not edges: raise ValueError("No edges after merge.")
    starts_raw = data.get("start_points", [])
    end_pt = data.get("end_point")
    if not end_pt or len(end_pt)!=3:
        raise ValueError("end_point missing or malformed")
    end_tuple = tuple(map(float,end_pt))

    # Append end vertex (no edge auto connect)
    if INCLUDE_END_POINT_VERTEX and not any(same_coord(tuple(v), end_tuple, SNAP_TOLERANCE) for v in vertices):
        vertices.append(list(end_tuple))
        dprint(f"End point appended index={len(vertices)-1}")

    starts=[]
    for sp in starts_raw:
        pt=sp.get("point")
        if not pt or len(pt)!=3: continue
        starts.append({
            "seq_index": sp.get("seq_index"),
            "element_id": sp.get("element_id"),
            "point": tuple(map(float, pt))
        })
    if starts and all(s["seq_index"] is not None for s in starts):
        starts.sort(key=lambda s:s["seq_index"])

    if mode=="chain":
        sequence_data, graph = process_chain(vertices, edges, starts, end_tuple) if starts else ({
            "legs":[], "total_length_raw":0.0,"successful_raw_legs":0,"failed_raw_legs":0,
            "cumulative_vertex_path_xyz":[]
        }, None)
        chain_segments_success = 0
    else: # direct
        sequence_data, graph, chain_segments = process_direct(vertices, edges, starts, end_tuple) if starts else ({
            "legs":[], "total_length_raw":0.0,"successful_raw_legs":0,"failed_raw_legs":0,
            "cumulative_vertex_path_xyz":[]
        }, None, [])
        chain_segments_success = sum(1 for s in chain_segments if s["success"])

    version = "raw_only"
    if USE_TOPOLOGIC:
        try:
            import topologicpy
            version = getattr(topologicpy,"__version__","unknown")
        except:
            version="topologic_unavailable"

    meta = {
        "version": version,
        "tolerance": TOLERANCE,
        "snap_tolerance": SNAP_TOLERANCE,
        "mode": mode,
        "vertices_input": len(vertices),
        "edges_input": len(edges),
        "start_points": len(starts),
        "legs": len(sequence_data["legs"]),
        "raw_success_legs": sequence_data["successful_raw_legs"],
        "raw_failed_legs": sequence_data["failed_raw_legs"],
        "raw_total_length": sequence_data["total_length_raw"],
        "topologic_enabled": USE_TOPOLOGIC,
        "edge_weights": USE_EDGE_WEIGHTS,
        "raw_weighted": RAW_WEIGHTED,
        "prefer_infra_edges": PREFER_INFRA_EDGES,
        "require_match": REQUIRE_MATCH,
        "include_end_point_vertex": INCLUDE_END_POINT_VERTEX,
        "vertical_bridging_fallback": ENABLE_VERTICAL_BRIDGING_FALLBACK,
        "final_stub_fallback": ENABLE_FINAL_STUB_FALLBACK,
        "chain_fallback_enabled": ENABLE_CHAIN_FALLBACK if mode=="direct" else False,
        "chain_segments_success": chain_segments_success if mode=="direct" else 0,
        "duration_sec": time.time()-start_time
    }

    payload={"meta": meta, "sequence": sequence_data}
    with open(output_path,"w") as f:
        json.dump(payload,f,indent=2)
    return payload

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def resolve_paths_and_mode():
    script_dir=os.path.dirname(os.path.abspath(__file__))
    args=[a for a in sys.argv[1:] if not a.startswith("--")]
    mode="direct"
    if "--chain" in sys.argv: mode="chain"
    if "--direct" in sys.argv: mode="direct"
    if args:
        candidate=os.path.abspath(args[0])
        if os.path.isdir(candidate):
            ip=os.path.join(candidate, DEFAULT_JSON_INPUT_FILENAME)
        else:
            ip=candidate
    else:
        ip=os.path.join(script_dir, DEFAULT_JSON_INPUT_FILENAME)
    if not os.path.isfile(ip):
        raise FileNotFoundError(f"Input JSON not found: {ip}")
    op=os.path.join(os.path.dirname(ip), JSON_OUTPUT_FILENAME)
    return ip, op, mode

if __name__=="__main__":
    in_path, out_path, mode = resolve_paths_and_mode()
    payload = process(in_path, out_path, mode)
    m=payload["meta"]
    print(f"Mode={m['mode']} legs={m['legs']} raw_ok={m['raw_success_legs']} raw_fail={m['raw_failed_legs']} Output={out_path}")
