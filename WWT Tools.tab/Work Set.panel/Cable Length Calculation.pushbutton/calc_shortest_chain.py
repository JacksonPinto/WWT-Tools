import json
import os
import math
import time
import sys
from collections import deque, defaultdict
from typing import List, Dict, Any, Tuple, Optional

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_JSON_INPUT_FILENAME = "topologic.JSON"
JSON_OUTPUT_FILENAME = "topologic_results.json"

TOLERANCE = 0.000001          # Very small to avoid unintended merges
SNAP_TOLERANCE = 1e-9         # Exact coordinate match requirement
EDGE_WEIGHT_KEY = "w"         # Edge cost dictionary key
USE_TOPOLOGIC = True          # Use Topologic.Graph in comparison
USE_EDGE_WEIGHTS = True       # Store geometric length in edge dictionaries
RAW_USE_WEIGHTS = True        # Dijkstra weighted by geometric length
REQUIRE_MATCH = False         # If True, discard Topologic path if deviates from raw
EXPORT_GRAPH_JSON = False
GRAPH_EXPORT_FILENAME = "topologic_script.json"
OVERWRITE_GRAPH_EXPORT = True

DEBUG = ("--debug" in sys.argv) or (os.environ.get("DEBUG") == "1")
def dprint(*a): 
    if DEBUG: print("[DEBUG]", *a)

# ------------------------------------------------------------------
# Load JSON
# ------------------------------------------------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

# ------------------------------------------------------------------
# Raw Graph (exact indices)
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
        a = self.vertices[u]; b = self.vertices[v]
        return math.dist(a,b)

    def shortest_path(self, s:int, t:int, weighted:bool) -> Tuple[List[int], List[int], float]:
        if s==t:
            return [s],[ -1 ],0.0
        if weighted:
            import heapq
            dist=[float("inf")]*len(self.vertices)
            prev=[(-1,-1)]*len(self.vertices) # (prevVertex, edgeId)
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
                        dist[v]=nd
                        prev[v]=(u,eid)
                        heapq.heappush(h,(nd,v))
            if dist[t]==float("inf"):
                return [],[],0.0
            path=[]; ep=[]
            cur=t
            while cur!=s:
                p,eid=prev[cur]
                path.append(cur)
                ep.append(eid)
                cur=p
            path.append(s)
            path.reverse()
            ep.reverse()
            return path,ep,dist[t]
        else:
            # BFS unweighted
            q=deque([s])
            prev={s:(-1,-1)}
            while q:
                u=q.popleft()
                if u==t: break
                for (v,eid) in self.adj[u]:
                    if v not in prev:
                        prev[v]=(u,eid)
                        q.append(v)
            if t not in prev:
                return [],[],0.0
            path=[]; ep=[]
            cur=t
            while cur!=s:
                p,eid=prev[cur]
                path.append(cur); ep.append(eid)
                cur=p
            path.append(s)
            path.reverse(); ep.reverse()
            # length = sum of geometric lengths
            length = sum(self.edge_length(eid) for eid in ep)
            return path, ep, length

# ------------------------------------------------------------------
# Topologic build (no wire simplification)
# ------------------------------------------------------------------
def build_topologic(vertices: List[List[float]], edges: List[List[int]]):
    try:
        import topologicpy
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
    except Exception as e:
        dprint("Topologic import failed:", e)
        return None,[],[]

    vertex_objs=[Vertex.ByCoordinates(*c) for c in vertices]
    edge_objs=[]
    for (u,v) in edges:
        try:
            e = Edge.ByStartVertexEndVertex(vertex_objs[u], vertex_objs[v])
            if USE_EDGE_WEIGHTS:
                length = math.dist(vertices[u], vertices[v])
                # assign length dictionary
                try:
                    Topology.SetDictionary(e, {EDGE_WEIGHT_KEY: length})
                except Exception:
                    pass
            edge_objs.append(e)
        except Exception as e:
            dprint("Edge create fail", (u,v), e)

    # Do NOT ask for topologyType="Wire" to avoid simplification
    geom = vertex_objs + edge_objs
    topo=None
    try:
        topo = Topology.ByGeometry(geometry=geom, tolerance=TOLERANCE)
    except TypeError:
        try:
            topo = Topology.ByGeometry(geom, TOLERANCE)
        except Exception:
            topo=None
    if topo is None:
        # fallback: structured (still may aggregate)
        topo = Topology.ByGeometry(vertices=vertices, edges=edges, faces=[], topologyType=None, tolerance=TOLERANCE)

    cluster = Cluster.ByTopologies([topo])
    graph=None
    try:
        graph = Graph.ByTopology(cluster, tolerance=TOLERANCE)
    except Exception as e:
        dprint("Graph build failed:", e)
        return None, vertex_objs, edge_objs
    return graph, vertex_objs, edge_objs

def export_graph(graph, directory):
    if not graph or not EXPORT_GRAPH_JSON: return
    try:
        from topologicpy.Graph import Graph
        path=os.path.join(directory, GRAPH_EXPORT_FILENAME)
        try:
            Graph.ExportToJSON(graph, path, overwrite=OVERWRITE_GRAPH_EXPORT)
        except TypeError:
            if OVERWRITE_GRAPH_EXPORT and os.path.exists(path):
                try: os.remove(path)
                except: pass
            Graph.ExportToJSON(graph, path)
    except Exception as e:
        dprint("Graph export failed:", e)

# ------------------------------------------------------------------
# Topologic path
# ------------------------------------------------------------------
def topo_shortest(graph, vertices_cache, start_coord, end_coord, use_weights:bool):
    if not graph: 
        return {"status":"no_graph"}
    try:
        from topologicpy.Graph import Graph
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
    except:
        return {"status":"no_graph"}

    gvs = vertices_cache or []
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
        # nearest
        best=None; best_d2=1e100
        for gv in gvs:
            d2=(Vertex.X(gv)-c[0])**2+(Vertex.Y(gv)-c[1])**2+(Vertex.Z(gv)-c[2])**2
            if d2<best_d2: best_d2=d2; best=gv
        return best

    sv = map_coord(start_coord)
    ev = map_coord(end_coord)
    if not sv or not ev:
        return {"status":"unmapped"}
    if sv==ev:
        return {"status":"same_vertex", "coords":[start_coord], "length":0.0}

    edgeKey = EDGE_WEIGHT_KEY if use_weights else ""
    try:
        wire = Graph.ShortestPath(graph, sv, ev, "", edgeKey)
    except Exception as e:
        dprint("Graph.ShortestPath error:", e)
        wire=None
    if not wire:
        return {"status":"no_path"}
    try:
        vs = Topology.Vertices(wire) or []
        coords=[(Vertex.X(v),Vertex.Y(v),Vertex.Z(v)) for v in vs]
    except Exception:
        coords=[]
    if len(coords)<2:
        return {"status":"no_path"}
    length=sum(math.dist(coords[i-1],coords[i]) for i in range(1,len(coords)))
    return {"status":"ok", "coords":coords, "length":length}

# ------------------------------------------------------------------
# Compare raw vs topologic paths
# ------------------------------------------------------------------
def compare_paths(raw_indices:List[int], topo_coords:List[Tuple[float,float,float]], vertices:List[List[float]]) -> bool:
    if not raw_indices or not topo_coords: 
        return False
    # Convert raw indices to coords
    raw_coords=[tuple(vertices[i]) for i in raw_indices]
    # Normalizing lengths
    if len(raw_coords)!=len(topo_coords):
        return False
    for a,b in zip(raw_coords, topo_coords):
        if any(abs(a[i]-b[i])>SNAP_TOLERANCE for i in range(3)):
            return False
    return True

# ------------------------------------------------------------------
# Sequence routing
# ------------------------------------------------------------------
def process_sequence(vertices, edges, starts, end_point):
    raw = RawGraph(vertices, edges)
    graph = None; gvs=[]
    if USE_TOPOLOGIC:
        graph, _, _ = build_topologic(vertices, edges)
        if graph:
            from topologicpy.Graph import Graph as TG
            try:
                gvs = TG.Vertices(graph) or []
            except:
                gvs=[]

    ordered = sorted(starts, key=lambda s: s.get("seq_index",0))
    legs=[]
    success=0
    failed=0
    cumulative=[]

    def coord_to_index(c):
        idx, d = raw.nearest_vertex(c)
        exact = d <= SNAP_TOLERANCE
        return idx, exact, d

    for i in range(len(ordered)-1):
        a=ordered[i]; b=ordered[i+1]
        ac=tuple(a["point"]); bc=tuple(b["point"])
        ai, a_exact, a_snap = coord_to_index(ac)
        bi, b_exact, b_snap = coord_to_index(bc)

        raw_path, raw_edges, raw_len = raw.shortest_path(ai, bi, RAW_USE_WEIGHTS)
        if raw_path:
            success += 1
            raw_coords=[vertices[j] for j in raw_path]
            if cumulative: cumulative.extend(raw_coords[1:])
            else: cumulative.extend(raw_coords)
        else:
            failed += 1
            raw_coords=[]; raw_len=0.0

        topo_info={"status":"skipped"}
        if USE_TOPOLOGIC:
            topo_info = topo_shortest(graph, gvs, ac, bc, USE_EDGE_WEIGHTS)
        matched = False
        if topo_info.get("status")=="ok":
            matched = compare_paths(raw_path, topo_info.get("coords",[]), vertices)
            if REQUIRE_MATCH and not matched:
                topo_info["status"]="mismatch_discarded"

        legs.append({
            "leg_type":"sequence",
            "from_seq_index":a.get("seq_index"),
            "to_seq_index":b.get("seq_index"),
            "from_element_id":a.get("element_id"),
            "to_element_id":b.get("element_id"),
            "raw_vertex_indices": raw_path,
            "raw_edge_indices": raw_edges,
            "raw_length": raw_len,
            "raw_success": bool(raw_path),
            "topologic_status": topo_info.get("status"),
            "topologic_length": topo_info.get("length"),
            "topologic_coords": topo_info.get("coords"),
            "topologic_matches_raw": matched,
            "start_exact": a_exact,
            "end_exact": b_exact,
            "start_snap_distance": a_snap,
            "end_snap_distance": b_snap
        })

    # Final leg to end
    if ordered:
        last=ordered[-1]
        lc=tuple(last["point"])
        ec=tuple(end_point)
        li, l_exact, l_snap = coord_to_index(lc)
        ei, e_exact, e_snap = coord_to_index(ec)

        raw_path, raw_edges, raw_len = raw.shortest_path(li, ei, RAW_USE_WEIGHTS)
        if raw_path:
            success += 1
            raw_coords=[vertices[j] for j in raw_path]
            if cumulative: cumulative.extend(raw_coords[1:])
            else: cumulative.extend(raw_coords)
        else:
            failed += 1
            raw_coords=[]; raw_len=0.0

        topo_info={"status":"skipped"}
        if USE_TOPOLOGIC:
            topo_info = topo_shortest(graph, gvs, lc, ec, USE_EDGE_WEIGHTS)
        matched=False
        if topo_info.get("status")=="ok":
            matched = compare_paths(raw_path, topo_info.get("coords",[]), vertices)
            if REQUIRE_MATCH and not matched:
                topo_info["status"]="mismatch_discarded"

        legs.append({
            "leg_type":"final_to_end",
            "from_seq_index": last.get("seq_index"),
            "to_end": True,
            "from_element_id": last.get("element_id"),
            "raw_vertex_indices": raw_path,
            "raw_edge_indices": raw_edges,
            "raw_length": raw_len,
            "raw_success": bool(raw_path),
            "topologic_status": topo_info.get("status"),
            "topologic_length": topo_info.get("length"),
            "topologic_coords": topo_info.get("coords"),
            "topologic_matches_raw": matched,
            "start_exact": l_exact,
            "end_exact": e_exact,
            "start_snap_distance": l_snap,
            "end_snap_distance": e_snap
        })

    return {
        "legs": legs,
        "total_length_raw": sum(l["raw_length"] for l in legs),
        "successful_raw_legs": sum(1 for l in legs if l["raw_success"]),
        "failed_raw_legs": sum(1 for l in legs if not l["raw_success"]),
        "cumulative_vertex_path_xyz": cumulative
    }, graph

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def process(input_path: str, output_path: str):
    start_time=time.time()
    data = load_json(input_path)
    vertices = data.get("vertices", [])
    edges = data.get("edges") or data.get("infra_edges") or []
    if not vertices or not edges:
        raise ValueError("Missing vertices or edges in JSON.")
    start_points = data.get("start_points", [])
    end_point = data.get("end_point")
    if not end_point or len(end_point)!=3:
        raise ValueError("end_point missing or malformed")

    seq_data, graph = process_sequence(vertices, edges, start_points, end_point)
    export_graph(graph, os.path.dirname(input_path))
    version = getattr(sys.modules.get("topologicpy",""),"__version__","unknown")

    # Evaluate topologic path usage stats
    topo_ok = sum(1 for l in seq_data["legs"] if l["topologic_status"]=="ok")
    topo_match = sum(1 for l in seq_data["legs"] if l.get("topologic_matches_raw"))
    meta = {
        "version": version,
        "tolerance": TOLERANCE,
        "snap_tolerance": SNAP_TOLERANCE,
        "vertices_input": len(vertices),
        "edges_input": len(edges),
        "start_points": len(start_points),
        "legs": len(seq_data["legs"]),
        "raw_success_legs": seq_data["successful_raw_legs"],
        "raw_failed_legs": seq_data["failed_raw_legs"],
        "raw_total_length": seq_data["total_length_raw"],
        "topologic_enabled": USE_TOPOLOGIC,
        "topologic_ok_legs": topo_ok,
        "topologic_match_raw_legs": topo_match,
        "edge_weights": USE_EDGE_WEIGHTS,
        "raw_weighted": RAW_USE_WEIGHTS,
        "require_match": REQUIRE_MATCH,
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
        raise FileNotFoundError(f"Input JSON not found: {ip}")
    op=os.path.join(os.path.dirname(ip), JSON_OUTPUT_FILENAME)
    return ip, op

if __name__=="__main__":
    in_path, out_path = resolve_paths()
    payload = process(in_path, out_path)
    m=payload["meta"]
    print(f"Legs={m['legs']} raw_ok={m['raw_success_legs']} raw_fail={m['raw_failed_legs']} "
          f"topo_ok={m['topologic_ok_legs']} topo_match_raw={m['topologic_match_raw_legs']} Output={out_path}")
