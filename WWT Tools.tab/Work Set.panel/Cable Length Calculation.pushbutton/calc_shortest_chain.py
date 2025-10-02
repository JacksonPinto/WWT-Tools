#! python3
# calc_shortest_chain.py (revised to preserve full edge path)
import os, sys, json, math, time
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Graph import Graph
from topologicpy.Wire import Wire
from topologicpy.Dictionary import Dictionary

# ---------------- CONFIG (adjust as needed) ----------------
TOLERANCE = 1e-6                 # smaller to avoid corner loss
MAP_TOLERANCE = 5
DEVICE_EDGE_PENALTY_FACTOR = 1e-4
PRESERVE_ALL_EDGES = True        # True => skip SelfMerge to keep original segmentation
INDEX_MATCH_TOL = 1e-9           # map full path coords back to original vertex indices
LOG_PREFIX = "[CALC]"
# -----------------------------------------------------------

def log(msg): print("{} {}".format(LOG_PREFIX, msg))

def dist3(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def nearest_vertex_index(pt, coords):
    md=float('inf'); mi=None
    for i,c in enumerate(coords):
        d=dist3(pt,c)
        if d<md:
            md=d; mi=i
    return mi,md

def map_coord_to_index(pt, original_vertices, tol=INDEX_MATCH_TOL):
    px,py,pz = pt
    for i,(x,y,z) in enumerate(original_vertices):
        if abs(x-px)<=tol and abs(y-py)<=tol and abs(z-pz)<=tol:
            return i
    idx,_ = nearest_vertex_index(pt, original_vertices)
    return idx

def main():
    t0=time.time()
    script_dir=os.path.dirname(os.path.abspath(__file__))
    json_path=os.path.join(script_dir,"topologic.JSON")
    if not os.path.isfile(json_path):
        log("ERROR missing topologic.JSON"); sys.exit(1)
    data=json.load(open(json_path,"r"))

    vertices      = data.get("vertices", [])
    infra_edges   = data.get("infra_edges", [])
    device_edges  = data.get("device_edges", [])
    combined      = data.get("edges", [])
    starts        = data.get("start_points", [])
    end_pt        = data.get("end_point", None)

    if infra_edges and device_edges:
        edges_source=(infra_edges, device_edges)
    else:
        edges_source=(combined, [])

    log("Vertices:{} InfraEdges:{} DeviceEdges:{} TotalEdges:{} ChainPoints:{}".format(
        len(vertices), len(infra_edges), len(device_edges), len(combined), len(starts)
    ))

    topo_vertices=[Vertex.ByCoordinates(*v) for v in vertices]

    def make_edge(i,j,cat):
        v1=topo_vertices[i]; v2=topo_vertices[j]
        e=Edge.ByVertices(v1,v2)
        length=dist3(vertices[i],vertices[j])
        cost=length*(DEVICE_EDGE_PENALTY_FACTOR if cat=="JUMPER" else 1.0)
        d=Dictionary.ByKeysValues(["category","length","cost"],[cat,length,cost])
        try: e.SetDictionary(d)
        except: pass
        return e

    edge_objs=[]
    for i,j in edges_source[0]:
        edge_objs.append(make_edge(i,j,"INFRA"))
    for i,j in edges_source[1]:
        edge_objs.append(make_edge(i,j,"JUMPER"))

    cluster = Cluster.ByTopologies(*edge_objs)
    if PRESERVE_ALL_EDGES:
        merged_or_cluster = cluster
    else:
        merged_or_cluster = Topology.SelfMerge(cluster, tolerance=TOLERANCE)

    graph = Graph.ByTopology(merged_or_cluster, tolerance=TOLERANCE)
    graph_vertices = Graph.Vertices(graph)
    coords=[v.Coordinates() for v in graph_vertices]
    log("Graph vertices: {}".format(len(coords)))

    # Map device start points
    mapped_points=[]
    for sp in starts:
        coord=sp.get("point")
        eid=sp.get("element_id")
        seq_index=sp.get("seq_index")
        if not (isinstance(coord,list) and len(coord)==3):
            continue
        idx,d=nearest_vertex_index(coord,coords)
        if d>MAP_TOLERANCE:
            log("WARN mapping {} distance {:.6f}".format(eid,d))
        mapped_points.append({
            "element_id":eid,
            "vertex":graph_vertices[idx],
            "coord":coord,
            "mapped_distance":d,
            "seq_index":seq_index
        })
    log("Start point sequence (seq_index -> element_id): {}".format(
        ["{}->{}".format(mp.get("seq_index"), mp.get("element_id")) for mp in mapped_points]
    ))

    if not (isinstance(end_pt,list) and len(end_pt)==3):
        log("ERROR invalid end point"); sys.exit(1)
    end_idx,end_d=nearest_vertex_index(end_pt,coords)
    if end_d>MAP_TOLERANCE:
        log("WARN end mapping distance {:.6f}".format(end_d))
    end_v=graph_vertices[end_idx]

    chain_list = mapped_points + [{
        "element_id": None,
        "vertex": end_v,
        "coord": end_pt,
        "mapped_distance": end_d,
        "seq_index": None
    }]

    results=[]
    success=0
    for i in range(len(chain_list)-1):
        orig=chain_list[i]; dest=chain_list[i+1]
        eid=orig["element_id"]
        seq_idx=orig.get("seq_index")
        sv=orig["vertex"]; dv=dest["vertex"]
        mdist=orig["mapped_distance"]
        try:
            wire=Graph.ShortestPath(graph, sv, dv, edgeKey="cost", tolerance=TOLERANCE)
            if not wire:
                results.append({
                    "start_index":i,
                    "element_id":eid,
                    "seq_index":seq_idx,
                    "length":None,
                    "length_key":None,
                    "length_full":None,
                    "key_vertices_path_xyz":[],
                    "full_edge_path_xyz":[],
                    "full_edge_path_indices":[],
                    "mapped_distance":mdist
                })
                continue

            # Simplified key vertices
            key_vertices = Wire.Vertices(wire)
            key_path_xyz = [v.Coordinates() for v in key_vertices]

            # Edge sequence (full detail)
            edges_seq = Wire.Edges(wire)
            full_coords=[]
            # accumulate vertices per edge preserving order
            for e in edges_seq:
                evs = Edge.Vertices(e)
                if not evs: continue
                c0 = evs[0].Coordinates()
                c1 = evs[-1].Coordinates()
                if not full_coords:
                    full_coords.append(c0)
                # avoid duplicate
                if full_coords[-1] != c0:
                    full_coords.append(c0)
                full_coords.append(c1)

            # Deduplicate consecutive
            cleaned=[]
            for c in full_coords:
                if not cleaned or cleaned[-1]!=c:
                    cleaned.append(c)
            full_coords = cleaned

            # Compute lengths
            length_key = 0.0
            for j in range(len(key_path_xyz)-1):
                length_key += dist3(key_path_xyz[j], key_path_xyz[j+1])

            # True length from edge objects (avoid chord under-estimation)
            length_full = 0.0
            for e in edges_seq:
                try:
                    # If dictionary stored length
                    dct = e.Dictionary()
                    if dct:
                        keys = dct.Keys()
                        vals = dct.Values()
                        if "length" in keys:
                            length_full += vals[keys.index("length")]
                            continue
                except:
                    pass
                # Fallback geometric length
                evs = Edge.Vertices(e)
                if len(evs)==2:
                    cA = evs[0].Coordinates()
                    cB = evs[1].Coordinates()
                    length_full += dist3(cA,cB)

            # Map to original vertices indices
            full_indices = [map_coord_to_index(c, vertices) for c in full_coords]

            results.append({
                "start_index":i,
                "element_id":eid,
                "seq_index":seq_idx,
                "length":length_key,              # legacy (simplified)
                "length_key":length_key,
                "length_full":length_full,        # accurate
                "key_vertices_path_xyz":key_path_xyz,
                "full_edge_path_xyz":full_coords,
                "full_edge_path_indices":full_indices,
                "mapped_distance":mdist
            })
            success += 1
        except Exception as ex:
            log("ERROR path element {}: {}".format(eid,ex))
            results.append({
                "start_index":i,
                "element_id":eid,
                "seq_index":seq_idx,
                "length":None,
                "length_key":None,
                "length_full":None,
                "key_vertices_path_xyz":[],
                "full_edge_path_xyz":[],
                "full_edge_path_indices":[],
                "mapped_distance":mdist
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
            "graph_vertices":len(coords),
            "chain_points":len(chain_list),
            "mapped_chain":len(mapped_points),
            "paths_success":success,
            "paths_failed":len(chain_list)-1-success,
            "duration_sec":time.time()-t0,
            "preserve_all_edges":PRESERVE_ALL_EDGES
        },
        "results":results
    }
    out_path=os.path.join(script_dir,"topologic_results.json")
    json.dump(out, open(out_path,"w"), indent=2)
    log("Results written: {}".format(out_path))
    log("Summary: {} success / {} total".format(success, len(chain_list)-1))

if __name__ == "__main__":
    main()