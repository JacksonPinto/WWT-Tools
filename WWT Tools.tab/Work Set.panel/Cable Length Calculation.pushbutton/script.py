# -*- coding: utf-8 -*-
# Export Infrastructure + Projected L Jumpers (Edge Splitting)
# Version: 3.0.0 (IronPython/Revit 2022+ and 2026 tested)
#
# (Unchanged header text omitted for brevity)
from Autodesk.Revit.DB import (
    FilteredElementCollector, BuiltInCategory, UV,
    FamilyInstance, LocationCurve, FamilySymbol, BuiltInParameter
)
from Autodesk.Revit.UI.Selection import ObjectType
from pyrevit import forms
import os, json, math, traceback, datetime

# ---------------- CONFIG ----------------
SHOW_SCRIPT_PATH = False

USE_INFRA_CATEGORY_DIALOG  = True
USE_DEVICE_CATEGORY_DIALOG = True

# Geometry sampling & tolerances
SUBDIVIDE_ARCS = True
CURVE_SEG_LENGTH_FT = 0.25
MERGE_TOL = 5e-4
PROJECTION_ENDPOINT_TOL = 1e-2
ROUND_PREC = 6

# Fittings
FITTING_CONNECTOR_TO_POINT = True
INCLUDE_FITTING_WITHOUT_CONNECTORS = True

# L jumper segment creation
CREATE_VERTICAL_SEGMENT = True
CREATE_HORIZONTAL_SEGMENT = True
VERTICAL_MIN_LEN = 1e-6

# External processing
PYTHON3_PATH = r"C:\Users\JacksonAugusto\AppData\Local\Programs\Python\Python312\python.exe"
RUN_CALC_SHORTEST = True
CALC_SCRIPT_NAME  = "calc_shortest.py"
CALC_ARGS         = []
FORCE_INTERNAL_CALC   = False

RUN_UPDATE_LENGTHS = True
UPDATE_SCRIPT_NAME = "update_cable_lengths.py"
UPDATE_ARGS        = []
FORCE_INTERNAL_UPDATE = True

FALLBACK_INTERNAL_IF_EXTERNAL_FAILS = True
OPEN_FOLDER_AFTER = False

# Debug / console
PRINT_TO_CONSOLE = True
DEBUG_PROJECTION = False
DEBUG_DEVICE     = False

# --- NEW FLAG: enforce strict sequential manual picking (one-by-one) ---
STRICT_MANUAL_SEQUENCE = True  # Set to False to revert to original multi-pick behavior

# ----------------------------------------
uidoc = __revit__.ActiveUIDocument
doc   = uidoc.Document
active_view = uidoc.ActiveView
if not doc or not active_view:
    forms.alert("Need active document + active view.", exitscript=True)

def cprint(*args):
    if PRINT_TO_CONSOLE:
        try:
            print("[EXPORT]", " ".join([str(a) for a in args]))
        except:
            pass

def to_int_id(obj):
    try:
        if hasattr(obj,'Id'):
            return int(str(obj.Id))
        if hasattr(obj,'IntegerValue'):
            return obj.IntegerValue
        return int(obj)
    except:
        return None

def dist3(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def sample_curve(curve):
    pts=[]
    try:
        sp=curve.GetEndPoint(0); ep=curve.GetEndPoint(1)
        length=curve.Length
        name=curve.GetType().Name.lower()
        if (("line" in name) and ("arc" not in name)) or not SUBDIVIDE_ARCS:
            return [(sp.X,sp.Y,sp.Z),(ep.X,ep.Y,ep.Z)]
        segs=max(2,int(math.ceil(length/CURVE_SEG_LENGTH_FT)))
        i=0
        while i<=segs:
            p=curve.Evaluate(float(i)/segs,True)
            pts.append((p.X,p.Y,p.Z))
            i+=1
        return pts
    except:
        return pts

def norm_key(p):
    return (round(p[0],ROUND_PREC),round(p[1],ROUND_PREC),round(p[2],ROUND_PREC))

def get_all_connectors(el):
    pts=[]; seen=set()
    def add(xyz):
        t=(xyz.X,xyz.Y,xyz.Z)
        if t not in seen:
            seen.add(t); pts.append(t)
    try:
        mep=getattr(el,'MEPModel',None)
        if mep:
            cm=getattr(mep,'ConnectorManager',None)
            if cm and getattr(cm,'Connectors',None):
                for c in cm.Connectors: add(c.Origin)
    except: pass
    try:
        cm2=getattr(el,'ConnectorManager',None)
        if cm2 and getattr(cm2,'Connectors',None):
            for c in cm2.Connectors: add(c.Origin)
    except: pass
    return pts

def get_fitting_location_point(el):
    loc=getattr(el,"Location",None)
    if loc and hasattr(loc,"Point") and getattr(loc,"Point",None):
        p=loc.Point
        return (p.X,p.Y,p.Z),"LocationPoint"
    try:
        if isinstance(el,FamilyInstance):
            tr=el.GetTransform()
            if tr:
                o=tr.Origin
                return (o.X,o.Y,o.Z),"TransformOrigin"
    except: pass
    return None,"None"

def project_point_to_segment(pt, a, b):
    ax,ay,az = a; bx,by,bz = b; px,py,pz = pt
    ab=(bx-ax, by-ay, bz-az); ap=(px-ax, py-ay, pz-az)
    ab2=ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2]
    if ab2==0: return a, 0.0, dist3(pt,a)
    t=(ap[0]*ab[0]+ap[1]*ab[1]+ap[2]*ab[2])/ab2
    if t<0: t=0
    elif t>1: t=1
    cx=ax+ab[0]*t; cy=ay+ab[1]*t; cz=az+ab[2]*t
    d=dist3(pt,(cx,cy,cz))
    return (cx,cy,cz), t, d

def get_symbol_name(symbol):
    try:
        return symbol.Name
    except:
        param = symbol.get_Parameter(BuiltInParameter.SYMBOL_NAME_PARAM)
        if param:
            return param.AsString()
    return "Unknown"

infra_category_map=[
    ("Cable Trays", BuiltInCategory.OST_CableTray),
    ("Cable Tray Fittings", BuiltInCategory.OST_CableTrayFitting),
    ("Conduits", BuiltInCategory.OST_Conduit),
    ("Conduit Fittings", BuiltInCategory.OST_ConduitFitting),
]
device_category_map=[
    ("Electrical Fixtures", BuiltInCategory.OST_ElectricalFixtures),
    ("Electrical Equipment", BuiltInCategory.OST_ElectricalEquipment),
    ("Lighting Fixtures", BuiltInCategory.OST_LightingFixtures),
    ("Data Devices", BuiltInCategory.OST_DataDevices),
    ("Lighting Devices", BuiltInCategory.OST_LightingDevices),
    ("Communication Devices", BuiltInCategory.OST_CommunicationDevices),
    ("Fire Alarm Devices", BuiltInCategory.OST_FireAlarmDevices),
    ("Security Devices", BuiltInCategory.OST_SecurityDevices),
    ("Nurse Call Devices", BuiltInCategory.OST_NurseCallDevices),
    ("Telephone Devices", BuiltInCategory.OST_TelephoneDevices),
]

if USE_INFRA_CATEGORY_DIALOG:
    infra_names=[n for n,_ in infra_category_map]
    chosen=forms.SelectFromList.show(infra_names,title="Select Infrastructure Categories",multiselect=True)
    if not chosen: forms.alert("No infrastructure categories.", exitscript=True)
    infra_cats=[cid for n,cid in infra_category_map if n in chosen]
else:
    infra_cats=[cid for _,cid in infra_category_map]

if USE_DEVICE_CATEGORY_DIALOG:
    dev_names=[n for n,_ in device_category_map]
    dchosen=forms.SelectFromList.show(dev_names,title="Select Device Categories",multiselect=True)
    if not dchosen: forms.alert("No device categories.", exitscript=True)
    device_cat_ids=[cid for n,cid in device_category_map if n in dchosen]
else:
    dchosen=[n for n,_ in device_category_map]
    device_cat_ids=[cid for _,cid in device_category_map]

selection_modes = [
    "Manual Pick Devices",
    "All Devices In Active View",
    "Devices Type In Active View"
]
mode = forms.SelectFromList.show(selection_modes, multiselect=False, title="Device Selection Mode")
if not mode:
    forms.alert("Device selection canceled.", exitscript=True)
selection_mode = (
    "manual" if mode.startswith("Manual")
    else "all" if mode.startswith("All")
    else "bytype"
)

selected_type_ids = None
if selection_mode == "bytype":
    type_choices = []
    type_map = {}
    for cid in device_cat_ids:
        col = FilteredElementCollector(doc, active_view.Id).OfCategory(cid).WhereElementIsNotElementType()
        for el in col:
            symbol_id = el.GetTypeId()
            symbol = doc.GetElement(symbol_id)
            if symbol is not None:
                name = get_symbol_name(symbol)
                cat_name = symbol.Category.Name if symbol.Category else str(cid)
                pretty = "{} - {}".format(cat_name, name)
                if pretty not in type_map:
                    type_choices.append(pretty)
                    type_map[pretty] = symbol_id
    if not type_choices:
        forms.alert("No device types found in active view.", exitscript=True)
    selected_types = forms.SelectFromList.show(
        type_choices, title="Select Device Types In Active View", multiselect=True
    )
    if not selected_types:
        forms.alert("No device types selected.", exitscript=True)
    selected_type_ids = [type_map[name] for name in selected_types]

fitting_cat_set=set([BuiltInCategory.OST_CableTrayFitting, BuiltInCategory.OST_ConduitFitting])
linear_cat_set =set([BuiltInCategory.OST_CableTray, BuiltInCategory.OST_Conduit])

fittings=[]; linears=[]
for cat in infra_cats:
    col=(FilteredElementCollector(doc, active_view.Id).OfCategory(cat).WhereElementIsNotElementType())
    for el in col:
        if cat in fitting_cat_set: fittings.append(el)
        elif cat in linear_cat_set: linears.append(el)

devices=[]

# ---------------- PATCH: Strict sequential manual picking ----------------
if selection_mode == "manual":
    if STRICT_MANUAL_SEQUENCE:
        forms.alert("Sequential Manual Pick: Click devices in desired order. Press ESC when done.")
        seq = 0
        seen_ids = set()
        while True:
            try:
                ref = uidoc.Selection.PickObject(ObjectType.Element, "Pick device #{0} (ESC to finish)".format(seq+1))
                el = doc.GetElement(ref.ElementId)
                if selected_type_ids is not None and el.GetTypeId() not in selected_type_ids:
                    cprint("Skipped (type filter) ElementId", to_int_id(el))
                    continue
                eid = to_int_id(el)
                if eid in seen_ids:
                    cprint("Duplicate pick ignored ElementId", eid)
                    continue
                devices.append(el)
                seen_ids.add(eid)
                cprint("Picked seq {0} ElementId {1}".format(seq, eid))
                seq += 1
            except:
                break
        if not devices:
            forms.alert("No devices picked.", exitscript=True)
    else:
        # original multi-pick approach (unchanged)
        try:
            refs=uidoc.Selection.PickObjects(ObjectType.Element,"Pick device elements")
            for r in refs:
                el=doc.GetElement(r.ElementId)
                if selected_type_ids is not None:
                    if el.GetTypeId() in selected_type_ids:
                        devices.append(el)
                else:
                    devices.append(el)
        except Exception as e:
            forms.alert("Device pick aborted:\n{0}".format(e), exitscript=True)
        if not devices:
            forms.alert("No devices picked.", exitscript=True)
elif selection_mode == "all":
    for cid in device_cat_ids:
        col = FilteredElementCollector(doc, active_view.Id).OfCategory(cid).WhereElementIsNotElementType()
        for el in col:
            devices.append(el)
elif selection_mode == "bytype":
    try:
        for type_id in selected_type_ids:
            col = FilteredElementCollector(doc, active_view.Id).OfTypeId(type_id).WhereElementIsNotElementType()
            for el in col:
                devices.append(el)
    except AttributeError:
        for cid in device_cat_ids:
            col = FilteredElementCollector(doc, active_view.Id).OfCategory(cid).WhereElementIsNotElementType()
            for el in col:
                if el.GetTypeId() in selected_type_ids:
                    devices.append(el)
# --------------- END PATCH ----------------

cprint("Final device order (ElementIds):", [to_int_id(d) for d in devices])

vertices=[]
vertex_map={}
infra_edges=[]
fitting_points=[]
device_edges=[]
start_points=[]

calc_modes = [
    "Individual Single Cable",
    "Daisy Chain Connection"
]
user_mode = forms.SelectFromList.show(
    calc_modes,
    title="Modo de cálculo",
    multiselect=False
)
if not user_mode:
    forms.alert("Cálculo cancelado pelo usuário.", exitscript=True)

if user_mode.startswith("Daisy"):
    CALC_SCRIPT_NAME = "calc_shortest_chain.py"
else:
    CALC_SCRIPT_NAME = "calc_shortest.py"

def add_vertex(pt):
    nk=norm_key(pt)
    idx=vertex_map.get(nk)
    if idx is not None: return idx
    i=0
    while i < len(vertices):
        v=vertices[i]
        if (abs(v[0]-pt[0])<=MERGE_TOL and
            abs(v[1]-pt[1])<=MERGE_TOL and
            abs(v[2]-pt[2])<=MERGE_TOL):
            vertex_map[nk]=i
            return i
        i+=1
    vertices.append([pt[0],pt[1],pt[2]])
    idx=len(vertices)-1
    vertex_map[nk]=idx
    return idx

def add_edge(i,j,edge_list):
    if i==j: return
    if i>j: i,j=j,i
    for (a,b) in edge_list:
        if a==i and b==j: return
    edge_list.append([i,j])

fittings_total=0
fittings_lp_locationpoint=0
fittings_lp_transform=0
fittings_lp_none=0

for el in fittings:
    fittings_total+=1
    lp, method = get_fitting_location_point(el)
    if not lp:
        fittings_lp_none+=1
        continue
    if method=="LocationPoint": fittings_lp_locationpoint+=1
    elif method=="TransformOrigin": fittings_lp_transform+=1
    fitting_points.append({
        "element_id": to_int_id(el),
        "method": method,
        "point": [lp[0],lp[1],lp[2]]
    })
    cons=get_all_connectors(el)
    if cons:
        base_idx=add_vertex(lp)
        for c in cons:
            ci=add_vertex(c)
            add_edge(ci, base_idx, infra_edges if FITTING_CONNECTOR_TO_POINT else infra_edges)
    elif INCLUDE_FITTING_WITHOUT_CONNECTORS:
        add_vertex(lp)

linear_total=0; linear_curve_edges=0
for el in linears:
    loc=getattr(el,"Location",None)
    if not (loc and isinstance(loc,LocationCurve) and loc.Curve): continue
    pts=sample_curve(loc.Curve)
    if len(pts)<2: continue
    linear_total+=1
    for a,b in zip(pts, pts[1:]):
        if dist3(a,b)>1e-9:
            ia=add_vertex(a); ib=add_vertex(b)
            add_edge(ia,ib,infra_edges)
            linear_curve_edges+=1

device_points=[]
for d in devices:
    loc=getattr(d,"Location",None)
    if loc and hasattr(loc,"Point") and loc.Point:
        p=loc.Point
        device_points.append((d,(p.X,p.Y,p.Z)))

if not device_points:
    forms.alert("No device point locations found.", exitscript=True)

forms.alert("Pick a FACE for End Point (sink)")
try:
    face_ref=uidoc.Selection.PickObject(ObjectType.Face,"Pick End Face")
except Exception as e:
    forms.alert("End face selection aborted:\n{0}".format(e), exitscript=True)

def face_center(ref):
    el=doc.GetElement(ref.ElementId)
    geom=el.GetGeometryObjectFromReference(ref)
    if not geom: return None
    try:
        bbox=geom.GetBoundingBox()
        uv=UV((bbox.Min.U+bbox.Max.U)/2.0,(bbox.Min.V+bbox.Max.V)/2.0)
        xyz=geom.Evaluate(uv)
        return (xyz.X,xyz.Y,xyz.Z)
    except: return None

end_xyz=face_center(face_ref)
if not end_xyz:
    forms.alert("Failed computing end point.", exitscript=True)
end_idx=add_vertex(end_xyz)

def split_edge(edge_index, proj_pt, infra_edges):
    a,b = infra_edges[edge_index]
    del infra_edges[edge_index]
    p_idx = add_vertex(proj_pt)
    add_edge(a, p_idx, infra_edges)
    add_edge(p_idx, b, infra_edges)
    return p_idx

def project_device(device_pt):
    best_edge_index=None
    best_proj=None
    best_t=None
    best_dist=None
    i=0
    while i < len(infra_edges):
        a_idx,b_idx = infra_edges[i]
        a=vertices[a_idx]; b=vertices[b_idx]
        proj, t, d = project_point_to_segment(device_pt, a, b)
        if best_dist is None or d < best_dist:
            best_dist=d; best_edge_index=i; best_proj=proj; best_t=t
        i+=1
    return best_edge_index, best_proj, best_t, best_dist

# Device processing with seq_index (unchanged from earlier patch)
for seq_idx, (d,(dx,dy,dz)) in enumerate(device_points):
    dev_id=to_int_id(d)
    device_vertex_idx = add_vertex((dx,dy,dz))
    start_points.append({"element_id": dev_id if dev_id is not None else -1,
                         "point":[dx,dy,dz],
                         "seq_index": seq_idx})
    edge_idx, proj_pt, t_val, pdist = project_device((dx,dy,dz))
    if edge_idx is None:
        continue
    a_idx,b_idx = infra_edges[edge_idx]
    a=vertices[a_idx]; b=vertices[b_idx]
    if dist3(proj_pt,a) <= PROJECTION_ENDPOINT_TOL:
        proj_vertex_idx = a_idx
    elif dist3(proj_pt,b) <= PROJECTION_ENDPOINT_TOL:
        proj_vertex_idx = b_idx
    else:
        proj_vertex_idx = split_edge(edge_idx, proj_pt, infra_edges)
        if DEBUG_PROJECTION:
            cprint("Split edge; new vertex", proj_vertex_idx)
    if CREATE_VERTICAL_SEGMENT:
        v_pt = (dx,dy,vertices[proj_vertex_idx][2])
        if abs(v_pt[2]-dz) <= VERTICAL_MIN_LEN:
            vertical_idx = device_vertex_idx
        else:
            vertical_idx = add_vertex(v_pt)
            add_edge(device_vertex_idx, vertical_idx, device_edges)
        if CREATE_HORIZONTAL_SEGMENT:
            add_edge(vertical_idx, proj_vertex_idx, device_edges)
        else:
            if vertical_idx != proj_vertex_idx:
                add_edge(vertical_idx, proj_vertex_idx, device_edges)
    else:
        add_edge(device_vertex_idx, proj_vertex_idx, device_edges)

cprint("Manual pick sequence (seq_index -> element_id):",
       ["{}->{}".format(sp["seq_index"], sp["element_id"]) for sp in start_points])

meta={
    "version":"3.0.0",
    "vertex_count":len(vertices),
    "infra_edge_count":len(infra_edges),
    "device_edge_count":len(device_edges),
    "device_count":len(start_points),
    "projection_endpoint_tol":PROJECTION_ENDPOINT_TOL,
    "merge_tol":MERGE_TOL,
    "vertical_segment":CREATE_VERTICAL_SEGMENT,
    "horizontal_segment":CREATE_HORIZONTAL_SEGMENT
}

combined_edges = infra_edges + device_edges

graph_data={
    "meta":meta,
    "vertices":vertices,
    "edges":combined_edges,
    "infra_edges":infra_edges,
    "device_edges":device_edges,
    "start_points":start_points,
    "end_point":[end_xyz[0],end_xyz[1],end_xyz[2]]
}

script_dir=os.path.dirname(__file__)
json_path=os.path.join(script_dir,"topologic.JSON")
with open(json_path,"w") as f:
    json.dump(graph_data,f,indent=2)

cprint("EXPORT SUMMARY vertices={0} infraEdges={1} deviceEdges={2} totalEdges={3} devices={4}".format(
    len(vertices), len(infra_edges), len(device_edges), len(combined_edges), len(start_points)
))

forms.alert(
    "GRAPH DONE\nVertices:{0}\nInfraEdges:{1}\nDeviceEdges:{2}\nTotalEdges:{3}\nDevices:{4}\nJSON:\n{5}".format(
        len(vertices), len(infra_edges), len(device_edges), len(combined_edges), len(start_points), json_path
    ),
    title="Cable Length Calculation"
)

import subprocess
def run_external_or_internal(script_path, interpreter, args, allow_fallback, force_internal, log_basename,
                             injected_globals=None):
    import subprocess
    external_ok=False
    stdout_data=""; stderr_data=""
    status=""
    log_path=os.path.join(os.path.dirname(script_path), log_basename)
    if (not force_internal) and interpreter and os.path.isfile(interpreter):
        try:
            cmd=[interpreter, script_path] + list(args)
            proc=subprocess.Popen(cmd, cwd=os.path.dirname(script_path),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True)
            out,err=proc.communicate()
            stdout_data, stderr_data = out, err
            rc=proc.returncode
            if rc==0:
                external_ok=True
                status="External OK rc=0"
            else:
                status="External FAILED rc={0}".format(rc)
        except Exception as ex:
            status="External exception: {0}".format(ex)
    elif not force_internal:
        status="External interpreter invalid: {0}".format(interpreter)

    if (force_internal or (not external_ok and allow_fallback)):
        try:
            g={}
            if injected_globals: g.update(injected_globals)
            g['__file__']=script_path; g['__name__']='__main__'
            code=open(script_path,'r').read()
            exec(compile(code, script_path, 'exec'), g, g)
            if external_ok: status += "\nInternal EXEC SUCCESS."
            else: status += "\nInternal fallback SUCCESS."
        except Exception as ie:
            tb=traceback.format_exc()
            status += "\nInternal EXEC FAILED: {0}".format(ie)
            stderr_data += "\n[INTERNAL TRACEBACK]\n{0}".format(tb)

    try:
        lf=open(log_path,"a")
        lf.write("\n=== {0} | {1} ===\n".format(os.path.basename(script_path),
                                                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        lf.write("STATUS: {0}\n".format(status))
        if stdout_data:
            lf.write("--- STDOUT ---\n"); lf.write(stdout_data); lf.write("\n")
        if stderr_data:
            lf.write("--- STDERR ---\n"); lf.write(stderr_data); lf.write("\n")
        lf.close()
    except: pass

    return status, external_ok, stdout_data[:1200], stderr_data[:1200], log_path

injected_globals = {
    '__revit__': __revit__,
    'uidoc': uidoc,
    'doc': doc,
    'active_view': active_view
}

def run_step(run_flag, script_name, force_internal, args_list, title):
    if not run_flag:
        return
    spath=os.path.join(script_dir, script_name)
    if not os.path.isfile(spath):
        forms.alert("{0} not found: {1}".format(title, spath)); return
    status, ext_ok, out_snip, err_snip, logpath = run_external_or_internal(
        spath, PYTHON3_PATH, args_list, FALLBACK_INTERNAL_IF_EXTERNAL_FAILS,
        force_internal, script_name + ".log",
        injected_globals=injected_globals
    )
    forms.alert("{0} Step:\n{1}\n\nSTDOUT(first 600):\n{2}".format(title, status, out_snip[:600]),
                title="{0} Result".format(script_name))

run_step(RUN_CALC_SHORTEST, CALC_SCRIPT_NAME, FORCE_INTERNAL_CALC, CALC_ARGS, "Shortest Path")
run_step(RUN_UPDATE_LENGTHS, UPDATE_SCRIPT_NAME, FORCE_INTERNAL_UPDATE, UPDATE_ARGS, "Update Lengths")

if OPEN_FOLDER_AFTER:
    try:
        subprocess.Popen(r'explorer /select,"{0}"'.format(json_path))
    except: pass