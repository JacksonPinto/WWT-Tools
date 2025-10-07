# -*- coding: utf-8 -*-
# Export Infrastructure + Projected L Jumpers (Edge Splitting)
# Version: 3.x (integrated calc_shortest_topologic launcher)

from Autodesk.Revit.DB import (
    FilteredElementCollector, BuiltInCategory, UV,
    FamilyInstance, LocationCurve, BuiltInParameter
)
from Autodesk.Revit.UI.Selection import ObjectType
from pyrevit import forms
import os, json, math, traceback, datetime, sys

# ---------------- CONFIG ----------------
SHOW_SCRIPT_PATH = False

USE_INFRA_CATEGORY_DIALOG  = True
USE_DEVICE_CATEGORY_DIALOG = True

SUBDIVIDE_ARCS = True
CURVE_SEG_LENGTH_FT = 0.25
MERGE_TOL = 1e-6
PROJECTION_ENDPOINT_TOL = 1e-4
ROUND_PREC = 9

FITTING_CONNECTOR_TO_POINT = True
INCLUDE_FITTING_WITHOUT_CONNECTORS = True

CREATE_VERTICAL_SEGMENT = True
CREATE_HORIZONTAL_SEGMENT = True
VERTICAL_MIN_LEN = 1e-4

# ---------------- External processing (PATCH) ----------------
# Auto-detect a Python 3 interpreter so script works for any user profile.
def _detect_python3():
    import sys, os
    # 1. Environment variable override (CUSTOM_PYTHON3 or PYTHON3_PATH)
    for env_var in ("CUSTOM_PYTHON3", "PYTHON3_PATH"):
        env_override = os.environ.get(env_var)
        if env_override and os.path.isfile(env_override):
            return env_override

    home = os.path.expanduser("~")
    local_appdata = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
    candidates = []

    # 2. Typical user-local installs
    for ver in ("Python312","Python311","Python310","Python39","Python38"):
        candidates.append(os.path.join(local_appdata, "Programs", "Python", ver, "python.exe"))

    # 3. Program Files installs
    pf = os.environ.get("ProgramFiles")
    if pf:
        for ver in ("Python312","Python311","Python310","Python39","Python38"):
            candidates.append(os.path.join(pf, ver, "python.exe"))
            candidates.append(os.path.join(pf, "Python", ver, "python.exe"))

    # 4. sys.executable (exclude IronPython / None)
    exe = getattr(sys, "executable", None)
    if exe and isinstance(exe, basestring if 'basestring' in dir(__builtins__) else str):
        lower = exe.lower()
        if lower.endswith("python.exe") and ("ironpython" not in lower):
            candidates.append(exe)

    # 5. PATH search (very lightweight)
    path_env = os.environ.get("PATH","")
    for p in path_env.split(os.pathsep):
        cand = os.path.join(p, "python.exe")
        if os.path.isfile(cand):
            candidates.append(cand)

    # Deduplicate preserving order
    seen=set(); ordered=[]
    for c in candidates:
        if c and c not in seen:
            seen.add(c); ordered.append(c)

    for c in ordered:
        try:
            if os.path.isfile(c):
                return c
        except:
            pass
    return ""  # empty => internal fallback only

PYTHON3_PATH = _detect_python3()

RUN_CALC_SHORTEST = True
CALC_SCRIPT_NAME  = "calc_shortest_topologic.py"
CALC_ARGS         = []  # will be set to --direct or --chain later
FORCE_INTERNAL_CALC   = False

RUN_UPDATE_LENGTHS = True
UPDATE_SCRIPT_NAME = "update_cable_lengths.py"
UPDATE_ARGS        = []
FORCE_INTERNAL_UPDATE = True

FALLBACK_INTERNAL_IF_EXTERNAL_FAILS = True
OPEN_FOLDER_AFTER = False

PRINT_TO_CONSOLE = True
DEBUG_PROJECTION = False
DEBUG_DEVICE     = False

STRICT_MANUAL_SEQUENCE = True

uidoc = __revit__.ActiveUIDocument
doc   = uidoc.Document
active_view = uidoc.ActiveView
if not doc or not active_view:
    forms.alert("Need active document + active view.", exitscript=True)

def cprint(*args):
    if PRINT_TO_CONSOLE:
        try: print("[EXPORT]", " ".join([str(a) for a in args]))
        except: pass

def to_int_id(obj):
    try:
        if hasattr(obj,'Id'): return int(str(obj.Id))
        if hasattr(obj,'IntegerValue'): return obj.IntegerValue
        return int(obj)
    except: return None

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
        for i in range(segs+1):
            p=curve.Evaluate(float(i)/segs,True)
            pts.append((p.X,p.Y,p.Z))
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
        p=loc.Point; return (p.X,p.Y,p.Z),"LocationPoint"
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
    try: return symbol.Name
    except:
        param = symbol.get_Parameter(BuiltInParameter.SYMBOL_NAME_PARAM)
        if param: return param.AsString()
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

# Sequential manual pick
if selection_mode == "manual":
    if STRICT_MANUAL_SEQUENCE:
        forms.alert("Sequential Manual Pick: Click devices in desired order. ESC to finish.")
        seq=0; seen=set()
        while True:
            try:
                ref=uidoc.Selection.PickObject(ObjectType.Element, "Pick device #{}".format(seq+1))
                el=doc.GetElement(ref.ElementId)
                if selected_type_ids and el.GetTypeId() not in selected_type_ids:
                    continue
                eid=to_int_id(el)
                if eid in seen: continue
                devices.append(el); seen.add(eid); seq+=1
            except: break
        if not devices:
            forms.alert("No devices picked.", exitscript=True)
    else:
        try:
            refs=uidoc.Selection.PickObjects(ObjectType.Element,"Pick device elements")
            for r in refs:
                el=doc.GetElement(r.ElementId)
                if selected_type_ids and el.GetTypeId() not in selected_type_ids: continue
                devices.append(el)
        except Exception as e:
            forms.alert("Device pick aborted:\n{}".format(e), exitscript=True)
        if not devices: forms.alert("No devices picked.", exitscript=True)
elif selection_mode == "all":
    for cid in device_cat_ids:
        col=FilteredElementCollector(doc, active_view.Id).OfCategory(cid).WhereElementIsNotElementType()
        for el in col: devices.append(el)
elif selection_mode == "bytype":
    try:
        for type_id in selected_type_ids:
            col=FilteredElementCollector(doc, active_view.Id).OfTypeId(type_id).WhereElementIsNotElementType()
            for el in col: devices.append(el)
    except AttributeError:
        for cid in device_cat_ids:
            col=FilteredElementCollector(doc, active_view.Id).OfCategory(cid).WhereElementIsNotElementType()
            for el in col:
                if el.GetTypeId() in selected_type_ids: devices.append(el)

cprint("Final device order:", [to_int_id(d) for d in devices])

vertices=[]
vertex_map={}
infra_edges=[]
device_edges=[]
start_points=[]

# Mode selection for calculation (PATCH: map to --direct / --chain)
calc_modes=[
    "Individual Single Cable",
    "Daisy Chain Connection"
]
user_mode = forms.SelectFromList.show(calc_modes, title="Calculation Mode", multiselect=False)
if not user_mode:
    forms.alert("Calculation mode canceled.", exitscript=True)

# Integrated script always used
CALC_SCRIPT_NAME = "calc_shortest_topologic.py"
if user_mode.startswith("Daisy"):
    CALC_ARGS = ["--chain"]
else:
    CALC_ARGS = ["--direct"]

def add_vertex(pt):
    nk=norm_key(pt)
    idx=vertex_map.get(nk)
    if idx is not None: return idx
    for i,v in enumerate(vertices):
        if (abs(v[0]-pt[0])<=MERGE_TOL and abs(v[1]-pt[1])<=MERGE_TOL and abs(v[2]-pt[2])<=MERGE_TOL):
            vertex_map[nk]=i; return i
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

def get_symbol_loc(el):
    loc=getattr(el,"Location",None)
    if loc and hasattr(loc,"Point") and loc.Point:
        p=loc.Point; return (p.X,p.Y,p.Z)
    return None

# Fittings
for el in fittings:
    lp, method = get_fitting_location_point(el)
    if not lp: continue
    base_idx=add_vertex(lp)
    cons=get_all_connectors(el)
    if cons:
        for c in cons:
            ci=add_vertex((c[0],c[1],c[2]))
            add_edge(ci, base_idx, infra_edges)
    elif INCLUDE_FITTING_WITHOUT_CONNECTORS:
        pass

# Linear elements
for el in linears:
    loc=getattr(el,"Location",None)
    if not (loc and isinstance(loc,LocationCurve) and loc.Curve): continue
    pts=sample_curve(loc.Curve)
    if len(pts)<2: continue
    for a,b in zip(pts, pts[1:]):
        if dist3(a,b)>1e-9:
            ia=add_vertex(a); ib=add_vertex(b)
            add_edge(ia,ib,infra_edges)

# Devices & start points
for seq,(el) in enumerate(devices):
    p=get_symbol_loc(el)
    if not p: continue
    add_vertex(p)
    start_points.append({"element_id": to_int_id(el), "point":[p[0],p[1],p[2]], "seq_index": seq})

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
        from Autodesk.Revit.DB import UV
        uv=UV((bbox.Min.U+bbox.Max.U)/2.0,(bbox.Min.V+bbox.Max.V)/2.0)
        xyz=geom.Evaluate(uv)
        return (xyz.X,xyz.Y,xyz.Z)
    except: return None

end_xyz=face_center(face_ref)
if not end_xyz: forms.alert("Failed computing end point.", exitscript=True)
end_idx=add_vertex(end_xyz)

# Minimal device projection / jumper edges (reuse original simplified logic)
def project_point_to_segment(pt, a, b):
    ax,ay,az=a; bx,by,bz=b; px,py,pz=pt
    ab=(bx-ax,by-ay,bz-az); ap=(px-ax,py-ay,pz-az)
    ab2=ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2]
    if ab2==0: return a,0.0,dist3(pt,a)
    t=(ap[0]*ab[0]+ap[1]*ab[1]+ap[2]*ab[2])/ab2
    t=max(0,min(1,t))
    cx=ax+ab[0]*t; cy=ay+ab[1]*t; cz=az+ab[2]*t
    return (cx,cy,cz), t, dist3(pt,(cx,cy,cz))

def split_edge(idx, proj_pt, edges_list):
    a,b=edges_list[idx]
    del edges_list[idx]
    p_idx=add_vertex(proj_pt)
    add_edge(a,p_idx,edges_list)
    add_edge(p_idx,b,edges_list)
    return p_idx

def project_device(device_pt):
    best=None; best_proj=None; best_t=None; best_d=None
    for i,(a_idx,b_idx) in enumerate(infra_edges):
        a=vertices[a_idx]; b=vertices[b_idx]
        proj,t,d=project_point_to_segment(device_pt,a,b)
        if best_d is None or d<best_d:
            best=i; best_proj=proj; best_t=t; best_d=d
    return best, best_proj, best_t, best_d

for sp in start_points:
    dx,dy,dz = sp["point"]
    edge_idx, proj_pt, t_val, pdist = project_device((dx,dy,dz))
    if edge_idx is None: continue
    a_idx,b_idx = infra_edges[edge_idx]
    a=vertices[a_idx]; b=vertices[b_idx]
    from_pt = (dx,dy,dz)
    if dist3(proj_pt,a)<=PROJECTION_ENDPOINT_TOL:
        pj=a_idx
    elif dist3(proj_pt,b)<=PROJECTION_ENDPOINT_TOL:
        pj=b_idx
    else:
        pj=split_edge(edge_idx, proj_pt, infra_edges)
    # vertical + horizontal (device_edges)
    if CREATE_VERTICAL_SEGMENT:
        v_pt=(dx,dy,vertices[pj][2])
        if abs(v_pt[2]-dz)<=VERTICAL_MIN_LEN:
            v_idx=add_vertex(from_pt)
        else:
            v_idx=add_vertex(v_pt)
            add_edge(add_vertex(from_pt), v_idx, device_edges)
        add_edge(v_idx, pj, device_edges)
    else:
        add_edge(add_vertex(from_pt), pj, device_edges)

combined_edges = infra_edges + device_edges

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

cprint("EXPORT SUMMARY vertices={} infraEdges={} deviceEdges={} totalEdges={} devices={}".format(
    len(vertices), len(infra_edges), len(device_edges), len(combined_edges), len(start_points)
))

forms.alert(
    "GRAPH DONE\nVertices:{0}\nInfraEdges:{1}\nDeviceEdges:{2}\nTotalEdges:{3}\nDevices:{4}\nMode: {5}\nPython3: {6}\nJSON:\n{7}".format(
        len(vertices), len(infra_edges), len(device_edges), len(combined_edges), len(start_points),
        CALC_ARGS[0] if CALC_ARGS else "N/A", (PYTHON3_PATH or "Internal Only"), json_path
    ),
    title="Cable Length Calculation"
)

import subprocess, traceback, datetime

def run_external_or_internal(script_path, interpreter, args, allow_fallback, force_internal, log_basename,
                             injected_globals=None):
    external_ok=False; stdout_data=""; stderr_data=""; status=""
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
            if rc==0: external_ok=True; status="External OK rc=0"
            else: status="External FAILED rc={}".format(rc)
        except Exception as ex:
            status="External exception: {}".format(ex)
    elif not force_internal:
        status="External interpreter invalid: {}".format(interpreter)

    if (force_internal or (not external_ok and allow_fallback)):
        try:
            g={}
            if injected_globals: g.update(injected_globals)
            g['__file__']=script_path; g['__name__']='__main__'
            code=open(script_path,'r').read()
            # Insert argv simulation for internal execution
            old_argv=sys.argv
            sys.argv=[script_path]+list(args)
            try:
                exec(compile(code, script_path, 'exec'), g, g)
            finally:
                sys.argv=old_argv
            status += ("\nInternal EXEC SUCCESS." if external_ok else "\nInternal fallback SUCCESS.")
        except Exception as ie:
            tb=traceback.format_exc()
            status += "\nInternal EXEC FAILED: {}".format(ie)
            stderr_data += "\n[INTERNAL TRACEBACK]\n{}".format(tb)

    try:
        with open(log_path,"a") as lf:
            lf.write("\n=== {} | {} ===\n".format(os.path.basename(script_path),
                                                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            lf.write("STATUS: {}\n".format(status))
            if stdout_data: lf.write("--- STDOUT ---\n{}\n".format(stdout_data))
            if stderr_data: lf.write("--- STDERR ---\n{}\n".format(stderr_data))
    except: pass
    return status, external_ok, stdout_data[:1200], stderr_data[:1200], log_path

injected_globals = {
    '__revit__': __revit__,
    'uidoc': uidoc,
    'doc': doc,
    'active_view': active_view
}

def run_step(run_flag, script_name, force_internal, args_list, title):
    if not run_flag: return
    spath=os.path.join(script_dir, script_name)
    if not os.path.isfile(spath):
        forms.alert("{} not found: {}".format(title, spath)); return
    status, ext_ok, out_snip, err_snip, logpath = run_external_or_internal(
        spath, PYTHON3_PATH, args_list, FALLBACK_INTERNAL_IF_EXTERNAL_FAILS,
        force_internal, script_name + ".log",
        injected_globals=injected_globals
    )
    forms.alert("{} Step:\n{}\n\nSTDOUT(first 600):\n{}".format(title, status, out_snip[:600]),
                title="{} Result".format(script_name))

run_step(RUN_CALC_SHORTEST, CALC_SCRIPT_NAME, FORCE_INTERNAL_CALC, CALC_ARGS, "Shortest Path")
run_step(RUN_UPDATE_LENGTHS, UPDATE_SCRIPT_NAME, FORCE_INTERNAL_UPDATE, UPDATE_ARGS, "Update Lengths")

if OPEN_FOLDER_AFTER:
    try: subprocess.Popen(r'explorer /select,"{0}"'.format(json_path))
    except: pass