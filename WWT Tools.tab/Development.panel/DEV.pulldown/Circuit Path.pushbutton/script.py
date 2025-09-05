# Set Circuit Path for Any Electrical Circuit Type (Power, Data, Comms, Telephone, etc.)
# Version: 18 (2025-08-16)
# - Fix: wrap Document.Regenerate in a short transaction to avoid "Modification of the document is forbidden"
# - Robust circuit dedup (UniqueId / str(Id)); avoids ElementId.IntegerValue
# - Debug prints in both feet and millimeters
# - Verifies path inside transaction and again after commit

from pyrevit import revit, DB, forms
from Autodesk.Revit.UI.Selection import ObjectType

TOL = 1e-6          # geometric tolerance in feet
CMP_TOL = 1e-5      # path comparison tolerance in feet

# --- Unit helpers (feet <-> mm) ---
def ft_to_mm(v):
    try:
        return DB.UnitUtils.Convert(v, DB.UnitTypeId.Feet, DB.UnitTypeId.Millimeters)
    except Exception:
        try:
            return DB.UnitUtils.ConvertFromInternalUnits(v, DB.DisplayUnitType.DUT_MILLIMETERS)
        except Exception:
            return v * 304.8

def xyz_str_both(xyz):
    return ("ft: ({:.3f}, {:.3f}, {:.3f}) | mm: ({:.1f}, {:.1f}, {:.1f})"
            .format(xyz.X, xyz.Y, xyz.Z, ft_to_mm(xyz.X), ft_to_mm(xyz.Y), ft_to_mm(xyz.Z)))

# --- General helpers ---
def get_param(element, param_name):
    try:
        p = element.LookupParameter(param_name)
        if p:
            return p.AsString()
    except Exception:
        pass
    return None

def _dedup_by_key(elements, keyfunc):
    seen = set()
    out = []
    for el in elements:
        try:
            k = keyfunc(el)
        except Exception:
            k = None
        if k is None:
            k = repr(el)
        if k not in seen:
            out.append(el)
            seen.add(k)
    return out

def get_circuits_from_element(element):
    circuits = []
    if hasattr(element, "GetAssignedElectricalSystems"):
        try:
            found = element.GetAssignedElectricalSystems()
            if found:
                circuits += list(found)
        except Exception:
            pass
    if hasattr(element, "GetElectricalSystems"):
        try:
            found = element.GetElectricalSystems()
            if found:
                circuits += list(found)
        except Exception:
            pass
    if hasattr(element, "MEPModel") and element.MEPModel:
        try:
            found = element.MEPModel.AssignedElectricalSystems
            if found:
                circuits += list(found)
        except Exception:
            pass
    # Fallback: element is BaseEquipment (e.g., panel)
    if not circuits:
        all_circuits = DB.FilteredElementCollector(revit.doc).OfClass(DB.Electrical.ElectricalSystem)
        for c in all_circuits:
            try:
                if c.BaseEquipment and c.BaseEquipment.Id == element.Id:
                    circuits.append(c)
            except Exception:
                pass
    # Fallback by Panel/Circuit Number (devices)
    circuit_number = get_param(element, "Circuit Number")
    panel_name = get_param(element, "Panel")
    if not circuits and circuit_number and panel_name:
        all_circuits = DB.FilteredElementCollector(revit.doc).OfClass(DB.Electrical.ElectricalSystem)
        for c in all_circuits:
            try:
                num_param = c.LookupParameter("Circuit Number")
                panel_param = c.LookupParameter("Panel")
                if num_param and panel_param:
                    if num_param.AsString() == circuit_number and panel_param.AsString() == panel_name:
                        circuits.append(c)
            except Exception:
                pass
    # Robust dedup (avoid ElementId.IntegerValue)
    def keyfunc(sys):
        try:
            if getattr(sys, "UniqueId", None):
                return sys.UniqueId
        except Exception:
            pass
        try:
            return str(sys.Id)
        except Exception:
            return None
    return _dedup_by_key(circuits, keyfunc)

def get_circuit_description(circuit):
    def val(el, name):
        p = el.LookupParameter(name)
        return p.AsString() if p else ""
    return "Panel: {} | Circuit: {} | Name: {} | Type: {}".format(
        val(circuit, "Panel"),
        val(circuit, "Circuit Number"),
        val(circuit, "Name"),
        str(getattr(circuit, "SystemType", "Unknown"))
    )

def get_xyz_from_element(element):
    try:
        loc = element.Location
        if hasattr(loc, "Point") and loc.Point:
            return loc.Point
        elif hasattr(loc, "Curve") and loc.Curve:
            # Use start point as representative
            return loc.Curve.GetEndPoint(0)
    except Exception:
        pass
    bbox = element.get_BoundingBox(revit.doc.ActiveView) if revit.doc.ActiveView else element.get_BoundingBox(None)
    if bbox:
        return (bbox.Min + bbox.Max) / 2.0
    return None

def catid_to_int(cat_id):
    # Tolerant across API/engines
    for getter in (
        lambda c: getattr(c, "IntegerValue", None),
        lambda c: c.get_IntegerValue() if hasattr(c, "get_IntegerValue") else None,
        lambda c: int(str(c)),
    ):
        try:
            val = getter(cat_id)
            if val is not None:
                return int(val)
        except Exception:
            pass
    return -1

def refine_points_axis_aligned(points):
    """Ensure each segment is horizontal (Z const) or vertical (X,Y const); insert elbows as needed."""
    if not points or len(points) < 2:
        return points
    refined = [points[0]]
    for pt in points[1:]:
        last = refined[-1]
        horiz = abs(pt.Z - last.Z) < TOL
        vert = abs(pt.X - last.X) < TOL and abs(pt.Y - last.Y) < TOL
        if horiz or vert:
            refined.append(pt)
        else:
            elbow = DB.XYZ(last.X, last.Y, pt.Z)  # go vertical first, then horizontal
            # avoid zero-length segment
            if abs(elbow.X - last.X) > TOL or abs(elbow.Y - last.Y) > TOL or abs(elbow.Z - last.Z) > TOL:
                refined.append(elbow)
            refined.append(pt)
    # Remove consecutive duplicates
    deduped = []
    for n in refined:
        if not deduped or (abs(n.X - deduped[-1].X) > TOL or abs(n.Y - deduped[-1].Y) > TOL or abs(n.Z - deduped[-1].Z) > TOL):
            deduped.append(n)
    return deduped

def same_path(path1, path2, tol=CMP_TOL):
    if len(path1) != len(path2):
        return False
    for p1, p2 in zip(path1, path2):
        if abs(p1.X - p2.X) > tol or abs(p1.Y - p2.Y) > tol or abs(p1.Z - p2.Z) > tol:
            return False
    return True

def safe_regenerate(doc):
    """Regenerate inside a short transaction to avoid 'document is forbidden'."""
    try:
        with revit.Transaction("Post-commit Regenerate"):
            doc.Regenerate()
    except Exception:
        # Fallback: just try to refresh the active view
        try:
            revit.uidoc.RefreshActiveView()
        except Exception:
            pass

# --- 1) Pick element (panel or device) ---
try:
    ref = revit.uidoc.Selection.PickObject(ObjectType.Element, "Step 1: Pick a panel or device to get available circuits")
    picked_element = revit.doc.GetElement(ref.ElementId)
except Exception:
    forms.alert("No element was selected. Script will exit.", exitscript=True)
    picked_element = None

# --- 2) Choose circuit ---
circuits = get_circuits_from_element(picked_element)
if not circuits:
    forms.alert("No circuits found on the selected element. Exiting.", exitscript=True)

forms.alert("DEBUG: Found {} circuits on element.\n{}".format(
    len(circuits),
    "\n".join([get_circuit_description(c) for c in circuits])
))

if len(circuits) > 1:
    descs = [get_circuit_description(c) for c in circuits]
    sel = forms.SelectFromList.show(descs, multiselect=False, title="Select Circuit", button_name="Select")
    if sel is None:
        forms.alert("No circuit selected. Exiting.", exitscript=True)
    circuit = circuits[descs.index(sel)]
else:
    circuit = circuits[0]

panel = circuit.LookupParameter("Panel").AsString() if circuit.LookupParameter("Panel") else ""
cnum = circuit.LookupParameter("Circuit Number").AsString() if circuit.LookupParameter("Circuit Number") else ""

# --- 3) Original path ---
orig_path = list(circuit.GetCircuitPath())
if len(orig_path) < 2:
    forms.alert("Circuit does not have a valid path (needs at least two nodes). Exiting.", exitscript=True)

forms.alert("DEBUG: Original circuit path (ft | mm):\n" + "\n".join(
    ["Node {}: {}".format(i+1, xyz_str_both(p)) for i, p in enumerate(orig_path)]
))

start_xyz = orig_path[0]
end_xyz = orig_path[-1]

# --- 4) Pick infrastructure elements (midpoints sources) ---
try:
    refs = revit.uidoc.Selection.PickObjects(
        ObjectType.Element,
        "Step 2: Select objects for the new circuit path (Cable trays, fittings, conduits). Right-click Finish to confirm."
    )
    infra_elements = [revit.doc.GetElement(r.ElementId) for r in refs]
except Exception:
    forms.alert("No elements selected for path. Exiting.", exitscript=True)
    infra_elements = []

if not infra_elements:
    forms.alert("At least one infrastructure element is required. Exiting.", exitscript=True)

# Validate categories
allowed_categories = [
    DB.BuiltInCategory.OST_CableTray,
    DB.BuiltInCategory.OST_CableTrayFitting,
    DB.BuiltInCategory.OST_Conduit,
    DB.BuiltInCategory.OST_ConduitFitting,
]
allowed_cat_ids = [int(cat) for cat in allowed_categories]
invalids = [e for e in infra_elements if catid_to_int(e.Category.Id) not in allowed_cat_ids]
if invalids:
    names = [e.Name for e in invalids]
    forms.alert("Invalid selections (not Cable Tray/Conduit/Fitting):\n{}\nExiting.".format("\n".join(names)), exitscript=True)

# --- 5) Collect midpoints ---
midpoints = []
for elem in infra_elements:
    pt = get_xyz_from_element(elem)
    if pt:
        midpoints.append(pt)
    else:
        forms.alert("Could not determine location for element: {}. Exiting.".format(elem.Name), exitscript=True)

# --- 6) Build and refine path (API in feet) ---
proposed = [start_xyz] + midpoints + [end_xyz]
refined = refine_points_axis_aligned(proposed)

forms.alert("DEBUG: Refined (proposed) circuit path (ft | mm):\n" + "\n".join(
    ["Node {}: {}".format(i+1, xyz_str_both(p)) for i, p in enumerate(refined)]
))

# --- 7) Set path and verify (inside txn) ---
inside_success = False
inside_err = None

with revit.Transaction("Set Circuit Path"):
    try:
        from System.Collections.Generic import List
        np = List[DB.XYZ]()
        for p in refined:
            np.Add(DB.XYZ(p.X, p.Y, p.Z))  # feet to feet (internal units)
        circuit.SetCircuitPath(np)

        now_inside = list(circuit.GetCircuitPath())
        forms.alert("DEBUG: Path after SetCircuitPath (inside txn) (ft | mm):\n" + "\n".join(
            ["Node {}: {}".format(i+1, xyz_str_both(p)) for i, p in enumerate(now_inside)]
        ))
        inside_success = same_path(refined, now_inside)
        if not inside_success:
            inside_err = "Revit did not accept the path inside the transaction. Ensure all segments are axis-aligned and nodes are not too close together."
    except Exception as e:
        inside_err = "Exception during SetCircuitPath: {}".format(str(e))
        inside_success = False

# --- 8) Post-commit: safe regenerate + verify again ---
safe_regenerate(revit.doc)
after_commit = list(circuit.GetCircuitPath())
post_ok = same_path(refined, after_commit)

forms.alert("DEBUG: Path after commit (post Regenerate) (ft | mm):\n" + "\n".join(
    ["Node {}: {}".format(i+1, xyz_str_both(p)) for i, p in enumerate(after_commit)]
))

# --- Final message ---
if inside_success and post_ok:
    forms.alert(
        "SUCCESS: Circuit path updated for {} / Circuit {}.\nIf Edit Path UI was open, close it (Finish/Cancel) and re-open to see the change."
        .format(panel or "<Panel>", cnum or "<Number>"),
        exitscript=True
    )
else:
    msg = []
    if inside_err:
        msg.append(inside_err)
    if not post_ok:
        msg.append("After commit, the path differs from the proposed path. If Edit Path was open during run, close it and re-open. If it still reverts, Revit recalculated the route (invalid nodes / too-short segments / constraints).")
    forms.alert("FAILED to fully apply proposed path for {} / Circuit {}.\n\n{}".format(panel or "<Panel>", cnum or "<Number>", "\n".join(msg)), exitscript=True)