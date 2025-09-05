# -*- coding: utf-8 -*-
"""
PyRevit script to set family parameter values on selected family types,
prompting the user for parameter names and values,
and supporting unit input for numeric parameters (e.g., mm, cm, m, in, ft).
Bare numbers (no unit) are treated as millimeters by default.
"""
import re
from pyrevit import revit, DB, forms
from Autodesk.Revit.DB import StorageType


def parse_length(raw):
    """Convert a length string (e.g. '1000 mm', '3 ft', or plain number=mm) into internal feet."""
    m = re.match(r'^\s*([-+]?\d*\.?\d+)\s*([A-Za-z"\']*)\s*$', raw)
    if not m:
        raise ValueError("Invalid length format: '{0}'".format(raw))
    num = float(m.group(1))
    unit = m.group(2).lower()
    # Conversion factors to feet
    unit_map = {
        'mm': 0.00328084,
        'cm': 0.0328084,
        'm': 3.28084,
        'in': 0.0833333,
        '"': 0.0833333,
        "'": 1.0,
        'ft': 1.0,
        # default blank = mm
        '': 0.00328084
    }
    if unit not in unit_map:
        raise ValueError("Unknown unit: '{0}'".format(unit))
    return num * unit_map[unit]


def main():
    doc = revit.doc
    if not doc.IsFamilyDocument:
        forms.alert("ERROR: Must run in a Family document.", exitscript=True)

    fam_mgr = doc.FamilyManager

    # Ask for parameter names
    names_in = forms.ask_for_string(
        default="Conduit Diameter;AnotherParam",
        prompt="Enter FAMILY PARAMETER NAMES (semicolon-separated)",
        title="Parameter Names"
    )
    if not names_in:
        forms.alert("No parameter names entered.", exitscript=True)

    # Ask for corresponding raw values
    vals_in = forms.ask_for_string(
        default="19;TextValue",
        prompt="Enter VALUES (same order; lengths support mm, cm, m, in, ft; plain numbers=mm)",
        title="Parameter Values"
    )
    if not vals_in:
        forms.alert("No values entered.", exitscript=True)

    names = [n.strip() for n in names_in.split(';') if n.strip()]
    raws  = [v.strip() for v in vals_in.split(';') if v.strip()]
    if len(names) != len(raws):
        forms.alert("ERROR: Number of names and values must match.", exitscript=True)

    raw_map = dict(zip(names, raws))

    # Select family types
    types = list(fam_mgr.Types)
    type_names = [t.Name for t in types]
    sel = forms.SelectFromList.show(
        type_names,
        title="Select Family Types to Modify",
        multiselect=True
    )
    if not sel:
        forms.alert("No types selected.", exitscript=True)
    if isinstance(sel, str):
        sel = [sel]
    sel_types = [t for t in types if t.Name in sel]

    # Store original type
    orig = fam_mgr.CurrentType
    missing = set()
    fail = []

    tr = DB.Transaction(doc, "Set Family Parameters")
    tr.Start()
    for ftype in sel_types:
        fam_mgr.CurrentType = ftype
        for pname, raw in raw_map.items():
            p = next((p for p in fam_mgr.Parameters if p.Definition.Name == pname), None)
            if not p:
                missing.add(pname)
                continue
            st = p.StorageType
            try:
                if st == StorageType.Double:
                    # Numeric parameter: parse length (with units) or plain mm
                    val = parse_length(raw)
                elif st == StorageType.Integer:
                    val = int(raw)
                elif st == StorageType.String:
                    val = raw
                else:
                    raise ValueError("Unsupported storage type {0}".format(st))
                fam_mgr.Set(p, val)
            except Exception as e:
                fail.append((ftype.Name, pname, str(e)))
    # restore and commit
    fam_mgr.CurrentType = orig
    tr.Commit()

    # Report
    msgs = []
    if not missing and not fail:
        msgs.append("SUCCESS: All parameters set successfully.")
    else:
        if missing:
            msgs.append("WARNING: Missing params: {0}".format(
                ", ".join(sorted(missing))))
        if fail:
            fmsg = "\n".join(
                "Type '{0}', Param '{1}': {2}".format(t, p, e)
                for t, p, e in fail
            )
            msgs.append("ERROR: Failed to set:\n" + fmsg)
    forms.alert("\n\n".join(msgs))

if __name__ == '__main__':
    main()
