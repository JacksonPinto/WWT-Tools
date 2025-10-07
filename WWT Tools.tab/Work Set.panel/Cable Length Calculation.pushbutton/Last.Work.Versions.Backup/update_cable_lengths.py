# -*- coding: utf-8 -*-
# update_cable_lengths.py
# Version: 6.6.0 (2025-10-07)
# Author: JacksonPinto
#
# UPDATE 6.6.0:
# - Added support for unified Topologic results schema:
#     { "meta": {...}, "sequence": { "legs": [ ... ] } }
# - Still supports legacy "results" list or direct list formats.
# - Selects length in priority: raw_length (if >0) else topologic_length (if status ok) else raw_length (even if 0) when raw_success True.
# - Colors Green on path success (raw_success True or topologic_status == "ok"), Red otherwise.
# - Safe handling of duplicate element IDs (keeps first non-zero length or longest).
# - Added basic console logging for diagnostics (printed to pyRevit output).
#
from Autodesk.Revit.DB import FilteredElementCollector, Transaction
from pyrevit import forms
import json, os

doc = __revit__.ActiveUIDocument.Document
script_dir = os.path.dirname(__file__)
results_path = os.path.join(script_dir, "topologic_results.json")

if not os.path.exists(results_path):
    forms.alert("Results file not found: {}".format(results_path), exitscript=True)

with open(results_path, "r") as f:
    data = json.load(f)

# ---------------------------
# Normalize results collection
# ---------------------------
# Supported patterns:
# 1) Unified: {"meta": {...}, "sequence": {"legs":[...]}}
# 2) Prior direct: {"results":[...]}
# 3) Legacy: [ {...}, {...} ]
if isinstance(data, dict):
    if "sequence" in data and isinstance(data["sequence"], dict) and "legs" in data["sequence"]:
        results = data["sequence"]["legs"]
    elif "results" in data and isinstance(data["results"], list):
        results = data["results"]
    else:
        # Attempt direct list fallback
        maybe_list = data.get("legs")
        if isinstance(maybe_list, list):
            results = maybe_list
        else:
            # Last resort: treat dict values that look like legs (not expected)
            forms.alert("Unsupported JSON schema in topologic_results.json", exitscript=True)
else:
    # Raw list
    results = data

if not isinstance(results, list):
    forms.alert("Parsed results are not a list. Abort.", exitscript=True)

# ---------------------------
# Collect categories in view
# ---------------------------
categories = set()
for el in FilteredElementCollector(doc, doc.ActiveView.Id).WhereElementIsNotElementType():
    try:
        if el.Category:
            categories.add(el.Category.Name)
    except:
        pass

categories = sorted(categories)
selected_cat = forms.SelectFromList.show(categories, title="Select Category to Update (Cable Length)", multiselect=False)
if not selected_cat:
    forms.alert("No category selected. Cancelled.", exitscript=True)
if isinstance(selected_cat, list):
    selected_cat = selected_cat[0]

elements = [
    el for el in FilteredElementCollector(doc, doc.ActiveView.Id).WhereElementIsNotElementType()
    if el.Category and el.Category.Name == selected_cat
]
if not elements:
    forms.alert("No elements of selected category in view.", exitscript=True)

# ---------------------------
# Parameter selections
# ---------------------------
param_names = [p.Definition.Name for p in elements[0].Parameters]
target_param = forms.SelectFromList.show(param_names, title="Select Parameter to store cable length", multiselect=False)
if not target_param:
    forms.alert("No target parameter chosen.", exitscript=True)

color_param = "Equipment Color"
if color_param not in param_names:
    candidate = [n for n in param_names if "Color" in n or "colour" in n]
    if not candidate:
        candidate = param_names
    picked = forms.SelectFromList.show(candidate, title="Select Color Parameter (Green/Red)", multiselect=False)
    if not picked:
        forms.alert("No color parameter chosen.", exitscript=True)
    color_param = picked if not isinstance(picked, list) else picked[0]

# ---------------------------
# Map Revit elements by ID
# ---------------------------
id_to_elem = {}
for el in elements:
    try:
        id_to_elem[int(str(el.Id))] = el
    except:
        pass

# ---------------------------
# Build element length map
# Handle duplicates: keep first non-zero; else larger value
# ---------------------------
elem_lengths = {}       # element_id -> length (float)
elem_success = {}       # element_id -> bool (success path flag)
skipped_no_id = 0
skipped_no_length = 0

for res in results:
    eid = res.get("from_element_id")
    if eid is None:
        skipped_no_id += 1
        continue

    raw_len = res.get("raw_length")
    raw_success = res.get("raw_success")
    topo_status = res.get("topologic_status")
    topo_len = res.get("topologic_length")

    # Determine success flag
    success_flag = bool(raw_success) or (topo_status == "ok")

    # Determine candidate length:
    # Priority: positive raw_length >0 -> else positive topo_length -> else raw_length (even 0 if success) -> else None
    length_val = None
    if isinstance(raw_len, (int, float)) and raw_len > 0:
        length_val = float(raw_len)
    elif topo_status == "ok" and isinstance(topo_len, (int, float)) and topo_len > 0:
        length_val = float(topo_len)
    elif success_flag and isinstance(raw_len, (int, float)):
        length_val = float(raw_len)

    if length_val is None:
        skipped_no_length += 1
        # Still record success flag so color can reflect failure
        if eid not in elem_success:
            elem_success[eid] = False
        continue

    # Merge logic
    if eid not in elem_lengths:
        elem_lengths[eid] = length_val
        elem_success[eid] = success_flag
    else:
        # Prefer existing non-zero; if existing zero and new >0 update; else keep larger
        if elem_lengths[eid] == 0.0 and length_val > 0:
            elem_lengths[eid] = length_val
            elem_success[eid] = success_flag
        else:
            # Keep larger length if that seems more complete
            if length_val > elem_lengths[eid]:
                elem_lengths[eid] = length_val
                elem_success[eid] = success_flag or elem_success[eid]

# ---------------------------
# Apply to Revit
# ---------------------------
updated = 0
green = 0
red = 0
skipped_missing = 0

t = Transaction(doc, "Update Cable Lengths")
t.Start()
for eid, length in elem_lengths.items():
    el = id_to_elem.get(eid)
    if not el:
        skipped_missing += 1
        continue

    success_flag = elem_success.get(eid, False)
    # Set length
    p = el.LookupParameter(target_param)
    if p:
        try:
            p.Set(length)
            updated += 1
        except Exception as e:
            print("[WARN] Could not set length for {}: {}".format(eid, e))

    # Color
    cp = el.LookupParameter(color_param)
    if cp:
        try:
            if success_flag:
                cp.Set("Green")
                green += 1
            else:
                cp.Set("Red")
                red += 1
        except Exception as e:
            print("[WARN] Could not set color for {}: {}".format(eid, e))
t.Commit()

# ---------------------------
# Report
# ---------------------------
forms.alert(
    "Cable Length Update Complete\n"
    "Category: {}\n"
    "Length Param: {}\n"
    "Elements Updated: {}\n"
    "Green: {}  Red: {}\n"
    "Skipped (no element in view): {}\n"
    "Skipped (no id in leg): {}\n"
    "Skipped (no usable length): {}\n"
    "Leg Records Processed: {}\n"
    "Unique Element IDs with lengths: {}"
    .format(
        selected_cat,
        target_param,
        updated,
        green,
        red,
        skipped_missing,
        skipped_no_id,
        skipped_no_length,
        len(results),
        len(elem_lengths)
    )
)

# Console diagnostics (pyRevit output)
print("[INFO] Results file:", results_path)
print("[INFO] Legs processed:", len(results))
print("[INFO] Unique element length assignments:", len(elem_lengths))
print("[INFO] Updated elements:", updated)
print("[INFO] Success (Green):", green, " Fail (Red):", red)
print("[INFO] Skipped no element:", skipped_missing,
      "no id:", skipped_no_id, "no length:", skipped_no_length)