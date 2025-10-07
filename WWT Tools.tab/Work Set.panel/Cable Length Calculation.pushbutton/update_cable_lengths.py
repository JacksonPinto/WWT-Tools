# -*- coding: utf-8 -*-
# update_cable_lengths.py
# Version: 6.5.2 (2025-08-27)
# Author: JacksonPinto
#
# UPDATE 6.5.2:
# - Adjusted reader for new results JSON structure (results list under 'results', includes meta).
# - Verifies count of elements found vs results.
# - Logs skipped elements (no path).
# - Keeps existing behavior coloring Green (has length) / Red (no length).
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

# Backward compatibility: if 'results' not present assume full list
if "results" in data:
    results = data["results"]
else:
    results = data  # old format list

# Collect categories in active view
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

# Parameter selection
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

# Map elements
id_to_elem = {}
for el in elements:
    try:
        id_to_elem[int(str(el.Id))] = el
    except:
        pass

# Apply
updated = 0
green = 0
red = 0
skipped_missing = 0

t = Transaction(doc, "Update Cable Lengths")
t.Start()
for res in results:
    eid = res.get("from_element_id")
    length = res.get("raw_length")
    el = id_to_elem.get(eid)
    if not el:
        skipped_missing += 1
        continue
    # set length
    if length is not None:
        p = el.LookupParameter(target_param)
        if p:
            try:
                p.Set(length)  # values are in internal feet
                updated += 1
            except Exception as e:
                print("[WARN] Could not set length for {}: {}".format(eid, e))
    # set color
    cp = el.LookupParameter(color_param)
    if cp:
        try:
            if length is not None:
                cp.Set("Green")
                green += 1
            else:
                cp.Set("Red")
                red += 1
        except Exception as e:
            print("[WARN] Could not set color for {}: {}".format(eid, e))
t.Commit()

forms.alert(
    "Update complete.\nCategory: {}\nLength Param: {}\nUpdated Lengths: {}\nGreen: {}  Red: {}\nSkipped (no element in view): {}\nTotal Results: {}"
    .format(selected_cat, target_param, updated, green, red, skipped_missing, len(results))
)