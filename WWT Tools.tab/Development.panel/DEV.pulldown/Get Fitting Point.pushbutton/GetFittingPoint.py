# -*- coding: utf-8 -*-
import clr
import json
import os

clr.AddReference("RevitAPI")
from Autodesk.Revit.DB import FilteredElementCollector, BuiltInCategory

# Access Revit document and UI document via __revit__
uidoc = __revit__.ActiveUIDocument
doc = uidoc.Document

# Validate context
if not doc or not uidoc:
    raise Exception("❌ Revit document or UI document not found. Make sure you're running this inside Revit.")

active_view = uidoc.ActiveView
if not active_view:
    raise Exception("❌ Active view not found. Please open a view before running the script.")

# Collect Cable Tray Fittings in the active view
collector = FilteredElementCollector(doc, active_view.Id)
collector = collector.OfCategory(BuiltInCategory.OST_CableTrayFitting).WhereElementIsNotElementType()

# Extract XYZ points
points = []
for element in collector:
    location = element.Location
    if location and hasattr(location, 'Point') and location.Point:
        pt = location.Point
        points.append({'X': pt.X, 'Y': pt.Y, 'Z': pt.Z})

# Get script directory using __file__
script_dir = os.path.dirname(__file__)
output_path = os.path.join(script_dir, "CableTrayFittingPoints.json")

# Write to JSON
with open(output_path, 'w') as f:
    json.dump(points, f, indent=4)

# Feedback in PyRevit console
print("✅ Exported {} points to:\n{}".format(len(points), output_path))