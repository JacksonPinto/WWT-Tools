# -*- coding: utf-8 -*-
# Copyright: World Wide Technology
# Version: 3.0
# Author: Jackson Pinto

import clr
clr.AddReference('RevitServices')
clr.AddReference('RevitAPI')
clr.AddReference('RevitNodes')
clr.AddReference('System')

from RevitServices.Persistence import DocumentManager
from Autodesk.Revit.DB import *
from Autodesk.Revit.UI import *
from Autodesk.Revit.UI.Selection import ObjectType
from pyrevit import forms
import csv
import os

# Get Revit Document
uiapp = __revit__
uidoc = uiapp.ActiveUIDocument
doc = uidoc.Document

# Step 1: Notify user before selection
forms.alert("Please select a Rack element to export parameters.", title="Rack Selection", exitscript=False)

# Step 2: Ask user to select an element
selection = uidoc.Selection
try:
    selected_ref = selection.PickObject(ObjectType.Element, "Select an element to export parameters")
    selected_element = doc.GetElement(selected_ref.ElementId)
except Exception as e:
    forms.alert("No element selected or selection error: {}".format(str(e)), exitscript=True)

# Step 3: Retrieve 'Rack Spaces RU' parameter value from the Type Parameter
selected_type = doc.GetElement(selected_element.GetTypeId())
rack_spaces_param = selected_type.LookupParameter("Rack Spaces RU")
if not rack_spaces_param:
    forms.alert("Parameter 'Rack Spaces RU' not found in the selected element type.", exitscript=True)

try:
    rack_spaces_count = rack_spaces_param.AsDouble()  # Handle numeric parameter correctly
    rack_spaces_count = int(rack_spaces_count)  # Convert to integer
except (ValueError, TypeError) as e:
    forms.alert("Invalid value in 'Rack Spaces RU' parameter: {}".format(str(e)), exitscript=True)

# Step 4: Get script folder path and set filename
script_folder = os.path.dirname(__file__)
csv_file_path = os.path.join(script_folder, "ParameterNames.csv")

# Step 5: Generate parameter names dynamically
header = ["RackUnit [Integer]", "EquipDepth [Length]", "EquipModel [Text]", "EquipSchedule [Text]"]
parameter_data = [header]

for i in range(rack_spaces_count, 0, -1):
    row = [
        "RU{:02d}SPACE".format(i),   # First Column (RackUnit)
        "RU{:02d}DEPTH".format(i),   # Second Column (EquipDepth)
        "MODEL{:02d}".format(i),    # Third Column (EquipModel)
        "RU{:02d}SCH".format(i)     # Fourth Column (EquipSchedule)
    ]
    parameter_data.append(row)

# Step 6: Write to CSV file (IronPython compatible way)
try:
    with open(csv_file_path, 'wb') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in parameter_data:
            writer.writerow([col.encode('utf-8') for col in row])
    forms.alert("Parameter names successfully exported to:\n{}".format(csv_file_path))
except Exception as e:
    forms.alert("Error writing CSV file: {}".format(str(e)), exitscript=True)
    
# Step 7: Run Rack Populate Script    
rackpopulate_script = os.path.join(script_folder, "rackpopulate.py")

# Check if the script exists
if os.path.exists(rackpopulate_script):
    try:
        # Run the script using IronPython's execfile()
        execfile(rackpopulate_script)
        forms.alert("rackpopulate.py has been executed successfully!")
    except Exception as e:
        forms.alert("Error executing rackpopulate.py: {}".format(str(e)), exitscript=True)
else:
    forms.alert("rackpopulate.py not found in script folder!", exitscript=True)
