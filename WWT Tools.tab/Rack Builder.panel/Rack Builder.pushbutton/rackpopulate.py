# -*- coding: utf-8 -*-
# Copyright: World Wide Technology
# Version: 1.8
# Author: Jackson Pinto

import clr
clr.AddReference('RevitServices')
clr.AddReference('RevitAPI')
clr.AddReference('RevitNodes')
clr.AddReference('System')

from RevitServices.Persistence import DocumentManager
from RevitServices.Transactions import TransactionManager
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
forms.alert("Please select a Rack Element again.", title="Rack Selection", exitscript=False)

# Step 2: Ask user to select an element
selection = uidoc.Selection
try:
    selected_ref = selection.PickObject(ObjectType.Element, "Select an element to list parameters")
    selected_element = doc.GetElement(selected_ref.ElementId)
except Exception as e:
    forms.alert("No element selected or selection error: {}".format(str(e)), exitscript=True)

# Step 3: Auto-load 'ParameterNames.csv' from script folder
script_folder = os.path.dirname(__file__)
csv_names_file = os.path.join(script_folder, "ParameterNames.csv")

if not os.path.exists(csv_names_file):
    forms.alert("File 'ParameterNames.csv' not found in script folder!\nExiting script.", exitscript=True)

# Step 4: Ask user to select a CSV file with parameter values
csv_values_file = forms.pick_file(file_ext='csv', title='Select CSV file with Parameter Values')
if not csv_values_file:
    forms.alert("No CSV file selected. Exiting script.", exitscript=True)

# Step 5: Prompt user for unit input for length parameters
selected_unit = forms.ask_for_string(prompt='Enter unit for length parameters:', default='mm')

# Step 6: Read parameter names from CSV (Using First Row as Index)
parameter_names = []
try:
    with open(csv_names_file, 'r') as file:
        reader = csv.reader(file)
        index_row = next(reader)  # Read the first row (Index line)
        valid_columns = [i for i, col in enumerate(index_row) if col.strip()]
        for row in reader:
            parameter_names.append([row[i].strip() for i in valid_columns if i < len(row) and row[i].strip()])
except Exception as e:
    forms.alert("Error reading ParameterNames.csv: {}".format(str(e)), exitscript=True)

# Step 7: Read parameter values from CSV (Mapping Data Positions)
parameter_values = []
try:
    with open(csv_values_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            parameter_values.append([row[i].strip() for i in valid_columns if i < len(row) and row[i].strip()])
except Exception as e:
    forms.alert("Error reading ParameterValues.csv: {}".format(str(e)), exitscript=True)

# Step 8: Start transaction and update parameters
with Transaction(doc, "Update Parameters") as t:
    try:
        t.Start()
        for row_index, param_row in enumerate(parameter_names):
            for col_index, param_name in enumerate(param_row):
                for param in selected_element.Parameters:
                    if param.Definition.Name == param_name:
                        param_value = parameter_values[row_index][col_index]
                        if param.StorageType == StorageType.String:
                            param.Set(str(param_value))
                        elif param.StorageType == StorageType.Integer:
                            param.Set(int(float(param_value)))
                        elif param.StorageType == StorageType.Double:
                            formatted_value = UnitUtils.ConvertToInternalUnits(float(param_value), UnitTypeId.Millimeters)
                            param.Set(formatted_value)
                        elif param.StorageType == StorageType.ElementId:
                            param.Set(ElementId(int(float(param_value))))
                        print("Successfully updated {} for instance {} with value {}".format(param_name, selected_element.Id.IntegerValue, param_value))
        t.Commit()
        forms.alert("Parameters successfully updated!")
    except Exception as e:
        t.RollBack()
        forms.alert("Transaction failed: {}".format(str(e)), exitscript=True)