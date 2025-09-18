# -*- coding: utf-8 -*-
"""
Bulk Insert Shared Parameters from Configured File (with Robust API Handling)

- Scans the shared parameters file currently set in Revit (Application.SharedParametersFilename).
- Shows all shared parameters (API, not text parsing!) in a table UI.
- Lets you select which to add, set group and instance/type per parameter.
- Handles all import and enum issues.
- Works in pyRevit and IronPython (requires correct API imports!).

Author: Jackson Pinto & Copilot (2024)
"""

from Autodesk.Revit.DB import BuiltInParameterGroup, Transaction
from pyrevit import revit, forms, script
import os

# --- BuiltInParameterGroup mapping (UI label to BIP enum) ---
GROUP_BIP_LOOKUP = {
    "Analysis Results": BuiltInParameterGroup.PG_ANALYSIS_RESULTS,
    "Constraints": BuiltInParameterGroup.PG_CONSTRAINTS,
    "Construction": BuiltInParameterGroup.PG_CONSTRUCTION,
    "Data": BuiltInParameterGroup.PG_DATA,
    "Dimensions": BuiltInParameterGroup.PG_GEOMETRY,
    "General": BuiltInParameterGroup.PG_GENERAL,
    "Graphics": BuiltInParameterGroup.PG_GRAPHICS,
    "Identity Data": BuiltInParameterGroup.PG_IDENTITY_DATA,
    "IFC Parameters": BuiltInParameterGroup.PG_IFC,
    "Materials and Finishes": BuiltInParameterGroup.PG_MATERIALS,
    "Model Properties": BuiltInParameterGroup.PG_ADSK_MODEL_PROPERTIES,
    "Other": BuiltInParameterGroup.INVALID,
    "Text": BuiltInParameterGroup.PG_TEXT,
    "Title Text": BuiltInParameterGroup.PG_TITLE,
    "Visibility": BuiltInParameterGroup.PG_VISIBILITY,
}

def get_group_names():
    return list(GROUP_BIP_LOOKUP.keys())

def bip_from_group_name(group_name):
    # Return the BuiltInParameterGroup enum for a group name string
    return GROUP_BIP_LOOKUP.get(group_name, BuiltInParameterGroup.INVALID)

def group_name_from_bip(bip):
    # Return the UI group name for a BuiltInParameterGroup enum value
    for k, v in GROUP_BIP_LOOKUP.items():
        if v == bip:
            return k
    return "Other"

def get_sharedparams_api_groups_and_defs(sp_file):
    """Returns list of dicts with shared param info using only Revit API."""
    params = []
    if not sp_file:
        return params

    for group in sp_file.Groups:
        group_name = group.Name  # e.g. "InfoComm", "Invisible Parameters"
        for definition in group.Definitions:
            try:
                params.append({
                    "name": definition.Name,
                    "guid": str(definition.GUID),
                    "datatype": str(definition.ParameterType) if hasattr(definition, "ParameterType") else "(unknown)",
                    "group_name": group_name,
                    "definition": definition
                })
            except Exception:
                continue
    return params

class ParamRow(forms.TemplateListItem):
    def __init__(self, param):
        super(ParamRow, self).__init__(param)
        self.selected = False
        self.group = "Other"  # User must choose group for each
        self.is_instance = False  # Default to No (Type)

    @property
    def name(self):
        return self.item["name"]

    @property
    def group_choices(self):
        return get_group_names()

    def set_group(self, value):
        self.group = value

    def set_instance(self, value):
        self.is_instance = (value == "Yes")

    def values_for_ui(self):
        return [self.selected,
                self.name,
                self.group,
                "Yes" if self.is_instance else "No"]

    def update_from_ui(self, vals):
        self.selected = vals[0]
        self.set_group(vals[2])
        self.set_instance(vals[3])

def multi_column_table(params):
    columns = ["Select", "Shared Parameter Name", "Parameter Group", "Instance (Yes/No)"]
    rows = [ParamRow(p) for p in params]
    table_data = [r.values_for_ui() for r in rows]

    edited_data = forms.edit_table(
        columns=columns,
        data=table_data,
        title="Bulk Shared Parameter Inserter",
        description="Check parameters to insert, set group and instance as needed, then press OK."
    )

    if not edited_data:
        return []

    selected_rows = []
    for row, vals in zip(rows, edited_data):
        row.update_from_ui(vals)
        if row.selected:
            selected_rows.append(row)
    return selected_rows

def add_shared_parameter_to_family(doc, definition, group, is_instance):
    fam_mgr = doc.FamilyManager
    for fam_param in fam_mgr.Parameters:
        if fam_param.Definition.Name == definition.Name:
            return fam_param  # Already exists
    return fam_mgr.AddParameter(definition, group, is_instance)

def main():
    doc = revit.doc
    output = script.get_output()
    if not doc.IsFamilyDocument:
        forms.alert("This script only works in Revit Family Documents.", exitscript=True)

    app = doc.Application
    sp_filepath = app.SharedParametersFilename

    if not sp_filepath or not os.path.exists(sp_filepath):
        forms.alert(
            "No shared parameters file is configured in Revit, or the file does not exist.\n"
            "Set the shared parameters file in Revit (File > Options > File Locations > Shared Parameters), then try again.",
            exitscript=True
        )

    sp_file = app.OpenSharedParameterFile()
    output.print_md("**Shared Parameters file path:** `{}`".format(sp_filepath))

    all_params = get_sharedparams_api_groups_and_defs(sp_file)
    if not all_params:
        forms.alert("No parameters found in shared parameters file.", exitscript=True)

    # Show table UI for parameter selection
    selected_rows = multi_column_table(all_params)
    if not selected_rows:
        forms.alert("No parameters selected.", exitscript=True)

    # Insert parameters in transaction
    t = Transaction(doc, "Add Shared Parameters")
    t.Start()
    results = []
    for row in selected_rows:
        definition = row.item["definition"]
        if not definition:
            results.append(u"Parameter '{}' definition not found.".format(row.name))
            continue
        group_enum = bip_from_group_name(row.group)
        try:
            fam_param = add_shared_parameter_to_family(doc, definition, group_enum, row.is_instance)
            results.append(u"Added: '{}' as {} under '{}'".format(row.name, "Instance" if row.is_instance else "Type", row.group))
        except Exception as ex:
            results.append(u"Error adding '{}': {}".format(row.name, str(ex)))
    t.Commit()

    forms.alert('\n'.join(results), title="Results")

if __name__ == "__main__":
    main()