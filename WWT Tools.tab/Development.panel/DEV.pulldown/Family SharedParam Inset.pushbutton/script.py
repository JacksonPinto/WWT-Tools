# -*- coding: utf-8 -*-
"""
pyRevit Script: Advanced UI Bulk Shared Parameter Inserter

- Scans current configured shared parameters file (Revit .NET API only).
- Lists all parameters with checkboxes, Parameter Group dropdown, BIP Name auto (live lookup), Instance (Yes/No).
- After user presses OK, inserts checked parameters with chosen settings into the open family.

Author: Jackson Pinto & Copilot
"""

from pyrevit import revit, DB, forms, script
import os

# --- Lookup table for Group Name -> BIP Enum ---
GROUP_BIP_LOOKUP = {
    "Analysis Results": DB.BuiltInParameterGroup.PG_ANALYSIS_RESULTS,
    "Constraints": DB.BuiltInParameterGroup.PG_CONSTRAINTS,
    "Construction": DB.BuiltInParameterGroup.PG_CONSTRUCTION,
    "Data": DB.BuiltInParameterGroup.PG_DATA,
    "Dimensions": DB.BuiltInParameterGroup.PG_GEOMETRY,
    "General": DB.BuiltInParameterGroup.PG_GENERAL,
    "Graphics": DB.BuiltInParameterGroup.PG_GRAPHICS,
    "Identity Data": DB.BuiltInParameterGroup.PG_IDENTITY_DATA,
    "IFC Parameters": DB.BuiltInParameterGroup.PG_IFC,
    "Materials and Finishes": DB.BuiltInParameterGroup.PG_MATERIALS,
    "Model Properties": DB.BuiltInParameterGroup.PG_ADSK_MODEL_PROPERTIES,
    "Other": DB.BuiltInParameterGroup.INVALID,
    "Text": DB.BuiltInParameterGroup.PG_TEXT,
    "Title Text": DB.BuiltInParameterGroup.PG_TITLE,
    "Visibility": DB.BuiltInParameterGroup.PG_VISIBILITY,
}

def get_group_names():
    return list(GROUP_BIP_LOOKUP.keys())

def bip_from_group_name(group_name):
    return GROUP_BIP_LOOKUP.get(group_name, DB.BuiltInParameterGroup.INVALID)

def group_name_from_bip(bip):
    for k, v in GROUP_BIP_LOOKUP.items():
        if v == bip:
            return k
    return "Other"

# --- API: Get shared parameter groups and definitions ---
def get_sharedparams_api_groups_and_defs(sp_file):
    params = []
    if not sp_file:
        return params

    for group in sp_file.Groups:
        group_name = group.Name
        for definition in group.Definitions:
            try:
                params.append({
                    "name": definition.Name,
                    "guid": str(definition.GUID),
                    "datatype": str(definition.ParameterType) if hasattr(definition, "ParameterType") else "(unknown)",
                    "group_name": group_name,
                    "default_bip": bip_from_group_name(group_name),
                    "definition": definition
                })
            except Exception as ex:
                params.append({
                    "name": "(error)",
                    "guid": "(error)",
                    "datatype": "(error)",
                    "group_name": group_name,
                    "default_bip": DB.BuiltInParameterGroup.INVALID,
                    "definition": None
                })
    return params

# --- Custom multi-column UI ---
class ParamRow(forms.TemplateListItem):
    def __init__(self, param):
        super(ParamRow, self).__init__(param)
        self.selected = False
        self.group = group_name_from_bip(param["default_bip"])  # editable per row
        self.bip = param["default_bip"]                        # auto-updated
        self.is_instance = False                               # default No

    @property
    def name(self):
        return self.item["name"]

    @property
    def group_choices(self):
        return get_group_names()

    @property
    def bip_name(self):
        return str(self.bip).replace("BuiltInParameterGroup.", "PG_")

    def set_group(self, value):
        self.group = value
        self.bip = bip_from_group_name(value)

    def set_instance(self, value):
        self.is_instance = (value == "Yes")

    def values_for_ui(self):
        # For UI: [checkbox, name, dropdown group, auto bip, instance dropdown]
        return [self.selected,
                self.name,
                self.group,
                self.bip_name,
                "Yes" if self.is_instance else "No"]

    def update_from_ui(self, vals):
        self.selected = vals[0]
        self.set_group(vals[2])
        self.set_instance(vals[4])

# --- Multi-column table UI ---
def multi_column_table(params):
    columns = ["Select", "Shared Parameter Name", "Parameter Group", "BIP Name (auto)", "Instance (Yes/No)"]
    rows = [ParamRow(p) for p in params]

    # Build table content for UI, row by row
    table_data = [r.values_for_ui() for r in rows]

    # The main UI: editable table
    edited_data = forms.edit_table(
        columns=columns,
        data=table_data,
        title="Bulk Shared Parameter Inserter",
        description="Check parameters to insert, set group, instance as needed, then press OK."
    )

    if not edited_data:
        return []

    # Update each row from UI and collect selected rows
    selected_rows = []
    for row, vals in zip(rows, edited_data):
        row.update_from_ui(vals)
        if row.selected:
            selected_rows.append(row)
    return selected_rows

# --- Insert parameters into family ---
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

    # Advanced table UI for parameter selection and settings
    selected_rows = multi_column_table(all_params)
    if not selected_rows:
        forms.alert("No parameters selected.", exitscript=True)

    # Insert parameters in transaction
    t = DB.Transaction(doc, "Add Shared Parameters")
    t.Start()
    results = []
    for row in selected_rows:
        definition = row.item["definition"]
        if not definition:
            results.append(u"Parameter '{}' definition not found.".format(row.name))
            continue
        try:
            fam_param = add_shared_parameter_to_family(doc, definition, row.bip, row.is_instance)
            results.append(u"Added: '{}' as {} under '{}'".format(row.name, "Instance" if row.is_instance else "Type", row.group))
        except Exception as ex:
            results.append(u"Error adding '{}': {}".format(row.name, str(ex)))
    t.Commit()

    forms.alert('\n'.join(results), title="Results")

if __name__ == "__main__":
    main()