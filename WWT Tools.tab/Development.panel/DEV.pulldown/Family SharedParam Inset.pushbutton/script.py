# -*- coding: utf-8 -*-
"""
Bulk Insert Shared Parameters for Revit 2025+ (GroupTypeId API)

- Compatible with Revit 2025 and newer (uses GroupTypeId, not BuiltInParameterGroup)
- Scans the shared parameters file currently set in Revit (Application.SharedParametersFilename).
- Shows all shared parameters (API, not text parsing!) in a table UI.
- Lets you select which to add, set group and instance/type per parameter.
"""

from Autodesk.Revit.DB import GroupTypeId, Transaction
from pyrevit import revit, forms, script
import os

GROUP_BIP_LOOKUP = {
    "Constraints": GroupTypeId.Constraints,
    "Construction": GroupTypeId.Construction,
    "Data": GroupTypeId.Data,
    "Dimensions": GroupTypeId.Geometry,
    "General": GroupTypeId.General,
    "Graphics": GroupTypeId.Graphics,
    "Identity Data": GroupTypeId.IdentityData,
    "IFC Parameters": GroupTypeId.IFC,
    "Materials and Finishes": GroupTypeId.Materials,
    "Model Properties": GroupTypeId.ADSKModelProperties,
    "Other": GroupTypeId.Invalid,
    "Text": GroupTypeId.Text,
    "Title Text": GroupTypeId.Title,
    "Visibility": GroupTypeId.Visibility,
}

def get_group_names():
    return list(GROUP_BIP_LOOKUP.keys())

def bip_from_group_name(group_name):
    return GROUP_BIP_LOOKUP.get(group_name, GroupTypeId.Invalid)

def group_name_from_bip(bip):
    for k, v in GROUP_BIP_LOOKUP.items():
        if v == bip:
            return k
    return "Other"

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
                    "definition": definition
                })
            except Exception:
                continue
    return params

class ParamRow(forms.TemplateListItem):
    def __init__(self, param):
        super(ParamRow, self).__init__(param)
        self.selected = False
        self.group = "Other"
        self.is_instance = False

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
        title="Bulk Shared Parameter Inserter (Revit 2025+)",
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
            return fam_param
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

    selected_rows = multi_column_table(all_params)
    if not selected_rows:
        forms.alert("No parameters selected.", exitscript=True)

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