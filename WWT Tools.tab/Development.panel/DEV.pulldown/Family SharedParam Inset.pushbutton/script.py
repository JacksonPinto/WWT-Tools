# -*- coding: utf-8 -*-
"""
Bulk Insert Shared Parameters for Revit 2025+ (GroupTypeId API, robust and minimal group set, per-parameter Instance selection)

- Compatible with Revit 2025 and newer (uses GroupTypeId, not BuiltInParameterGroup)
- Only the main parameter groups are available (see GROUP_BIP_LOOKUP).
- Scans the shared parameters file currently set in Revit (Application.SharedParametersFilename).
- Lets you select which parameters to add (basic selection UI, compatible with all pyRevit builds).
- Prompts for Instance/Type per parameter.
"""

from Autodesk.Revit.DB import GroupTypeId, Transaction
from pyrevit import revit, forms, script
import os

# Minimal group set, robust for all 2025+ APIs
GROUP_BIP_LOOKUP = {
    "Constraints": GroupTypeId.Constraints,
    "Construction": GroupTypeId.Construction,
    "Data": GroupTypeId.Data,
    "Dimensions": GroupTypeId.Geometry,
    "General": GroupTypeId.General,
    "Graphics": GroupTypeId.Graphics,
    "Identity Data": GroupTypeId.IdentityData,
    "Materials and Finishes": GroupTypeId.Materials,
    "Text": GroupTypeId.Text,
    "Visibility": GroupTypeId.Visibility,
}
GROUP_NAMES = list(GROUP_BIP_LOOKUP.keys())

def get_sharedparams_api_groups_and_defs(sp_file):
    """Returns list of dicts with shared param info using only Revit API."""
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

def multi_select_params(params):
    param_names = [u"[{}] {}".format(p["group_name"], p["name"]) for p in params]
    selected = forms.SelectFromList.show(param_names, multiselect=True, title="Select Shared Parameters to Add")
    if not selected:
        return []
    return [p for p, label in zip(params, param_names) if label in selected]

def choose_group(default="Data"):
    return forms.ask_for_one_item(GROUP_NAMES, default=default, prompt="Choose Parameter Group:")

def choose_instance_type(param_name, default="Instance"):
    return forms.ask_for_one_item(["Instance", "Type"], default=default, prompt="Should parameter '{}' be Instance or Type?".format(param_name))

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

    selected_params = multi_select_params(all_params)
    if not selected_params:
        forms.alert("No parameters selected.", exitscript=True)

    # Prompt user ONCE for group selection for all (for simplicity)
    chosen_group_name = choose_group()
    if not chosen_group_name or chosen_group_name not in GROUP_BIP_LOOKUP:
        forms.alert("No group selected, or group not mapped. Defaulting to 'Data'.")
        chosen_group_name = "Data"
    group_enum = GROUP_BIP_LOOKUP[chosen_group_name]

    # Prompt for Instance/Type per parameter
    param_instance_map = {}
    for param in selected_params:
        inst_type = choose_instance_type(param["name"])
        if inst_type is None:
            inst_type = "Instance"
        param_instance_map[param["name"]] = (inst_type == "Instance")

    t = Transaction(doc, "Add Shared Parameters")
    t.Start()
    results = []
    for param in selected_params:
        definition = param["definition"]
        if not definition:
            results.append(u"Parameter '{}' definition not found.".format(param["name"]))
            continue
        is_instance = param_instance_map.get(param["name"], True)
        try:
            fam_param = add_shared_parameter_to_family(doc, definition, group_enum, is_instance)
            results.append(u"Added: '{}' as {} under '{}'".format(param["name"], "Instance" if is_instance else "Type", chosen_group_name))
        except Exception as ex:
            results.append(u"Error adding '{}': {}".format(param["name"], str(ex)))
    t.Commit()

    forms.alert('\n'.join(results), title="Results")

if __name__ == "__main__":
    main()