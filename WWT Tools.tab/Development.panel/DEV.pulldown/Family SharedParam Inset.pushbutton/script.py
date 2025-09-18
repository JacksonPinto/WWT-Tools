# -*- coding: utf-8 -*-
"""
pyRevit Script: Bulk Shared Parameters Insertion for Revit Families

Features:
- Scan shared parameters file (*.txt) and list all parameters with checkboxes for selection (multi-select).
- UI for selecting one or more shared parameters to add to current open family document.
- For each parameter, user can:
    - Choose to add as Type or Instance parameter.
    - Choose "Group parameter under" from the list of valid Revit parameter groups.
- Uses pyRevit, Revit API (.NET), and works under IronPython.

Author: Jackson Pinto (for pyRevit)
"""

from pyrevit import revit, DB, forms, script
import clr
import os

# Load Windows Forms for file dialog
clr.AddReference('System.Windows.Forms')
from System.Windows.Forms import OpenFileDialog

__title__ = 'Bulk Shared Params'
__author__ = 'Jackson Pinto'
__doc__ = 'Bulk insert shared parameters into Revit Family with type/instance, group selection.'

# Helper to open file dialog to select shared params file
def select_shared_params_file():
    dialog = OpenFileDialog()
    dialog.Title = "Select Shared Parameters File"
    dialog.Filter = "TXT files (*.txt)|*.txt"
    dialog.Multiselect = False
    if dialog.ShowDialog() == 1:
        return dialog.FileName
    return None

# Parse shared params file, returns list of dicts with name, guid, type, group, etc.
def parse_shared_params_file(sp_filepath):
    params = []
    if not os.path.exists(sp_filepath):
        return params

    section = None
    groups = {}
    params_data = []
    with open(sp_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('['):
                section = line.lower()
                continue
            if not line or line.startswith('#'):
                continue
            if section == '[groups]':
                # Format: ID\tGroupName
                try:
                    group_id, group_name = line.split('\t')
                    groups[group_id] = group_name
                except:
                    continue
            elif section == '[parameters]':
                # Format: GUID\tName\tGroupID\tType\tDatatype\tVisible\tDesc
                data = line.split('\t')
                if len(data) >= 3:
                    param = {
                        'guid': data[0],
                        'name': data[1],
                        'group_id': data[2],
                        'type': data[3] if len(data) > 3 else '',
                        'datatype': data[4] if len(data) > 4 else '',
                        'visible': data[5] if len(data) > 5 else '',
                        'desc': data[6] if len(data) > 6 else '',
                        'group_name': groups.get(data[2], '')
                    }
                    params.append(param)
    return params

# Show multi-select checkbox UI for parameter selection
def select_parameters_ui(params):
    class ParamOption(forms.TemplateListItem):
        @property
        def name(self):
            return self.item['name']
        @property
        def description(self):
            return "Type: {} | GUID: {}".format(self.item['datatype'], self.item['guid'])

    param_options = [ParamOption(p) for p in params]
    res = forms.SelectFromList.show(param_options,
                                    title="Select Shared Parameters to Add",
                                    multiselect=True,
                                    button_name='Add Selected')
    if not res:
        return []
    return [x.item for x in res]

# For each selected parameter, get type/instance and group UI
def get_param_settings_ui(selected_params):
    # Get all built-in parameter groups (Revit API)
    param_groups = [g for g in DB.BuiltInParameterGroup]
    group_names = [DB.LabelUtils.GetLabelFor(g) for g in param_groups]
    settings = []
    for param in selected_params:
        # Ask user for Type/Instance and group
        values = forms.CommandSwitchWindow.show(
            ['Instance', 'Type'],
            message="Insert parameter '{}' as:".format(param['name']),
            default='Instance'
        )
        param_type = DB.BuiltInParameterGroup.INVALID
        group_choice = forms.SelectFromList.show(
            group_names,
            title="Group parameter '{}' under:".format(param['name']),
            button_name='Select'
        )
        if not group_choice:
            group_choice = group_names[0]
        # Map group name back to BuiltInParameterGroup enum
        group_enum = param_groups[group_names.index(group_choice)]

        settings.append({
            'param': param,
            'is_instance': (values == 'Instance'),
            'group': group_enum
        })
    return settings

# Actual insertion of shared parameter
def add_shared_parameter_to_family(doc, definition, group, is_instance):
    app = doc.Application
    fam_mgr = doc.FamilyManager

    # Need to check if param already exists
    for fam_param in fam_mgr.Parameters:
        if fam_param.Definition.Name == definition.Name:
            return fam_param # Already exists

    fam_param = fam_mgr.AddParameter(definition, group, is_instance)
    return fam_param

# Main Routine
def main():
    # Only works for family documents
    doc = revit.doc
    if not doc.IsFamilyDocument:
        forms.alert("This script only works in Revit Family Documents.", exitscript=True)
    # Step 1: Select shared params file
    sp_filepath = select_shared_params_file()
    if not sp_filepath:
        forms.alert("No shared parameters file selected.", exitscript=True)

    # Step 2: Parse file and show parameter selection UI
    all_params = parse_shared_params_file(sp_filepath)
    if not all_params:
        forms.alert("No parameters found in shared parameters file.", exitscript=True)

    selected_params = select_parameters_ui(all_params)
    if not selected_params:
        forms.alert("No parameters selected.", exitscript=True)

    # Step 3: For each param, get type/instance and group
    param_settings = get_param_settings_ui(selected_params)

    # Step 4: Open shared params file in Revit API (must match filepath)
    app = doc.Application
    app.SharedParametersFilename = sp_filepath
    sp_file = app.OpenSharedParameterFile()
    if not sp_file:
        forms.alert("Could not open shared parameters file via API.", exitscript=True)

    # Create a map for guid to Definition
    guid2def = {}
    for group in sp_file.Groups:
        for definition in group.Definitions:
            guid2def[str(definition.GUID)] = definition

    # Step 5: Insert parameters in transaction
    t = DB.Transaction(doc, "Add Shared Parameters")
    t.Start()
    results = []
    for setting in param_settings:
        param = setting['param']
        is_instance = setting['is_instance']
        group_enum = setting['group']

        definition = guid2def.get(param['guid'], None)
        if not definition:
            results.append("Parameter '{}' not found in API file.".format(param['name']))
            continue
        try:
            fam_param = add_shared_parameter_to_family(doc, definition, group_enum, is_instance)
            results.append("Added: '{}' as {} under '{}'".format(param['name'], "Instance" if is_instance else "Type", DB.LabelUtils.GetLabelFor(group_enum)))
        except Exception as ex:
            results.append("Error adding '{}': {}".format(param['name'], str(ex)))
    t.Commit()

    # Show results
    forms.alert('\n'.join(results), title="Results")

if __name__ == "__main__":
    main()