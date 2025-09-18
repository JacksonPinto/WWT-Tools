# -*- coding: utf-8 -*-
"""
pyRevit Script: Bulk Insert Shared Parameters from Configured File

- Uses the shared parameters file currently configured in Revit (Application.SharedParametersFilename).
- Lists all available shared parameters in that file for user selection (multi-select).
- Allows bulk insertion of selected parameters into the open family, with type/instance and group options.
- No file dialog: always uses the configured file for this Revit session.

Author: Jackson Pinto
"""

from pyrevit import revit, DB, forms, script
import os

__title__ = 'Bulk SharedParams (Config File)'
__author__ = 'Jackson Pinto'

def parse_shared_params_file(sp_filepath):
    params = []
    if not os.path.exists(sp_filepath):
        return params

    section = None
    groups = {}
    with open(sp_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('*'):
                if 'GROUP' in line.upper():
                    section = 'groups'
                elif 'PARAM' in line.upper():
                    section = 'params'
                else:
                    section = None
                continue
            if not line or line.startswith('#') or section is None:
                continue

            if section == 'groups':
                # Format: GROUP\tID\tNAME
                if line.startswith('GROUP'):
                    try:
                        _, group_id, group_name = line.split('\t', 2)
                        groups[group_id] = group_name
                    except Exception:
                        continue
            elif section == 'params':
                # Format: PARAM\tGUID\tNAME\tDATATYPE\tDATACATEGORY\tGROUP\tVISIBLE\tDESCRIPTION\tUSERMODIFIABLE\tHIDEWHENNOVALUE
                if line.startswith('PARAM'):
                    parts = line.split('\t')
                    if len(parts) >= 6:
                        param = {
                            'guid': parts[1],
                            'name': parts[2],
                            'datatype': parts[3],
                            'datacategory': parts[4],
                            'group_id': parts[5],
                            'visible': parts[6] if len(parts) > 6 else '',
                            'desc': parts[7] if len(parts) > 7 else '',
                            'usermodifiable': parts[8] if len(parts) > 8 else '',
                            'hidewhennovalue': parts[9] if len(parts) > 9 else '',
                            'group_name': groups.get(parts[5], parts[5])
                        }
                        params.append(param)
    return params

def select_parameters_ui(params):
    class ParamOption(forms.TemplateListItem):
        @property
        def name(self):
            return self.item['name']
        @property
        def description(self):
            return "Type: {} | GUID: {} | Group: {}".format(self.item['datatype'], self.item['guid'], self.item['group_name'])

    param_options = [ParamOption(p) for p in params]
    res = forms.SelectFromList.show(param_options,
                                    title="Select Shared Parameters to Add",
                                    multiselect=True,
                                    button_name='Add Selected')
    if not res:
        return []
    return [x.item for x in res]

def get_param_settings_ui(selected_params):
    # Get all built-in parameter groups (Revit API)
    param_groups = [g for g in DB.BuiltInParameterGroup if g != DB.BuiltInParameterGroup.INVALID]
    group_names = [DB.LabelUtils.GetLabelFor(g) for g in param_groups]
    settings = []
    for param in selected_params:
        # Ask user for Type/Instance and group
        paramtype_choice = forms.CommandSwitchWindow.show(
            ['Instance', 'Type'],
            message="Insert parameter '{}' as:".format(param['name']),
            default='Instance'
        )
        group_choice = forms.SelectFromList.show(
            group_names,
            title="Group parameter '{}' under:".format(param['name']),
            button_name='Select'
        )
        if not group_choice:
            group_choice = group_names[0]
        group_enum = param_groups[group_names.index(group_choice)]
        settings.append({
            'param': param,
            'is_instance': (paramtype_choice == 'Instance'),
            'group': group_enum
        })
    return settings

def add_shared_parameter_to_family(doc, definition, group, is_instance):
    fam_mgr = doc.FamilyManager
    # Check if param already exists
    for fam_param in fam_mgr.Parameters:
        if fam_param.Definition.Name == definition.Name:
            return fam_param # Already exists
    fam_param = fam_mgr.AddParameter(definition, group, is_instance)
    return fam_param

def find_definition_by_guid(sp_file, guid):
    # GUID comparison is case-insensitive
    for group in sp_file.Groups:
        for definition in group.Definitions:
            if str(definition.GUID).lower() == guid.lower():
                return definition
    return None

def main():
    doc = revit.doc
    if not doc.IsFamilyDocument:
        forms.alert("This script only works in Revit Family Documents.", exitscript=True)

    app = doc.Application
    sp_filepath = app.SharedParametersFilename

    # Defensive: ensure path is string and exists
    if not sp_filepath or not os.path.exists(sp_filepath):
        forms.alert(
            "No shared parameters file is configured in Revit, or the file does not exist.\n"
            "Set the shared parameters file in Revit (File > Options > File Locations > Shared Parameters), then try again.",
            exitscript=True
        )

    # Parse file, list parameters for selection
    all_params = parse_shared_params_file(sp_filepath)
    if not all_params:
        forms.alert("No parameters found in shared parameters file:\n{}".format(sp_filepath), exitscript=True)

    selected_params = select_parameters_ui(all_params)
    if not selected_params:
        forms.alert("No parameters selected.", exitscript=True)

    # For each param, get type/instance and group
    param_settings = get_param_settings_ui(selected_params)

    # Open shared params file in Revit API (must match filepath)
    sp_file = app.OpenSharedParameterFile()
    if not sp_file:
        forms.alert("Could not open shared parameters file via API.", exitscript=True)

    # Insert parameters in transaction
    t = DB.Transaction(doc, "Add Shared Parameters")
    t.Start()
    results = []
    for setting in param_settings:
        param = setting['param']
        is_instance = setting['is_instance']
        group_enum = setting['group']

        definition = find_definition_by_guid(sp_file, param['guid'])
        if not definition:
            results.append(u"Parameter '{}' not found in API file.".format(param['name']))
            continue
        try:
            fam_param = add_shared_parameter_to_family(doc, definition, group_enum, is_instance)
            results.append(u"Added: '{}' as {} under '{}'".format(param['name'], "Instance" if is_instance else "Type", DB.LabelUtils.GetLabelFor(group_enum)))
        except Exception as ex:
            results.append(u"Error adding '{}': {}".format(param['name'], str(ex)))
    t.Commit()

    forms.alert('\n'.join(results), title="Results")

if __name__ == "__main__":
    main()