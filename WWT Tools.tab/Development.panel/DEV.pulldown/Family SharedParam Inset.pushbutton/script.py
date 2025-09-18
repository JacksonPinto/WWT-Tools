from pyrevit import revit, forms
import os

doc = revit.doc
app = doc.Application
sp_filepath = app.SharedParametersFilename

if not sp_filepath or not os.path.exists(sp_filepath):
    forms.alert("No shared parameters file set or file not found:\n{}".format(sp_filepath), exitscript=True)

sp_file = app.OpenSharedParameterFile()
if not sp_file:
    forms.alert("Revit API could not open the shared parameters file (null object).\n\nTry copying the file to C:\\Temp, set that as your shared parameter file, and try again.", exitscript=True)

group_count = len(list(sp_file.Groups))
params_per_group = [len(list(g.Definitions)) for g in sp_file.Groups]
param_count = sum(params_per_group)
first_group = sp_file.Groups[0].Name if group_count > 0 else "(none)"
first_param = sp_file.Groups[0].Definitions[0].Name if group_count > 0 and params_per_group[0] > 0 else "(none)"

msg = (
    "Path: {}\n"
    "Groups found: {}\n"
    "Parameters per group: {}\n"
    "Total parameters: {}\n"
    "First group: {}\n"
    "First parameter: {}\n"
    .format(
        sp_filepath,
        group_count,
        params_per_group,
        param_count,
        first_group,
        first_param
    )
)
forms.alert(msg, title="Shared Parameter File Diagnostic")