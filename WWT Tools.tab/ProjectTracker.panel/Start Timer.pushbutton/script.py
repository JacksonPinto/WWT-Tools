# -*- coding: utf-8 -*-
from pyrevit import script, revit, forms
import time

# Get the active Revit document
doc = revit.doc
user = doc.Application.Username  # <-- Fixed

# Prompt for project metadata (comma separated)
metadata = forms.ask_for_string(
    default="ProjectName,Discipline,Task",
    prompt="Enter project name, discipline, task (comma separated):"
)
if metadata:
    parts = [x.strip() for x in metadata.split(",")]
    while len(parts) < 3:
        parts.append("Unknown")
    project_name, discipline, task = parts
else:
    project_name, discipline, task = "Unknown", "Unknown", "Unknown"

context = {
    "file_name": doc.Title,
    "project_name": getattr(doc.ProjectInformation, "Name", ""),
    "view_name": getattr(doc.ActiveView, "Name", ""),
    "view_type": str(getattr(doc.ActiveView, "ViewType", "")),
}

# Store start time and metadata using script.store_data
script.store_data("projecttracker_start_time", time.time())
script.store_data("projecttracker_metadata", {
    "project_name": project_name,
    "discipline": discipline,
    "task": task
})
script.store_data("projecttracker_context", context)

forms.alert(
    "Started time tracking for project '{}'.\nDon't forget to Stop when you're done!".format(project_name),
    exitscript=False
)