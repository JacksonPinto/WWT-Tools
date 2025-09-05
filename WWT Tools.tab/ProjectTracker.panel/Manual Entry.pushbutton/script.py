# -*- coding: utf-8 -*-
from pyrevit import forms, script, revit
import time
import os
import csv

doc = revit.doc
user = doc.Application.Username  # <-- Fixed

fields = forms.fields(
    forms.TextBox("Project Name", default="Unknown"),
    forms.TextBox("Discipline", default="Unknown"),
    forms.TextBox("Task", default="Meeting"),
    forms.TextBox("Minutes", default="30")
)
if forms.Form(fields, title="Manual Time Entry").show():
    project_name = fields[0].value
    discipline = fields[1].value
    task = fields[2].value
    try:
        minutes = int(fields[3].value)
    except Exception:
        forms.alert("Invalid minutes value. Please enter a number.", exitscript=True)

    context = {
        "file_name": doc.Title,
        "view_name": doc.ActiveView.Name,
        "view_type": str(doc.ActiveView.ViewType)
    }

    log_data = {
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "username": user,
        "project_name": project_name,
        "discipline": discipline,
        "task": task,
        "file_name": context["file_name"],
        "view_name": context["view_name"],
        "view_type": context["view_type"],
        "log_type": "Manual",
        "elapsed_seconds": minutes * 60
    }

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    csv_path = os.path.abspath(os.path.join(data_dir, "project_log.csv"))

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "datetime", "username", "project_name", "discipline", "task",
            "file_name", "view_name", "view_type", "log_type", "elapsed_seconds"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)

    forms.alert("Manual entry of {} minutes logged for project '{}'.".format(minutes, project_name))