# -*- coding: utf-8 -*-
from pyrevit import script, forms, revit
import time
import os
import csv

doc = revit.doc
user = doc.Application.Username  # <-- Fixed

# Retrieve stored data using script.retrieve_data
start_time = script.retrieve_data("projecttracker_start_time")
metadata = script.retrieve_data("projecttracker_metadata")
context = script.retrieve_data("projecttracker_context")

if not start_time:
    forms.alert("No tracking session in progress!", exitscript=True)

elapsed = int(time.time() - start_time)

log_data = {
    "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
    "username": user,
    "project_name": metadata["project_name"],
    "discipline": metadata["discipline"],
    "task": metadata["task"],
    "file_name": context["file_name"],
    "view_name": context["view_name"],
    "view_type": context["view_type"],
    "log_type": "Automatic",
    "elapsed_seconds": elapsed
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

# Clean up data
script.store_data("projecttracker_start_time", None)
script.store_data("projecttracker_metadata", None)
script.store_data("projecttracker_context", None)

forms.alert("Tracking stopped.\nLogged {} seconds for project '{}'.".format(elapsed, metadata["project_name"]))