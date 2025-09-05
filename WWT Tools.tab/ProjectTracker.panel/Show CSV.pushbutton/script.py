# -*- coding: utf-8 -*-
# ProjectTracker - Show CSV Button
#
# Opens the project_log.csv file in the default CSV viewer (e.g., Excel).
# Reads settings.json for custom log folder if set.

from pyrevit import forms
import os
import json
import subprocess
import sys

# Determine data directory and settings file
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
settings_path = os.path.join(data_dir, "settings.json")
csv_filename = "project_log.csv"

# Default: use data_dir for CSV unless overridden in settings
log_folder = data_dir

# Load custom folder from settings if present
if os.path.exists(settings_path):
    with open(settings_path, "r") as f:
        try:
            settings = json.load(f)
            if "log_folder" in settings and os.path.isdir(settings["log_folder"]):
                log_folder = settings["log_folder"]
        except Exception:
            pass

csv_path = os.path.join(log_folder, csv_filename)

# Check if CSV exists
if not os.path.isfile(csv_path):
    forms.alert("No log file found at:\n{}\n\nYou may need to start tracking or set the log folder in Settings.".format(csv_path))
else:
    try:
        # Windows: os.startfile, else use subprocess
        if sys.platform.startswith("win"):
            os.startfile(csv_path)
        elif sys.platform.startswith("darwin"):  # macOS
            subprocess.Popen(["open", csv_path])
        else:  # Linux/other
            subprocess.Popen(["xdg-open", csv_path])
    except Exception as e:
        forms.alert("Failed to open log file:\n{}\n\nError: {}".format(csv_path, str(e)))