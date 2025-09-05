# -*- coding: utf-8 -*-
# ProjectTracker - Settings Button
#
# Allows the user to pick the folder where the time log CSV file will be stored.
# This is useful for sharing the log via network/cloud storage.

from pyrevit import forms
import os
import json

# Determine where to keep settings file (in extension's data folder)
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
settings_path = os.path.join(data_dir, "settings.json")

# Load current settings if available
settings = {}
if os.path.exists(settings_path):
    with open(settings_path, "r") as f:
        try:
            settings = json.load(f)
        except Exception:
            settings = {}

# Let user pick (or re-pick) the log folder
folder = forms.pick_folder(
    title="Select folder to store the project log (CSV) file.\n(Use a cloud folder for auto-sync!)"
)
if folder and os.path.isdir(folder):
    settings["log_folder"] = folder
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    forms.alert("Log folder set to:\n{}".format(folder))
else:
    forms.alert("No folder selected. Log folder unchanged.")