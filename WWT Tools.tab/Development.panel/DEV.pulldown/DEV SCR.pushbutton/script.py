# check_python_version.py
# Script for pyRevit to print Python sys.version and sys.version_info

import sys
from pyrevit import script

output = script.get_output()

output.print_md("## Python Version Info (pyRevit)")

output.print_md("**sys.version:** `{}`".format(sys.version))
output.print_md("**sys.version_info:** `{}`".format(sys.version_info))

# Also print to Revit command line output
print("sys.version:", sys.version)
print("sys.version_info:", sys.version_info)
# check_cpython_version.py
# Script for pyRevit to check and print the Python version used by CPython (external Python 3.x)

import subprocess
from pyrevit import script

output = script.get_output()

# Path to your Python 3.x interpreter
PYTHON3_PATH = r"C:\Users\JacksonAugusto\AppData\Local\Programs\Python\Python38\python.exe"

output.print_md("## Checking CPython (external Python 3.x) version")
try:
    # Run a small Python script to print version info
    result = subprocess.check_output(
        [PYTHON3_PATH, "-c", "import sys; print('sys.version:', sys.version); print('sys.version_info:', sys.version_info)"],
        universal_newlines=True
    )
    output.print_md("**CPython Output**")
    output.print_md("```text\n{}\n```".format(result))
except Exception as e:
    output.print_md("**Error:** Could not run external Python.\nException: `{}`".format(str(e)))
