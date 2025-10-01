# WWTpyRevit

This repository contains custom pyRevit tools designed specifically for MEP (Mechanical, Electrical, and Plumbing) projects.

## About pyRevit

[pyRevit](https://www.pyrevit.com/) is an IronPython script library for Revit. It extends Autodesk Revit with a powerful set of tools and a flexible scripting environment, allowing users to automate repetitive tasks and create custom workflows within Revit.

## Requirements

To use the tools in this repository, you will need the following:

- **Python 3.12.3**  
  (Required for Topologic node integration. Download from [python.org](https://www.python.org/downloads/release/python-3123/))

- **pyRevit 5.2.0.25181**  
  (You can download this specific version from [pyRevit releases](https://github.com/eirannejad/pyRevit/releases/tag/v5.2.0.25181))

  - **topologicpy 0.8.58**  
  (You can install this specific version from python console pip install topologicpy --upgrade)

## How to Use

1. **Install Python 3.12.3**  
   Make sure to add Python to your system PATH during installation.

2. **Install pyRevit 5.2.0.25181**  
   Follow the official installation instructions for pyRevit and make sure you are using the specified version.

3. **Clone this repository**  
   ```
   git clone https://github.com/JacksonPinto/WWTpyRevit.git
   ```

4. **Add the repository as an extension in pyRevit**  
   Open pyRevit CLI or manage extensions through the pyRevit UI to add this folder as an extension.

## Notes

- These tools are tailored for MEP workflows in Revit.
- The Topologic node requires Python 3.12.3 specifically.
- Ensure all dependencies are installed before using the add-ins.

For more information on pyRevit, visit the [official documentation](https://www.pyrevit.com/).
