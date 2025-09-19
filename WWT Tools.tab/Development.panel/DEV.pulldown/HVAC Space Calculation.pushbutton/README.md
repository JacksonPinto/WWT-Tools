# pyRevit Script: HVAC System Calculation for Revit 2025+ (ASHRAE Compliant)

This script is designed for Revit 2025+ (CPython, not IronPython) and automates HVAC space calculations according to ASHRAE standards. It scans all Spaces in the project, collects their dimensions, and fills in key HVAC calculation parameters.

## Features

- **Scans all Spaces** in the active Revit project.
- **Extracts dimension values**: Area, Perimeter, Unbounded Height, Volume (see image 3).
- **Considers "Space Type"** for ASHRAE lookup/calculation.
- **Performs ASHRAE-based calculations** for:
  - Specified Supply/Return/Exhaust Airflow
  - ASHRAE Occupant Count Input
  - ASHRAE Zone Air Distribution Efficiency
  - Design Heating/Cooling Load
  - Design Air Changes per Hour (ACH)
  - Number of People
- **Writes results** to the corresponding Space parameters (see images 1 & 2 for parameter mapping).

## Requirements

- Revit 2025+ (with CPython scripting support)
- pyRevit 5.0+
- Python 3.10+ (CPython)
- `RevitAPI` and `RevitServices` (from pyRevit/revitpythonwrapper)
- ASHRAE design guide reference values (for default airflow/person, occupancy, etc.)

## ASHRAE Reference

See [ASHRAE Design Guides](https://www.ashrae.org/technical-resources/bookstore/ashrae-design-guides) for detailed standards and recommended values.

---

## Usage

1. Load the script in pyRevit (Tools > pyRevit > Add Script).
2. Run on a model with Spaces and MEP settings configured.
3. Script will update all relevant parameters for each Space.

---

## Parameter Mapping

| Revit Parameter Name                  | Purpose                                                 | Example Source (Image)       |
|---------------------------------------|---------------------------------------------------------|------------------------------|
| Specified Supply Airflow              | L/s; script-calculated supply airflow                   | Image 1                      |
| Specified Return Airflow              | L/s; script-calculated return airflow                   | Image 1                      |
| Specified Exhaust Airflow             | L/s; script-calculated exhaust airflow                  | Image 1                      |
| ASHRAE Occupant Count Input           | Calculated occupant count (ASHRAE)                      | Image 1                      |
| ASHRAE Zone Air Distribution Eff      | ASHRAE Ef (from table or default)                       | Image 1                      |
| Design Heating Load                   | kW; calculated heating load                             | Image 2                      |
| Design Cooling Load                   | kW; calculated cooling load                             | Image 2                      |
| Design ACH                           | Air changes per hour                                    | Image 2                      |
| Number of People                      | Occupants; calculated                                   | Image 2                      |
| Area, Perimeter, Height, Volume       | Source values for calculations                          | Image 3                      |

---

## Calculation Notes

- Airflow, occupancy, and load calculations are based on ASHRAE tables, using the "Space Type" parameter to determine defaults.
- You may customize the ASHRAE lookup table in the script for your region/project.

---

## Limitations

- Only works with Revit Spaces, not Rooms.
- Parameter names must match exactly as in your Revit template (customizations may require script edits).