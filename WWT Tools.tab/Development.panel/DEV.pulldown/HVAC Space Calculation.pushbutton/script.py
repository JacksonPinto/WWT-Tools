# -*- coding: utf-8 -*-
# pyRevit Script: HVAC System Calculation for Spaces (Revit 2025+ API, CPython)
# Author: JacksonPinto / Copilot
# Reference: ASHRAE Design Guides, pyRevit, Revit 2025+ API

from Autodesk.Revit.DB import (
    FilteredElementCollector, BuiltInCategory, BuiltInParameter, Transaction,
    StorageType, UnitTypeId, ElementId
)
from Autodesk.Revit.DB.Mechanical import Space
from pyrevit import revit, DB, script

# ASHRAE default values by Space Type (customize for your project)
ASHRAE_DEFAULTS = {
    'Office': {
        'airflow_per_person': 2.5,   # L/s/person
        'airflow_per_area': 0.3,     # L/s/m²
        'occupant_density': 5,       # people/100m²
        'zone_efficiency': 0.8,      # Distribution Efficiency
        'ach': 1.0,                  # Air Changes per Hour
        'sensible_heat': 70,         # W/m²
        'latent_heat': 25            # W/m²
    },
    'Conference': {
        'airflow_per_person': 3.8,
        'airflow_per_area': 0.6,
        'occupant_density': 15,
        'zone_efficiency': 0.8,
        'ach': 1.0,
        'sensible_heat': 70,
        'latent_heat': 25
    },
    # Add more types as needed
    'Default': {
        'airflow_per_person': 2.5,
        'airflow_per_area': 0.3,
        'occupant_density': 5,
        'zone_efficiency': 0.8,
        'ach': 1.0,
        'sensible_heat': 70,
        'latent_heat': 25
    }
}

def get_param_value(element, built_in_param):
    param = element.get_Parameter(built_in_param)
    if param:
        if param.StorageType == StorageType.Double:
            try:
                return param.AsDouble()
            except Exception:
                return None
        elif param.StorageType == StorageType.String:
            return param.AsString()
        elif param.StorageType == StorageType.Integer:
            return param.AsInteger()
        elif param.StorageType == StorageType.ElementId:
            return param.AsElementId()
    return None

def set_param_value(element, param_name, value):
    param = element.LookupParameter(param_name)
    if param and not param.IsReadOnly:
        try:
            if param.StorageType == StorageType.Double:
                param.Set(float(value))
            elif param.StorageType == StorageType.Integer:
                param.Set(int(round(value)))
            elif param.StorageType == StorageType.String:
                param.Set(str(value))
            elif param.StorageType == StorageType.ElementId:
                if isinstance(value, ElementId):
                    param.Set(value)
        except Exception as e:
            pass

def get_space_type_defaults(space_type):
    if not space_type:
        return ASHRAE_DEFAULTS['Default']
    for k in ASHRAE_DEFAULTS.keys():
        if k.lower() in space_type.lower():
            return ASHRAE_DEFAULTS[k]
    return ASHRAE_DEFAULTS['Default']

def calculate_hvac(space, defaults, area, volume):
    # Occupant count (ASHRAE): area [m²] * (occupant_density/100)
    num_people = area * (defaults['occupant_density'] / 100.0)
    # Outdoor airflow required (L/s): per ASHRAE (people + area)
    outdoor_airflow = (num_people * defaults['airflow_per_person'] +
                       area * defaults['airflow_per_area'])
    # Design ACH: outdoor_airflow [L/s] * 3.6 / volume [m³]
    design_ach = 0
    if volume > 0:
        design_ach = (outdoor_airflow * 3.6) / volume
    # Sensible and latent heat loads
    design_heating_load = area * defaults['sensible_heat'] / 1000.0  # kW
    design_cooling_load = area * (defaults['sensible_heat'] + defaults['latent_heat']) / 1000.0 # kW

    return {
        'Specified Supply Airflow': outdoor_airflow,                      # L/s
        'Specified Return Airflow': outdoor_airflow * 0.8,                # Example: 80% of supply
        'Specified Exhaust Airflow': outdoor_airflow * 0.2,               # Example: 20% of supply
        'ASHRAE Occupant Count Input': num_people,
        'ASHRAE Zone Air Distribution Eff': defaults['zone_efficiency'],
        'Design Heating Load': design_heating_load,                       # kW
        'Design Cooling Load': design_cooling_load,                       # kW
        'Design ACH': design_ach,
        'Number of People': num_people
    }

PARAM_MAP = {
    'Specified Supply Airflow': "Specified Supply Airflow",
    'Specified Return Airflow': "Specified Return Airflow",
    'Specified Exhaust Airflow': "Specified Exhaust Airflow",
    'ASHRAE Occupant Count Input': "ASHRAE Occupant Count Input",
    'ASHRAE Zone Air Distribution Eff': "ASHRAE Zone Air Distribution Eff",
    'Design Heating Load': "Design Heating Load",
    'Design Cooling Load': "Design Cooling Load",
    'Design ACH': "Design ACH",
    'Number of People': "Number of People"
}

def main():
    doc = revit.doc
    output = script.get_output()
    spaces = FilteredElementCollector(doc).OfCategory(BuiltInCategory.OST_MEPSpaces).WhereElementIsNotElementType().ToElements()
    updated_spaces = 0
    with Transaction(doc, "HVAC Calculation for Spaces") as t:
        t.Start()
        for space in spaces:
            # Get space type (BuiltInParameter.ROOM_DEPARTMENT is typically used for Space Type)
            space_type = get_param_value(space, BuiltInParameter.ROOM_DEPARTMENT)
            defaults = get_space_type_defaults(space_type)

            # Get dimensions (Area, Perimeter, Height, Volume)
            area = get_param_value(space, BuiltInParameter.ROOM_AREA) or 0
            volume = get_param_value(space, BuiltInParameter.ROOM_VOLUME) or 0
            perimeter = get_param_value(space, BuiltInParameter.ROOM_PERIMETER) or 0
            height = get_param_value(space, BuiltInParameter.ROOM_UNBOUNDED_HEIGHT) or 0

            # Convert from Revit internal units to meters (if needed)
            area = DB.UnitUtils.ConvertFromInternalUnits(area, UnitTypeId.SquareMeters)
            volume = DB.UnitUtils.ConvertFromInternalUnits(volume, UnitTypeId.CubicMeters)
            perimeter = DB.UnitUtils.ConvertFromInternalUnits(perimeter, UnitTypeId.Meters)
            height = DB.UnitUtils.ConvertFromInternalUnits(height, UnitTypeId.Meters)

            # Calculate all HVAC values
            results = calculate_hvac(space, defaults, area, volume)
            for key, param_name in PARAM_MAP.items():
                set_param_value(space, param_name, results[key])
            updated_spaces += 1
        t.Commit()
    output.print_md("**Updated {} spaces with HVAC calculations.**".format(updated_spaces))

if __name__ == "__main__":
    main()