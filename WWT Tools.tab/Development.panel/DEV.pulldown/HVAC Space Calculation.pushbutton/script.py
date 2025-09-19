# -*- coding: utf-8 -*-
# pyRevit Script: HVAC System Calculation for Spaces (IronPython & CPython Compatible)
# Author: JacksonPinto / Copilot

from Autodesk.Revit.DB import (
    FilteredElementCollector, BuiltInCategory, BuiltInParameter, Transaction,
    StorageType, UnitTypeId, ElementId
)
from Autodesk.Revit.DB.Mechanical import Space
from pyrevit import revit, DB, script

# ASHRAE default values by Space Type
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

def get_param_value_by_name(element, param_name):
    param = element.LookupParameter(param_name)
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
        except Exception:
            pass

def get_space_type_defaults(space_type):
    if not space_type:
        return ASHRAE_DEFAULTS['Default']
    for k in ASHRAE_DEFAULTS.keys():
        if k.lower() in space_type.lower():
            return ASHRAE_DEFAULTS[k]
    return ASHRAE_DEFAULTS['Default']

def calculate_hvac(space, defaults, area, volume):
    # Use area and volume in m² and m³
    # If height is needed, calculate as height = volume / area (if area > 0)
    height = volume / area if (area > 0 and volume > 0) else 0

    num_people = area * (defaults['occupant_density'] / 100.0)
    outdoor_airflow = (num_people * defaults['airflow_per_person'] +
                       area * defaults['airflow_per_area'])
    design_ach = 0
    if volume > 0:
        design_ach = (outdoor_airflow * 3.6) / volume
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
        'Number of People': num_people,
        'Space Height': height                                            # For debug/reference, not set in Revit
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
    # 'Space Height': -- Not a Revit parameter, internal only
}

def main():
    doc = revit.doc
    output = script.get_output()
    spaces = FilteredElementCollector(doc).OfCategory(BuiltInCategory.OST_MEPSpaces).WhereElementIsNotElementType().ToElements()
    updated_spaces = 0
    with Transaction(doc, "HVAC Calculation for Spaces") as t:
        t.Start()
        for space in spaces:
            space_type = get_param_value(space, BuiltInParameter.ROOM_DEPARTMENT)
            defaults = get_space_type_defaults(space_type)

            # Use BuiltInParameter for Area, Volume, Perimeter
            area = get_param_value(space, BuiltInParameter.ROOM_AREA) or 0
            volume = get_param_value(space, BuiltInParameter.ROOM_VOLUME) or 0
            perimeter = get_param_value(space, BuiltInParameter.ROOM_PERIMETER) or 0

            # Convert from Revit internal units to meters (if needed)
            try:
                area = DB.UnitUtils.ConvertFromInternalUnits(area, UnitTypeId.SquareMeters)
                volume = DB.UnitUtils.ConvertFromInternalUnits(volume, UnitTypeId.CubicMeters)
                perimeter = DB.UnitUtils.ConvertFromInternalUnits(perimeter, UnitTypeId.Meters)
            except Exception:
                # For IronPython/older API, fallback: values are in project units
                pass

            results = calculate_hvac(space, defaults, area, volume)
            for key, param_name in PARAM_MAP.items():
                set_param_value(space, param_name, results[key])
            updated_spaces += 1
        t.Commit()
    output.print_md("**Updated {} spaces with HVAC calculations.**".format(updated_spaces))

if __name__ == "__main__":
    main()