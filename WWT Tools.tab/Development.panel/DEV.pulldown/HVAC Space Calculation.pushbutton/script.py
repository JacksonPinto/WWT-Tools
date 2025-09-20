# -*- coding: utf-8 -*-
import sys

# --- ENGINE DETECTION ---
is_cpython = sys.version_info.major >= 3
if is_cpython:
    from Autodesk.Revit.DB import UnitTypeId, UnitUtils
else:
    from Autodesk.Revit.DB import DisplayUnitType, UnitUtils

from Autodesk.Revit.DB import (
    FilteredElementCollector, BuiltInCategory, BuiltInParameter, Transaction,
    StorageType, ElementId
)
from pyrevit import revit, DB, script

# --- ASHRAE DEFAULTS ---
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

# --- UNIT MAPS ---
if is_cpython:
    PARAM_UNIT_MAP = {
        "Specified Supply Airflow": UnitTypeId.HVAC_Airflow,            # L/s (internal: CFM)
        "Specified Return Airflow": UnitTypeId.HVAC_Airflow,
        "Specified Exhaust Airflow": UnitTypeId.HVAC_Airflow,
        "ASHRAE Occupant Count Input": UnitTypeId.Number,               # Unitless
        "ASHRAE Zone Air Distribution Eff": UnitTypeId.Number,          # Unitless
        "Design Heating Load": UnitTypeId.HVAC_Power,                   # kW (internal: W)
        "Design Cooling Load": UnitTypeId.HVAC_Power,                   # kW (internal: W)
        "Design ACH": UnitTypeId.Number,                                # Unitless
        "Number of People": UnitTypeId.Number                           # Unitless
    }
else:
    PARAM_UNIT_MAP = {
        "Specified Supply Airflow": DisplayUnitType.DUT_LITERS_PER_SECOND,
        "Specified Return Airflow": DisplayUnitType.DUT_LITERS_PER_SECOND,
        "Specified Exhaust Airflow": DisplayUnitType.DUT_LITERS_PER_SECOND,
        "ASHRAE Occupant Count Input": DisplayUnitType.DUT_GENERAL,
        "ASHRAE Zone Air Distribution Eff": DisplayUnitType.DUT_GENERAL,
        "Design Heating Load": DisplayUnitType.DUT_KILOWATTS,
        "Design Cooling Load": DisplayUnitType.DUT_KILOWATTS,
        "Design ACH": DisplayUnitType.DUT_GENERAL,
        "Number of People": DisplayUnitType.DUT_GENERAL
    }

def to_internal(value, unit_type):
    return UnitUtils.ConvertToInternalUnits(float(value), unit_type)

# --- PARAMETER ACCESS HELPERS ---
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

def get_param_display_value(element, param_name):
    param = element.LookupParameter(param_name)
    if param:
        try:
            return param.AsValueString()
        except Exception:
            return None
    return None

def set_param_value_with_unit(element, param_name, display_value):
    param = element.LookupParameter(param_name)
    if param and not param.IsReadOnly:
        try:
            unit_type = PARAM_UNIT_MAP.get(param_name, PARAM_UNIT_MAP.values()[0])
            internal_value = to_internal(display_value, unit_type)
            param.Set(internal_value)
        except Exception as e:
            print("Error setting parameter {}: {}".format(param_name, e))

def get_space_type_defaults(space_type):
    if not space_type:
        return ASHRAE_DEFAULTS['Default']
    for k in ASHRAE_DEFAULTS.keys():
        if k.lower() in space_type.lower():
            return ASHRAE_DEFAULTS[k]
    return ASHRAE_DEFAULTS['Default']

def calculate_hvac(space, defaults, area, volume):
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
        'Space Height': height                                            # Internal for debug
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
    debug_lines = []
    with Transaction(doc, "HVAC Calculation for Spaces") as t:
        t.Start()
        for space in spaces:
            space_type = get_param_value(space, BuiltInParameter.ROOM_DEPARTMENT)
            defaults = get_space_type_defaults(space_type)
            area = get_param_value(space, BuiltInParameter.ROOM_AREA) or 0
            volume = get_param_value(space, BuiltInParameter.ROOM_VOLUME) or 0
            perimeter = get_param_value(space, BuiltInParameter.ROOM_PERIMETER) or 0

            # Convert from Revit internal units to meters (if needed)
            try:
                if is_cpython:
                    area = UnitUtils.ConvertFromInternalUnits(area, UnitTypeId.SquareMeters)
                    volume = UnitUtils.ConvertFromInternalUnits(volume, UnitTypeId.CubicMeters)
                    perimeter = UnitUtils.ConvertFromInternalUnits(perimeter, UnitTypeId.Meters)
                else:
                    area = UnitUtils.ConvertFromInternalUnits(area, DisplayUnitType.DUT_SQUARE_METERS)
                    volume = UnitUtils.ConvertFromInternalUnits(volume, DisplayUnitType.DUT_CUBIC_METERS)
                    perimeter = UnitUtils.ConvertFromInternalUnits(perimeter, DisplayUnitType.DUT_METERS)
            except Exception:
                pass

            results = calculate_hvac(space, defaults, area, volume)
            # Set all mapped parameters using correct unit conversion
            for key, param_name in PARAM_MAP.items():
                set_param_value_with_unit(space, param_name, results[key])
            updated_spaces += 1

            # Get space name for debug, fallback to ID if no name
            space_name = get_param_value_by_name(space, "Name")
            if not space_name:
                space_name = "SpaceId: {}".format(space.Id.IntegerValue)
            # Collect parameter values for debug
            debug_lines.append('\n{}:'.format(space_name))
            for key, param_name in PARAM_MAP.items():
                display_val = get_param_display_value(space, param_name)
                debug_lines.append('    "{}" = {}'.format(param_name, display_val))
        t.Commit()

    output.print_md("**Updated {} spaces with HVAC calculations.**".format(updated_spaces))
    output.print_md("---\n**DEBUG VALUES:**\n" + "\n".join(debug_lines))

if __name__ == "__main__":
    main()