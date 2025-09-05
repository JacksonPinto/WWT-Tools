# coding: utf-8
# -*- coding: utf-8 -*-
__title__   = 'Units Converter'
__tooltip__ = 'Toggle Common & Electrical units Imperial ⇆ Metric'
__author__  = 'ChatGPT'
__version__ = '2.2.7'

from Autodesk.Revit.DB import (
    Transaction, TransactionStatus, FormatOptions,
    SpecTypeId, UnitTypeId, SymbolTypeId, UnitUtils
)
from Autodesk.Revit.UI import TaskDialog

doc   = __revit__.ActiveUIDocument.Document
units = doc.GetUnits()

# 1) Friendly names for every SpecTypeId we actually support
SPEC_NAMES = {
    # Common
    SpecTypeId.Angle:                'Angle',
    SpecTypeId.Distance:             'Distance',
    SpecTypeId.Length:               'Length',
    SpecTypeId.Area:                 'Area',
    SpecTypeId.Volume:               'Volume',
    SpecTypeId.MassDensity:          'Mass Density',
    SpecTypeId.RotationAngle:        'Rotation Angle',
    SpecTypeId.Slope:                'Slope',
    SpecTypeId.Speed:                'Speed',
    SpecTypeId.Time:                 'Time',
    # Electrical
    SpecTypeId.ApparentPower:        'Apparent Power',
    SpecTypeId.CableTraySize:       'Cable Tray Sizes',
    SpecTypeId.ConduitSize:          'Conduit Size',
    SpecTypeId.Current:              'Current',
    SpecTypeId.ElectricalFrequency:  'Frequency',
    SpecTypeId.Illuminance:          'Illuminance',
    SpecTypeId.Luminance:            'Luminance',
    SpecTypeId.LuminousFlux:         'Luminous Flux',
    SpecTypeId.LuminousIntensity:    'Luminous Intensity',
    SpecTypeId.ElectricalPotential:  'Electrical Potential',
    SpecTypeId.ElectricalPower:      'Power',
    SpecTypeId.ElectricalPowerDensity:'Power Density',
    SpecTypeId.PowerPerLength:       'Power per Length',
    SpecTypeId.ElectricalResistivity:'Electrical Resistivity',
    SpecTypeId.ElectricalTemperature:'Temperature',
    SpecTypeId.Wattage:              'Wattage',
    SpecTypeId.WireDiameter:         'Wire Diameter',
    # NOTE: SpecTypeId.ApparentPowerDensity **does not exist** in 2022 and is omitted
}

# 2) Imperial→Metric
imp_to_met = {
    # Common
    SpecTypeId.Angle:                   UnitTypeId.Degrees,
    SpecTypeId.Distance:                UnitTypeId.Meters,
    SpecTypeId.Length:                  UnitTypeId.Millimeters,
    SpecTypeId.Area:                    UnitTypeId.SquareMeters,
    SpecTypeId.Volume:                  UnitTypeId.CubicMeters,
    SpecTypeId.MassDensity:             UnitTypeId.KilogramsPerCubicMeter,
    SpecTypeId.RotationAngle:           UnitTypeId.Degrees,
    SpecTypeId.Slope:                   SymbolTypeId.Percent,
    SpecTypeId.Speed:                   UnitTypeId.MetersPerSecond,
    SpecTypeId.Time:                    UnitTypeId.Seconds,
    # Electrical
    SpecTypeId.ApparentPower:           UnitTypeId.VoltAmperes,
    SpecTypeId.CableTraySize:           UnitTypeId.Millimeters,
    SpecTypeId.ConduitSize:             UnitTypeId.Millimeters,
    SpecTypeId.Current:                 UnitTypeId.Amperes,
    SpecTypeId.ElectricalFrequency:     UnitTypeId.Hertz,
    SpecTypeId.ElectricalPotential:     UnitTypeId.Volts,
    SpecTypeId.ElectricalPower:         UnitTypeId.Watts,
    SpecTypeId.ElectricalPowerDensity:  UnitTypeId.WattsPerSquareMeter,
    SpecTypeId.PowerPerLength:          UnitTypeId.WattsPerMeter,
    SpecTypeId.ElectricalTemperature:   UnitTypeId.Celsius,
    SpecTypeId.WireDiameter:            UnitTypeId.Millimeters,
}

#  Metric→Imperial
met_to_imp = {
    # Common
    SpecTypeId.Angle:                   UnitTypeId.Degrees,
    SpecTypeId.Distance:                UnitTypeId.Feet,
    SpecTypeId.Length:                  UnitTypeId.Feet,
    SpecTypeId.Area:                    UnitTypeId.SquareFeet,
    SpecTypeId.Volume:                  UnitTypeId.CubicFeet,
    SpecTypeId.MassDensity:             UnitTypeId.PoundsMassPerCubicFoot,
    SpecTypeId.RotationAngle:           UnitTypeId.Degrees,
    SpecTypeId.Slope:                   SymbolTypeId.Percent,
    SpecTypeId.Speed:                   UnitTypeId.FeetPerSecond,
    SpecTypeId.Time:                    UnitTypeId.Seconds,
    # Electrical
    SpecTypeId.ApparentPower:           UnitTypeId.VoltAmperes,
    SpecTypeId.CableTraySize:           UnitTypeId.Inches,
    SpecTypeId.ConduitSize:             UnitTypeId.Inches,
    SpecTypeId.Current:                 UnitTypeId.Amperes,
    SpecTypeId.ElectricalFrequency:     UnitTypeId.Hertz,
    SpecTypeId.ElectricalPotential:     UnitTypeId.Volts,
    SpecTypeId.ElectricalPower:         UnitTypeId.Watts,
    SpecTypeId.ElectricalPowerDensity:  UnitTypeId.WattsPerSquareFoot,
    SpecTypeId.PowerPerLength:          UnitTypeId.WattsPerFoot,
    SpecTypeId.ElectricalTemperature:   UnitTypeId.Fahrenheit,
    SpecTypeId.WireDiameter:            UnitTypeId.FractionalInches,
}

# 3) Decide Imperial vs. Metric by looking at LENGTH
metric_uids = {UnitTypeId.Millimeters, UnitTypeId.Centimeters, UnitTypeId.Meters}
current_uid = units.GetFormatOptions(SpecTypeId.Length).GetUnitTypeId()
if current_uid in metric_uids:
    mapping, target = met_to_imp, 'Imperial'
else:
    mapping, target = imp_to_met,   'Metric'

# 4) Run it in one safe Transaction, skipping any truly invalid combos
t = Transaction(doc, 'Convert Common & Electrical → {}'.format(target))
skipped = []
try:
    t.Start()
    for spec_id, dst in mapping.iteritems():    # IronPython: .iteritems()
        if UnitUtils.IsValidUnit(spec_id, dst):
            units.SetFormatOptions(spec_id, FormatOptions(dst))
        elif spec_id in SPEC_NAMES:
            skipped.append(SPEC_NAMES[spec_id])
    doc.SetUnits(units)
    t.Commit()

    # Build feedback
    msg = 'Converted Common & Electrical units to {}'.format(target)
    if skipped:
        msg += '\n\nCould not convert:\n' + '\n'.join('- ' + n for n in skipped)
    TaskDialog.Show('Units Converter', msg)

except Exception as e:
    if t.GetStatus() == TransactionStatus.Started:
        t.RollBack()
    TaskDialog.Show(
        'Units Converter',
        'An unexpected error occurred during unit conversion: {}'.format(str(e))
    )
