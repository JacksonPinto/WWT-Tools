# -*- coding: utf-8 -*-
__title__ = "Read & Recreate Selected Solids"
__author__ = "JacksonPinto & Copilot"
__doc__ = "Reads selected solid(s) in a family and recreates them as DirectShape."

from pyrevit import revit, DB
from Autodesk.Revit.DB import Transaction, DirectShape, ElementId, BuiltInCategory, Solid

uidoc = revit.uidoc
doc = revit.doc

def get_selected_solids():
    """Get all Solid objects from the current selection."""
    selection_ids = uidoc.Selection.GetElementIds()
    solids = []
    for eid in selection_ids:
        el = doc.GetElement(eid)
        geo_elem = el.get_Geometry(DB.Options())
        if geo_elem:
            for geo_obj in geo_elem:
                # Some geometry objects are GeometryInstance
                if isinstance(geo_obj, DB.GeometryInstance):
                    inst_geo = geo_obj.GetInstanceGeometry()
                    for inst_obj in inst_geo:
                        if isinstance(inst_obj, Solid) and inst_obj.Volume > 0:
                            solids.append(inst_obj)
                elif isinstance(geo_obj, Solid) and geo_obj.Volume > 0:
                    solids.append(geo_obj)
    return solids

def create_directshapes_from_solids(solids):
    """Create DirectShape objects from given solids."""
    category_id = ElementId(BuiltInCategory.OST_GenericModel)
    for solid in solids:
        ds = DirectShape.CreateElement(doc, category_id)
        ds.SetShape([solid])

def main():
    solids = get_selected_solids()
    if not solids:
        print("No valid Solid geometry found in selection.")
        return
    t = Transaction(doc, "Recreate Selected Solids as DirectShape")
    t.Start()
    create_directshapes_from_solids(solids)
    t.Commit()
    print("DirectShape objects created from selected solids.")

if __name__ == "__main__":
    main()