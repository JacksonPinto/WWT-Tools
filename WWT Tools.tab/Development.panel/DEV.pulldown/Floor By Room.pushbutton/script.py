# pyRevit script: Create Floor by Room with Floor Type mapping and Select All/None buttons (IronPython)
from pyrevit import script
from Autodesk.Revit.DB import (
    FilteredElementCollector, BuiltInCategory, SpatialElementBoundaryOptions,
    FloorType, Transaction, CurveLoop, Floor, BuiltInParameter, SpatialElementBoundaryLocation
)
from Autodesk.Revit.UI import TaskDialog

import clr
import os

clr.AddReference('PresentationFramework')
from System.Windows.Markup import XamlReader
from System.Windows import Window
from System.Collections.ObjectModel import ObservableCollection

doc = __revit__.ActiveUIDocument.Document
output = script.get_output()

def get_room_name(room):
    try:
        name_param = room.LookupParameter("Name")
        if name_param:
            return name_param.AsString()
    except Exception:
        pass
    return "?"

def get_rooms_in_view(view):
    return list(
        FilteredElementCollector(doc, view.Id)
        .OfCategory(BuiltInCategory.OST_Rooms)
        .WhereElementIsNotElementType()
        .ToElements()
    )

def get_floor_types():
    floor_types = list(FilteredElementCollector(doc).OfClass(FloorType))
    floor_type_names = []
    for ft in floor_types:
        param = ft.get_Parameter(BuiltInParameter.SYMBOL_NAME_PARAM)
        if param:
            floor_type_names.append(param.AsString())
        else:
            floor_type_names.append("Unnamed Type")
    return zip(floor_type_names, floor_types)

def get_room_boundary(room):
    options = SpatialElementBoundaryOptions()
    options.SpatialElementBoundaryLocation = SpatialElementBoundaryLocation.Finish
    boundaries = room.GetBoundarySegments(options)
    if not boundaries:
        return None
    outer_loop = []
    for segment in boundaries[0]:
        outer_loop.append(segment.GetCurve())
    return outer_loop

def get_room_level(room):
    return doc.GetElement(room.LevelId)

def create_floor(boundary_curves, floor_type, level):
    curve_loop = CurveLoop.Create(boundary_curves)
    curve_loops = [curve_loop]
    floor = Floor.Create(doc, curve_loops, floor_type.Id, level.Id)
    return floor

class RoomRow(object):
    def __init__(self, room, floor_types):
        self.IsSelected = False
        self.RoomNumber = getattr(room, "Number", "?")
        self.RoomName = get_room_name(room)
        self.RoomObj = room
        self.SelectedFloorType = floor_types[0] if floor_types else ""

def main():
    view = doc.ActiveView
    rooms = get_rooms_in_view(view)
    if not rooms:
        TaskDialog.Show("Create Floor by Room", "No rooms found in the active view.")
        return

    # Floor types
    floor_type_list = list(get_floor_types())
    floor_type_names = [ft[0] for ft in floor_type_list]
    floor_types_by_name = dict(floor_type_list)
    if not floor_type_names:
        TaskDialog.Show("Create Floor by Room", "No floor types found in the project.")
        return

    # Prepare data for DataGrid
    room_rows = [RoomRow(room, floor_type_names) for room in rooms]
    data = ObservableCollection[object]()
    for row in room_rows:
        data.Add(row)

    # Load XAML
    xaml_path = os.path.join(os.path.dirname(__file__), "FloorByRoom.xaml")
    with open(xaml_path, 'r') as xaml_file:
        xaml_string = xaml_file.read()
    window = XamlReader.Parse(xaml_string)

    # Bind data to DataGrid
    roomgrid = window.FindName("RoomGrid")
    roomgrid.ItemsSource = data

    # Fix for IronPython: set ItemsSource for ComboBox column to a .NET List[str]
    import System
    from System.Collections.Generic import List as NetList

    for col in roomgrid.Columns:
        if col.Header == "Floor Type":
            net_floor_types = NetList[str]()
            for name in floor_type_names:
                net_floor_types.Add(name)
            col.ItemsSource = net_floor_types

    ok_clicked = {'result': False}
    def Ok_Click(sender, args):
        ok_clicked['result'] = True
        window.Close()
    def Cancel_Click(sender, args):
        window.Close()
    window.FindName("okButton").Click += Ok_Click
    window.FindName("cancelButton").Click += Cancel_Click

    # Select All / Select None Buttons
    def SelectAll_Click(sender, args):
        for row in data:
            row.IsSelected = True
        roomgrid.Items.Refresh()

    def SelectNone_Click(sender, args):
        for row in data:
            row.IsSelected = False
        roomgrid.Items.Refresh()

    window.FindName("selectAllButton").Click += SelectAll_Click
    window.FindName("selectNoneButton").Click += SelectNone_Click

    # Show window
    window.ShowDialog()

    if not ok_clicked['result']:
        TaskDialog.Show("Create Floor by Room", "Cancelled.")
        return

    # Collect selections
    selected_mappings = []
    for row in data:
        if row.IsSelected:
            room = row.RoomObj
            chosen_ftype_name = row.SelectedFloorType
            selected_mappings.append((room, chosen_ftype_name))

    if not selected_mappings:
        TaskDialog.Show("Create Floor by Room", "No rooms selected.")
        return

    created_count = 0
    t = Transaction(doc, "Create Floors by Room (Mapping)")
    t.Start()
    for room, floor_type_name in selected_mappings:
        floor_type = floor_types_by_name.get(floor_type_name)
        if not floor_type:
            output.print_md("Floor type '{0}' not found for room **{1} - {2}**. Skipped.".format(
                floor_type_name, getattr(room, "Number", "?"), get_room_name(room)))
            continue
        boundary = get_room_boundary(room)
        if not boundary:
            output.print_md("Room **{0} - {1}** has no valid boundary. Skipped.".format(
                getattr(room, "Number", "?"), get_room_name(room)))
            continue
        level = get_room_level(room)
        try:
            create_floor(boundary, floor_type, level)
            created_count += 1
        except Exception as e:
            output.print_md("Error creating floor for room **{0} - {1}**: {2}".format(
                getattr(room, "Number", "?"), get_room_name(room), str(e)))
    t.Commit()
    TaskDialog.Show("Create Floor by Room", "Created floors for {0} room(s).".format(created_count))

if __name__ == "__main__":
    main()