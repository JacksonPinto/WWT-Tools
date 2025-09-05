# -*- coding: utf-8 -*-
from pyrevit import script, revit
from Autodesk.Revit.DB import (
    FilteredElementCollector, View3D, View, Transaction, XYZ, BuiltInParameter, ElementId
)
from Autodesk.Revit.UI import TaskDialog
import clr
import os

clr.AddReference('PresentationFramework')
from System.Windows.Markup import XamlReader
from System.Collections.ObjectModel import ObservableCollection

doc = revit.doc
uidoc = revit.uidoc

def mm_to_ft(mm):
    return mm / 304.8

def get_perspective_views(doc):
    return [
        v for v in FilteredElementCollector(doc)
        .OfClass(View3D)
        .WhereElementIsNotElementType()
        if not v.IsTemplate and v.CanBePrinted and v.IsPerspective
    ]

def get_view_templates(doc):
    # Only 3D view templates, sorted by name
    return sorted(
        [v for v in FilteredElementCollector(doc)
            .OfClass(View3D)
            .WhereElementIsNotElementType()
            if v.IsTemplate],
        key=lambda vt: vt.Name
    )

class CameraRow(object):
    def __init__(self, view, view_templates):
        self.ViewObj = view
        self.ViewName = view.Name
        self.IsSelected = False
        self.ViewTemplates = view_templates
        self.SelectedViewTemplate = None
        template_id = view.ViewTemplateId
        for t in view_templates:
            if t.Id == template_id:
                self.SelectedViewTemplate = t
                break

def set_crop_size(view, height_mm, aspect_ratio):
    height_ft = mm_to_ft(height_mm)
    crop = view.CropBox
    center = XYZ(
        (crop.Min.X + crop.Max.X) / 2.0,
        (crop.Min.Y + crop.Max.Y) / 2.0,
        (crop.Min.Z + crop.Max.Z) / 2.0
    )
    width_ft = height_ft * aspect_ratio
    dx = width_ft / 2.0
    dy = height_ft / 2.0
    crop.Min = XYZ(center.X - dx, center.Y - dy, crop.Min.Z)
    crop.Max = XYZ(center.X + dx, center.Y + dy, crop.Max.Z)
    view.CropBox = crop

def set_far_clip_enabled(view, enabled):
    param = view.LookupParameter("Far Clip Active")
    if not param:
        try:
            param = view.get_Parameter(BuiltInParameter.VIEWER_BOUND_FAR_CLIPPING)
        except:
            param = None
    if param and not param.IsReadOnly:
        try:
            param.Set(1 if enabled else 0)
        except Exception as e:
            print("Failed to set far clip for {}: {}".format(view.Name, e))

def set_target_elevation(view, elev_mm):
    elev_ft = mm_to_ft(elev_mm)
    param = view.LookupParameter("Camera Target Z")
    if param and not param.IsReadOnly:
        try:
            param.Set(elev_ft)
        except Exception as e:
            print("Failed to set target elevation for {}: {}".format(view.Name, e))

def get_template_by_name(template_name):
    for v in FilteredElementCollector(doc).OfClass(View):
        if v.IsTemplate and v.Name == template_name:
            return v
    return None

def set_view_template(view, template):
    # Only set if template is not None and not already assigned
    if template and view.ViewTemplateId != template.Id:
        try:
            view.ViewTemplateId = template.Id
        except Exception as e:
            print("Failed to set view template for {}: {}".format(view.Name, e))

def main():
    # Load XAML UI
    xaml_path = os.path.join(os.path.dirname(__file__), "CameraAdjust.xaml")
    with open(xaml_path, 'r') as xaml_file:
        xaml_string = xaml_file.read()
    window = XamlReader.Parse(xaml_string)

    # Gather all perspective cameras and model 3D view templates
    all_cameras = get_perspective_views(doc)
    view_templates = get_view_templates(doc)
    camera_rows = [CameraRow(v, view_templates) for v in all_cameras]
    data = ObservableCollection[object]()
    for row in camera_rows:
        data.Add(row)

    grid = window.FindName("CameraGrid")
    grid.ItemsSource = data

    filter_box = window.FindName("filterBox")
    def refresh_grid():
        filter_text = filter_box.Text.strip().lower()
        grid.ItemsSource = ObservableCollection[object]([
            row for row in camera_rows
            if filter_text in row.ViewName.lower()
        ])
        grid.Items.Refresh()
    def filter_changed(sender, args):
        refresh_grid()
    filter_box.TextChanged += filter_changed

    def select_all(sender, args):
        filter_text = filter_box.Text.strip().lower()
        for row in camera_rows:
            if filter_text in row.ViewName.lower():
                row.IsSelected = True
        refresh_grid()
    def select_none(sender, args):
        filter_text = filter_box.Text.strip().lower()
        for row in camera_rows:
            if filter_text in row.ViewName.lower():
                row.IsSelected = False
        refresh_grid()
    window.FindName("selectAllButton").Click += select_all
    window.FindName("selectNoneButton").Click += select_none

    ok_clicked = {'result': False}
    def ok_click(sender, args):
        ok_clicked['result'] = True
        window.Close()
    def cancel_click(sender, args):
        window.Close()
    window.FindName("okButton").Click += ok_click
    window.FindName("cancelButton").Click += cancel_click

    refresh_grid()
    window.ShowDialog()

    if not ok_clicked['result']:
        TaskDialog.Show("Camera Adjust", "Operation cancelled.")
        return

    aspect_idx = window.FindName("aspectCombo").SelectedIndex
    aspect_ratio = 4.0/3.0 if aspect_idx == 0 else 16.0/9.0

    try:
        height_mm = float(window.FindName("heightBox").Text)
        if height_mm <= 0:
            raise ValueError
    except Exception:
        TaskDialog.Show("Camera Adjust", "Invalid height value.")
        return

    far_clip_enabled = window.FindName("farClipBox").IsChecked

    try:
        elev_mm = float(window.FindName("targetElevationBox").Text)
    except Exception:
        elev_mm = 1900.0

    selected = [row for row in camera_rows if row.IsSelected]
    if not selected:
        TaskDialog.Show("Camera Adjust", "No cameras selected.")
        return

    # Remove template, set properties, then set template, all in a single transaction
    with Transaction(doc, "Camera Adjust") as t:
        t.Start()
        for row in selected:
            view = row.ViewObj

            # Remove view template if any (unlock all properties)
            if view.ViewTemplateId != ElementId.InvalidElementId:
                view.ViewTemplateId = ElementId.InvalidElementId

            # Always switch to ortho to allow changes
            if view.IsPerspective:
                view.ToggleToIsometric()
            set_crop_size(view, height_mm, aspect_ratio)
            set_far_clip_enabled(view, far_clip_enabled)
            set_target_elevation(view, elev_mm)
            if not view.IsPerspective:
                view.ToggleToPerspective()

            # Set the template if one is selected
            selected_template = row.SelectedViewTemplate
            if selected_template and view.ViewTemplateId != selected_template.Id:
                set_view_template(view, selected_template)

        t.Commit()

    TaskDialog.Show(
        "Camera Adjust",
        "Set field of view for {} cameras to {:.0f}mm x {:.0f}mm (aspect {:.2f}), Far Clip {}, Target Elevation {:.0f}mm.".format(
            len(selected), height_mm * aspect_ratio, height_mm, aspect_ratio,
            "ENABLED" if far_clip_enabled else "DISABLED", elev_mm)
    )

if __name__ == "__main__":
    main()