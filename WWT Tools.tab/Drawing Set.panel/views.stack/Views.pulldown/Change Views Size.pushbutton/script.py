# pyRevit script: adjust_crop_boxes_aspect.py

from pyrevit import revit, DB, forms
from Autodesk.Revit.DB import View, BoundingBoxXYZ, XYZ, Transaction

def adjust_crop_box(view, crop_width, crop_height):
    crop_box = view.CropBox
    center = crop_box.Min + ((crop_box.Max - crop_box.Min) * 0.5)

    half_width = crop_width / 2.0
    half_height = crop_height / 2.0

    new_min = XYZ(center.X - half_width, center.Y - half_height, crop_box.Min.Z)
    new_max = XYZ(center.X + half_width, center.Y + half_height, crop_box.Max.Z)

    new_crop = BoundingBoxXYZ()
    new_crop.Min = new_min
    new_crop.Max = new_max
    new_crop.Transform = crop_box.Transform

    view.CropBox = new_crop
    view.CropBoxActive = True
    view.CropBoxVisible = True

# Aspect ratio options
aspect_ratios = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "16:9": (16, 9)
}

# Select aspect ratio
selected_ratio_label = forms.SelectFromList.show(
    sorted(aspect_ratios.keys()),
    title="Select Aspect Ratio",
    button_name="Select"
)

if not selected_ratio_label:
    forms.alert("No aspect ratio selected.", exitscript=True)

# Ask for height
height_input = forms.ask_for_string(
    default='1.0', prompt='Enter crop box height (feet):'
)

try:
    crop_height = float(height_input)
except (ValueError, TypeError) as e:
    forms.alert('Invalid height. Please enter a numeric value: {}'.format(str(e)), exitscript=True)

# Calculate width from selected ratio
ratio_w, ratio_h = aspect_ratios[selected_ratio_label]
crop_width = (crop_height * ratio_w) / ratio_h

# Select views
selected_views = forms.select_views(title='Select Views to Adjust Crop Boxes')

if not selected_views:
    forms.alert('No views selected.', exitscript=True)

with revit.Transaction("Adjust Crop Boxes with Ratio"):
    for view in selected_views:
        if view.CropBoxActive and not view.IsTemplate:
            adjust_crop_box(view, crop_width, crop_height)