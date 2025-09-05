# -*- coding: utf-8 -*-
"""
pyRevit script: Set 'Equipment Color' parameter with a custom popup and Select Elements button.
- Prompts for the value and lets you pick elements in the order you want.
- Sets the parameter for each, in the order picked.
- Uses WPF for a modern UI.
"""

from pyrevit import revit, script
from Autodesk.Revit.DB import Transaction
from System.Collections.Generic import List

import clr
clr.AddReference('PresentationFramework')
clr.AddReference('PresentationCore')
clr.AddReference('WindowsBase')

from System.Windows import Window, Application
from System.Windows.Controls import StackPanel, TextBox, Button, Label
from System.Windows import Thickness, HorizontalAlignment, VerticalAlignment, SizeToContent

output = script.get_output()

class EquipmentColorWindow(Window):
    def __init__(self):
        self.Title = "Set Equipment Color"
        self.Width = 350
        self.Height = 120
        self.SizeToContent = SizeToContent.Height
        self.WindowStartupLocation = 1  # CenterScreen

        self.selected_elements = []
        self.param_value = None

        panel = StackPanel()
        panel.Margin = Thickness(15)
        self.Content = panel

        label = Label()
        label.Content = "Enter value for 'Equipment Color':"
        panel.Children.Add(label)

        self.textbox = TextBox()
        self.textbox.Margin = Thickness(0, 0, 0, 10)
        self.textbox.Height = 28
        panel.Children.Add(self.textbox)

        btn_select = Button()
        btn_select.Content = "Select Elements"
        btn_select.Height = 32
        btn_select.Click += self.select_elements
        panel.Children.Add(btn_select)

        self.status_label = Label()
        self.status_label.Content = ""
        self.status_label.Margin = Thickness(0, 8, 0, 0)
        panel.Children.Add(self.status_label)

    def select_elements(self, sender, args):
        self.param_value = self.textbox.Text
        if not self.param_value:
            self.status_label.Content = "Please enter a value before selecting elements."
            return

        # Hide window before picking to avoid UI issues
        self.Hide()
        try:
            picked_refs = revit.uidoc.Selection.PickObjects(
                Autodesk.Revit.UI.Selection.ObjectType.Element,
                "Select elements to set 'Equipment Color' (ESC when done)"
            )
            self.selected_elements = [revit.doc.GetElement(r.ElementId) for r in picked_refs]
            self.status_label.Content = "{} elements selected.".format(len(self.selected_elements))
        except Exception as ex:
            self.status_label.Content = "Selection cancelled or failed."
            self.selected_elements = []
        finally:
            self.ShowDialog()  # Show the dialog again

        # If succeeded, run parameter setting and close window
        if self.selected_elements:
            self.Close()

def set_equipment_color(element, value):
    # Try instance parameter first
    param = element.LookupParameter("Equipment Color")
    if param and not param.IsReadOnly:
        param.Set(str(value))
        return True
    # Try type parameter (family symbol)
    if hasattr(element, "Symbol"):
        symbol = element.Symbol
        param = symbol.LookupParameter("Equipment Color")
        if param and not param.IsReadOnly:
            param.Set(str(value))
            return True
    return False

def main():
    # Show the WPF UI for input and selection
    app = Application()
    win = EquipmentColorWindow()
    app.Run(win)

    value = win.param_value
    picked = win.selected_elements

    if not value or not picked:
        script.get_output().print_md("**No value entered or no elements picked.**")
        return

    # Set parameter in the same order as picked
    set_count = 0
    not_found = []
    with revit.Transaction("Set Equipment Color on Picked Elements"):
        for el in picked:
            if set_equipment_color(el, value):
                set_count += 1
                output.print_md("Set on ID {} ('{}')".format(el.Id.IntegerValue, el.Name if hasattr(el, "Name") else ""))
            else:
                not_found.append(el.Id.IntegerValue)
                output.print_md("Skipped ID {} (no writable 'Equipment Color')".format(el.Id.IntegerValue))

    msg = "Set 'Equipment Color' = '{}' on {} elements.".format(value, set_count)
    if not_found:
        msg += "\nSkipped {} (no writable parameter): {}".format(len(not_found), ", ".join(map(str, not_found)))
    from pyrevit import forms
    forms.alert(msg)

if __name__ == "__main__":
    main()