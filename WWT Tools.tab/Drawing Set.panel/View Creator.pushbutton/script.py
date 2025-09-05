# -*- coding: utf-8 -*-
#pylint: disable=unused-argument,too-many-lines
#pylint: disable=missing-function-docstring,missing-class-docstring

import re
from pyrevit import coreutils
from pyrevit import revit, DB
from pyrevit import forms
from pyrevit import script

logger = script.get_logger()


class BatchViewDuplicatorWindow(forms.WPFWindow):
    def __init__(self, xaml_file_name):
        forms.WPFWindow.__init__(self, xaml_file_name)
        self.view_list = []  # 🔹 Use list instead of dict
        self.views_tb.Focus()

    def _process_view_entries(self):
        """Reads user input and parses tab-separated view names."""
        for view_entry in str(self.views_tb.Text).split("\n"):
            if coreutils.is_blank(view_entry):
                continue

            if "\t" not in view_entry:
                logger.warning("View name must be separated by a tab: {}".format(view_entry))
                return False

            view_entry = re.sub("\t+", "\t", view_entry).strip()
            orig_view, new_view = view_entry.split("\t")
            self.view_list.append((orig_view, new_view))  # 🔹 Store as tuple

        return True

    def _get_existing_views(self):
        """Collects all non-template views that can be duplicated."""
        return {
            v.Name: v for v in DB.FilteredElementCollector(revit.doc)
            .OfClass(DB.View)
            .WhereElementIsNotElementType()
            .ToElements()
            if not v.IsTemplate and v.CanViewBeDuplicated(DB.ViewDuplicateOption.AsDependent)
        }

    def _duplicate_view(self, original_view, new_view_name):
        """Duplicates a view as dependent and renames it."""
        with DB.Transaction(revit.doc, "Duplicate View") as t:
            t.Start()
            try:
                new_view_id = original_view.Duplicate(DB.ViewDuplicateOption.AsDependent)
                new_view = revit.doc.GetElement(new_view_id)
                new_view.Name = new_view_name
                t.Commit()
                logger.info("✅ Created view: {} → {}".format(original_view.Name, new_view_name))
            except Exception as err:
                t.RollBack()
                logger.error("❌ Error duplicating view {}: {}".format(original_view.Name, err))

    def duplicate_views(self, sender, args):
        """Main function: processes input, finds views, and duplicates them."""
        self.Close()

        if not self._process_view_entries():
            logger.error("🚨 Aborted due to input errors.")
            return

        existing_views = self._get_existing_views()

        with DB.TransactionGroup(revit.doc, "Batch Duplicate Views") as tg:
            tg.Start()
            for orig_view_name, new_view_name in self.view_list:  # 🔹 Now iterating over a list
                if orig_view_name in existing_views:
                    self._duplicate_view(existing_views[orig_view_name], new_view_name)
                else:
                    logger.warning("⚠️ Skipping: View '{}' not found.".format(orig_view_name))
            tg.Assimilate()


BatchViewDuplicatorWindow("ViewCreator.xaml").ShowDialog()
