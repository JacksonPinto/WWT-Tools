# -*- coding: utf-8 -*-
"""
PyRevit script (Electrical Fixtures & Data Devices only):
1) Let user pick a linked model if multiple are loaded.
2) Scan chosen linked model for FamilyInstances in ElectricalFixtures and DataDevices.
3) Scan host document for loaded FamilySymbol types in those categories.
4) Present a DataGridView mapping linked types to host types.
5) Activate host symbols and place mapped instances at original insertion points.
"""
from pyrevit import revit, forms, script
import clr
from Autodesk.Revit.DB import (
    FilteredElementCollector,
    RevitLinkInstance,
    FamilyInstance,
    FamilySymbol,
    Level,
    BuiltInCategory,
    BuiltInParameter
)
from Autodesk.Revit.DB.Structure import StructuralType
from System.Windows.Forms import (
    Form,
    DataGridView,
    DataGridViewTextBoxColumn,
    DataGridViewComboBoxColumn,
    Button,
    DockStyle,
    FormStartPosition,
    DialogResult,
    DataGridViewAutoSizeColumnsMode,
    DataGridViewColumnHeadersHeightSizeMode
)
import System.Drawing as sd

doc = revit.doc
out = script.get_output()

# Categories to scan
elec_cat = BuiltInCategory.OST_ElectricalFixtures
data_cat = BuiltInCategory.OST_DataDevices
cat_ids = set([int(elec_cat), int(data_cat)])

# 1) Select which linked model to scan
link_instances = list(FilteredElementCollector(doc).OfClass(RevitLinkInstance))
if not link_instances:
    forms.alert('No linked Revit models found.', exitscript=True)
if len(link_instances) > 1:
    choices = ['{0}. {1}'.format(i+1, li.Name) for i, li in enumerate(link_instances)]
    sel = forms.SelectFromList.show(choices, title='Select Linked Model', multiselect=False)
    if not sel:
        script.exit()
    sel_text = sel[0]
    try:
        idx = int(sel_text.split('.', 1)[0]) - 1
        chosen_link = link_instances[idx]
    except (ValueError, IndexError) as e:
        forms.alert('Invalid selection: {}. Exiting.'.format(str(e)), exitscript=True)
    links_to_scan = [chosen_link]
else:
    links_to_scan = link_instances

# 2) Gather linked instances
linked_data = {}
for link in links_to_scan:
    ldoc = link.GetLinkDocument()
    xfm = link.GetTransform()
    for inst in FilteredElementCollector(ldoc).OfClass(FamilyInstance):
        cat = inst.Category
        if not cat or cat.Id.IntegerValue not in cat_ids:
            continue
        param = inst.Symbol.get_Parameter(BuiltInParameter.SYMBOL_NAME_PARAM)
        tname = param.AsString() if param else inst.Symbol.Name
        key = '{0}: {1}'.format(inst.Symbol.Family.Name, tname)
        loc = inst.Location
        if not hasattr(loc, 'Point'):
            continue
        pt = xfm.OfPoint(loc.Point)
        lvl_elem = ldoc.GetElement(inst.LevelId)
        lvl_name = lvl_elem.Name if lvl_elem else ''
        linked_data.setdefault(key, []).append((pt, lvl_name))
if not linked_data:
    forms.alert('No ElectricalFixtures/DataDevices found in the selected link.', exitscript=True)
all_keys = sorted(linked_data.keys())

# 3) Gather host family symbols and levels
host_levels = {lvl.Name: lvl for lvl in FilteredElementCollector(doc).OfClass(Level)}
host_types = {}
for sym in FilteredElementCollector(doc).OfClass(FamilySymbol):
    cat = sym.Category
    if not cat or cat.Id.IntegerValue not in cat_ids:
        continue
    param = sym.get_Parameter(BuiltInParameter.SYMBOL_NAME_PARAM)
    tname = param.AsString() if param else sym.Name
    key = '{0}: {1}'.format(sym.Family.Name, tname)
    host_types[key] = sym
if not host_types:
    forms.alert('No host FamilySymbols loaded.', exitscript=True)

# 4) Build mapping UI
form = Form()
form.Text = 'Map Linked Types to Host Types'
form.Width = 1200
form.Height = 800
form.StartPosition = FormStartPosition.CenterScreen

dgv = DataGridView()
dgv.Dock = DockStyle.Fill
dgv.AutoGenerateColumns = False
dgv.RowHeadersVisible = False
# Bold headers
dgv.ColumnHeadersDefaultCellStyle.Font = sd.Font(dgv.Font.FontFamily, dgv.Font.Size, sd.FontStyle.Bold)
dgv.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.DisableResizing
dgv.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill

col_link = DataGridViewTextBoxColumn()
col_link.HeaderText = 'Linked Type'
col_link.ReadOnly = True
col_link.FillWeight = 50

col_host = DataGridViewComboBoxColumn()
col_host.HeaderText = 'Host Type'
col_host.FillWeight = 50
for key in sorted(host_types.keys()):
    col_host.Items.Add(key)

dgv.Columns.Add(col_link)
dgv.Columns.Add(col_host)
for key in all_keys:
    dgv.Rows.Add(key, None)

# Adjust row/header heights
btn_ok = Button(Text='OK')
btn_h = btn_ok.Height * 3
row_count = dgv.Rows.Count
if row_count:
    avail_h = form.ClientSize.Height - btn_h - 20
    row_h = avail_h // row_count
    for row in dgv.Rows:
        row.Height = row_h
    dgv.ColumnHeadersHeight = row_h * 3

form.Controls.Add(dgv)
btn_ok.Height = btn_h
btn_ok.Dock = DockStyle.Bottom
form.Controls.Add(btn_ok)

def on_ok(s, e):
    form.DialogResult = DialogResult.OK
    form.Close()
btn_ok.Click += on_ok
if form.ShowDialog() != DialogResult.OK:
    script.exit()

# 5) Read mappings
mappings = {}
for row in dgv.Rows:
    if not row.IsNewRow:
        lk = row.Cells[0].Value
        hv = row.Cells[1].Value
        if hv:
            mappings[lk] = hv
if not mappings:
    script.exit()

# 6) Activate host symbols
with revit.Transaction('Activate Host Symbols'):
    for key in set(mappings.values()):
        sym = host_types[key]
        if not sym.IsActive:
            sym.Activate()
    doc.Regenerate()

# 7) Place mapped instances
with revit.Transaction('Place Mapped Instances'):
    out.print_md('**Placing Mapped Instances...**')
    for lk, pts in linked_data.items():
        if lk not in mappings:
            continue
        sym = host_types[mappings[lk]]
        out.print_md('- {0} → {1} ({2} pts)'.format(lk, mappings[lk], len(pts)))
        for pt, lvl_name in pts:
            lvl = host_levels.get(lvl_name, doc.ActiveView.GenLevel)
            doc.Create.NewFamilyInstance(pt, sym, lvl, StructuralType.NonStructural)

forms.alert('Families successfully inserted!', title='Done')
