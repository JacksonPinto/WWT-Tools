# -*- coding: utf-8 -*-
import os
import csv
import xlrd
import codecs
from pyrevit import forms, script

# Prompt user to select an Excel file
excel_path = forms.pick_file(file_ext='xlsx')

# Exit if no file is selected
if not excel_path:
    forms.alert("No file selected! Exiting script.")
    script.exit()

# Prompt user to enter the sheet name
sheet_name = forms.ask_for_string(
    prompt="Enter the name of the sheet to extract data from:",
    title="Sheet Name Input"
)

# Exit if no sheet name is provided
if not sheet_name:
    forms.alert("No sheet name provided! Exiting script.")
    script.exit()

# Ask user to select a folder to save CSV
csv_folder = forms.pick_folder(title="Select folder to save CSV file")
if not csv_folder:
    forms.alert("No folder selected! Exiting script.")
    script.exit()

# Set fixed CSV filename
csv_filename = "RackBuilder_ParameterValues.csv"
csv_path = os.path.join(csv_folder, csv_filename)

try:
    # Open the Excel workbook
    workbook = xlrd.open_workbook(excel_path)

    # Check if the sheet exists
    if sheet_name not in workbook.sheet_names():
        forms.alert("Sheet '{}' not found in the workbook.".format(sheet_name))
        script.exit()

    # Access the specified sheet
    worksheet = workbook.sheet_by_name(sheet_name)

    # Read all rows and columns
    data = [worksheet.row_values(row_idx) for row_idx in range(worksheet.nrows)]

    # Identify the range of data to extract
    ru_start_index = None
    ru_end_index = None
    header_index = None
    rear_elevation_index = None

    # Find RU01 (start), the last RU{number} (end), and 'Rear Elevation' row
    for row_idx in range(worksheet.nrows):
        first_col_value = str(worksheet.cell_value(row_idx, 0)).strip()

        if first_col_value.startswith("RU") and first_col_value[2:].isdigit():
            if ru_start_index is None:
                ru_start_index = row_idx
            ru_end_index = row_idx

        if first_col_value == "Rear Elevation":
            rear_elevation_index = row_idx
            break  # Stop scanning after finding 'Rear Elevation'

    # Adjust the end index if 'Rear Elevation' exists
    if rear_elevation_index is not None:
        ru_end_index = min(ru_end_index, rear_elevation_index - 1)

    # Find the header row (the row immediately above RU start index)
    if ru_start_index is not None and ru_start_index > 0:
        header_index = ru_start_index - 1
    else:
        forms.alert("Could not find 'RU01' in the first column.")
        script.exit()

    # Extract column indexes for required fields
    required_headers = ["RU Height", "Device Depth", "Hostname", "Device Part #"]
    header_row = data[header_index]

    column_indices = {col_name: header_row.index(col_name) for col_name in required_headers if col_name in header_row}

    if len(column_indices) != len(required_headers):
        missing = [col for col in required_headers if col not in column_indices]
        forms.alert("Missing required columns: {}".format(missing))
        script.exit()

    # Extract the filtered data
    filtered_data = [required_headers]
    for row_idx in range(ru_start_index, ru_end_index + 1):
        row = data[row_idx]
        ru_height = row[column_indices["RU Height"]]
        device_depth = row[column_indices["Device Depth"]]
        hostname = row[column_indices["Hostname"]]
        device_part = row[column_indices["Device Part #"]]

        # Convert Device Depth to a proper float format
        if isinstance(device_depth, (float, int)):
            device_depth = "{:.2f}".format(device_depth)  # Ensure two decimal places
        else:
            device_depth = ""

        # Handle RU Height adjustments
        try:
            ru_height = float(ru_height)
            if ru_height > 1:
                for i in range(int(ru_height)):
                    filtered_data.append(["0", device_depth, hostname, device_part])
                filtered_data[-1][0] = str(int(ru_height))  # Assign height to the last row
            else:
                filtered_data.append([str(int(ru_height)), device_depth, hostname, device_part])
        except ValueError:
            filtered_data.append(["", device_depth, hostname, device_part])

    # Remove empty lines before saving
    filtered_data = [row for row in filtered_data if any(cell and str(cell).strip() for cell in row)]

    # Write data to CSV using codecs (IronPython-compatible)
    with codecs.open(csv_path, mode='w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(filtered_data)

    forms.alert("Filtered CSV file successfully created:\n{}".format(csv_path))

except Exception as e:
    forms.alert("Failed to process Excel file:\n{}".format(e))
    script.exit()
