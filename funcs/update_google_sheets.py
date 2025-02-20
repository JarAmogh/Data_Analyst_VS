import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
from funcs.config_table import load_data, state_order_counts, metro_vs_non_metro_counts, avg_order_cost_per_state, avg_order_cost_per_city_category

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

creds = ServiceAccountCredentials.from_json_keyfile_name("sheets.json", scope)
client = gspread.authorize(creds)

sheet = client.open_by_key("1aN4zGckDyiYrZReG80uy94eBSh5DQsavSzbCRdzR7iQ")

def update_sheet(worksheet, data, start_row, end_row, start_col, end_col):
    cell_range = f"{chr(65+start_col-1)}{start_row}:{chr(65+end_col-1)}{end_row}"
    worksheet.batch_clear([cell_range])
    time.sleep(2)
    worksheet.update(cell_range.split(":")[0], data)
    print(f" i have pdated rows {start_row}-{end_row} successfully.")

df = load_data()

worksheet = sheet.sheet1

state_orders = [["State", "Total Orders"]] + list(state_order_counts(df).items())
update_sheet(worksheet, state_orders, 1, 13, 1, 2)

metro_vs_non_metro = [["Category", "Orders"]] + list(metro_vs_non_metro_counts(df).items())
update_sheet(worksheet, metro_vs_non_metro, 15, 17, 1, 2)

avg_order_cost = [["State", "Average Order Cost"]] + avg_order_cost_per_state(df).reset_index().values.tolist()
update_sheet(worksheet, avg_order_cost, 19, 32, 1, 2)

avg_order_cost_city_category = [["City Category", "Average Order Cost"]] + avg_order_cost_per_city_category(df).reset_index().values.tolist()
update_sheet(worksheet, avg_order_cost_city_category, 34, 34 + len(avg_order_cost_city_category), 1, 2)