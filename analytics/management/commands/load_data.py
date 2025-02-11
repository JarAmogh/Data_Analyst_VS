from django.core.management.base import BaseCommand

import pandas as pd
from analytics.models import (
    AmazonSaleReport,
    ExpenseReport,
    SalesReport,
    CloudWarehouseComparison,
    InternationalSaleReport,
    May2022Report,
    PLMarch2021Report
)
import random

# class Command(BaseCommand):
#     help = "Load all cleaned data from CSV files into MongoDB"

#     def handle(self, *args, **kwargs):
#         # ✅ Insert Data from amazon_sale_report.csv
#         self.stdout.write("Loading amazon_sale_report.csv...")
#         amazon_sales_df = pd.read_csv('amazon_sale_report.csv', low_memory=False, dtype=str).fillna("Unknown")
        
#         for _, row in amazon_sales_df.iterrows():
#             order_id = str(row['Order ID'])
#             # Check if the record exists
#             existing_record = AmazonSaleReport.objects.filter(order_id=order_id).first()

#             if existing_record:
#                 # If record exists, update it
#                 existing_record.date = row['Date']
#                 existing_record.status = row['Status']
#                 existing_record.amount = float(row['Amount']) if row['Amount'].replace('.', '', 1).isdigit() else 0.0
#                 existing_record.ship_city = str(row['ship-city'])
#                 existing_record.ship_state = str(row['ship-state'])
#                 existing_record.save()  # Save the updated record
#             else:
#                 # If record doesn't exist, create a new one
#                 AmazonSaleReport(
#                     order_id=order_id,
#                     date=row['Date'],
#                     status=row['Status'],
#                     amount=float(row['Amount']) if row['Amount'].replace('.', '', 1).isdigit() else 0.0,
#                     ship_city=str(row['ship-city']),
#                     ship_state=str(row['ship-state']),
#                 ).save()  # Save the new record

#         self.stdout.write(f"Processed {len(amazon_sales_df)} records for AmazonSaleReport.")
        
        # After this part is finished, COMMENT OUT the section above
        # --------------------------------------------
 
# class Command(BaseCommand):
#     help = "Load cloud_warehouse_comparison_chart.csv into MongoDB"

#     def handle(self, *args, **kwargs):
#         # ✅ Insert Data from cloud_warehouse_comparison_chart.csv
#         self.stdout.write("Loading cloud_warehouse_comparison_chart.csv...")
#         cloud_warehouse_df = pd.read_csv('cloud_warehouse_comparison_chart.csv', low_memory=False, dtype=str).fillna("Unknown")
        
#         # Print columns to check for the correct column names
#         print("Columns in cloud_warehouse_comparison_chart.csv:", cloud_warehouse_df.columns)

#         # Rename 'Unnamed: 1' to 'warehouse' as it's the most likely candidate
#         cloud_warehouse_df.rename(columns={'Unnamed: 1': 'warehouse'}, inplace=True)

#         cloud_warehouse_records = []
#         for _, row in cloud_warehouse_df.iterrows():
#             cloud_warehouse_records.append(CloudWarehouseComparison(
#                 warehouse=str(row['warehouse']),
#                 shiprocket=float(row['Shiprocket']) if row['Shiprocket'].replace('.', '', 1).isdigit() else 0.0,
#                 increff=float(row['INCREFF']) if row['INCREFF'].replace('.', '', 1).isdigit() else 0.0,
#             ))
#         CloudWarehouseComparison.objects.insert(cloud_warehouse_records)
#         self.stdout.write(f"Processed {len(cloud_warehouse_records)} records for CloudWarehouseComparison.")


# class Command(BaseCommand):
#     help = "Load expense_iigf.csv into MongoDB"

#     def handle(self, *args, **kwargs):
#         # ✅ Insert Data from expense_iigf.csv
#         self.stdout.write("Loading expense_iigf.csv...")
#         expense_df = pd.read_csv('expense_iigf.csv', low_memory=False, dtype=str).fillna("Unknown")
        
#         # Print columns to check for the correct column names
#         print("Columns in expense_iigf.csv:", expense_df.columns)

#         # Check if 'Expense Value' exists
#         if 'Expense Value' not in expense_df.columns:
#             self.stdout.write("Column 'Expense Value' not found. Skipping this column.")
#             expense_df.rename(columns={
#                 "Recived Amount": "Received Amount",
#                 "Expance": "Expense"
#             }, inplace=True)
#             expense_df = expense_df.loc[:, ~expense_df.columns.str.contains('Unnamed', na=False)]
#             expense_records = []
#             for _, row in expense_df.iterrows():
#                 expense_records.append(ExpenseReport(
#                     received_amount=str(row['Received Amount']),
#                     received_amount_value=0.0,  # No value, as the column was missing
#                     expense=str(row['Expense']),
#                     expense_value=0.0,  # Default value since 'Expense Value' is missing
#                 ))
#             ExpenseReport.objects.insert(expense_records)
#             self.stdout.write(f"Processed {len(expense_records)} records for ExpenseReport.")
#         else:
#             expense_df.rename(columns={
#                 "Recived Amount": "Received Amount",
#                 "Expance": "Expense"
#             }, inplace=True)
#             expense_df = expense_df.loc[:, ~expense_df.columns.str.contains('Unnamed', na=False)]
#             expense_records = []
#             for _, row in expense_df.iterrows():
#                 expense_records.append(ExpenseReport(
#                     received_amount=str(row['Received Amount']),
#                     received_amount_value=float(row['Received Amount Value']) if row['Received Amount Value'].replace('.', '', 1).isdigit() else 0.0,
#                     expense=str(row['Expense']),
#                     expense_value=float(row['Expense Value']) if row['Expense Value'].replace('.', '', 1).isdigit() else 0.0,
#                 ))
#             ExpenseReport.objects.insert(expense_records)
#             self.stdout.write(f"Processed {len(expense_records)} records for ExpenseReport.")


# class Command(BaseCommand):
#     help = "Load international_sale_report.csv into MongoDB"

#     def handle(self, *args, **kwargs):
#         # ✅ Insert Data from international_sale_report.csv
#         self.stdout.write("Loading international_sale_report.csv...")
#         international_sale_df = pd.read_csv('international_sale_report.csv', low_memory=False, dtype=str).fillna("Unknown")
        
#         # Print columns to check for the correct column names
#         print("Columns in international_sale_report.csv:", international_sale_df.columns)

#         # Strip any leading/trailing whitespace from column names
#         international_sale_df.columns = international_sale_df.columns.str.strip()

#         # Print columns again to ensure they are stripped
#         print("Cleaned Columns:", international_sale_df.columns)

#         # Ensure required columns exist in the data
#         required_columns = ['DATE', 'Months', 'CUSTOMER', 'Style', 'SKU', 'Size', 'PCS', 'RATE', 'GROSS AMT']
#         missing_columns = [col for col in required_columns if col not in international_sale_df.columns]
        
#         if missing_columns:
#             self.stdout.write(f"Missing columns: {missing_columns}. Adding random values for missing columns.")
        
#         # Add random values for 'Qty' if it's missing
#         if 'Qty' not in international_sale_df.columns:
#             international_sale_df['Qty'] = [random.randint(1, 10) for _ in range(len(international_sale_df))]
        
#         # Insert records into MongoDB
#         international_sale_records = []
#         for _, row in international_sale_df.iterrows():
#             international_sale_records.append(InternationalSaleReport(
#                 date=row['DATE'],
#                 months=row['Months'],
#                 customer=row['CUSTOMER'],
#                 style=row['Style'],
#                 sku=row['SKU'],
#                 size=row['Size'],
#                 pcs=int(row['Qty']),  # Handle Qty as integer
#                 rate=float(row['RATE']) if row['RATE'].replace('.', '', 1).isdigit() else 0.0,
#                 gross_amt=float(row['GROSS AMT']) if row['GROSS AMT'].replace('.', '', 1).isdigit() else 0.0,
#             ))

#         # Insert into MongoDB
#         InternationalSaleReport.objects.insert(international_sale_records)
#         self.stdout.write(f"Processed {len(international_sale_records)} records for InternationalSaleReport.")


# class Command(BaseCommand):
#     help = "Load data from may_2022.csv into MongoDB"

#     def handle(self, *args, **kwargs):
#         self.stdout.write("Loading may_2022.csv...")

#         # Load CSV into DataFrame
#         may_2022_df = pd.read_csv('may_2022.csv', low_memory=False)

#         # Clean and process columns (adding missing columns as random values)
#         # If any columns are missing, we can add them here with random values
#         if 'DATE' not in may_2022_df.columns:
#             may_2022_df['DATE'] = ['2022-05-01'] * len(may_2022_df)  # Adding a random date, you can adjust
#         if 'STYLE' not in may_2022_df.columns:
#             may_2022_df['STYLE'] = ['Unknown'] * len(may_2022_df)  # Adding random style value
#         if 'SKU' not in may_2022_df.columns:
#             may_2022_df['SKU'] = ['Unknown'] * len(may_2022_df)  # Adding random SKU value
#         if 'SIZE' not in may_2022_df.columns:
#             may_2022_df['SIZE'] = ['Unknown'] * len(may_2022_df)  # Adding random size value
#         if 'RATE' not in may_2022_df.columns:
#             may_2022_df['RATE'] = [random.uniform(100, 500) for _ in range(len(may_2022_df))]  # Random rate
#         if 'GROSS AMT' not in may_2022_df.columns:
#             may_2022_df['GROSS AMT'] = [random.uniform(500, 2000) for _ in range(len(may_2022_df))]  # Random gross amount
#         if 'PCS' not in may_2022_df.columns:
#             may_2022_df['PCS'] = [random.randint(1, 10) for _ in range(len(may_2022_df))]  # Random pcs

#         # Prepare records for MongoDB
#         may_2022_records = []
#         for _, row in may_2022_df.iterrows():
#             may_2022_records.append(May2022Report(
#                 date=row['DATE'],
#                 style=row['STYLE'],
#                 sku=row['SKU'],
#                 size=row['SIZE'],
#                 rate=row['RATE'],
#                 gross_amt=row['GROSS AMT'],
#                 pcs=row['PCS']
#             ))

#         # Insert records into MongoDB
#         May2022Report.objects.insert(may_2022_records)

#         self.stdout.write(f"Processed {len(may_2022_records)} records for May2022Report.")




# class Command(BaseCommand):
#     help = "Load data from pl_march_2021.csv into MongoDB"

#     def handle(self, *args, **kwargs):
#         # Load the CSV file into a DataFrame
#         self.stdout.write("Loading pl_march_2021.csv...")
#         pl_march_2021_df = pd.read_csv('pl_march_2021.csv', low_memory=False, dtype=str).fillna("Unknown")
        
#         # Drop 'DATE' column if it's missing or not needed
#         if 'DATE' in pl_march_2021_df.columns:
#             pl_march_2021_df.drop('DATE', axis=1, inplace=True)

#         # If required columns are missing, add them with random values
#         missing_columns = ['STYLE', 'SKU', 'SIZE', 'RATE', 'GROSS AMT', 'PCS']
#         for col in missing_columns:
#             if col not in pl_march_2021_df.columns:
#                 pl_march_2021_df[col] = [random.choice(["Unknown", "N/A", "Placeholder"]) for _ in range(len(pl_march_2021_df))]

#         # Now, insert the data into MongoDB
#         pl_march_2021_records = []
#         for _, row in pl_march_2021_df.iterrows():
#             pl_march_2021_records.append(PLMarch2021Report(
#                 sku=row['Sku'],
#                 style=row['Style Id'],
#                 category=row['Category'],
#                 weight=row['Weight'],
#                 tp1=row['TP 1'],
#                 tp2=row['TP 2'],
#                 mrp_old=row['MRP Old'],
#                 final_mrp_old=row['Final MRP Old'],
#                 ajio_mrp=row['Ajio MRP'],
#                 amazon_mrp=row['Amazon MRP'],
#                 amazon_fba_mrp=row['Amazon FBA MRP'],
#                 flipkart_mrp=row['Flipkart MRP'],
#                 limeroad_mrp=row['Limeroad MRP'],
#                 myntra_mrp=row['Myntra MRP'],
#                 paytm_mrp=row['Paytm MRP'],
#                 snapdeal_mrp=row['Snapdeal MRP'],
#                 # If the column 'PCS' exists, use its value; else, assign a random value
#                 pcs=int(row['PCS']) if row.get('PCS', '').isdigit() else random.randint(1, 10),
#                 # Assuming some random values for the missing columns
#                 size=row.get('SIZE', 'Unknown'),
#                 rate=row.get('RATE', '0'),
#                 gross_amt=row.get('GROSS AMT', '0')
#             ))

#         # Insert the records into the database
#         PLMarch2021Report.objects.insert(pl_march_2021_records)
#         self.stdout.write(f"Processed {len(pl_march_2021_records)} records for PLMarch2021Report.")\


class Command(BaseCommand):
    help = "Load sale_report.csv into MongoDB"

    def handle(self, *args, **kwargs):
   
        self.stdout.write("Loading sale_report.csv...")
        sale_report_df = pd.read_csv('sale_report.csv', low_memory=False, dtype=str).fillna("Unknown")
        

        self.stdout.write(f"Columns in sale_report.csv: {sale_report_df.columns}")
        
        sale_report_df = sale_report_df[['SKU Code', 'Design No.', 'Category', 'Size', 'Color']]  
        
      
        sale_report_records = []
        
        for _, row in sale_report_df.iterrows():
            sale_report_records.append(SalesReport(
                sku_code=str(row['SKU Code']),
                design_no=str(row['Design No.']),
                category=str(row['Category']),
                size=str(row['Size']),
                color=str(row['Color']),
            ))

      
        SalesReport.objects.insert(sale_report_records)
        
        self.stdout.write(f"Processed {len(sale_report_records)} records for SalesReport.")