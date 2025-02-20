from celery import shared_task
import pymongo
import pandas as pd
import os
from django.conf import settings
import subprocess

@shared_task
def load_clean_data_into_mongo():
    """
    
    """
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["mydatabase"]
    collection = db["ecommerce"]

    clean_dir = os.path.join(settings.BASE_DIR, "Clean_Dataset")

    datasets = {
        "clean_sale_report": "clean_sale_report.csv",
        "clean_march_2021": "clean_march_2021.csv",
        "clean_may_2022": "clean_may_2022.csv",
        "clean_international_sale_df": "clean_international_sale_df.csv",
        "clean_cloud_warehouse_df": "clean_cloud_warehouse_df.csv",
        "clean_amazon_sale_report_df": "clean_amazon_sale_report_df.csv",
        "clean_expense_iigf_df": "clean_expense_iigf_df.csv",
    }

    for dataset_name, filename in datasets.items():
        file_path = os.path.join(clean_dir, filename)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            df.columns = df.columns.str.replace(".", "_", regex=False)

            df.dropna(inplace=True)
            data = df.to_dict(orient="records")

            for record in data:
                record["dataset_name"] = dataset_name  

            if data:
                collection.insert_many(data)
                print(f" Inserted {len(data)} records from '{dataset_name}' into 'ecommerce' collection")
        else:
            print(f" File not found: {file_path}")

    print("Periodic MongoDB is beonn updte now")


@shared_task
def update_google_sheets_task():
    project_dir = settings.BASE_DIR  
    script_path = os.path.join(project_dir, "funcs", "update_google_sheets.py")
    
    try:
        result = subprocess.run(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"My Output:\n{result.stdout}")
        print(f" Error:\n{result.stderr}")
    except Exception as e:
        print(f"Error running update_google_sheets.py: {e}")

