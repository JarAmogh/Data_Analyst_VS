import sqlite3

# Define the path to the SQLite database
db_path = '/Users/amogh/Documents/DataAnalyst/Data_Analyst_VS/db.sqlite3'

conn = sqlite3.connect(db_path)


cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()


for table in tables:
    table_name = table[0]
    print(f"Contents of table: {table_name}")
    

    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    
    for row in rows:
        print(row)
    
    print("\n" + "="*50 + "\n")


cursor.close()
conn.close()

print("Database inspection complete.")