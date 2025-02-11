#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# In[6]:


# # Clear all DataFrames by setting them to None
# # amazon_sale_report_df = None
# cloud_warehouse_comparison_chart_df = None
# expense_iigf_df = None
# international_sale_report_df = None
# may_2022_df = None
# pl_march_2021_df = None
# sale_report_df = None


# In[7]:


import pandas as pd

# Specify the data types for columns in Amazon Sale Report
dtype_spec = {
    'Order ID': 'str',
    'Date': 'str',
    'Status': 'str',
    'Fulfilment': 'str',
    'Sales Channel': 'str',
    'ship-service-level': 'str',
    'Style': 'str',
    'SKU': 'str',
    'Category': 'str',
    'Size': 'str',
    'ASIN': 'str',
    'Courier Status': 'str',
    'Qty': 'float',
    'currency': 'str',
    'Amount': 'float',
    'ship-city': 'str',
    'ship-state': 'str',
    'ship-postal-code': 'str',
    'ship-country': 'str',
    'promotion-ids': 'str',
    'B2B': 'bool',
    'fulfilled-by': 'str'
}

# Reload the CSV files into DataFrames with specified data types

amazon_sale_report_df = pd.read_csv('amazon_sale_report.csv', dtype=dtype_spec, low_memory=False)
cloud_warehouse_comparison_chart_df = pd.read_csv('cloud_warehouse_comparison_chart.csv')
expense_iigf_df = pd.read_csv('expense_iigf.csv')
international_sale_report_df = pd.read_csv('international_sale_report.csv')
may_2022_df = pd.read_csv('may_2022.csv')
pl_march_2021_df = pd.read_csv('pl_march_2021.csv')
sale_report_df = pd.read_csv('sale_report.csv')




# Data Cleaning 

# In[8]:


amazon_sale_report_df


# In[9]:


print(amazon_sale_report_df.isnull().sum())


# In[10]:


# Handle missing values
amazon_sale_report_df['Courier Status'] = amazon_sale_report_df['Courier Status'].fillna('unknown')
amazon_sale_report_df['currency'] = amazon_sale_report_df['currency'].fillna('unknown')
amazon_sale_report_df['Amount'] = amazon_sale_report_df['Amount'].fillna(0)
amazon_sale_report_df['ship-city'] = amazon_sale_report_df['ship-city'].fillna('unknown')
amazon_sale_report_df['ship-state'] = amazon_sale_report_df['ship-state'].fillna('unknown')
amazon_sale_report_df['ship-postal-code'] = amazon_sale_report_df['ship-postal-code'].fillna('unknown')
amazon_sale_report_df['ship-country'] = amazon_sale_report_df['ship-country'].fillna('unknown')
amazon_sale_report_df['promotion-ids'] = amazon_sale_report_df['promotion-ids'].fillna('none')
amazon_sale_report_df['fulfilled-by'] = amazon_sale_report_df['fulfilled-by'].fillna('unknown')


# In[11]:


amazon_sale_report_df.drop_duplicates(inplace=True)


# In[12]:


amazon_sale_report_df['Category'] = amazon_sale_report_df['Category'].str.lower().str.strip()
amazon_sale_report_df['Status'] = amazon_sale_report_df['Status'].str.lower().str.strip()
amazon_sale_report_df['Fulfilment'] = amazon_sale_report_df['Fulfilment'].str.lower().str.strip()
amazon_sale_report_df['Style'] = amazon_sale_report_df['Style'].str.lower().str.strip()
amazon_sale_report_df['SKU'] = amazon_sale_report_df['SKU'].str.lower().str.strip()
amazon_sale_report_df['Courier Status'] = amazon_sale_report_df['Courier Status'].str.lower().str.strip()
amazon_sale_report_df['currency'] = amazon_sale_report_df['currency'].str.lower().str.strip()

# Convert 'Date' column to datetime format
amazon_sale_report_df['Date'] = pd.to_datetime(amazon_sale_report_df['Date'], format='%m-%d-%y')

# Create new columns
amazon_sale_report_df['Month'] = amazon_sale_report_df['Date'].dt.month
amazon_sale_report_df['Weekday'] = amazon_sale_report_df['Date'].dt.weekday


# In[13]:


amazon_sale_report_df.drop(columns=['Unnamed: 22'], inplace=True)


# In[14]:


amazon_sale_report_df


# In[15]:


print(cloud_warehouse_comparison_chart_df.isnull().sum())


# In[16]:


# Convert columns to numeric, forcing non-numeric values to NaN
cloud_warehouse_comparison_chart_df['Shiprocket'] = pd.to_numeric(cloud_warehouse_comparison_chart_df['Shiprocket'], errors='coerce')
cloud_warehouse_comparison_chart_df['INCREFF'] = pd.to_numeric(cloud_warehouse_comparison_chart_df['INCREFF'], errors='coerce')

# Handle missing values without using inplace=True
cloud_warehouse_comparison_chart_df['Shiprocket'] = cloud_warehouse_comparison_chart_df['Shiprocket'].fillna(cloud_warehouse_comparison_chart_df['Shiprocket'].mean())
cloud_warehouse_comparison_chart_df['INCREFF'] = cloud_warehouse_comparison_chart_df['INCREFF'].fillna(cloud_warehouse_comparison_chart_df['INCREFF'].mean())

# Remove duplicate rows
cloud_warehouse_comparison_chart_df.drop_duplicates(inplace=True)

# Drop the 'Unnamed: 1' column
if 'Unnamed: 1' in cloud_warehouse_comparison_chart_df.columns:
    cloud_warehouse_comparison_chart_df.drop(columns=['Unnamed: 1'], inplace=True)


# In[17]:


cloud_warehouse_comparison_chart_df['Shiprocket'] = cloud_warehouse_comparison_chart_df['Shiprocket'].fillna(cloud_warehouse_comparison_chart_df['Shiprocket'].mean())


# In[18]:


cloud_warehouse_comparison_chart_df.drop(columns=['Shiprocket'], inplace=True)


# Cleaning of EXPENSE_IIGF.CSV
# 

# In[19]:


print(expense_iigf_df.isnull().sum())


# In[20]:


# Handle missing values without using inplace=True
expense_iigf_df['Recived Amount'] = expense_iigf_df['Recived Amount'].fillna('unknown')
expense_iigf_df['Unnamed: 1'] = expense_iigf_df['Unnamed: 1'].fillna(0)
expense_iigf_df['Expance'] = expense_iigf_df['Expance'].fillna('unknown')


# In[21]:


# Drop the first row containing headers (if needed)
# expense_iigf_df.drop(0, inplace=True)

# Handle missing values without using inplace=True
expense_iigf_df['Recived Amount'] = expense_iigf_df['Recived Amount'].fillna('unknown')
expense_iigf_df['Unnamed: 1'] = expense_iigf_df['Unnamed: 1'].fillna(0)
expense_iigf_df['Expance'] = expense_iigf_df['Expance'].fillna('unknown')

# Rename columns
expense_iigf_df.rename(columns={
    'Recived Amount': 'Received_Amount',
    'Unnamed: 1': 'Received_Amount_Value',
    'Expance': 'Expense',
    'Unnamed: 3': 'Expense_Value'
}, inplace=True)

# Remove duplicate rows
expense_iigf_df.drop_duplicates(inplace=True)


# In[22]:


print(expense_iigf_df.isnull().sum())


# Data cleaning for the international_sale_report.csv

# In[23]:


print(international_sale_report_df.isnull().sum())


# In[24]:


# Handle missing values without using inplace=True
international_sale_report_df['DATE'] = international_sale_report_df['DATE'].fillna('unknown')
international_sale_report_df['Months'] = international_sale_report_df['Months'].fillna('unknown')
international_sale_report_df['CUSTOMER'] = international_sale_report_df['CUSTOMER'].fillna('unknown')
international_sale_report_df['Style'] = international_sale_report_df['Style'].fillna('unknown')
international_sale_report_df['SKU'] = international_sale_report_df['SKU'].fillna('unknown')
international_sale_report_df['Size'] = international_sale_report_df['Size'].fillna('unknown')
international_sale_report_df['PCS'] = international_sale_report_df['PCS'].fillna(0)
international_sale_report_df['RATE'] = international_sale_report_df['RATE'].fillna(0)
international_sale_report_df['GROSS AMT'] = international_sale_report_df['GROSS AMT'].fillna(0)


# In[25]:


# Remove duplicate rows
international_sale_report_df.drop_duplicates(inplace=True)

# Standardize text columns
international_sale_report_df['CUSTOMER'] = international_sale_report_df['CUSTOMER'].str.lower().str.strip()
international_sale_report_df['Style'] = international_sale_report_df['Style'].str.lower().str.strip()
international_sale_report_df['SKU'] = international_sale_report_df['SKU'].str.lower().str.strip()
international_sale_report_df['Size'] = international_sale_report_df['Size'].str.lower().str.strip()

# Convert data types if necessary
international_sale_report_df['DATE'] = pd.to_datetime(international_sale_report_df['DATE'], errors='coerce')
international_sale_report_df['PCS'] = pd.to_numeric(international_sale_report_df['PCS'], errors='coerce')
international_sale_report_df['RATE'] = pd.to_numeric(international_sale_report_df['RATE'], errors='coerce')
international_sale_report_df['GROSS AMT'] = pd.to_numeric(international_sale_report_df['GROSS AMT'], errors='coerce')

# Create new columns (if necessary)
international_sale_report_df['Month'] = international_sale_report_df['DATE'].dt.month
international_sale_report_df['Year'] = international_sale_report_df['DATE'].dt.year


# In[26]:


# Convert data types with specified format for DATE
date_format = '%Y-%m-%d'  # Specify the appropriate date format
international_sale_report_df['DATE'] = pd.to_datetime(international_sale_report_df['DATE'], format=date_format, errors='coerce')

# Create new columns
international_sale_report_df['Month'] = international_sale_report_df['DATE'].dt.month
international_sale_report_df['Year'] = international_sale_report_df['DATE'].dt.year


# In[27]:


print(international_sale_report_df.isnull().sum())


# In[28]:


# Convert DATE column to datetime with more robust parsing
international_sale_report_df['DATE'] = pd.to_datetime(international_sale_report_df['DATE'], errors='coerce')

# Fill missing DATE values with a placeholder or an estimated date
international_sale_report_df['DATE'] = international_sale_report_df['DATE'].fillna(pd.Timestamp('1900-01-01'))

# Recreate Month and Year columns
international_sale_report_df['Month'] = international_sale_report_df['DATE'].dt.month
international_sale_report_df['Year'] = international_sale_report_df['DATE'].dt.year

# Fill remaining missing values
international_sale_report_df['PCS'] = international_sale_report_df['PCS'].fillna(0)
international_sale_report_df['RATE'] = international_sale_report_df['RATE'].fillna(0)
international_sale_report_df['GROSS AMT'] = international_sale_report_df['GROSS AMT'].fillna(0)


# In[29]:


print(international_sale_report_df.isnull().sum())


# Data cleaning for the may_2022

# In[30]:


print(may_2022_df.isnull().sum())


# Data Cleaning for pl_march_2021.csv

# In[31]:


print(pl_march_2021_df.isnull().sum())


# Data Cleaning sale_report.csv
# 

# In[32]:


print(sale_report_df.isnull().sum())


# In[33]:


# Handle missing values
sale_report_df['SKU Code'] = sale_report_df['SKU Code'].fillna('unknown')
sale_report_df['Design No.'] = sale_report_df['Design No.'].fillna('unknown')
sale_report_df['Stock'] = sale_report_df['Stock'].fillna(0)
sale_report_df['Category'] = sale_report_df['Category'].fillna('unknown')
sale_report_df['Size'] = sale_report_df['Size'].fillna('unknown')
sale_report_df['Color'] = sale_report_df['Color'].fillna('unknown')


# In[34]:


# Remove duplicate rows
sale_report_df.drop_duplicates(inplace=True)


# In[35]:


# Standardize text columns
sale_report_df['SKU Code'] = sale_report_df['SKU Code'].str.lower().str.strip()
sale_report_df['Design No.'] = sale_report_df['Design No.'].str.lower().str.strip()
sale_report_df['Category'] = sale_report_df['Category'].str.lower().str.strip()
sale_report_df['Size'] = sale_report_df['Size'].str.lower().str.strip()
sale_report_df['Color'] = sale_report_df['Color'].str.lower().str.strip()


# In[36]:


print(sale_report_df.isnull().sum())


# Data Cleaning is completed

# 🔹 Step 1: Enrich Sales Transactions with Product Details
# 
# 
# 
# 📌 Objective: Identify best-selling product categories and stock turnover.
# 

# In[37]:


import pandas as pd


# In[38]:


# Merge the tables on SKU and SKU Code
combined_sales_df = pd.merge(amazon_sale_report_df, sale_report_df, left_on='SKU', right_on='SKU Code', how='left')

# View the new DataFrame
print("Combined Sales DataFrame:")
print(combined_sales_df.head())


# In[39]:


# Calculate total sales per product category and store it in a DataFrame
total_sales_per_category = combined_sales_df.groupby('Category_y')['Amount'].sum().reset_index()
total_sales_per_category.columns = ['Category', 'Total Sales']

# View the DataFrame
print("Total Sales per Product Category DataFrame:")
print(total_sales_per_category)


# KPI calculations
# 
# Total Sales per Product Category: SUM(Amount) grouped by Category_y
# 
# Top-Selling Products: SUM(Quantity Ordered) grouped by SKU and Design No.
# 
# Sales Correlation Heatmap: Correlation Matrix of numerical columns visualized using sns.heatmap()

# These lines group the merged dataset by product category and sum the total sales (Amount) for each category. The resulting DataFrame renames the columns for better readability, showing total sales per category.

# In[40]:


combined_sales_df


# In[41]:


# Check column names in the DataFrame
print("Columns in combined_sales_df:")
print(combined_sales_df.columns)



# In[42]:


# Calculate top-selling products by quantity ordered and store it in a DataFrame
top_selling_products = combined_sales_df.groupby(['SKU', 'Design No.'])['Qty'].sum().reset_index()
top_selling_products = top_selling_products.sort_values(by='Qty', ascending=False).head(10)

# View the DataFrame
print("Top-Selling Products DataFrame:")
print(top_selling_products)


# In[43]:


# Select only numeric columns for the correlation calculation
numeric_columns = combined_sales_df.select_dtypes(include=['number'])

# Calculate correlation matrix
correlation_matrix = numeric_columns.corr()


# In[44]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Sales Correlation Heatmap')
plt.show()


# MAIN VISUALISATION
# 

# In[45]:


import pandas as pd

# Group by SKU and Category, sum the quantities, and reset the index
grouped_products = combined_sales_df.groupby(['SKU', 'Category_y'])['Qty'].sum().reset_index()

# Sort by Quantity and get the top products per category
top_products_per_category = grouped_products.sort_values(by='Qty', ascending=False).groupby('Category_y').head(1)

# View the DataFrame
print("Top Products per Category DataFrame:")
print(top_products_per_category)


# In[46]:


import pandas as pd

# Group by SKU and Category, sum the quantities, and reset the index
grouped_products = combined_sales_df.groupby(['SKU', 'Category_y'])['Qty'].sum().reset_index()

# Sort by Quantity and get the top products per category
top_products_per_category = grouped_products.sort_values(by='Qty', ascending=False).groupby('Category_y').head(1)

# View the DataFrame
print("Top Products per Category DataFrame:")
print(top_products_per_category)


# In[47]:


# Calculate total quantity for the top products
total_qty = top_products_per_category['Qty'].sum()

# Calculate the percentage of each category
top_products_per_category['Percentage'] = (top_products_per_category['Qty'] / total_qty) * 100

# Split into main categories and "Other"
main_categories = top_products_per_category[top_products_per_category['Percentage'] >= 2]
other_categories = top_products_per_category[top_products_per_category['Percentage'] < 2]

# Aggregate the "Other" categories
other_row = pd.DataFrame([{
    'SKU': 'Other',
    'Category_y': 'Other',
    'Qty': other_categories['Qty'].sum(),
    'Percentage': other_categories['Percentage'].sum()
}])

# Combine the main categories with the "Other" category
final_products = pd.concat([main_categories, other_row])

# View the DataFrame
print("Final Products DataFrame:")
print(final_products)


# In[48]:


import matplotlib.pyplot as plt

# Pie chart for top products per category with category labels
plt.figure(figsize=(10, 6))
plt.pie(final_products['Qty'], labels=final_products['Category_y'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Top Products by Quantity Ordered (Categories with "Other")')
plt.axis('equal')
plt.show()


# In[49]:


# Create legend for "Other" categories
other_categories_list = other_categories[['SKU', 'Category_y']].values.tolist()
other_categories_legend = '\n'.join([f"{sku}: {category}" for sku, category in other_categories_list])

print("Other Categories Included:")
print(other_categories_legend)


# Business Insights:
# 
# Sales Dominance: "Kurta" and "Kurta Set" categories are the top performers, indicating strong market demand.
# 
# Diversification: Categories with less than 2% contribution are grouped under "Other," revealing areas for potential growth and diversification.
# 
# Balanced View: The visualization helps in identifying key revenue-driving categories while acknowledging smaller yet significant segments.

#                                     stacked bar chart for revenue contribution by products.

# In[50]:


combined_sales_df = pd.merge(amazon_sale_report_df, sale_report_df, left_on='SKU', right_on='SKU Code', how='left')

# Calculate total sales amount by category and SKU
revenue_contribution_df = combined_sales_df.groupby(['Category_y', 'SKU'])['Amount'].sum().unstack().fillna(0)

# Stacked bar chart for revenue contribution by products within categories
plt.figure(figsize=(14, 8))
revenue_contribution_df.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis', legend=False)
plt.xlabel('Product Categories')
plt.ylabel('Total Sales Amount (in currency units)')
plt.title('Revenue Contribution by Products within Categories')
plt.xticks(rotation=45)

plt.show()


# In[51]:


# View the DataFrame to check the exact values
print("Revenue Contribution by Products within Categories DataFrame:")
print(revenue_contribution_df)


# In[52]:


combined_sales_df = pd.merge(amazon_sale_report_df, sale_report_df, left_on='SKU', right_on='SKU Code', how='left')

# Calculate total sales amount by category and SKU
revenue_contribution_df = combined_sales_df.groupby(['Category_y', 'SKU'])['Amount'].sum().unstack().fillna(0)

# View the DataFrame to check the exact values
print("Revenue Contribution by Products within Categories DataFrame:")
print(revenue_contribution_df)

# Stacked bar chart for revenue contribution by products within categories with currency format on y-axis
plt.figure(figsize=(14, 8))
revenue_contribution_df.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis', legend=False)
plt.xlabel('Product Categories')
plt.ylabel('Total Sales Amount (INR)')
plt.title('Revenue Contribution by Products within Categories')
plt.xticks(rotation=45)

# Add currency format to y-axis
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "₹{:,.0f}".format(x)))

plt.show()


# Business Insights:
# 
# Top Revenue Drivers: "Kurta" and "Kurta Set" are the top revenue contributors, indicating their high demand.
# 
# Category Performance: Revenue is highly concentrated in a few key categories, suggesting a potential focus for marketing and inventory management.
# 
# Product Variability: The stacked bar chart clearly shows the revenue distribution among different products within each category, highlighting the significant contributors.

#                                     Scatter Plot: Total Sales vs. Profit Margin

# In[53]:


# Display first few rows of Amazon Sale Report
display(amazon_sale_report_df.head())


# In[54]:


# Display the first few rows of Sale Report DataFrame
display(sale_report_df.head())


# In[55]:


# Merge Amazon Sales Report with Sale Report on SKU = SKU Code
merged_df = amazon_sale_report_df.merge(
    sale_report_df, left_on="SKU", right_on="SKU Code", how="left"
)

# Display the first few rows of the merged DataFrame
display(merged_df.head())


# KPI Calculation
# 

# In[56]:


# Calculate Total Sales per Product Category
sales_per_category = (
    merged_df.groupby("Category_x")["Amount"]
    .sum()
    .reset_index()
    .sort_values(by="Amount", ascending=False)
)

# Display the KPI DataFrame
display(sales_per_category.head())


# In[61]:


# 📊 Visualization: Total Sales per Product Category (Bar Chart)
plt.figure(figsize=(12, 6))

# Create the bar chart with accurate x-axis values
sns.barplot(x="Amount", y="Category_x", hue="Category_x", data=sales_per_category, palette="coolwarm", legend=False)

# Labels and title
plt.xlabel("Total Sales (₹)", fontsize=12)
plt.ylabel("Product Category", fontsize=12)
plt.title("Total Sales per Product Category", fontsize=14)

# Format x-axis with commas for readability
plt.xticks(rotation=45)  # Rotate for better readability
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'₹{int(x):,}'))  # Format with commas

# Show value labels on bars
for index, value in enumerate(sales_per_category["Amount"]):
    plt.text(value, index, f'₹{int(value):,}', va='center', fontsize=10)

# Show the plot
plt.show()


# 1️⃣ Total Sales per Product Category
# 	•	Highest Revenue: Set (₹39.2M), Kurta (₹21.3M), Western Dress (₹11.2M) → These categories contribute the most to total revenue.
# 	•	Lowest Revenue: Ethnic Dress (₹0.79M) → Least profitable category, may need promotional strategies.
# 	•	Actionable Insight: Focus on top categories (Sets & Kurtas) for marketing & inventory optimization.

# In[62]:


# 📊 Visualization: Top-Selling Product Categories by Quantity Ordered
top_selling_products_category = (
    merged_df.groupby("Category_x")["Qty"]
    .sum()
    .reset_index()
    .sort_values(by="Qty", ascending=False)
    .head(10)  # Display top 10 product categories instead of SKU
)

plt.figure(figsize=(12, 6))

# Create the bar chart
sns.barplot(x="Qty", y="Category_x", hue="Category_x", data=top_selling_products_category, palette="magma", legend=False)

# Labels and title
plt.xlabel("Total Quantity Ordered (Units Sold)", fontsize=12)
plt.ylabel("Product Category", fontsize=12)
plt.title("Top-Selling Product Categories by Quantity Ordered", fontsize=14)

# Format x-axis with commas for readability
plt.xticks(rotation=45)  # Rotate for better readability
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{int(x):,}'))  # Format with commas

# Show value labels on bars
for index, value in enumerate(top_selling_products_category["Qty"]):
    plt.text(value, index, f'{int(value):,}', va='center', fontsize=10)

# Show the plot
plt.show()


#  Top-Selling Product Categories by Quantity Ordered
# 	•	Most Ordered: Kurta & Sets → High demand, optimize stock & pricing strategies.
# 	•	Lower Order Volume: Western Dress, Ethnic Dress → Consider discounts or promotions to boost sales.
# 	•	Actionable Insight: Increase stock & advertising for high-volume products, improve visibility for lower-selling categories.
# 

#                                         Sales Channel Performance Analysis

# In[73]:


# Display the first few rows of Amazon Sales Report
display(amazon_sale_report_df.head())


# In[74]:


# Display the first few rows of Cloud Warehouse Comparison Chart
display(cloud_warehouse_comparison_chart_df.head())


# In[75]:


# Merge Amazon Sales Report with Cloud Warehouse Comparison Chart on Sales Channel
sales_channel_df = amazon_sale_report_df.merge(
    cloud_warehouse_comparison_chart_df, left_index=True, right_index=True, how="left"
)

# Display the first few rows of the merged DataFrame
display(sales_channel_df.head())


# In[77]:


# Print all column names
print(sales_channel_df.columns)


# In[78]:


# Remove spaces from column names
sales_channel_df.rename(columns=lambda x: x.strip(), inplace=True)


# In[80]:


# List unique sales channels
unique_sales_channels = sales_channel_df["Sales Channel"].unique()
print(unique_sales_channels)


# In[81]:


# Count sales records per platform
sales_channel_counts = sales_channel_df["Sales Channel"].value_counts()
display(sales_channel_counts)


# In[82]:


# Count total orders per Sales Channel
orders_per_channel = sales_channel_df["Sales Channel"].value_counts().reset_index()
orders_per_channel.columns = ["Sales Channel", "Total Orders"]

# Display the DataFrame
display(orders_per_channel)


# In[83]:


import matplotlib.pyplot as plt

# 📊 Visualization: Sales Distribution Across Platforms
plt.figure(figsize=(8, 6))
plt.pie(orders_per_channel["Total Orders"], labels=orders_per_channel["Sales Channel"], 
        autopct='%1.1f%%', colors=sns.color_palette("coolwarm"))

# Title
plt.title("Sales Distribution Across Platforms")

# Show the plot
plt.show()


# In[84]:


# Check if Amount column exists in Cloud Warehouse dataset
print(cloud_warehouse_comparison_chart_df.columns)


# 1️⃣ The dataset does not contain an Amount column.
# 	•	This means we do not have revenue (₹) data for platforms like INCREFF, Shiprocket, etc.
# 	•	We cannot confirm that these platforms had zero sales—only that their sales data is missing from this dataset.

#  Business Insights: Sales Platform Analysis
# 
# ✅ 1. Amazon.in is the only recorded sales platform
# 	•	Amazon.in contributes ₹78,592,678.30, which is 100% of revenue in our dataset.
# 	•	No other platform has revenue (Amount), but this does not mean other platforms had zero sales.

#  2. Missing Revenue Data for Non-Amazon Platforms
# 	•	The INCREFF column exists, but does not have sales revenue (Amount).
# 	•	This means we cannot measure profitability for INCREFF, Shiprocket, or other platforms.
# 	•	If we need their sales performance, we must find another dataset that tracks their Amount.
# 
# 🔍 3. INCREFF Data Might Represent Another KPI
# 	•	The INCREFF column has numeric values (e.g., 10.166667, 15.500000, etc.)
# 	•	This suggests it might be a performance metric (e.g., stock movement, processing time, efficiency) rather than sales.

#                                         Monthly Sales Trend Analysis 📈

# In[86]:


# Convert 'Date' to datetime format
sales_channel_df["Date"] = pd.to_datetime(sales_channel_df["Date"])

# Extract Year-Month
sales_channel_df["Year-Month"] = sales_channel_df["Date"].dt.to_period("M")

# Group by Month and Sum Sales Amount
monthly_sales = sales_channel_df.groupby("Year-Month")["Amount"].sum().reset_index()

# Convert back to string for plotting
monthly_sales["Year-Month"] = monthly_sales["Year-Month"].astype(str)


# In[87]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.lineplot(x="Year-Month", y="Amount", data=monthly_sales, marker="o", color="b")

plt.xticks(rotation=45)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Total Sales (₹)", fontsize=12)
plt.title("Monthly Sales Trend", fontsize=14)
plt.grid(True)

plt.show()


# In[92]:


# Find all unique months in the dataset
unique_months = sales_channel_df["Month"].unique()
print(unique_months)


# In[93]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Extract Year and Month for heatmap structure
sales_channel_df["Year"] = sales_channel_df["Date"].dt.year
sales_channel_df["Month"] = sales_channel_df["Date"].dt.month

# Map month numbers to month names
month_mapping = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}
sales_channel_df["Month"] = sales_channel_df["Month"].map(month_mapping)

# Count total orders per month
monthly_orders = sales_channel_df.groupby(["Year", "Month"]).size().reset_index(name="Total Orders")

# Pivot table for heatmap
orders_pivot = monthly_orders.pivot(index="Month", columns="Year", values="Total Orders")

# Ensure all months are included
all_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
orders_pivot = orders_pivot.reindex(all_months, fill_value=0)

# Plot Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(orders_pivot, cmap="coolwarm", annot=True, fmt=".0f", linewidths=0.5, cbar_kws={'label': 'Total Orders'})

plt.xlabel("Year", fontsize=12)
plt.ylabel("Month", fontsize=12)
plt.title("Monthly Order Volume Heatmap", fontsize=14)

plt.show()


# Business Insights 
# 
#  Peak Sales Months: Orders are highest in April-June, indicating seasonal demand or promotions.
# 
# 
# 
# Slow Sales Periods: Other months have low or no orders, requiring off-season discounts or campaigns.
# 
# Year-over-Year Growth: Comparing years helps track sales improvement or decline for better planning.
# 
# 
#  Action Plan: Focus on inventory, marketing, and discounts before peak months to maximize revenue. 
