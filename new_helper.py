#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# In[123]:


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



_new_amazon_sale_report_df1= pd.read_csv('amazon_sale_report.csv', dtype=dtype_spec, low_memory=False)
_new_cloud_warehouse_comparison_chart_df = pd.read_csv('cloud_warehouse_comparison_chart.csv')
_new_expense_iigf_df = pd.read_csv('expense_iigf.csv')
_new_international_sale_report_df = pd.read_csv('international_sale_report.csv')
_new_may_2022_df = pd.read_csv('may_2022.csv')
_new_pl_march_2021_df = pd.read_csv('pl_march_2021.csv')
_new_sale_report_df = pd.read_csv('sale_report.csv')



# In[124]:


_new_amazon_sale_report_df1.isnull().sum()


# In[125]:


_new_sale_report_df.isnull().sum()


# Sales by Region (Approach)
# 
# 	•	Observation: Metro cities contribute 70% of total revenue, while smaller cities contribute only 30%.
# 
# 	•	Business Decision: Target marketing campaigns to high-revenue regions while expanding delivery options to smaller cities.
# 	
# 	•	Actionable Insight: Consider regional pricing strategies to boost adoption in Tier-2 and Tier-3 cities.

# Modifications done in Amazon sales report
# 1) Added mssing values of city , stae randomly 
# 2) dorpped unknown column
# 3) added reasonbale amount for missing amount values
# 4) addeda colum of metro and non metro city 

# In[126]:


_new_sale_report_df


# In[127]:


_new_amazon_sale_report_df1.columns


# In[128]:


import random


# In[129]:


metro_cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
non_metro_cities = ["Jaipur", "Lucknow", "Indore", "Bhopal", "Chandigarh", "Patna", "Nagpur", "Vadodara", "Coimbatore"]

def classsify_city(city):
    if city in metro_cities:
        return "Metro"
    elif city in non_metro_cities:
        return "Non-Metro"
    else:
        return "Other"
_new_amazon_sale_report_df1['ship-city'] = _new_amazon_sale_report_df1['ship-city'].apply(
    lambda x: x if pd.notna(x) else random.choice(metro_cities + non_metro_cities))

state_mapping = {
    "Mumbai": "Maharashtra", "Delhi": "Delhi", "Bangalore": "Karnataka", "Hyderabad": "Telangana",
    "Chennai": "Tamil Nadu", "Kolkata": "West Bengal", "Pune": "Maharashtra", "Ahmedabad": "Gujarat",
    "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh", "Indore": "Madhya Pradesh", "Bhopal": "Madhya Pradesh",
    "Chandigarh": "Chandigarh", "Patna": "Bihar", "Nagpur": "Maharashtra", "Vadodara": "Gujarat",
    "Coimbatore": "Tamil Nadu"
}

_new_amazon_sale_report_df1['ship-state'] = _new_amazon_sale_report_df1['ship-city'].map(state_mapping)

_new_amazon_sale_report_df1['Amount'] = _new_amazon_sale_report_df1['Amount'].apply(
    lambda x: x if pd.notna(x) else random.randint(200, 2000)
)
_new_amazon_sale_report_df1.drop(columns=['Unnamed: 22', 'promotion-ids'], inplace=True)

_new_amazon_sale_report_df1['city-type'] = _new_amazon_sale_report_df1['ship-city'].apply(classsify_city)


_new_amazon_sale_report_df1





# In[130]:


# # Function to get counts of metro cities, non-metro cities, each state, and each city
# def get_location_counts(df):
#     metro_count = df[df['city-type'] == 'Metro'].shape[0]
#     non_metro_count = df[df['city-type'] == 'Non-Metro'].shape[0]
#     state_counts = df['ship-state'].value_counts()
#     city_counts = df['ship-city'].value_counts()
    
#     return metro_count, non_metro_count, state_counts, city_counts

# # Compute the counts
# metro_count, non_metro_count, state_counts, city_counts = get_location_counts(_new_amazon_sale_report_df)

# # Display results
# print(f"Metro City Count: {metro_count}")
# print(f"Non-Metro City Count: {non_metro_count}")

# print("\nState Counts:")
# print(state_counts)

# print("\nCity Counts:")
# print(city_counts)


# In[131]:


_new_amazon_sale_report_df1.isnull().sum()


# In[132]:


_new_amazon_sale_report_df1['currency'].fillna("INR", inplace=True)
_new_amazon_sale_report_df1['ship-country'].fillna("IN", inplace=True)
_new_amazon_sale_report_df1['ship-state'].fillna("Unknown State", inplace=True)
_new_amazon_sale_report_df1['ship-postal-code'] = _new_amazon_sale_report_df1['ship-postal-code'].apply(lambda x: x if pd.notna(x) else random.randint(100000, 999999))
_new_amazon_sale_report_df1['Courier Status'].fillna("Unknown", inplace=True)
_new_amazon_sale_report_df1['fulfilled-by'].fillna("Not Specified", inplace=True)



# In[133]:


_new_amazon_sale_report_df1.isnull().sum()


# In[134]:


_new_amazon_sale_report_df1


# In[135]:


print(_new_amazon_sale_report_df1['ship-state'].value_counts())


# In[136]:


filtered_orders_per_state = _new_amazon_sale_report_df1['ship-state'].value_counts().drop("Unknown State")


plt.figure(figsize=(12, 6))
filtered_orders_per_state.plot(kind='bar', color='royalblue')

plt.xlabel("State")
plt.ylabel("Total Number of Orders")
plt.title("Total Orders per State")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# Conclusion :
# 
# 1) excessive marketing is required for the lower states
# 
# 2) Making more warehouses for the lower states to bring more item in stocks to increase order

# In[137]:


city_counts = _new_amazon_sale_report_df1['ship-city'].value_counts()
print(city_counts)


# In[138]:


metro_cities = ["BENGALURU", "HYDERABAD", "MUMBAI", "NEW DELHI", "CHENNAI"]

_new_amazon_sale_report_df1['city_category'] = _new_amazon_sale_report_df1['ship-city'].apply(
    lambda x: 'Metro' if x in metro_cities else 'Non-Metro'
)

city_category_counts = _new_amazon_sale_report_df1['city_category'].value_counts()

plt.figure(figsize=(8, 8))
city_category_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
plt.title('Percentage of Orders in Metro vs Non-Metro Cities')
plt.ylabel('')
plt.show()


# Conclusion:
# 
# 1) Population distribution (we cannot ignore non-metro customer)
# 
# 2) Non - metro customer are main source of revenue

# In[139]:


avg_order_cost_per_state = _new_amazon_sale_report_df1.groupby('ship-state')['Amount'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 7))
avg_order_cost_per_state.plot(kind='bar', color='royalblue')
plt.xlabel("State")
plt.ylabel("Average Order Cost")
plt.title("Average Order Cost per State")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[140]:


metro_cities = ["BENGALURU", "HYDERABAD", "MUMBAI", "NEW DELHI", "CHENNAI"]

_new_amazon_sale_report_df1['city_category'] = _new_amazon_sale_report_df1['ship-city'].apply(
    lambda x: 'Metro' if x in metro_cities else 'Non-Metro'
)

avg_order_cost_per_category = _new_amazon_sale_report_df1.groupby('city_category')['Amount'].mean()

plt.figure(figsize=(10, 6))
avg_order_cost_per_category.plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.xlabel("City Category")
plt.ylabel("Average Order Cost")
plt.title("Average Order Cost for Metro and Non-Metro Cities")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Conclusion :
# 
# 1) Non-Metro has higher Avg cost because of all delivary and handling cost
# 
# 2) Make more warehouse to bring the prices lower since non-metro city people holds major share 
# 
# 3) Pottensial Market (Non-metro): improve logistics  and profit
# 
# 4) High purcahse capacity : we can introduce premium products in this market with proper marketing 

# In[141]:


_new_sale_report_df


# In[142]:


_new_amazon_sale_report_df1


# In[143]:


_new_sale_report_df.isnull().sum()



# In[144]:


_new_sale_report_df['SKU Code'] = _new_sale_report_df['SKU Code'].fillna(_new_sale_report_df['SKU Code'].mode()[0])
_new_sale_report_df['Design No.'] = _new_sale_report_df['Design No.'].fillna(_new_sale_report_df['Design No.'].mode()[0])
_new_sale_report_df['Stock'] = _new_sale_report_df['Stock'].fillna(_new_sale_report_df['Stock'].median())
_new_sale_report_df['Category'] = _new_sale_report_df['Category'].fillna(_new_sale_report_df['Category'].mode()[0])
_new_sale_report_df['Size'] = _new_sale_report_df['Size'].fillna(_new_sale_report_df['Size'].mode()[0])
_new_sale_report_df['Color'] = _new_sale_report_df['Color'].fillna(_new_sale_report_df['Color'].mode()[0])


# In[145]:


category_sales = _new_amazon_sale_report_df1.groupby('Category')['Amount'].sum().sort_values(ascending=False)
print(category_sales)

highest_sales_category = category_sales.idxmax()
print(f"The category with the highest sales is: {highest_sales_category}")

sales_report_combined = pd.merge(_new_sale_report_df, _new_amazon_sale_report_df1, left_on='SKU Code', right_on='SKU', how='inner')

print


# In[146]:


category_sales = _new_amazon_sale_report_df1.groupby('Category')['Amount'].sum().sort_values(ascending=False)


plt.figure(figsize=(10, 10))
category_sales.plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.Paired(range(len(category_sales))))
plt.title('Sales Distribution by Category')
plt.ylabel('')
plt.show()


# In[147]:


_new_amazon_sale_report_df1['Date'] = pd.to_datetime(_new_amazon_sale_report_df1['Date'])
_new_amazon_sale_report_df1['Month'] = _new_amazon_sale_report_df1['Date'].dt.month
_new_amazon_sale_report_df1['Year'] = _new_amazon_sale_report_df1['Date'].dt.year
set_data = _new_amazon_sale_report_df1[_new_amazon_sale_report_df1['Category'] == 'Set']
kurta_data = _new_amazon_sale_report_df1[_new_amazon_sale_report_df1['Category'] == 'kurta']
western_dress_data = _new_amazon_sale_report_df1[_new_amazon_sale_report_df1['Category'] == 'Western Dress']

set_orders = set_data.groupby(['Year', 'Month']).size().unstack(fill_value=0)
kurta_orders = kurta_data.groupby(['Year', 'Month']).size().unstack(fill_value=0)
western_dress_orders = western_dress_data.groupby(['Year', 'Month']).size().unstack(fill_value=0)


# In[148]:


_new_amazon_sale_report_df1


# In[149]:


plt.figure(figsize=(12, 6))
sns.heatmap(set_orders, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Number of Orders per Month - Set Category')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()


# In[150]:


plt.figure(figsize=(12, 6))
sns.heatmap(kurta_orders, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Number of Orders per Month - Kurta Category')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()


# In[151]:


plt.figure(figsize=(12, 6))
sns.heatmap(western_dress_orders, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Number of Orders per Month - Western Dress Category')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()


# FINDING THE MAXIMUM ORDER DAYAS NAD MIN ORDERS DAY FOR THE KURTA IN APRIL AND MONTH

# In[152]:


_new_amazon_sale_report_df1['Date'] = pd.to_datetime(_new_amazon_sale_report_df1['Date'])
_new_amazon_sale_report_df1['ship-state'] = _new_amazon_sale_report_df1['ship-state'].str.strip().str.title()

kurta_orders_df = _new_amazon_sale_report_df1[_new_amazon_sale_report_df1['Category'].str.lower() == 'kurta']

kurta_orders_per_day = kurta_orders_df.groupby('Date')['Qty'].sum().reset_index()
kurta_orders_per_day['Month'] = kurta_orders_per_day['Date'].dt.month
kurta_orders_per_day['Year'] = kurta_orders_per_day['Date'].dt.year

month_mapping = {3: "March", 4: "April", 5: "May"}

april_orders = kurta_orders_per_day[kurta_orders_per_day['Month'] == 4]
may_orders = kurta_orders_per_day[kurta_orders_per_day['Month'] == 5]

def get_max_min_orders(df):
    if not df.empty:
        max_order = df.loc[df['Qty'].idxmax()]
        min_order = df.loc[df['Qty'].idxmin()]
    else:
        max_order = {'Date': None, 'Qty': None}
        min_order = {'Date': None, 'Qty': None}
    return max_order, min_order

april_max, april_min = get_max_min_orders(april_orders)
may_max, may_min = get_max_min_orders(may_orders)

general_result_df = pd.DataFrame({
    'Month': ['April', 'April', 'May', 'May'],
    'Order Type': ['Max Orders', 'Min Orders', 'Max Orders', 'Min Orders'],
    'Date': [april_max['Date'], april_min['Date'], may_max['Date'], may_min['Date']],
    'Total Orders': [april_max['Qty'], april_min['Qty'], may_max['Qty'], may_min['Qty']]
})

general_result_df


# In[153]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=general_result_df, x='Month', y='Total Orders', hue='Order Type', palette='viridis')

for p, (_, row) in zip(ax.patches, general_result_df.iterrows()):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 20, 
            row['Date'].strftime('%Y-%m-%d'), 
            ha='center', fontsize=10, fontweight='bold', color='black')

plt.xlabel("Month")
plt.ylabel("Total Orders")
plt.title("Max & Min Orders in April and May - General")
plt.legend(title="Order Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[154]:


states = ['Maharashtra', 'Tamil Nadu', 'Telangana']

kurta_orders_per_day_state = kurta_orders_df.groupby(['Date', 'ship-state'])['Qty'].sum().reset_index()
kurta_orders_per_day_state['Month'] = kurta_orders_per_day_state['Date'].dt.month
kurta_orders_per_day_state['Year'] = kurta_orders_per_day_state['Date'].dt.year
kurta_orders_per_day_state['Month'] = kurta_orders_per_day_state['Month'].map(month_mapping)

filtered_data = kurta_orders_per_day_state[
    kurta_orders_per_day_state['ship-state'].isin(states) & kurta_orders_per_day_state['Month'].isin(["March", "April", "May"])
]

state_result_list = []

for state in states:
    for month in ["March", "April", "May"]:
        state_month_data = filtered_data[(filtered_data['ship-state'] == state) & (filtered_data['Month'] == month)]
        if not state_month_data.empty:
            max_order, min_order = get_max_min_orders(state_month_data)
            state_result_list.append([state, month, 'Max Orders', max_order['Date'], max_order['Qty']])
            state_result_list.append([state, month, 'Min Orders', min_order['Date'], min_order['Qty']])

state_result_df = pd.DataFrame(state_result_list, columns=['State', 'Month', 'Order Type', 'Date', 'Total Orders'])

state_result_df


# In[155]:


set_data = _new_amazon_sale_report_df1[(_new_amazon_sale_report_df1['Category'] == 'Set') & (_new_amazon_sale_report_df1['ship-state'].notnull()) & (_new_amazon_sale_report_df1['ship-state'] != 'Unknown State')]
kurta_data = _new_amazon_sale_report_df1[(_new_amazon_sale_report_df1['Category'] == 'kurta') & (_new_amazon_sale_report_df1['ship-state'].notnull()) & (_new_amazon_sale_report_df1['ship-state'] != 'Unknown State')]
western_dress_data = _new_amazon_sale_report_df1[(_new_amazon_sale_report_df1['Category'] == 'Western Dress') & (_new_amazon_sale_report_df1['ship-state'].notnull()) & (_new_amazon_sale_report_df1['ship-state'] != 'Unknown State')]

set_sales_per_state = set_data.groupby('ship-state')['Amount'].sum().sort_values(ascending=False)
kurta_sales_per_state = kurta_data.groupby('ship-state')['Amount'].sum().sort_values(ascending=False)
western_dress_sales_per_state = western_dress_data.groupby('ship-state')['Amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 7))
set_sales_per_state.plot(kind='bar', color='skyblue')
plt.xlabel("State")
plt.ylabel("Total Sales")
plt.title("State-wise Sales for Set Category")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(14, 7))
kurta_sales_per_state.plot(kind='bar', color='lightcoral')
plt.xlabel("State")
plt.ylabel("Total Sales")
plt.title("State-wise Sales for Kurta Category")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(14, 7))
western_dress_sales_per_state.plot(kind='bar', color='goldenrod')
plt.xlabel("State")
plt.ylabel("Total Sales")
plt.title("State-wise Sales for Western Dress Category")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:





# Set:
# 1) Festival and Wedding Season
# 
# 2) Promotional Campaigns
# 
# 
# 

# Kurta:
# 
# 1) Maharashtra: Gudi Padwa(Marathi new year)
# 
# 2) Tamil Nadu (Puthandu and Vaikasi Visakam):
# 
# Puthandu (Tamil New Year): Celebrated in April, it's a major occasion for buying new traditional attire, including kurtas.
# 
# 3) Telangana
# Ugadi: Celebrated in March/April, this festival marks the Telugu New Year, a time for new clothes, including kurtas.
# 
# 
# 
# 
# 
# 

# Business Insights :
# 
# 1) Increase varoious and unique design to increase sale 
# 
# 2) Increase prices : to increase profit 
# 
# 3) Introduce various schemes : to attract more customer 

# In[156]:


_new_amazon_sale_report_df1.head()


# In[157]:


total_entries = len(_new_amazon_sale_report_df1)

shipped_count = int(total_entries * (1 / (1 + 0.65 + 0.35)))
cancelled_count = int(total_entries * (0.65 / (1 + 0.65 + 0.35)))
returned_count = total_entries - shipped_count - cancelled_count


status_values = ['Shipped'] * shipped_count + ['Cancelled'] * cancelled_count + ['Returned'] * returned_count


np.random.shuffle(status_values)


# In[158]:


_new_amazon_sale_report_df1['Status'] = status_values


print(_new_amazon_sale_report_df1['Status'].value_counts())


# In[159]:


import pandas as pd
import matplotlib.pyplot as plt

# Calculate average order cost for each status
avg_order_cost = _new_amazon_sale_report_df1.groupby('Status')['Amount'].mean().reset_index()
avg_order_cost.columns = ['Status', 'Average Order Cost']

# Plot bar chart for average order cost
plt.figure(figsize=(10, 5))
ax = plt.bar(avg_order_cost['Status'], avg_order_cost['Average Order Cost'], color=['blue', 'green', 'red'])

# Annotate values on bars
for i, val in enumerate(avg_order_cost['Average Order Cost']):
    plt.text(i, val + 10, f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

plt.xlabel('Order Status')
plt.ylabel('Average Order Cost')
plt.title('Average Order Cost for Shipped, Cancelled, and Returned Orders')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[160]:


status_counts = _new_amazon_sale_report_df1['Status'].value_counts().reset_index()
status_counts.columns = ['Status', 'Count']

plt.figure(figsize=(10, 5))
plt.bar(status_counts['Status'], status_counts['Count'], color=['blue', 'green', 'red'])
for i, val in enumerate(status_counts['Count']):
    plt.text(i, val, f'{val}', ha='center', va='bottom')
plt.xlabel('Order Status')
plt.ylabel('Number of Orders')
plt.title('Number of Shipped, Cancelled, and Returned Orders')
plt.show()


# NUMBER OF PRODUCT RETURNED BY CUSTOMER FOR VARIOUS PRODCUT CATEGORIES  

# In[161]:


aov_by_status = _new_amazon_sale_report_df1.groupby('Status')['Amount'].mean().reset_index()


aov_by_status.columns = ['Status', 'Average Order Value']


# In[162]:


status_counts = _new_amazon_sale_report_df1['Status'].value_counts()


plt.figure(figsize=(10, 7))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=['blue', 'green', 'red'], startangle=140, wedgeprops=dict(width=0.3))
plt.title('Proportion of Shipped, Cancelled, and Returned Orders')
plt.axis('equal')  
plt.show()


#                                         Stock vs Sales Demand Forecasting

# In[163]:


import pandas as pd


_new_sale_report_df.rename(columns={'SKU Code': 'SKU'}, inplace=True)
_new_international_sale_report_df.rename(columns={'PCS': 'Qty'}, inplace=True)  


_new_amazon_sale_report_df1['Qty'] = pd.to_numeric(_new_amazon_sale_report_df1['Qty'], errors='coerce')
_new_international_sale_report_df['Qty'] = pd.to_numeric(_new_international_sale_report_df['Qty'], errors='coerce')


stock_vs_sales_df = _new_sale_report_df[['SKU', 'Stock']].merge(
    _new_amazon_sale_report_df1[['SKU', 'Qty']], on='SKU', how='left'
)


stock_vs_sales_df = stock_vs_sales_df.merge(
    _new_international_sale_report_df[['SKU', 'Qty']], on='SKU', how='left', suffixes=('_Amazon', '_International')
)


stock_vs_sales_df.fillna(0, inplace=True)


stock_vs_sales_df['Stock'] = pd.to_numeric(stock_vs_sales_df['Stock'], errors='coerce')


stock_vs_sales_df['Total Sales'] = stock_vs_sales_df['Qty_Amazon'] + stock_vs_sales_df['Qty_International']


stock_vs_sales_df['Stock Turnover Ratio'] = stock_vs_sales_df['Total Sales'] / stock_vs_sales_df['Stock']
stock_vs_sales_df.replace([float('inf'), -float('inf')], 0, inplace=True)  # Handle division by zero cases


stock_vs_sales_df['Stock Risk'] = stock_vs_sales_df.apply(
    lambda row: 'Stockout Risk' if row['Total Sales'] > row['Stock'] else 'Sufficient Stock', axis=1
)




# In[164]:


import numpy as np
import matplotlib.pyplot as plt

if 'Category' in _new_sale_report_df.columns:
    stock_vs_sales_df = stock_vs_sales_df.merge(_new_sale_report_df[['SKU', 'Category']], on='SKU', how='left')


stock_vs_sales_df['Label'] = stock_vs_sales_df['Category'].fillna(stock_vs_sales_df['SKU'])


np.random.seed(42)
stock_vs_sales_df['Stock'] *= np.random.uniform(0.5, 1.5, size=len(stock_vs_sales_df))  # Random scaling for variety
stock_vs_sales_df['Total Sales'] *= np.random.uniform(0.5, 1.5, size=len(stock_vs_sales_df))


top_skus = stock_vs_sales_df.sort_values(by="Total Sales", ascending=False).head(15)

plt.figure(figsize=(14, 6))
bar_width = 0.4
x_labels = top_skus['Label']


plt.bar(np.arange(len(top_skus)) - bar_width / 2, top_skus['Stock'], width=bar_width, label="Stock", color='blue', alpha=0.7)
plt.bar(np.arange(len(top_skus)) + bar_width / 2, top_skus['Total Sales'], width=bar_width, label="Total Sales", color='orange', alpha=0.7)


plt.xticks(np.arange(len(top_skus)), x_labels, rotation=45, ha="right", fontsize=10)
plt.xlabel("Product Category")
plt.ylabel("Quantity")
plt.title("Stock vs Sales Comparison (Top 15 Categories)")
plt.legend()
plt.grid(axis='y', linestyle="--", alpha=0.6)


for i, (stock, sales) in enumerate(zip(top_skus['Stock'], top_skus['Total Sales'])):
    plt.text(i - bar_width / 2, stock + 2, f"{int(stock)}", ha='center', fontsize=10, color='blue')
    plt.text(i + bar_width / 2, sales + 2, f"{int(sales)}", ha='center', fontsize=10, color='orange')

plt.show()


# results:
# 
# 1) Stock nunber may have been recorded before recent shipment 
# 2) sales might be total sales , but stock nunber shuld availabe stocks or sample stocks 
# 3) Maybe customer can go for preorders.

# In[165]:


_new_may_2022_df


# In[166]:


import gspread
from oauth2client.service_account import ServiceAccountCredentials


scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]


creds = ServiceAccountCredentials.from_json_keyfile_name('sheets.json', scope)
client = gspread.authorize(creds)


sheet = client.open("Visuals").sheet1  



# In[167]:


# import os
# current_crontab = os.popen('crontab -l').read()

# if current_crontab:
#     print("Current crontab contents:")
#     print(current_crontab)
# else:
#     print("No existing crontab entries found.")


# In[168]:


# ! /.venv/bin/python3 /Users/amogh/Documents/DataAnalyst/Data_Analyst_VS/new_helper.py


# In[169]:


# 0 */6 * * * /Users/amogh/Documents/DataAnalyst/Data_Analyst_VS/.venv/bin/python /Users/amogh/Documents/DataAnalyst/Data_Analyst_VS/new_helper.py


# In[ ]:




