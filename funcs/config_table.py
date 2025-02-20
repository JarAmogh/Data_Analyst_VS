import pandas as pd
import random

def load_data():
    return pd.read_csv("amazon_sale_report.csv", low_memory=False)

def state_order_counts(df):
    metro_cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
    non_metro_cities = ["Jaipur", "Lucknow", "Indore", "Bhopal", "Chandigarh", "Patna", "Nagpur", "Vadodara", "Coimbatore"]

    def classify_city(city):
        if city in metro_cities:
            return "Metro"
        elif city in non_metro_cities:
            return "Non-Metro"
        else:
            return "Other"

    state_mapping = {
        "Mumbai": "Maharashtra", "Delhi": "Delhi", "Bangalore": "Karnataka", "Hyderabad": "Telangana",
        "Chennai": "Tamil Nadu", "Kolkata": "West Bengal", "Pune": "Maharashtra", "Ahmedabad": "Gujarat",
        "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh", "Indore": "Madhya Pradesh", "Bhopal": "Madhya Pradesh",
        "Chandigarh": "Chandigarh", "Patna": "Bihar", "Nagpur": "Maharashtra", "Vadodara": "Gujarat",
        "Coimbatore": "Tamil Nadu"
    }

    df["ship-city"] = df["ship-city"].apply(lambda x: x if pd.notna(x) else random.choice(metro_cities + non_metro_cities))
    df["ship-state"] = df["ship-city"].map(state_mapping)
    df["Amount"] = df["Amount"].apply(lambda x: x if pd.notna(x) else random.randint(200, 2000))
    df["city-type"] = df["ship-city"].apply(classify_city)

    df["currency"].fillna("INR", inplace=True)
    df["ship-country"].fillna("IN", inplace=True)
    df["ship-state"].fillna("Unknown State", inplace=True)
    df["ship-postal-code"] = df["ship-postal-code"].apply(lambda x: x if pd.notna(x) else random.randint(100000, 999999))
    df["Courier Status"].fillna("Unknown", inplace=True)
    df["fulfilled-by"].fillna("Not Specified", inplace=True)

    return df["ship-state"].value_counts().drop("Unknown State", errors="ignore")

def metro_vs_non_metro_counts(df):
    metro_cities = ["BENGALURU", "HYDERABAD", "MUMBAI", "NEW DELHI", "CHENNAI"]
    df["city_category"] = df["ship-city"].apply(lambda x: "Metro" if x in metro_cities else "Non-Metro")
    return df["city_category"].value_counts()

def avg_order_cost_per_state(df):
    return df.groupby("ship-state")["Amount"].mean().sort_values(ascending=False).dropna()

def avg_order_cost_per_city_category(df):
    metro_cities = ["BENGALURU", "HYDERABAD", "MUMBAI", "NEW DELHI", "CHENNAI"]
    df["city_category"] = df["ship-city"].apply(lambda x: "Metro" if x in metro_cities else "Non-Metro")
    return df.groupby("city_category")["Amount"].mean()