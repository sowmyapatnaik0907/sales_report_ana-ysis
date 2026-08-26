import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("sales_data.csv")

# Convert date
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Calculate gross sales
df["Gross_Sales"] = df["Quantity"] * df["Unit_Price"]

# Calculate discount amount
df["Discount_Amount"] = df["Gross_Sales"] * df["Discount"]

# Calculate final sales
df["Net_Sales"] = df["Gross_Sales"] - df["Discount_Amount"]

# Extract date information
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.strftime("%B")

print(df.head())

print("\nTotal Gross Sales:")
print(df["Gross_Sales"].sum())

print("\nTotal Net Sales:")
print(df["Net_Sales"].sum())

print("\nAverage Order Value:")
print(df["Net_Sales"].mean()) 

#==total revenue====
total_revenue = df["Net_Sales"].sum()

print("Total Revenue:", total_revenue)
#==product category highest revenue
category_sales = df.groupby("Category")["Net_Sales"].sum()

print(category_sales)

print("\nBest Category:")
print(category_sales.idxmax())
#==city generates most sales
city_sales = df.groupby("City")["Net_Sales"].sum()

print(city_sales)

print("\nBest City:")
print(city_sales.idxmax())


#==best selling product==
product_sales = df.groupby("Product")["Net_Sales"].sum()

print(product_sales)

print("\nBest Product:")
print(product_sales.idxmax())

#==which salesmans generates highest value
salesperson_sales = df.groupby("Salesperson")["Net_Sales"].sum()

print(salesperson_sales)

print("\nTop Salesperson:")
print(salesperson_sales.idxmax())

#==which payment method is most popular
payment_count = df["Payment_Mode"].value_counts()

print(payment_count)

print("\nMost Popular Payment Method:")
print(payment_count.idxmax())

#percentage of orders return
return_rate = (df["Returned"] == "Yes").mean() * 100

print("Return Rate:", return_rate, "%")

#==does discounts affect sales
discount_analysis = df.groupby("Discount")["Net_Sales"].mean()

print(discount_analysis)

#which age group spends the most
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 45, 100],
    labels=["18-25", "26-35", "36-45", "46+"]
)

age_sales = df.groupby("Age_Group", observed=True)["Net_Sales"].sum()

print(age_sales)

print("\nHighest Spending Age Group:")
print(age_sales.idxmax())

#==which month have highest revenue
monthly_sales = df.groupby("Month_Name")["Net_Sales"].sum()

print(monthly_sales)

#==category sales
plt.figure(figsize=(8, 5))

sns.barplot(
    x=category_sales.index,
    y=category_sales.values
)

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Net Sales")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#==city sales
plt.figure(figsize=(8, 5))

sns.barplot(
    x=city_sales.index,
    y=city_sales.values
)

plt.title("Sales by City")
plt.xlabel("City")
plt.ylabel("Net Sales")

plt.tight_layout()
plt.show()

#==revenue distribtion
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Net_Sales"],
    kde=True
)

plt.title("Revenue Distribution")
plt.xlabel("Net Sales")

plt.tight_layout()
plt.show()

#==correlative analysis
numeric_columns = [
    "Age",
    "Quantity",
    "Unit_Price",
    "Discount",
    "Customer_Rating",
    "Gross_Sales",
    "Net_Sales"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()
