import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Sales Analytics",
    page_icon="📊",
    layout="wide"
)

# Load model
model = joblib.load("model.pkl")

# Load dataset
df = pd.read_csv("sales_data.csv")

# Create sales columns
df["Gross_Sales"] = (
    df["Quantity"] *
    df["Unit_Price"]
)

df["Discount_Amount"] = (
    df["Gross_Sales"] *
    df["Discount"]
)

df["Net_Sales"] = (
    df["Gross_Sales"] -
    df["Discount_Amount"]
)

# Title
st.title("📊 Sales Performance & Customer Insights")

st.write(
    "Data Science Hackathon Project"
)

# KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Revenue",
        f"₹{df['Net_Sales'].sum():,.0f}"
    )

with col2:
    st.metric(
        "Total Orders",
        len(df)
    )

with col3:
    st.metric(
        "Average Order Value",
        f"₹{df['Net_Sales'].mean():,.0f}"
    )

with col4:
    return_rate = (
        df["Returned"].eq("Yes").mean() * 100
    )

    st.metric(
        "Return Rate",
        f"{return_rate:.1f}%"
    )

# Category analysis
st.subheader("Sales by Category")

category_sales = (
    df.groupby("Category")["Net_Sales"]
    .sum()
)

st.bar_chart(category_sales)

# City analysis
st.subheader("Sales by City")

city_sales = (
    df.groupby("City")["Net_Sales"]
    .sum()
)

st.bar_chart(city_sales)

# Raw data
st.subheader("Dataset")

st.dataframe(df)