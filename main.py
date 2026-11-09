import pandas as pd

file_path = "data/online retail.xlsx"

sales_data = pd.read_excel(file_path)

print("Welcome to TrendNest!")
print("AI-Powered Retail Demand Forecasting")

print("\nFirst 5 rows of the dataset:")
print(sales_data.head())

print("\nColumn names:")
print(sales_data.columns.tolist())

print("\nDataset size:")
print(f"Rows: {sales_data.shape[0]}")
print(f"Columns: {sales_data.shape[1]}")
print("\n--- Data Quality Check ---")

print("\nMissing values in each column:")
print(sales_data.isnull().sum())

print("\nNumber of cancelled invoices:")
print(sales_data["InvoiceNo"].astype(str).str.startswith("C").sum())

print("\nSales quantity summary:")
print(sales_data["Quantity"].describe())


print("\n--- Additional Data Quality Checks ---")

# Check for duplicate rows
print("Exact duplicate rows:", sales_data.duplicated().sum())

# Check for missing product codes
print("Missing stock codes:", sales_data["StockCode"].isnull().sum())

# Check whether invoice dates are valid
valid_dates = pd.to_datetime(
    sales_data["InvoiceDate"], errors="coerce"
)
print("Invalid invoice dates:", valid_dates.isnull().sum())

# Check whether quantities and prices are valid numbers
quantities = pd.to_numeric(
    sales_data["Quantity"], errors="coerce"
)
prices = pd.to_numeric(
    sales_data["UnitPrice"], errors="coerce"
)

print("Invalid quantities:", quantities.isnull().sum())
print("Invalid unit prices:", prices.isnull().sum())

# Check for zero or negative prices
print("Zero or negative unit prices:", (prices <= 0).sum())

# Display the 10 largest quantities for investigation
print("Largest 10 quantities:")
print(sales_data["Quantity"].nlargest(10).tolist())

print("\n--- Cleaning the Dataset ---")

# Make a copy so the original data remains unchanged
clean_data = sales_data.copy()
# Remove exact duplicate rows
clean_data = clean_data.drop_duplicates()

# Remove rows without a product description
clean_data = clean_data.dropna(subset=["Description"])

# Exclude cancelled invoices
clean_data = clean_data[
    ~clean_data["InvoiceNo"].astype(str).str.startswith("C")
]

# Keep only positive quantities
clean_data = clean_data[clean_data["Quantity"] > 0]
# Keep only transactions with a positive unit price
clean_data = clean_data[clean_data["UnitPrice"] > 0]
# Check zero or negative prices in the cleaned data
print(
    "Zero or negative prices after cleaning:",
    (clean_data["UnitPrice"] <= 0).sum()
)
# Inspect transactions with zero or negative prices
print("\nExamples of zero or negative price records:")
print(
    clean_data.loc[
        clean_data["UnitPrice"] <= 0,
        ["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice"]
    ].head(10)
)
# Display the cleaning results
print("Original number of rows:", len(sales_data))
print("Rows after cleaning:", len(clean_data))
print("Rows removed:", len(sales_data) - len(clean_data))

print("\nFirst 5 rows of cleaned data:")
print(clean_data.head())
# Save the cleaned dataset
clean_data.to_csv("data/cleaned_online_retail.csv", index=False)

print("Cleaned dataset saved successfully!")
# Prepare daily sales data
clean_data["InvoiceDate"] = pd.to_datetime(clean_data["InvoiceDate"])

clean_data["Date"] = clean_data["InvoiceDate"].dt.date

daily_sales = clean_data.groupby(
    ["Date", "StockCode", "Description"]
)["Quantity"].sum().reset_index()

# Display the daily sales data
print("\nFirst 10 rows of daily sales:")
print(daily_sales.head(10))

print("\nTotal daily sales records:", len(daily_sales))
# Save daily sales data
daily_sales.to_csv("data/daily_sales.csv", index=False)

print("Daily sales data saved successfully!")
# Find products with the most daily sales records
product_counts = daily_sales.groupby(
    ["StockCode", "Description"]
).size().sort_values(ascending=False)

print("\nTop 5 products by number of sales records:")
print(product_counts.head(5))