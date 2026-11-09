import pandas as pd

# Load the daily sales dataset
daily_sales = pd.read_csv("data/daily_sales.csv")
# Convert Date column into datetime format
daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])

print("\nDate column converted successfully!")
print(daily_sales.dtypes)
# Display the dataset information
print("Welcome to TrendNest Forecasting!")

print("\nFirst 5 records:")
print(daily_sales.head())

print("\nDataset information:")
print(daily_sales.info())

print("\nTotal records:", len(daily_sales))
# Find the top 5 products by number of daily records
product_counts = daily_sales.groupby(
    ["StockCode", "Description"]
).size().sort_values(ascending=False)

print("\nTop 5 products:")
print(product_counts.head(5))
# Select the product for forecasting
selected_product = daily_sales[
    daily_sales["StockCode"] == "85123A"
].copy()

# Sort sales by date
selected_product = selected_product.sort_values("Date")

print("\nSelected product:")
print("WHITE HANGING HEART T-LIGHT HOLDER")

print("\nNumber of daily records:")
print(len(selected_product))

print("\nProduct sales history:")
print(selected_product.head(10))
# Prepare daily time series
selected_product = selected_product.set_index("Date")

# Create a continuous daily date range
selected_product = selected_product.resample("D")["Quantity"].sum()

# Fill days without recorded sales with zero
selected_product = selected_product.fillna(0)

print("\nPrepared daily sales history:")
print(selected_product.head(10))

print("\nTotal number of calendar days:", len(selected_product))

# Create features using previous days' sales
forecast_data = pd.DataFrame(selected_product)

forecast_data["Yesterday"] = forecast_data["Quantity"].shift(1)
forecast_data["Last 7 Days Average"] = (
    forecast_data["Quantity"].shift(1).rolling(window=7).mean()
)

# Remove rows that do not have enough previous sales data
forecast_data = forecast_data.dropna()

print("\nForecasting features:")
print(forecast_data.head(10))

print("\nTotal records available for forecasting:")
print(len(forecast_data))

# Split the data into training and testing sets
train_data = forecast_data.iloc[:-30]
test_data = forecast_data.iloc[-30:]

# Select the input features
X_train = train_data[["Yesterday", "Last 7 Days Average"]]
X_test = test_data[["Yesterday", "Last 7 Days Average"]]

# Select the target (what the model must predict)
y_train = train_data["Quantity"]
y_test = test_data["Quantity"]

print("\nTraining and testing data prepared!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Import the machine-learning model
from sklearn.ensemble import RandomForestRegressor

# Create the forecasting model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions using the testing data
predictions = model.predict(X_test)

print("\nTrendNest forecasting model trained successfully!")

print("\nFirst 10 predicted sales quantities:")
print(predictions[:10])

print("\nFirst 10 actual sales quantities:")
print(y_test.head(10).values)

# Import evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate model performance
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nTRENDNEST MODEL PERFORMANCE")
print("---------------------------")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

# Save model evaluation results
evaluation_results = pd.DataFrame({
    "Metric": ["MAE", "RMSE"],
    "Value": [mae, rmse]
})

evaluation_results.to_csv(
    "data/model_evaluation.csv",
    index=False
)

print("\nModel evaluation results saved successfully!")

# Generate a 7-day future forecast

# Get the most recent sales information
last_sales = list(selected_product.values)

future_predictions = []

# Predict the next 7 days
for day in range(7):

    yesterday = last_sales[-1]
    last_7_average = sum(last_sales[-7:]) / 7

    # Prepare input for the model
    next_input = pd.DataFrame({
        "Yesterday": [yesterday],
        "Last 7 Days Average": [last_7_average]
    })

    # Predict sales
    next_prediction = model.predict(next_input)[0]

    # Save prediction
    future_predictions.append(round(next_prediction, 2))

    # Add prediction to history for the next day
    last_sales.append(next_prediction)

# Display the forecast
print("\nTRENDNEST - 7 DAY SALES FORECAST")
print("--------------------------------")

for day, prediction in enumerate(future_predictions, start=1):
    print(f"Day {day}: {prediction} units")

# Save the 7-day forecast
forecast_results = pd.DataFrame({
    "Day": [f"Day {day}" for day in range(1, 8)],
    "Predicted Quantity": future_predictions
})

forecast_results.to_csv(
    "data/7_day_forecast.csv",
    index=False
)

print("\n7-day forecast saved successfully!")
