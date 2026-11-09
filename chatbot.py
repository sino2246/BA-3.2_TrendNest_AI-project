
import pandas as pd

# Load forecasting results
forecast = pd.read_csv("data/7_day_forecast.csv")
evaluation = pd.read_csv("data/model_evaluation.csv")

print("====================================")
print("       WELCOME TO TRENDNEST AI")
print("  Your Retail Demand Forecasting Bot")
print("====================================")

print("\nHello! I am the TrendNest chatbot.")
print("I can answer questions about your sales forecast.")
print("\nType 'help' to see what I can answer.")
print("Type 'exit' to leave the chatbot.")

while True:

    question = input("\nYou: ").lower().strip()

    if question == "exit":
        print("TrendNest: Goodbye! Thank you for using TrendNest.")
        break

    elif question == "help":
        print("\nYou can ask me:")
        print("- What product are we forecasting?")
        print("- What is the total predicted demand?")
        print("- Which day has the highest demand?")
        print("- Which day has the lowest demand?")
        print("- What is MAE?")
        print("- What is RMSE?")

    elif "what product" in question:
        print("TrendNest: WHITE HANGING HEART T-LIGHT HOLDER")

    elif "total" in question or "predicted demand" in question:
        total = forecast["Predicted Quantity"].sum()
        print(f"TrendNest: The predicted 7-day demand is {total:.2f} units.")

    elif "highest" in question:
        highest = forecast.loc[forecast["Predicted Quantity"].idxmax()]
        print(
            f"TrendNest: The highest predicted demand is "
            f"{highest['Predicted Quantity']:.2f} units on {highest['Day']}."
        )

    elif "lowest" in question:
        lowest = forecast.loc[forecast["Predicted Quantity"].idxmin()]
        print(
            f"TrendNest: The lowest predicted demand is "
            f"{lowest['Predicted Quantity']:.2f} units on {lowest['Day']}."
        )

    elif "mae" in question:
        mae = evaluation.loc[evaluation["Metric"] == "MAE", "Value"].iloc[0]
        print(f"TrendNest: The Mean Absolute Error (MAE) is {mae:.2f}.")

    elif "rmse" in question:
        rmse = evaluation.loc[evaluation["Metric"] == "RMSE", "Value"].iloc[0]
        print(f"TrendNest: The Root Mean Squared Error (RMSE) is {rmse:.2f}.")

    else:
        print("TrendNest: Sorry, I don't understand that question yet.")
        print("Type 'help' to see the questions I can answer.")
