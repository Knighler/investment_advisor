import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv("XAU_1Month_data.csv")
df = df['Date;Open;High;Low;Close;Volume'].str.split(';', expand=True)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Year'] = df['Date'].dt.year
gold_yearly = df.groupby('Year')['Close'].mean().reset_index()

X = gold_yearly[['Year']]
y = gold_yearly['Close']
model = LinearRegression()
model.fit(X, y)

future_years = np.array([2025, 2026, 2027]).reshape(-1, 1)
predicted_prices = model.predict(future_years)

price_2024 = gold_yearly[gold_yearly['Year'] == 2024]['Close'].values[0]
gold_roi_percent = ((predicted_prices - price_2024) / price_2024) * 100
usd_roi_percent = np.array([-3, -6, -9])

while True:
    try:
        budget = float(input("Enter your investment budget in USD: "))
        if budget <= 0:
            raise ValueError
        break
    except ValueError:
        print("Please enter a valid positive number.")

print("\n Investment ROI & Profit Comparison (from 2024)")
for i, year in enumerate([2025, 2026, 2027]):
    gold_roi = gold_roi_percent[i]
    usd_roi = usd_roi_percent[i]
    gold_profit = budget * gold_roi / 100
    usd_loss = budget * abs(usd_roi) / 100

    print(f"\nYear: {year}")
    print(f"  Gold expected ROI : {gold_roi:.2f}% → Profit: ${gold_profit:.2f}")
    print(f"  USD expected ROI  : {usd_roi:.2f}% → Loss  : ${usd_loss:.2f}")
    if gold_roi > abs(usd_roi):
        print("  ✅ Recommendation: INVEST in GOLD")
    else:
        print("  ✅ Recommendation: HOLD USD")

plt.plot(gold_yearly['Year'], gold_yearly['Close'], label='Historical Gold Price', marker='o')
plt.plot(future_years.flatten(), predicted_prices, 'ro--', label='Predicted Price (2025–2027)')
plt.axhline(y=price_2024, color='gray', linestyle='--', label='2024 Base Price')
plt.xlabel("Year")
plt.ylabel("Gold Price (USD)")
plt.title("Gold Price Trend Forecast")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
