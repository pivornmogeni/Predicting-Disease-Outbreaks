
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
data = pd.read_csv('data/co2_emissions.csv')

# Filter Kenya data
kenya_data = data[data['Entity'] == 'Kenya'].dropna()

# Features and target
X = kenya_data[['Year']]
y = kenya_data['Annual CO₂ emissions (tonnes )']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Kenya CO₂ Emissions Forecast')
plt.xlabel('Year')
plt.ylabel('CO₂ Emissions (tonnes)')
plt.legend()
plt.grid(True)
plt.savefig('kenya_co2_forecast_plot.png')
plt.show()
