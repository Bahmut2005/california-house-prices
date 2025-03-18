# california-house-prices
import numpy as np
import pandas as pd

# Generate synthetic dataset with 10,000 rows and 6 columns
np.random.seed(42)
data = {
    "MedInc": np.random.uniform(1, 15, 10000),  # Median income in $10,000s
    "HouseAge": np.random.randint(1, 52, 10000),  # Age of the house in years
    "AveRooms": np.random.uniform(1, 10, 10000),  # Average number of rooms
    "AveBedrms": np.random.uniform(1, 5, 10000),  # Average number of bedrooms
    "Population": np.random.randint(100, 5000, 10000),  # Population of the area
    "Price": np.random.uniform(0.5, 5.0, 10000) * 100000  # Price in $100,000s
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_path = "/mnt/data/california_housing_synthetic.csv"
df.to_csv(csv_path, index=False)

# Display confirmation
csv_path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(csv_path)

# Plot 4 different types of visualizations
plt.figure(figsize=(12, 6))
sns.histplot(df['Price'], bins=30, kde=True, color='blue')
plt.title("Distribution of House Prices")
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x=df['MedInc'], y=df['Price'], alpha=0.5)
plt.title("Median Income vs House Prices")
plt.xlabel("Median Income ($10,000s)")
plt.ylabel("House Price ($)")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

sns.pairplot(df.sample(500), vars=['MedInc', 'HouseAge', 'AveRooms', 'Price'])
plt.show()

# Data Preprocessing
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2 Score": r2_score(y_test, y_pred)
    })

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)

# Display results
import ace_tools as tools
tools.display_dataframe_to_user(name="ML Model Performance", dataframe=results_df)
