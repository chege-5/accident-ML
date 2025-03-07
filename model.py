import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

file_path = "acc.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=["Accident_Severity"])
y = df["Accident_Severity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nMean Squared Error: {mse:.4f}\nR-squared Score: {r2:.4f}")

joblib.dump(model, "accident_severity_model.pkl")
print("\nModel saved as accident_severity_model.pkl")

example_data = np.array([50, 1, 1, 3, 1])
example_df = pd.DataFrame([example_data], columns=X.columns)

predicted_severity = model.predict(example_df)

print(f"\nPredicted Accident Severity: {predicted_severity[0]:.2f}")

coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
coefficients.sort_values(by='Coefficient', ascending=False, inplace=True)

plt.figure(figsize=(8, 5))
sns.barplot(x=coefficients.index, y=coefficients["Coefficient"], palette="viridis")
plt.xticks(rotation=45)
plt.title("Feature Importance in Predicting Accident Severity")
plt.savefig("feature_importance.png")
plt.show()
