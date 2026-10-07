import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

file_path = "ec2dataset.csv"
data = pd.read_csv(file_path)

print(data.info())
print(data.head())

cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace("[$, hourly]", "", regex=True),
        errors="coerce"
    )

print(data[cost_columns].isnull().sum())

cost_summary = data[cost_columns].describe()
print(cost_summary)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.boxplot(data=data[cost_columns], palette="Set2")

plt.title("Cost Comparison of Amazon EC2 Instances (Hourly)", fontsize=16)
plt.ylabel("Cost (USD)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=12)

plt.tight_layout()
plt.savefig("ec2_cost_boxplot.png")
plt.close()

print("Boxplot saved as ec2_cost_boxplot.png")

def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

outliers_on_demand = detect_outliers("On Demand")
print(outliers_on_demand)

cost_comparison = data[
    ["Name", "On Demand", "Linux Reserved cost"]
].dropna().sort_values("On Demand")

print(cost_comparison.head(10))

def filter_instance_family(family):
    return data[data["Name"].str.startswith(family)]

t2_instances = filter_instance_family("T2")
t3_instances = filter_instance_family("T3")

t2_summary = t2_instances[cost_columns].describe()
t3_summary = t3_instances[cost_columns].describe()

print("T2 Instance Costs Summary:\n", t2_summary)
print("\nT3 Instance Costs Summary:\n", t3_summary)

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=t2_instances[cost_columns],
    palette="Blues",
    showmeans=True
)

plt.title("Cost Distribution for T2 Instances", fontsize=16)
plt.ylabel("Cost (USD)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=12)

plt.tight_layout()
plt.savefig("t2_cost_distribution.png")
plt.close()

print("T2 cost distribution saved as t2_cost_distribution.png")

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=t3_instances[cost_columns],
    palette="Greens",
    showmeans=True
)

plt.title("Cost Distribution for T3 Instances", fontsize=16)
plt.ylabel("Cost (USD)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=12)

plt.tight_layout()
plt.savefig("t3_cost_distribution.png")
plt.close()

print("T3 cost distribution saved as t3_cost_distribution.png")

comparison = pd.concat([
    t2_instances[["Name", "On Demand", "Linux Reserved cost"]],
    t3_instances[["Name", "On Demand", "Linux Reserved cost"]]
])

comparison_sorted = comparison.dropna().sort_values("On Demand")

print(comparison_sorted.head(10))

data["Instance Memory"] = pd.to_numeric(
    data["Instance Memory"].str.replace(" GiB", ""),
    errors="coerce"
)

data["vCPUs"] = pd.to_numeric(
    data["vCPUs"].str.extract(r"(\d+)")[0],
    errors="coerce"
)

print(data[["Instance Memory", "vCPUs"]].head())

data_cleaned = data.dropna(
    subset=["On Demand", "Instance Memory", "vCPUs"]
)

print(data_cleaned.isnull().sum())

X = data_cleaned[["Instance Memory", "vCPUs"]]
y = data_cleaned["On Demand"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(
    "Training samples:",
    len(X_train),
    "Testing samples:",
    len(X_test)
)

model = LinearRegression(
    positive=True,
    fit_intercept=False
)

model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7)

max_cost = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [0, max_cost],
    [0, max_cost]
)

plt.title("Actual vs Predicted On-Demand Costs")
plt.xlabel("Actual On-Demand Cost")
plt.ylabel("Predicted On-Demand Cost")

plt.tight_layout()
plt.savefig("actual_vs_predicted_cost.png")
plt.close()

print("Prediction graph saved as actual_vs_predicted_cost.png")

new_instance = pd.DataFrame(
    {
        "Instance Memory": [4],
        "vCPUs": [2]
    }
)

predicted_cost = model.predict(new_instance)

print(
    f"Predicted On-Demand Cost for 4 GiB, 2 vCPUs: "
    f"${predicted_cost[0]:.4f}"
)