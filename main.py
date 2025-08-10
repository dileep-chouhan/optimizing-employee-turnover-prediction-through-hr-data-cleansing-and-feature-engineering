import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
n_employees = 500
data = {
    'Age': np.random.randint(20, 60, size=n_employees),
    'Department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR'], size=n_employees),
    'YearsExperience': np.random.randint(0, 20, size=n_employees),
    'Salary': np.random.randint(40000, 120000, size=n_employees),
    'Satisfaction': np.random.randint(1, 11, size=n_employees), # 1-10 scale
    'WorkLifeBalance': np.random.randint(1, 11, size=n_employees), # 1-10 scale
    'Turnover': np.random.choice([0, 1], size=n_employees, p=[0.8, 0.2]) # 0: No Turnover, 1: Turnover
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
#Handle missing data (although synthetic data doesn't have missing values, this is a good practice)
#In real-world scenarios, you would replace this with appropriate imputation or removal techniques.
df.fillna(df.mean(), inplace=True)
# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Department'], drop_first=True)
#Feature Engineering: Create a composite score
df['OverallScore'] = (df['Satisfaction'] + df['WorkLifeBalance']) / 2
# --- 3. Data Splitting ---
X = df.drop('Turnover', axis=1)
y = df['Turnover']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 4. Model Training (Simple Logistic Regression for demonstration) ---
model = LogisticRegression()
model.fit(X_train, y_train)
# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
# --- 6. Visualization ---
plt.figure(figsize=(8, 6))
sns.countplot(x='Turnover', data=df)
plt.title('Employee Turnover Distribution')
plt.xlabel('Turnover (0: No, 1: Yes)')
plt.ylabel('Number of Employees')
plt.savefig('turnover_distribution.png')
print("Plot saved to turnover_distribution.png")
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")