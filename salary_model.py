import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. THE DATA (Usually you'd load a CSV, here we create it)
data = {
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940]
}
df = pd.DataFrame(data)

# 2. FEATURE SELECTION
# X = Independent Variable (Years), y = Dependent Variable (Salary)
X = df[['YearsExperience']] 
y = df['Salary']

# 3. TRAIN-TEST SPLIT (Crucial for Placement Interviews)
# We train on 80% of data and test on 20% to see if the model actually learned.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. INITIALIZE & TRAIN MODEL
model = LinearRegression() # This uses Ordinary Least Squares (Gradient Descent logic)
model.fit(X_train, y_train)

# 5. EVALUATION
y_pred = model.predict(X_test)

print(f"Model Accuracy (R2 Score): {r2_score(y_test, y_pred):.2f}")
print(f"Mean Error: ${mean_squared_error(y_test, y_pred):.2f}")

# 6. PRACTICAL PREDICTION
def predict_my_salary(years):
    salary = model.predict([[years]])
    print(f"Predicted Salary for {years} years experience: ${salary[0]:,.2f}")

# Test it
predict_my_salary(5.5)

# 7. VISUALIZATION
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Best Fit Line')
plt.title('Salary vs Experience')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.legend()
plt.show()