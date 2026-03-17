🚀 Salary Prediction using Linear Regression
📌 Project Overview
This project implements a Simple Linear Regression model to predict the salary of a professional based on their years of experience. This is a classic "Supervised Learning" problem where we use historical data to find a mathematical relationship between an independent variable (Experience) and a dependent variable (Salary).
🛠️ Key Machine Learning Steps Followed:
Data Preparation: Created a dataset using Pandas representing the correlation between work experience and annual pay.
Feature Selection: Defined the Independent Variable (
X
X
 = Years of Experience) and the Target Variable (
y
y
 = Salary).
Train-Test Split: Divided the data into 80% Training (to teach the model) and 20% Testing (to evaluate the model's performance on unseen data). This is crucial to prevent "overfitting."
Model Training: Used the LinearRegression algorithm from Scikit-Learn to find the "Best Fit Line."
Evaluation: Measured the model's accuracy using the R2 Score (Coefficient of Determination) and Mean Squared Error (MSE).
Prediction: Built a custom function to predict salaries for any given years of experience.
Visualization: Used Matplotlib to plot the training/testing data points and the resulting regression line.
💻 Installation & Usage
1. Prerequisites
Ensure you have Python installed. You will need the following libraries:
code
Bash
pip install pandas matplotlib scikit-learn
2. Running the script
code
Bash
python salary_model.py
📊 Results & Observations
Best Fit Line: The model successfully generated a linear trend line that minimizes the "Residuals" (distance between actual and predicted points).
Predictive Power: By passing 5.5 years of experience into the model, it returns a realistic salary estimate based on the learned patterns.
Visualization: The resulting plot clearly distinguishes between Training data (Blue) and Testing data (Green), showing how well the Red Line fits both.
