# Lab: Introduction to Supervised Learning

> [!NOTE]  
> Throughout this lab, add text blocks to explain any new concepts you encounter. This will help reinforce your understanding and ensure clarity in your learning process. If you're unsure about a function or concept, consider writing a short definition/explanation in a text block. In addition, you can use your favorite LLM  to get more details.



---
## Part 0: Introduction and Setup

**Learning Objectives:**
By the end of this lab, you will be able to:
✔ Define **supervised learning** and differentiate it from other ML types.
✔ Identify the two main supervised learning tasks: **classification** and **regression**.
✔ Apply a **standard machine learning pipeline** for supervised tasks.
✔ Implement basic classification (Decision Tree) and regression (Linear Regression) models using `scikit-learn`.
✔ Evaluate model performance using appropriate metrics (accuracy, classification report, MAE, MSE, R²).
✔ Understand the necessity of **data preprocessing** for real-world datasets.

**1. What is Supervised Learning?**
Supervised learning is a type of machine learning where an algorithm learns from a **labeled dataset**. This means for each data point (input features), there's a known correct output (label or target). The goal is to learn a mapping function that can predict the output for new, unseen input data.

*   **Classification:** Predicts a discrete category or class label.
    *   *Example:* Is this email spam or not? (Categories: Spam, Not Spam)
    *   *Example:* Which species does this Iris flower belong to? (Categories: Setosa, Versicolor, Virginica)
*   **Regression:** Predicts a continuous numerical value.
    *   *Example:* What is the predicted mileage (MPG) of this car? (Value: e.g., 25.5 MPG)
    *   *Example:* What is the expected progression of diabetes? (Value: e.g., 150.0)

**2. The Standard Machine Learning Pipeline**
Most supervised learning projects follow these general steps:

1.  **Load Data:** Get the data into your environment (e.g., using Pandas).
2.  **Explore & Preprocess Data:** Understand the data, handle missing values, encode categorical features, scale numerical features. (*Crucial step!*)
3.  **Split Data:** Divide the data into a **training set** (to build the model) and a **testing set** (to evaluate the model on unseen data).
4.  **Choose & Create Model:** Select an appropriate algorithm (e.g., Decision Tree for classification, Linear Regression for regression).
5.  **Train Model:** Fit the model to the **training data**. The model "learns" the patterns here.
6.  **Make Predictions:** Use the trained model to predict outputs for the **testing data**.
7.  **Evaluate Model:** Compare the model's predictions against the actual known labels in the **testing data** using relevant metrics.

**💡 Important Note for this Lab:**
In Parts 1-4, we will use datasets that have *already been preprocessed* (cleaned) for you. This allows you to focus on steps 3-7 (Split, Train, Predict, Evaluate) first. In Part 5, we will review Preprocessing (i.e. Step 2) and show you how real-world data often looks and how it gets cleaned.

**3. Datasets We Will Use:**

*   **Titanic (Classification):** Predict passenger survival. (Preprocessed)
*   **Iris (Classification):** Predict iris flower species.
*   **Auto MPG (Regression):** Predict car fuel efficiency (Miles Per Gallon). (Preprocessed version in Part 3, Raw version in Part 5)
*   **Diabetes (Regression):** Predict disease progression one year after baseline.

**4. Setup:**
Make sure you have the necessary libraries installed and imported.

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_iris, load_diabetes
from sklearn.preprocessing import StandardScaler, OneHotEncoder # We'll use these later in Part 5
from sklearn.compose import ColumnTransformer # Part 5
from sklearn.pipeline import Pipeline # Part 5

# Configure plots for better visibility
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

---

### Part 1: Guided Classification (Titanic Dataset)

**Goal:** Learn the classification pipeline by predicting survival on the Titanic dataset using a Decision Tree. We'll use a pre-cleaned version.

**Step 1.1: Load Pre-Cleaned Data**
<!-- This version has handled missing values and encoded categorical features numerically. -->

```python
# Load a Titanic dataset
url_titanic = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_df = pd.read_csv(url_titanic)

# Basic cleanup often needed even with "clean" sources:
# Drop columns that are not useful features or are identifiers
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, errors='ignore')
# Fill remaining missing Age with median
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
# Convert Sex to numeric (a simple form of encoding)
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
# Drop rows with any remaining NaNs (e.g. if Fare was missing)
titanic_df.dropna(inplace=True)


print("Titanic Data (first 5 rows):")
print(titanic_df.head())
print("\nData Info:")
titanic_df.info()
```

If you would like to save the dataset, you can use:
```python
# Save as CSV
titanic_df.to_csv('my_data.csv', index=False)
```

**Step 1.2: Define Features (X) and Target (y)**
We want to predict 'Survived'. All other columns will be our features.

```python
# Define features (X) and target (y)
X_titanic = titanic_df.drop('Survived', axis=1)
y_titanic = titanic_df['Survived']

print("\nFeatures (X):")
print(X_titanic.head(2))
print("\nTarget (y):")
print(y_titanic.head(2))
```

**Step 1.3: Split Data into Training and Testing Sets**
We'll use 80% for training and 20% for testing. `random_state` ensures reproducibility.

```python
# Split the data
X_train_titanic, X_test_titanic, y_train_titanic, y_test_titanic = train_test_split(
    X_titanic, y_titanic, test_size=0.2, random_state=42, stratify=y_titanic # stratify helps keep class proportions same in splits
)

print(f"\nTraining set shape: X={X_train_titanic.shape}, y={y_train_titanic.shape}")
print(f"Testing set shape: X={X_test_titanic.shape}, y={y_test_titanic.shape}")
```

**Step 1.4: Create and Train a Decision Tree Classifier**
We initialize the model and `fit` it to the training data. `max_depth` helps prevent overfitting.

```python
# Create a Decision Tree Classifier model
# max_depth restricts the tree's complexity
clf_titanic = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the model
clf_titanic.fit(X_train_titanic, y_train_titanic)

print("\nDecision Tree Classifier model trained successfully.")
```

**Step 1.5: Make Predictions on Test Data**
Use the trained model (`clf_titanic`) to predict survival for the unseen test set (`X_test_titanic`).

```python
# Make predictions
y_pred_titanic = clf_titanic.predict(X_test_titanic)

# Display first 10 predictions vs actual values
print("\nFirst 10 Predictions:", y_pred_titanic[:10])
print("First 10 Actual Values:", y_test_titanic[:10].values)
```

**Step 1.6: Evaluate the Model**
How well did our model do? We use `accuracy_score` and `classification_report`.

```python
# Evaluate the model
accuracy_titanic = accuracy_score(y_test_titanic, y_pred_titanic)
print(f"\nModel Accuracy: {accuracy_titanic:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_titanic, y_pred_titanic, target_names=['Did not Survive', 'Survived']))
```
*   **Accuracy:** Overall percentage of correct predictions.
*   **Precision:** Of those predicted to survive, how many actually did? (TP / (TP + FP))
*   **Recall:** Of those who actually survived, how many did the model correctly identify? (TP / (TP + FN))
*   **F1-Score:** Harmonic mean of Precision and Recall, good for balancing them.

**Step 1.7: Visualize the Decision Tree (Optional)**
Let's see how the model makes decisions.

```python
# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(clf_titanic,
          feature_names=X_titanic.columns,
          class_names=['Did not Survive', 'Survived'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Titanic Survival Prediction")
plt.show()
```

---

### Part 2: Practice Classification (Iris Dataset)

**Goal:** Apply the classification pipeline steps yourself using the Iris dataset. Use the steps from Part 1 as a guide.

**Step 2.1: Load Iris Data**
The Iris dataset is built into `scikit-learn`. It's already clean and numerical.

```python
# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])

# Map target numbers to species names for clarity (optional but good practice)
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Iris Data (first 5 rows):")
print(iris_df.head())
print("\nData Info:")
iris_df.info()
```

You can save the save the dataset:
```python
# Save as CSV
iris_df.to_csv('iris.csv', index=False)
```

**Step 2.2: Define Features (X) and Target (y)**
The target is 'target' (or 'species' if you prefer, but use the numeric 'target' for the model). Features are the measurements.

```python

# Define X_iris (features) and y_iris (target - use the numeric 'target' column)
X_iris = iris_df.drop(['target', 'species'], axis=1) # Drop both target columns
y_iris = iris_df['target']

# Print shapes to verify
print(f"\nIris features shape: {X_iris.shape}")
print(f"Iris target shape: {y_iris.shape}")
```

**Step 2.3: Split Data**
Split into 80% training, 20% testing. Use `random_state=42` and `stratify=y_iris`.

```python

# Split the data into X_train_iris, X_test_iris, y_train_iris, y_test_iris
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

# Print shapes to verify
print(f"\nTraining set shape: X={X_train_iris.shape}, y={y_train_iris.shape}")
print(f"Testing set shape: X={X_test_iris.shape}, y={y_test_iris.shape}")
```

**Step 2.4: Create and Train a Decision Tree Classifier**
Use `DecisionTreeClassifier` with `max_depth=4` and `random_state=42`. Train it on the Iris training data.

```python

# Create and train the model (call the variable clf_iris)
clf_iris = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_iris.fit(X_train_iris, y_train_iris)

print("\nIris Decision Tree model trained.")
```

**Step 2.5: Make Predictions**
Predict the species for the Iris test set.

```python

# Make predictions (call the variable y_pred_iris)
y_pred_iris = clf_iris.predict(X_test_iris)

# Display first 10 predictions vs actual values
print("\nFirst 10 Predictions:", y_pred_iris[:10])
print("First 10 Actual Values:", y_test_iris[:10].values)
```

**Step 2.6: Evaluate the Model**
Calculate accuracy and print the classification report using `iris.target_names` for labels.

```python

# Evaluate the model (accuracy_iris)
accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)
print(f"\nIris Model Accuracy: {accuracy_iris:.4f}")

# Print the classification report
print("\nIris Classification Report:")
print(classification_report(y_test_iris, y_pred_iris, target_names=iris.target_names))
```

**Reflection:** How did the model perform on Iris compared to Titanic? Why might it be different? (Hint: Iris is a simpler, cleaner dataset.)

---

### Part 3: Guided Regression (Auto MPG Dataset)

**Goal:** Learn the regression pipeline by predicting car mileage (MPG) using Linear Regression. We'll use a pre-cleaned version of the Auto MPG dataset.

**Step 3.1: Load Pre-Cleaned Auto MPG Data**
This version has handled missing values (often marked as '?') and potentially encoded the 'origin' feature. The target variable is 'mpg'.

```python
import pandas as pd
# Load pre-cleaned Auto MPG dataset
# You might need to find or create a cleaned version.
# For this example, let's assume 'origin' has been handled (e.g., one-hot encoded or dropped)
# and missing 'horsepower' has been imputed.
url_mpg = "https://raw.githubusercontent.com/ML-Course-2025/session3/main/datasets/cars/auto-mpg.csv"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']
mpg_df = pd.read_csv(url_mpg, names=column_names, na_values="?", header=0)

# Basic Cleaning (as foreshadowing for Part 5)
mpg_df = mpg_df.drop('car_name', axis=1) # Drop identifier
# Handle missing horsepower
mpg_df['horsepower'].fillna(mpg_df['horsepower'].median(), inplace=True)
# Handle categorical origin (simple mapping for now, one-hot is better)
# We will drop it here for simplicity in this part, but address encoding in Part 5
mpg_df = mpg_df.drop('origin', axis=1)
mpg_df.dropna(inplace=True) # Drop any other rows if necessary


print("Auto MPG Data (first 5 rows):")
print(mpg_df.head())
print("\nData Info:")
mpg_df.info()
```

**Step 3.2: Define Features (X) and Target (y)**
We want to predict 'mpg'.

```python
# Define features (X) and target (y)
X_mpg = mpg_df.drop('mpg', axis=1)
y_mpg = mpg_df['mpg']

print("\nFeatures (X):")
print(X_mpg.head(2))
print("\nTarget (y):")
print(y_mpg.head(2))
```

**Step 3.3: Split Data**
Split into 80% training, 20% testing. Use `random_state=42`. (Stratify isn't typically used for regression).

```python
# Split the data
X_train_mpg, X_test_mpg, y_train_mpg, y_test_mpg = train_test_split(
    X_mpg, y_mpg, test_size=0.2, random_state=42
)

print(f"\nTraining set shape: X={X_train_mpg.shape}, y={y_train_mpg.shape}")
print(f"Testing set shape: X={X_test_mpg.shape}, y={y_test_mpg.shape}")
```

**Step 3.4: Create and Train a Linear Regression Model**
We initialize `LinearRegression` and `fit` it to the training data.

```python
# Create a Linear Regression model
lr_mpg = LinearRegression()

# Train the model
lr_mpg.fit(X_train_mpg, y_train_mpg)

print("\nLinear Regression model trained successfully.")

# Optional: Inspect the coefficients
# print("\nModel Coefficients:", lr_mpg.coef_)
# print("Model Intercept:", lr_mpg.intercept_)
```

**Step 3.5: Make Predictions on Test Data**
Use the trained model (`lr_mpg`) to predict MPG for the test set (`X_test_mpg`).

```python
# Make predictions
y_pred_mpg = lr_mpg.predict(X_test_mpg)

# Display first 5 predictions vs actual values
print("\nFirst 5 Predictions:", y_pred_mpg[:5])
print("First 5 Actual Values:", y_test_mpg[:5].values)
```

**Step 3.6: Evaluate the Model**
For regression, we use different metrics: MAE, MSE, and R².

```python
# Evaluate the model
mae_mpg = mean_absolute_error(y_test_mpg, y_pred_mpg)
mse_mpg = mean_squared_error(y_test_mpg, y_pred_mpg)
rmse_mpg = np.sqrt(mse_mpg) # Root Mean Squared Error
r2_mpg = r2_score(y_test_mpg, y_pred_mpg)

print(f"\nMean Absolute Error (MAE): {mae_mpg:.4f}")
print(f"Mean Squared Error (MSE): {mse_mpg:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_mpg:.4f}")
print(f"R-squared (R²): {r2_mpg:.4f}")
```
*   **MAE:** Average absolute difference between predicted and actual values. (Easy to interpret)
*   **MSE:** Average squared difference. Penalizes larger errors more.
*   **RMSE:** Square root of MSE. In the same units as the target variable.
*   **R²:** Proportion of the variance in the target variable that's predictable from the features. (Ranges from -∞ to 1, higher is better, 1 is perfect).

**Step 3.7: Visualize Predictions vs Actuals**
A scatter plot helps see how well predictions match actual values.

```python
# Plot Actual vs Predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test_mpg, y_pred_mpg, alpha=0.7)
plt.plot([y_test_mpg.min(), y_test_mpg.max()], [y_test_mpg.min(), y_test_mpg.max()], '--r', linewidth=2) # Line of perfect prediction
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs. Predicted MPG')
plt.show()

# Plot Residuals
residuals_mpg = y_test_mpg - y_pred_mpg
plt.figure(figsize=(10, 6))
sns.histplot(residuals_mpg, kde=True)
plt.xlabel('Residuals (Actual MPG - Predicted MPG)')
plt.title('Distribution of Residuals')
plt.show()
# Ideally, residuals should be normally distributed around zero.
```

**Key Takeaways from Part 3:** You've trained and evaluated a regression model, learned about regression metrics (MAE, MSE, R²), and visualized its performance.

---

### Part 4: Practice Regression (Diabetes Dataset)

**Goal:** Apply the regression pipeline steps yourself using the Diabetes dataset from `sklearn`. Use Part 3 as a guide.

**Step 4.1: Load Diabetes Data**
This dataset is also built into `sklearn`. Features are already scaled.

```python
# Load the Diabetes dataset
diabetes = load_diabetes()
diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
diabetes_df['target'] = diabetes.target # Target is a quantitative measure of disease progression

print("Diabetes Data (first 5 rows):")
print(diabetes_df.head())
print("\nData Info:")
diabetes_df.info()
```

**Step 4.2: Define Features (X) and Target (y)**
The target is 'target'.

```python

# Define X_diabetes and y_diabetes
X_diabetes = diabetes_df.drop('target', axis=1)
y_diabetes = diabetes_df['target']

# Print shapes to verify
print(f"\nDiabetes features shape: {X_diabetes.shape}")
print(f"Diabetes target shape: {y_diabetes.shape}")
```

**Step 4.3: Split Data**
Split into 80% training, 20% testing. Use `random_state=42`.

```python

# Split the data into X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42
)

# Print shapes to verify
print(f"\nTraining set shape: X={X_train_diabetes.shape}, y={y_train_diabetes.shape}")
print(f"Testing set shape: X={X_test_diabetes.shape}, y={y_test_diabetes.shape}")
```

**Step 4.4: Create and Train a Linear Regression Model**
Use `LinearRegression`. Train it on the Diabetes training data.

```python

# Create and train the model (call the variable lr_diabetes)
lr_diabetes = LinearRegression()
lr_diabetes.fit(X_train_diabetes, y_train_diabetes)

print("\nDiabetes Linear Regression model trained.")
```

**Step 4.5: Make Predictions**
Predict the disease progression for the Diabetes test set.

```python

# Make predictions (call the variable y_pred_diabetes)
y_pred_diabetes = lr_diabetes.predict(X_test_diabetes)

# Display first 5 predictions vs actual values
print("\nFirst 5 Predictions:", y_pred_diabetes[:5])
print("First 5 Actual Values:", y_test_diabetes[:5].values)
```

**Step 4.6: Evaluate the Model**
Calculate MAE, MSE, RMSE, and R² for the Diabetes predictions.

```python
# Evaluate the model (mae_diabetes, mse_diabetes, rmse_diabetes, r2_diabetes)
mae_diabetes = mean_absolute_error(y_test_diabetes, y_pred_diabetes)
mse_diabetes = mean_squared_error(y_test_diabetes, y_pred_diabetes)
rmse_diabetes = np.sqrt(mse_diabetes)
r2_diabetes = r2_score(y_test_diabetes, y_pred_diabetes)


print(f"\nDiabetes Model Evaluation:")
print(f"MAE: {mae_diabetes:.4f}")
print(f"MSE: {mse_diabetes:.4f}")
print(f"RMSE: {rmse_diabetes:.4f}")
print(f"R²: {r2_diabetes:.4f}")
```

**Step 4.7: Visualize Predictions vs Actuals (Optional)**
Create the scatter plot and residual plot for the Diabetes model.

```python
# Plot Actual vs Predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test_diabetes, y_pred_diabetes, alpha=0.7)
plt.plot([y_test_diabetes.min(), y_test_diabetes.max()], [y_test_diabetes.min(), y_test_diabetes.max()], '--r', linewidth=2)
plt.xlabel('Actual Progression')
plt.ylabel('Predicted Progression')
plt.title('Actual vs. Predicted Diabetes Progression')
plt.show()

# Plot Residuals
residuals_diabetes = y_test_diabetes - y_pred_diabetes
plt.figure(figsize=(10, 6))
sns.histplot(residuals_diabetes, kde=True)
plt.xlabel('Residuals (Actual - Predicted)')
plt.title('Distribution of Residuals (Diabetes)')
plt.show()
```

**Reflection:** How does the R² score for the Diabetes model compare to the Auto MPG model? What does this tell you about how well each model fits its respective data?

---

### Part 5: The Reality - Data Preprocessing

**Goal:** Understand *why* preprocessing is crucial and see *how* the raw Auto MPG data needed cleaning before we could use it effectively in Part 3.

**The Scenario:** In Parts 1-4, we used data that was mostly ready for modeling. Real-world data is almost never that clean! It often has missing values, non-numeric data types, and features on different scales. Models usually require numerical input and perform better when data is scaled.

**Let's revisit the Auto MPG dataset, but load the *raw* version this time.**

**Step 5.1: Load RAW Data and Initial Inspection**
Notice the `na_values='?'` - this tells pandas to treat '?' as missing.

```python
import pandas as pd

# Load the raw data again
url_mpg = "https://raw.githubusercontent.com/ML-Course-2025/session3/main/datasets/cars/auto-mpg.csv"

# Define column names (to ensure consistency)
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']

# Read CSV correctly (comma-separated)
raw_mpg_df = pd.read_csv(url_mpg, names=column_names, na_values="?", header=0)

# Display first rows
print("Raw Auto MPG Data (first 5 rows):")
print(raw_mpg_df.head())

# Data info
print("\nRaw Data Info:")
raw_mpg_df.info()

# Check for missing values
print("\nMissing values count:")
print(raw_mpg_df.isnull().sum())
```
*   **Observation:** `horsepower` has missing values (6). Also notice `horsepower` is an `object` (text) type, likely because of the '?' before we handled `na_values`. `origin` is numeric but represents categories (1: USA, 2: Europe, 3: Japan). `car_name` is text and probably not useful as a raw feature.

**Step 5.2: Handling Missing Values (Imputation)**
We need to fill or remove missing values. For `horsepower`, let's fill with the median.

```python
# Calculate median horsepower (ignoring NaNs)
median_hp = raw_mpg_df['horsepower'].median()
print(f"\nMedian horsepower: {median_hp}")

# Fill missing horsepower values
raw_mpg_df['horsepower'].fillna(median_hp, inplace=True)

# Verify missing values are handled
print("\nMissing values count after imputation:")
print(raw_mpg_df.isnull().sum())
```

**Step 5.3: Handling Categorical Features (Encoding)**
`origin` is categorical. Treating it as a number (1, 2, 3) implies an order and distance that doesn't exist. We should use One-Hot Encoding. `car_name` is too unique; we'll drop it.

```python
# Drop the car name column
raw_mpg_df = raw_mpg_df.drop('car_name', axis=1)

# Use One-Hot Encoding for 'origin'
# This creates new columns like 'origin_1', 'origin_2', 'origin_3'
raw_mpg_df = pd.get_dummies(raw_mpg_df, columns=['origin'], prefix='origin', drop_first=False) # drop_first=False keeps all origins explicit

print("\nData after One-Hot Encoding 'origin' (first 5 rows):")
print(raw_mpg_df.head())
print("\nData Info after Encoding:")
raw_mpg_df.info()
```
*   **Observation:** `origin` is gone, replaced by `origin_1`, `origin_2`, `origin_3`. All columns are now numeric.

**Step 5.4: Feature Scaling (Standardization)**
Features like `weight` (thousands) and `acceleration` (tens) have vastly different scales. Many models (including Linear Regression, though it's less sensitive) benefit from scaling features to have zero mean and unit variance (Standardization).

```python
# Separate target from features *before* scaling
y_processed = raw_mpg_df['mpg']
X_processed = raw_mpg_df.drop('mpg', axis=1)

# Identify numerical columns to scale (exclude the one-hot encoded origin columns for this example, though scaling them doesn't hurt)
numerical_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']

# Create the scaler
scaler = StandardScaler()

# Fit and transform the numerical columns
X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])

print("\nFeatures after Standardization (first 5 rows):")
print(X_processed.head())

print("\nDescription of Scaled Features:")
print(X_processed[numerical_cols].describe()) # Mean should be close to 0, std dev close to 1
```

**Step 5.5: Data is Ready!**
*Now* the `X_processed` and `y_processed` dataframes are truly ready for the Split -> Train -> Predict -> Evaluate steps we performed in Part 3. The preprocessing steps (handling missing values, encoding categoricals, scaling) are essential for most real-world machine learning tasks.

**Key Takeaways from Part 5:**
*   Real-world data requires cleaning and preparation (preprocessing).
*   Common steps include handling missing data (imputation), converting categorical features to numbers (encoding), and scaling numerical features.
*   Tools like `pandas` for data manipulation and `scikit-learn`'s `StandardScaler` and `OneHotEncoder` (often used within `ColumnTransformer` and `Pipeline` for more complex workflows) are essential.
*   Preprocessing makes data suitable for modeling and often improves model performance.

---

**Lab Conclusion**

You have:
✔ Explored the fundamentals of supervised learning (classification and regression).
✔ Implemented the standard ML pipeline using `scikit-learn`.
✔ Trained and evaluated Decision Tree and Linear Regression models.
✔ Visualized model results.
✔ Gained an appreciation for the critical role of data preprocessing in real-world scenarios.

This lab provides a foundation. There are many more algorithms, evaluation techniques, and preprocessing methods to explore in the world of machine learning!

--------
<details>

<summary>Take home message</summary>

**1. What is Supervised Learning?**

*   **Definition:** A type of Machine Learning where models learn from **labeled data**. Each data point has input features and a known correct output (label/target).
*   **Goal:** To learn a function that maps inputs to outputs, enabling predictions on new, unseen data.
*   **Two Main Types:**
    *   **Classification:** Predicts a discrete **category** (e.g., Survive/Not Survive, Spam/Not Spam, Iris Species).
    *   **Regression:** Predicts a continuous **numerical value** (e.g., Car MPG, Diabetes Progression Score, House Price).

**2. The Standard Machine Learning Pipeline**

A consistent workflow applies to most supervised learning tasks:

1.  **Load Data:** Import data (e.g., using `pandas`).
2.  **Explore & Preprocess Data:** Understand, clean, and transform data (Crucial!).
3.  **Split Data:** Divide into **Training Set** (for learning) and **Testing Set** (for evaluation) using `train_test_split`.
4.  **Choose & Create Model:** Select an algorithm (e.g., `DecisionTreeClassifier`, `LinearRegression`).
5.  **Train Model:** Fit the model to the **training data** (`model.fit(X_train, y_train)`).
6.  **Make Predictions:** Use the trained model on the **test data** (`model.predict(X_test)`).
7.  **Evaluate Model:** Assess performance using appropriate metrics by comparing predictions to actual test labels.

**3. Classification Concepts (Parts 1 & 2)**

*   **Goal:** Assign data points to predefined categories.
*   **Example Model:** Decision Tree (`sklearn.tree.DecisionTreeClassifier`).
    *   Builds a tree-like structure to make decisions based on feature values.
    *   `max_depth` parameter helps control complexity and prevent overfitting.
*   **Evaluation Metrics:**
    *   **Accuracy:** Overall percentage of correct predictions.
    *   **Precision:** Accuracy of positive predictions (TP / (TP + FP)).
    *   **Recall (Sensitivity):** Ability to find all actual positive instances (TP / (TP + FN)).
    *   **F1-Score:** Harmonic mean of Precision and Recall, useful for balancing them.
    *   `sklearn.metrics.accuracy_score`, `sklearn.metrics.classification_report`.

**4. Regression Concepts (Parts 3 & 4)**

*   **Goal:** Predict a numerical value.
*   **Example Model:** Linear Regression (`sklearn.linear_model.LinearRegression`).
    *   Finds the best linear relationship between features and the target.
*   **Evaluation Metrics:**
    *   **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values. Easy to interpret.
    *   **Mean Squared Error (MSE):** Average squared difference. Penalizes larger errors more.
    *   **Root Mean Squared Error (RMSE):** Square root of MSE. In the same units as the target.
    *   **R-squared (R²):** Proportion of target variance explained by the model (0 to 1, higher is better).
    *   `sklearn.metrics.mean_absolute_error`, `sklearn.metrics.mean_squared_error`, `sklearn.metrics.r2_score`.
*   **Visualization:**
    *   **Actual vs. Predicted Plot:** Scatter plot to visually check prediction quality (points near diagonal line are good).
    *   **Residual Plot:** Histogram or scatter plot of errors (Actual - Predicted). Ideally centered around zero with no clear pattern.

**5. Data Preprocessing (Part 5)**

*   **Why it's Crucial:** Real-world data is often "messy". Models require specific data formats and often perform better with processed data.
*   **Common Steps:**
    *   **Handling Missing Values:**
        *   Identify using `.isnull().sum()`.
        *   Strategies: Remove rows/columns, or **Impute** (fill) with mean, median (for numerical), or mode (for categorical).
    *   **Encoding Categorical Features:**
        *   Convert text/categorical data into numbers.
        *   **One-Hot Encoding** (`pd.get_dummies` or `sklearn.preprocessing.OneHotEncoder`) is common, creating binary columns for each category.
    *   **Feature Scaling:**
        *   Bring numerical features to a similar scale. Important for distance-based algorithms and gradient descent.
        *   **Standardization** (`sklearn.preprocessing.StandardScaler`): Transforms data to have zero mean and unit variance.
        *   **Normalization:** Scales data to a specific range (e.g., 0 to 1).

**6. Key Tools Used**

*   **Pandas:** For data loading, manipulation, and exploration.
*   **Scikit-learn:** The core library for ML models, splitting, preprocessing, and evaluation.
*   **Matplotlib & Seaborn:** For data visualization.

**Conclusion:** This lab provided hands-on experience with the fundamental workflow of supervised machine learning, covering both classification and regression tasks, model evaluation, and the essential step of data preprocessing using standard Python libraries.
</details>