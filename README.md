# Python-project

# Heart Disease Prediction System (Logistic Regression)

## Title & Objective

**Title:** Heart Disease Prediction Using Machine Learning

**Objective:**
The goal of this project is to build a predictive system that can determine whether a person is likely to have heart disease based on medical attributes such as age, cholesterol levels, heart rate, and other clinical features. This helps demonstrate how machine learning can assist in early diagnosis and decision-making in healthcare.

---

## Quick Summary of Python

Python is a high-level, interpreted programming language known for its simplicity and readability. It is widely used in:

- **Data Science**
- **Machine Learning**
- **Web Development**
- **Automation**

In this project, Python is used alongside powerful libraries such as:

- **NumPy** → for numerical computations
- **Pandas** → for data manipulation and analysis
- **Seaborn & Matplotlib** → for data visualization
- **Scikit-learn** → for machine learning modeling

---

## System Requirements

### Operating Systems

- Windows 10/11
- macOS (10.14 or later)
- Linux (Ubuntu, Debian, etc.)

### Tools & Editors

- Jupyter Notebook / Google Colab (recommended)
- VS Code / PyCharm / Sublime Text

### Required Python Libraries

Install the following:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Installation & Setup Instructions

### Linux / macOS

1. Open Terminal
2. Install Python (if not installed):

```bash
sudo apt install python3 python3-pip  # Linux
brew install python                   # macOS
```

3. Install required libraries:

```bash
pip3 install numpy pandas matplotlib seaborn scikit-learn
```

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

### Windows

1. Download and install Python from: https://www.python.org
2. Open Command Prompt
3. Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

4. Install Jupyter:

```bash
pip install notebook
```

5. Run:

```bash
jupyter notebook
```

---

## AI Prompts Used to Develop This Project

Here are examples of prompts that guided this project:

- "Create a machine learning project to predict heart disease using Python"
- "Explain how to clean and explore a healthcare dataset"
- "How do I detect outliers using boxplots in Python?"
- "Build a logistic regression model step by step"
- "How do I evaluate a classification model using accuracy score?"
- "Help me create a predictive system from trained data"

---

## Step-by-Step Code Explanation

### 1. Import Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

These libraries handle data processing, visualization, and computation.

---

### 2. Load Dataset

```python
df = pd.read_csv("/content/heart_disease.csv")
df.head()
```

Reads the dataset and displays the first few rows.

---

### 3. Data Exploration

```python
df.shape
df.columns
df.info()
df.describe()
```

Understand dataset size, structure, and statistical summary.

---

### 4. Data Cleaning

```python
df.isnull().sum()
df.duplicated().sum()
```

- Confirms no missing values
- Detects duplicates (1 duplicate found)

---

### 5. Outlier Detection

#### Visualization

```python
sns.boxplot(x=df['age'])
plt.show()
```

#### Statistical Method

```python
Q1 = df['age'].quantile(0.25)
Q3 = df['sex'].quantile(0.75)
IQR = Q3 - Q1
```

#### Finding Outliers

```python
outliers = df[(df['age'] < Q1 - 1.5*IQR) & (df['sex'] > Q3 + 1.5*IQR)]
```

---

### 6. Check Dataset Balance

```python
df['target'].value_counts()
```

- 165 → Has disease
- 138 → No disease
- ✔️ Dataset is fairly balanced

---

### 7. Feature & Target Split

```python
x = df.drop(columns='target', axis=1)
y = df['target']
```

---

### 8. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2
)
```

- 80% training, 20% testing
- `stratify=y` ensures balanced class distribution

---

### 9. Model Training (Logistic Regression)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

Logistic Regression is used for binary classification.

---

### 10. Model Evaluation

```python
model.score(X_train, Y_train)
model.score(X_test, Y_test)
```

- Training Accuracy ≈ 85.5%
- Testing Accuracy ≈ 80.3%

---

### 11. Accuracy Score

```python
from sklearn.metrics import accuracy_score

x_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(x_train_prediction, Y_train)
```

---

### 12. Predictive System

```python
input_data = (57,0,0,120,354,0,1,163,1,0.6,2,0,2)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
```

#### Output Logic

```python
if (prediction[0] == 0):
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')
```

✔️ Example Output:

```
The Person has Heart Disease
```

---

## Conclusion

This project demonstrates how machine learning can be applied to healthcare data to predict heart disease. With further improvements and real-world validation, such systems can assist doctors in making faster and more accurate diagnoses.
