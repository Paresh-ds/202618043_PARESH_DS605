DS605 Fundamentals of Machine Learning --- Lab Assignment 5

Machine Learning with Scikit-learn and From Scratch

Student Name: Paresh Vaviya
Student ID: 202618043

1. Assignment Overview

This assignment implements Linear Regression and Logistic
Regression in two different ways:

Using Scikit-learn

From scratch using NumPy and Pandas

The main purpose is to understand not only how to use machine learning
libraries, but also how the underlying mathematical algorithms work.

The same fixed train-test split is used for the comparisons so that the
results are reproducible and fair.

2. Dataset

The dataset used is the UCI Productivity Prediction of Garment
Employees dataset.

The dataset contains information about garment production, including:

Team

Targeted productivity

SMV

WIP

Over time

Incentive

Idle time

Idle men

Number of style changes

Number of workers

Quarter

Department

Day

Actual productivity

Regression Target

For regression, the target variable is:

actual_productivity

The goal is to predict the actual productivity value.

Classification Target

For classification, a new target variable MeetsTarget is created:

MeetsTarget = 1 if actual_productivity >= targeted_productivity
MeetsTarget = 0 otherwise

actual_productivity is not used as an input feature for classification
because it is used to create the target.

3. Data Preprocessing

The data is prepared before training the models.

Missing Values

For numerical features, missing values are handled using the median of
the training data.

Categorical Features

Categorical features such as:

quarter

department

day

are converted into numerical form using One-Hot Encoding.

Feature Scaling

Numerical features are standardized using:

z = (x - mean) / standard deviation

The preprocessing parameters are learned from the training data and then
applied to the test data.

This avoids using information from the test set during training.

4. Part A --- Scikit-learn Implementation

Scikit-learn is used for:

Preprocessing

Linear Regression

Logistic Regression

Evaluation metrics

Timing

A Pipeline is used so that preprocessing and model training are
performed together.

Linear Regression

The Scikit-learn implementation uses:

Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

Logistic Regression

The Scikit-learn implementation uses:

Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

5. Part B --- From-Scratch Implementation

The second part implements the machine learning workflow without using
Scikit-learn's:

preprocessing utilities

regression models

classification models

train-test split utilities

evaluation metrics

NumPy and Pandas are used instead.

The same train-test indices are reused from the Scikit-learn
implementation.

6. Linear Regression From Scratch

Linear Regression predicts a continuous value using:

ŷ = Xβ

where:

X = feature matrix

β = coefficient vector

ŷ = predicted value

Normal Equation

The coefficients are calculated using the Normal Equation:

β = (XᵀX)⁻¹Xᵀy

The steps are:

XᵀX
  ↓
Calculate (XᵀX)⁻¹
  ↓
Xᵀy
  ↓
β = (XᵀX)⁻¹Xᵀy
  ↓
ŷ = Xβ

The implementation uses NumPy matrix operations.

Regression Metrics

The following metrics are calculated manually:

Mean Absolute Error

MAE = (1/n) Σ |yᵢ - ŷᵢ|

Root Mean Squared Error

RMSE = √[(1/n) Σ(yᵢ - ŷᵢ)²]

R² Score

R² = 1 - SSres/SStot

where:

SSres = Σ(yᵢ - ŷᵢ)²

SStot = Σ(yᵢ - ȳ)²

7. Logistic Regression From Scratch

Logistic Regression is used for binary classification.

The model first calculates a linear score:

z = Xβ

Sigmoid Function

The linear score is converted into a probability using the sigmoid
function:

σ(z) = 1 / (1 + e⁻ᶻ)

Therefore:

P(y = 1 | X) = σ(Xβ)

The probability lies between 0 and 1.

Classification Threshold

A threshold of 0.5 is used:

ŷ = 1  if P(y=1|X) >= 0.5
ŷ = 0  otherwise

Binary Cross-Entropy Loss

The loss function used for Logistic Regression is:

J(β) = -(1/m) Σ[
    yᵢ log(pᵢ) +
    (1-yᵢ) log(1-pᵢ)
]

where:

m = number of training samples

yᵢ = actual class

pᵢ = predicted probability

Gradient

The gradient of the loss function is:

∇J(β) = (1/m) Xᵀ(p-y)

Gradient Descent Update

The coefficients are updated using:

βnew = βold - α∇J(β)

where:

α = learning rate

∇J(β) = gradient

In this implementation, the coefficients are initialized to zero and
updated repeatedly for a fixed number of epochs.

8. Classification Metrics

The following metrics are calculated from:

True Positive (TP)

True Negative (TN)

False Positive (FP)

False Negative (FN)

Accuracy

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision

Precision = TP / (TP + FP)

Recall

Recall = TP / (TP + FN)

F1 Score

F1 = 2 × (Precision × Recall) / (Precision + Recall)

9. Runtime Comparison

Training and prediction times are measured using:

time.perf_counter()

The final comparison obtained in the experiment is:

Implementation                                             Training Time (ms)   Prediction Time (ms)

Logistic Regression --- Scikit-learn                                  66.8650                14.0041
Logistic Regression --- From Scratch                                  44.9386                 0.3553
Linear Regression --- Scikit-learn                                    33.7997                12.9261
Linear Regression --- From Scratch --- Normal Equation                 7.9121                 0.1438

Comparison Screenshot



The measured times show that the current from-scratch implementation has
lower measured prediction time. Training time depends on what portion of
the workflow is included in the timing and on the implementation
details.

10. Why Two Implementations?

The Scikit-learn implementation provides a practical and optimized way
to build machine learning models.

The from-scratch implementation helps understand the mathematical
operations behind the models.

Scikit-learn

Dataset
   ↓
Pipeline
   ↓
Preprocessing
   ↓
Optimized ML algorithm
   ↓
Prediction
   ↓
Metrics

From Scratch

Dataset
   ↓
Manual preprocessing
   ↓
NumPy matrix operations
   ↓
Mathematical algorithm
   ↓
Prediction
   ↓
Manual metrics

The comparison helps understand the difference between using a machine
learning library and implementing the main algorithmic steps manually.

11. Technologies Used

Python

Pandas

NumPy

Scikit-learn

Jupyter Notebook

Matplotlib (if used for data visualization)

12. Project Structure

Lab-Assignment-5/
│
├── README.md
├── comparison_results.png
├── dataset/
│   └── garment_worker_productivity.csv
│
└── notebook/
    └── Lab5.ipynb

The exact filenames may differ depending on the final GitHub repository
structure.

13. Conclusion

This assignment implements both regression and classification using
Scikit-learn and from-scratch approaches.

For Linear Regression, the from-scratch implementation uses the
Normal Equation:

β = (XᵀX)⁻¹Xᵀy

For Logistic Regression, the from-scratch implementation uses:

Sigmoid Function
       ↓
Binary Cross-Entropy
       ↓
Gradient
       ↓
Gradient Descent
       ↓
Probability
       ↓
Classification
	Implementation	                    Training Time (ms)	Prediction Time (ms)
0	logit regression Scikit-learn	         66.8650	        14.0041
1	logit regression From Scratch	         44.9386	        0.3553
2	Linear Regression Scikit-learn	         33.7997	        12.9261
3	Linear Regression From Scratch 	         7.9121	            0.1438
    noramal equation