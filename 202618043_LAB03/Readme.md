Assingment title :Scikit-learn: Data Preprocessing and Model Performance Evaluation
Name: Paresh Vaviya
Student ID : 202618043
data link: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

### Preprocessing Choices

The Hotel Booking Demand dataset was preprocessed using two different Scikit-learn pipelines to compare the effect of numerical feature scaling on Logistic Regression and Decision Tree models.

### 1. Handling Missing Values

Numerical columns were processed using KNNImputer(n_neighbors=5).
KNN imputation estimates missing numerical values using the five nearest observations.
Categorical columns were processed using SimpleImputer(strategy="most_frequent")
Missing categorical values were replaced with the most frequently occurring category.

### 2. Categorical Encoding

Categorical features were converted into numerical features using:

`OneHotEncoder(handle_unknown="ignore")`

The `handle_unknown="ignore"` option prevents errors when a category appears in the test data that was not present in the training data.

### 3. Pipeline A — StandardScaler

For numerical features, Pipeline A uses:

KNNImputer(n_neighbors=5) → StandardScaler()

StandardScaler standardizes numerical features based on the mean and standard deviation.

### 4. Pipeline B — MinMaxScaler

For numerical features, Pipeline B uses:

KNNImputer(n_neighbors=5) → MinMaxScaler()

MinMaxScaler scales numerical features to approximately the range 0 to 1.

### 5. ColumnTransformer

ColumnTransformer was used to apply different preprocessing operations to numerical and categorical columns.

### 6. Pipeline

Scikit-learn `Pipeline` was used to combine preprocessing with the machine-learning model. The preprocessing was fitted using the training data and then applied to the test data, helping prevent data leakage.

### 7. Train-Test Split

The same train-test split was used for all four experiments:

## 80% training data
## 20% testing data
## stratify=y
## random_state=42
This ensures that all four model-pipeline combinations are compared under the same conditions.

observation

Logistic Regression shows little evidence of overfitting because its training and testing accuracies are very close. The Decision Tree shows possible overfitting because its training accuracy is 99.62%, while its testing accuracy is approximately 86.12%, giving a large performance gap of about 13.5 percentage points.