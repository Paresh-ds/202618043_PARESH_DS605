# Airbnb Price Prediction -- Lab 4

## 1. Project Overview

This project is my **DS605 Fundamentals of Machine Learning -- Lab 4**
project.

The aim of this project is to build an end-to-end machine learning
system that can estimate the **nightly price of an Airbnb listing** in
New York City.

I started with the Airbnb dataset (`AB_NYC_2019.csv`), cleaned the data,
created useful features, trained and compared regression models,
improved the final model using feature engineering and hyperparameter
tuning, and finally connected the trained model to a Streamlit web
application.

The final application allows a user to enter information about an Airbnb
listing and get an estimated nightly price.

The assignment asks for a complete workflow including data preparation,
model training and evaluation, saving the final model/pipeline, and
building a Streamlit or Gradio application. The final GitHub repository
should contain the notebook, application, saved model, requirements
file, README, and supporting results/screenshots.

## 2. Dataset

**Dataset:** New York City Airbnb Open Data\
**File:** `AB_NYC_2019.csv`

The dataset contains Airbnb listing information such as:

-   Neighbourhood group
-   Neighbourhood
-   Latitude and longitude
-   Room type
-   Price
-   Minimum nights
-   Number of reviews
-   Reviews per month
-   Host listing count
-   Availability during the year
-   Last review date

The target variable for this project is:

``` text
price
```

The target represents the approximate nightly Airbnb price.

## 3. Project Workflow

The complete workflow is:

``` text
Raw Airbnb Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Outlier / Price Handling
       ↓
Train-Test Split
       ↓
KMeans Geographic Features
       ↓
Preprocessing Pipeline
       ↓
Regression Models
       ↓
Hyperparameter Tuning
       ↓
Final LightGBM Model
       ↓
Save model.pkl and kmeans.pkl
       ↓
Streamlit Application
       ↓
Predicted Airbnb Nightly Price
```

## 4. Data Cleaning

The following cleaning steps were used:

1.  Removed columns that were not useful for prediction:

    -   `id`
    -   `host_id`
    -   `host_name`

2.  Removed rows where the listing `name` was missing.

3.  Filled missing values in `reviews_per_month` with `0`.

4.  Converted `last_review` into a datetime column.

5.  Created date-based features from `last_review`.

6.  Removed listings where the price was zero.

7.  The price distribution contained extreme values, so the final
    modelling data used a **99th percentile price cap**.

The 99th percentile cap was calculated using:

``` python
price_cap = df["price"].quantile(0.99)
df = df[df["price"] <= price_cap].copy()
```

This was done to reduce the effect of extreme prices on model training.

## 5. Feature Engineering

Several additional features were created.

### Review date features

From `last_review`:

-   `review_year`
-   `review_month`
-   `review_dayofweek`
-   `days_since_review`

These features allow the model to use information about when the listing
was last reviewed.

### Geographic clustering

Latitude and longitude were used with **KMeans clustering**.

The final KMeans model used:

``` python
n_clusters = 10
random_state = 42
n_init = 10
```

Two geographic features were created:

-   `geo_cluster`
-   `geo_distance`

The purpose of this step was to give the model additional information
about the geographic location of the Airbnb.

### Room and location feature

A combined feature called `room_geo` was created by combining:

``` text
room_type + geo_cluster
```

This helps the model learn that the effect of a room type can be
different in different geographic areas.

## 6. Preprocessing

The final model uses a preprocessing pipeline.

### Numerical features

The numerical features include:

``` text
latitude
longitude
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
review_year
review_month
review_dayofweek
days_since_review
geo_cluster
geo_distance
```

Missing numerical values are handled using median imputation.

### Categorical features

The categorical features include:

``` text
neighbourhood_group
room_type
room_geo
```

Categorical missing values are handled using the most frequent value.

One-hot encoding is used with:

``` python
OneHotEncoder(
    handle_unknown="ignore",
    drop="first"
)
```

Using `handle_unknown="ignore"` is especially useful in the Streamlit
application because a new input should not cause the application to fail
simply because a category was not present in the training data.

## 7. Models

Different regression approaches were considered during the project.

The main final model used in the application is:

**LightGBM Regressor**

The final model parameters used in the modelling workflow were:

``` python
LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
```

## 8. Log Transformation of Price

Airbnb prices have a highly skewed distribution.

To reduce the effect of very large prices, the model was trained using:

``` python
np.log1p(y_train)
```

During prediction, the transformation is reversed using:

``` python
np.expm1(prediction)
```

Therefore, the user sees the prediction in the original price scale.

## 9. Saved Model Files

Two model files are used by the application.

### `model.pkl`

This file contains the trained machine learning pipeline.

It includes the preprocessing steps and the trained LightGBM model.

### `kmeans.pkl`

This file contains the trained KMeans model.

It is needed when a new latitude and longitude are entered in the
Streamlit application because the application must calculate the same
geographic features used during training.

The files are created using:

``` python
joblib.dump(pipeline, "model.pkl")
joblib.dump(kmeans, "kmeans.pkl")
```

## 10. Project Structure

The recommended project structure is:

``` text
202618043_LAB04/
│
├── .venv/
│
├── code/
│   ├── 202618043_LAB_4.ipynb
│   ├── create_models.py
│   ├── model.py
│   ├── app.py
│   ├── model.pkl
│   └── kmeans.pkl
│
├── data/
│   └── AB_NYC_2019.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

The `.venv` folder is used only for the local Python environment and
should not be uploaded to GitHub.

## 11. `model.py`

`model.py` is responsible for the prediction logic.

It:

1.  Loads `model.pkl`.
2.  Loads `kmeans.pkl`.
3.  Receives the values entered by the user.
4.  Creates the same engineered features used during training.
5.  Calculates `geo_cluster` and `geo_distance`.
6.  Creates `room_geo`.
7.  Sends the prepared data to the saved pipeline.
8.  Converts the log-scale prediction back to the original price scale.
9.  Returns the estimated nightly price.

The important point is that the same feature engineering logic used
during model training must also be used for new user inputs.

## 12. `app.py`

`app.py` is the Streamlit user interface.

The application accepts:

-   Neighbourhood group
-   Latitude
-   Longitude
-   Room type
-   Minimum nights
-   Number of reviews
-   Reviews per month
-   Host listing count
-   Availability for 365 days
-   Last review information

After the user clicks:

``` text
Predict Airbnb Price
```

the application calls `predict_price()` from `model.py` and displays the
estimated nightly price.

## 13. Setting Up the Project

### Step 1: Open the project

Open the project folder in VS Code.

### Step 2: Create the virtual environment

From the project root:

``` powershell
python -m venv .venv
```

### Step 3: Activate it

For PowerShell:

``` powershell
.venv\Scripts\Activate.ps1
```

For Command Prompt:

``` cmd
.venv\Scripts\activate
```

After activation, the terminal should show:

``` text
(.venv)
```

### Step 4: Install dependencies

``` powershell
pip install -r requirements.txt
```

If `requirements.txt` is not ready yet, the main packages are:

``` text
streamlit
pandas
numpy
scikit-learn
lightgbm
joblib
```

## 14. Creating the Model Files

Before running the Streamlit application, `model.pkl` and `kmeans.pkl`
must exist.

Run:

``` powershell
python create_models.py
```

The script loads the dataset, performs the required preprocessing and
feature engineering, trains KMeans, trains the LightGBM pipeline, and
saves:

``` text
code/model.pkl
code/kmeans.pkl
```

After successful completion, check:

``` powershell
dir *.pkl
```

Both files should be present and `model.pkl` should not have a size of 0
bytes.

## 15. Testing `model.py`

After the model files are created:

``` powershell
python model.py
```

`model.py` is mainly a reusable module, so it may finish without
displaying a prediction. The important result is that it should not give
a model-loading error.

## 16. Running the Streamlit Application

From the `code` folder:

``` powershell
streamlit run app.py
```

Alternatively, from the project root:

``` powershell
streamlit run code/app.py
```

Streamlit normally opens the application at:

``` text
http://localhost:8501
```

## 17. Problems Faced During Development

During the local setup, a few problems occurred.

### Problem 1: `FileNotFoundError`

The first error was:

``` text
FileNotFoundError: [Errno 2] No such file or directory:
...\\code\\model.pkl
```

The reason was simple: `model.py` was trying to load `model.pkl`, but
the file had not been created yet.

### Solution

The trained pipeline and KMeans model need to be saved first:

``` python
joblib.dump(pipeline, "model.pkl")
joblib.dump(kmeans, "kmeans.pkl")
```

The project was then changed so that the model creation step produces
these files before the application is started.

### Problem 2: `EOFError`

After creating a `model.pkl`, another error appeared:

``` text
EOFError
```

This means the pickle file could not be read correctly. In this case,
the file was incomplete, empty, or corrupted.

### Solution

The incorrect `.pkl` files were removed and the model was recreated
using:

``` powershell
python create_models.py
```

The important lesson is that a `.pkl` file should not be created as an
empty file. It must contain the actual serialized trained model.

### Problem 3: Running the files in the wrong order

The correct order is:

``` text
1. create_models.py
2. model.py
3. app.py
```

More precisely:

``` powershell
python create_models.py
python model.py
streamlit run app.py
```

`create_models.py` creates the saved model files.\
`model.py` loads those files and provides prediction logic.\
`app.py` provides the web interface.

## 18. Important Development Lesson

One of the main issues in this project was understanding the difference
between a Python source file and a saved machine learning model.

``` text
model.py
```

is Python code.

``` text
model.pkl
```

is a saved trained model.

``` text
kmeans.pkl
```

is a saved trained KMeans object.

The Streamlit application does not train the model every time a user
enters a value. Instead, it loads the already-trained models and uses
them for prediction.

This makes the application much faster and keeps the prediction process
consistent.

## 19. Model Evaluation

The project uses regression evaluation metrics such as:

-   R²
-   Mean Absolute Error (MAE)
-   Root Mean Squared Error (RMSE)

The final numerical results should be reported here after the final
notebook run.

### Final Model Results

  Metric             Final Result
  ------------- -----------------
  Training R²     Add final value
  Test R²         Add final value
  MAE             Add final value
  RMSE            Add final value

I am leaving these values as placeholders rather than writing numbers
that were not verified from the final run.

## 20. Application Testing

The Streamlit application should be tested using realistic Airbnb
listing information.

For example:

``` text
Neighbourhood Group: Manhattan
Room Type: Entire home/apt
Latitude: 40.72
Longitude: -73.99
Minimum Nights: 3
Number of Reviews: 10
Reviews per Month: 1.0
Host Listings Count: 1
Availability: 200
Last Review: 2019-06-30
```

The application then returns an estimated nightly price.

The exact prediction depends on the saved model and the entered values.

## 21. Limitations

This model has several limitations.

1.  Airbnb prices can change over time, while this dataset represents
    historical listings.

2.  The model does not know the current market price of a listing.

3.  Important factors such as detailed amenities, listing quality,
    photographs, host reputation, and special events may not be fully
    represented.

4.  Predictions for unusual or very different listings may be less
    reliable.

5.  The model is trained on New York City Airbnb data, so it should not
    automatically be used for another city.

6.  A machine learning prediction is an estimate, not a guaranteed
    Airbnb price.

## 22. GitHub Files

The repository should contain:

``` text
202618043_LAB04/
│
├── code/
│   ├── 202618043_LAB_4.ipynb
│   ├── create_models.py
│   ├── model.py
│   ├── app.py
│   ├── model.pkl
│   └── kmeans.pkl
│
├── data/
│   └── AB_NYC_2019.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

The assignment requires the public GitHub repository to contain the
complete notebook, Streamlit/Gradio application, saved model/pipeline,
`requirements.txt`, and README. Supporting model results, important
plots, application screenshots, and a deployed application link should
also be included when available.

## 23. GitHub

Check the repository status:

``` powershell
git status
```

Add the required files:

``` powershell
git add README.md requirements.txt .gitignore code data
```

Commit:

``` powershell
git commit -m "Complete Airbnb price prediction Lab 4"
```

Push:

``` powershell
git push origin main
```

If the repository uses `master` instead of `main`, use:

``` powershell
git push origin master
```

## 24. Final Run Checklist

Before submitting the project, I will check:

-   [ ] Notebook runs correctly
-   [ ] Dataset is available
-   [ ] Data cleaning is completed
-   [ ] Feature engineering is completed
-   [ ] Price outliers are handled
-   [ ] Final model is trained
-   [ ] Final model evaluation is recorded
-   [ ] `model.pkl` exists
-   [ ] `kmeans.pkl` exists
-   [ ] `model.py` loads both files
-   [ ] `app.py` runs successfully
-   [ ] Streamlit prediction works
-   [ ] `requirements.txt` is included
-   [ ] `.gitignore` is included
-   [ ] README is included
-   [ ] Application screenshot is included
-   [ ] GitHub repository is public
-   [ ] Deployed link is added if deployment is completed

## 25. Conclusion

This project takes the Airbnb dataset from raw data to a working machine
learning application.

The main work was not only training a regression model. The important
part was building the complete pipeline so that the same preprocessing
and feature engineering used during training can also be applied to new
Airbnb listings entered through the web application.

The final system uses geographic information through KMeans, categorical
and numerical preprocessing, engineered review/date features, a log
transformation of price, and a LightGBM regression model.

The final Streamlit application provides a simple way for a user to
enter listing information and receive an estimated nightly Airbnb price.
