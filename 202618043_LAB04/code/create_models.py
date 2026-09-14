import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMRegressor


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "AB_NYC_2019.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "code", "model.pkl"
)

KMEANS_PATH = os.path.join(
    BASE_DIR, "code", "kmeans.pkl"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. BASIC CLEANING
# ============================================================

df = df.drop(
    columns=["id", "host_id", "host_name"],
    errors="ignore"
)

df = df.dropna(subset=["name"])

df["reviews_per_month"] = df["reviews_per_month"].fillna(0)


# ============================================================
# 4. DATE FEATURES
# ============================================================

df["last_review"] = pd.to_datetime(
    df["last_review"],
    errors="coerce"
)

REFERENCE_DATE = df["last_review"].max()

df["review_year"] = df["last_review"].dt.year
df["review_month"] = df["last_review"].dt.month
df["review_dayofweek"] = df["last_review"].dt.dayofweek

df["days_since_review"] = (
    REFERENCE_DATE - df["last_review"]
).dt.days

df["days_since_review"] = df["days_since_review"].fillna(0)


# ============================================================
# 5. REMOVE INVALID PRICE
# ============================================================

df = df[df["price"] > 0].copy()


# ============================================================
# 6. PRICE CAPPING - 99%
# ============================================================

price_cap = df["price"].quantile(0.99)

df = df[df["price"] <= price_cap].copy()

print("99% Price Cap:", price_cap)
print("Data after price capping:", df.shape)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X = df.drop(
    columns=["price", "name", "last_review"],
    errors="ignore"
)

y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 8. KMEANS
# ============================================================

print("Training KMeans...")

kmeans = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)

kmeans.fit(
    X_train[["latitude", "longitude"]]
)


# ============================================================
# 9. GEO CLUSTER
# ============================================================

X_train = X_train.copy()
X_test = X_test.copy()

X_train["geo_cluster"] = kmeans.predict(
    X_train[["latitude", "longitude"]]
)

X_test["geo_cluster"] = kmeans.predict(
    X_test[["latitude", "longitude"]]
)


# ============================================================
# 10. GEO DISTANCE
# ============================================================

X_train["geo_distance"] = kmeans.transform(
    X_train[["latitude", "longitude"]]
).min(axis=1)

X_test["geo_distance"] = kmeans.transform(
    X_test[["latitude", "longitude"]]
).min(axis=1)


# ============================================================
# 11. ROOM + GEO FEATURE
# ============================================================

X_train["room_geo"] = (
    X_train["room_type"].astype(str)
    + "_"
    + X_train["geo_cluster"].astype(str)
)

X_test["room_geo"] = (
    X_test["room_type"].astype(str)
    + "_"
    + X_test["geo_cluster"].astype(str)
)


# ============================================================
# 12. DROP NEIGHBOURHOOD
# ============================================================

X_train = X_train.drop(
    columns=["neighbourhood"],
    errors="ignore"
)

X_test = X_test.drop(
    columns=["neighbourhood"],
    errors="ignore"
)


# ============================================================
# 13. FEATURES
# ============================================================

numeric_features = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "review_year",
    "review_month",
    "review_dayofweek",
    "days_since_review",
    "geo_cluster",
    "geo_distance"
]

categorical_features = [
    "neighbourhood_group",
    "room_type",
    "room_geo"
]


# ============================================================
# 14. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 15. LIGHTGBM MODEL
# ============================================================

lgbm_model = LGBMRegressor(
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


# ============================================================
# 16. PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", lgbm_model)
    ]
)


# ============================================================
# 17. TRAIN MODEL
# ============================================================

print("Training LightGBM...")

pipeline.fit(
    X_train,
    np.log1p(y_train)
)


# ============================================================
# 18. SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

joblib.dump(
    kmeans,
    KMEANS_PATH
)


# ============================================================
# 19. CHECK FILES
# ============================================================

print()
print("=" * 60)
print("MODEL CREATION COMPLETE")
print("=" * 60)

print("model.pkl:")
print(MODEL_PATH)

print()

print("kmeans.pkl:")
print(KMEANS_PATH)

print()
print("Files created successfully!")