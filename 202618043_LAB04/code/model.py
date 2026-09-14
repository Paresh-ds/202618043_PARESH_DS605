# ============================================================
# model.py
# Airbnb Price Prediction
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

KMEANS_PATH = os.path.join(BASE_DIR, "kmeans.pkl")


# ============================================================
# 2. LOAD SAVED MODELS
# ============================================================

model = joblib.load(MODEL_PATH)

kmeans = joblib.load(KMEANS_PATH)


# ============================================================
# 3. REFERENCE DATE
# ============================================================

REFERENCE_DATE = pd.Timestamp("2019-07-08")


# ============================================================
# 4. PREDICTION FUNCTION
# ============================================================

def predict_price(
    neighbourhood_group,
    room_type,
    latitude,
    longitude,
    minimum_nights,
    number_of_reviews,
    reviews_per_month,
    calculated_host_listings_count,
    availability_365,
    last_review
):

    # --------------------------------------------------------
    # Review date features
    # --------------------------------------------------------

    if last_review is None:

        review_year = 0
        review_month = 0
        review_dayofweek = 0
        days_since_review = 0

    else:

        last_review = pd.Timestamp(last_review)

        review_year = last_review.year

        review_month = last_review.month

        review_dayofweek = last_review.dayofweek

        days_since_review = (
            REFERENCE_DATE - last_review
        ).days

        if days_since_review < 0:
            days_since_review = 0


    # --------------------------------------------------------
    # Geographic input
    # --------------------------------------------------------

    geo_input = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude]
    })


    # --------------------------------------------------------
    # Geo cluster
    # --------------------------------------------------------

    geo_cluster = int(
        kmeans.predict(geo_input)[0]
    )


    # --------------------------------------------------------
    # Geo distance
    # --------------------------------------------------------

    geo_distance = float(
        kmeans.transform(geo_input)
        .min(axis=1)[0]
    )


    # --------------------------------------------------------
    # Room + geographic feature
    # --------------------------------------------------------

    room_geo = (
        str(room_type)
        + "_"
        + str(geo_cluster)
    )


    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "neighbourhood_group": [
            neighbourhood_group
        ],

        "latitude": [
            latitude
        ],

        "longitude": [
            longitude
        ],

        "room_type": [
            room_type
        ],

        "minimum_nights": [
            minimum_nights
        ],

        "number_of_reviews": [
            number_of_reviews
        ],

        "reviews_per_month": [
            reviews_per_month
        ],

        "calculated_host_listings_count": [
            calculated_host_listings_count
        ],

        "availability_365": [
            availability_365
        ],

        "review_year": [
            review_year
        ],

        "review_month": [
            review_month
        ],

        "review_dayofweek": [
            review_dayofweek
        ],

        "days_since_review": [
            days_since_review
        ],

        "geo_cluster": [
            geo_cluster
        ],

        "geo_distance": [
            geo_distance
        ],

        "room_geo": [
            room_geo
        ]
    })


    # ========================================================
    # 5. PREDICTION
    # ========================================================

    # The model was trained using:
    #
    # np.log1p(y_train)
    #
    # Therefore convert prediction back using:
    #
    # np.expm1()
    # ========================================================

    log_prediction = model.predict(
        input_data
    )


    predicted_price = np.expm1(
        log_prediction
    )[0]


    # --------------------------------------------------------
    # Prevent negative prediction
    # --------------------------------------------------------

    predicted_price = max(
        0,
        predicted_price
    )


    return predicted_price