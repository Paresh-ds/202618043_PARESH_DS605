# ============================================================
# AIRBNB PRICE PREDICTION - STREAMLIT APP
# ============================================================

import streamlit as st
from datetime import date

from model import predict_price


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Airbnb Price Prediction",
    #page_icon="🏠",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title(" Airbnb Price Prediction")

st.write(
    "Enter the Airbnb listing details below to estimate "
    "the nightly price."
)

st.divider()


# ============================================================
# LOCATION
# ============================================================

st.subheader(" Location")

neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    [
        "Bronx",
        "Brooklyn",
        "Manhattan",
        "Queens",
        "Staten Island"
    ]
)

latitude = st.number_input(
    "Latitude",
    min_value=40.4,
    max_value=40.95,
    value=40.72,
    format="%.6f"
)

longitude = st.number_input(
    "Longitude",
    min_value=-74.3,
    max_value=-73.65,
    value=-73.99,
    format="%.6f"
)


# ============================================================
# ROOM INFORMATION
# ============================================================

st.subheader(" Room Information")

room_type = st.selectbox(
    "Room Type",
    [
        "Entire home/apt",
        "Private room",
        "Shared room"
    ]
)


# ============================================================
# LISTING INFORMATION
# ============================================================

st.subheader(" Listing Information")

minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=365,
    value=3,
    step=1
)

number_of_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    max_value=1000,
    value=10,
    step=1
)

reviews_per_month = st.number_input(
    "Reviews per Month",
    min_value=0.0,
    max_value=100.0,
    value=1.0,
    step=0.1
)

calculated_host_listings_count = st.number_input(
    "Host Listings Count",
    min_value=1,
    max_value=1000,
    value=1,
    step=1
)

availability_365 = st.number_input(
    "Availability (365 days)",
    min_value=0,
    max_value=365,
    value=200,
    step=1
)


# ============================================================
# REVIEW DATE
# ============================================================

st.subheader(" Review Information")

has_review = st.checkbox(
    "Has previous review?",
    value=True
)

if has_review:

    last_review = st.date_input(
        "Last Review Date",
        value=date(2019, 6, 30),
        min_value=date(2010, 1, 1),
        max_value=date(2025, 12, 31)
    )

else:

    last_review = None


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

if st.button(
    " Predict Airbnb Price",
    type="primary",
    use_container_width=True
):

    try:

        price = predict_price(
            neighbourhood_group=neighbourhood_group,
            latitude=latitude,
            longitude=longitude,
            room_type=room_type,
            minimum_nights=minimum_nights,
            number_of_reviews=number_of_reviews,
            reviews_per_month=reviews_per_month,
            calculated_host_listings_count=calculated_host_listings_count,
            availability_365=availability_365,
            last_review=last_review
        )

        st.success("Prediction completed successfully!")

        st.metric(
            label=" Estimated Nightly Price",
            value=f"${price:,.2f}"
        )

    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Airbnb Price Prediction | Machine Learning Lab 4"
)