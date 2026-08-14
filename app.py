import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Forest Fire Prediction",
    page_icon="🔥",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🔥 Forest Fire Prediction System")
st.write("Predict the possibility of a forest fire using Random Forest.")


# ==========================================================
# LOAD DATASET AND TRAIN MODEL
# ==========================================================

@st.cache_resource
def train_model():

    data = pd.read_csv("forestfires.csv")

    # Create target
    data["fire"] = data["area"].apply(
        lambda x: 0 if x == 0 else 1
    )

    # Encode categorical columns separately
    month_encoder = LabelEncoder()
    day_encoder = LabelEncoder()

    data["month"] = month_encoder.fit_transform(
        data["month"]
    )

    data["day"] = day_encoder.fit_transform(
        data["day"]
    )

    # Features and target
    X = data.drop(
        ["area", "fire"],
        axis=1
    )

    y = data["fire"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, month_encoder, day_encoder, X.columns


# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    model, month_encoder, day_encoder, feature_columns = train_model()

    st.success("Random Forest model loaded successfully!")

except Exception as e:

    st.error("Unable to load forestfires.csv.")

    st.code(str(e))

    st.warning(
        "Make sure forestfires.csv is in the same folder as app.py."
    )

    st.stop()


# ==========================================================
# USER INPUT
# ==========================================================

st.subheader("Enter Forest Conditions")

col1, col2 = st.columns(2)


with col1:

    x_value = st.number_input(
        "X Coordinate",
        value=7,
        step=1
    )

    y_value = st.number_input(
        "Y Coordinate",
        value=5,
        step=1
    )

    month = st.selectbox(
        "Month",
        month_encoder.classes_
    )

    day = st.selectbox(
        "Day",
        day_encoder.classes_
    )

    ffmc = st.number_input(
        "FFMC",
        value=90.0
    )

    dmc = st.number_input(
        "DMC",
        value=100.0
    )


with col2:

    dc = st.number_input(
        "DC",
        value=500.0
    )

    isi = st.number_input(
        "ISI",
        value=10.0
    )

    temp = st.number_input(
        "Temperature",
        value=20.0
    )

    rh = st.number_input(
        "Relative Humidity",
        value=40,
        step=1
    )

    wind = st.number_input(
        "Wind Speed",
        value=4.0
    )

    rain = st.number_input(
        "Rainfall",
        value=0.0
    )


# ==========================================================
# PREDICTION
# ==========================================================

if st.button(
    "🔥 Predict Forest Fire",
    type="primary"
):

    month_encoded = month_encoder.transform(
        [month]
    )[0]

    day_encoded = day_encoder.transform(
        [day]
    )[0]

    input_data = pd.DataFrame(
        [[
            x_value,
            y_value,
            month_encoded,
            day_encoded,
            ffmc,
            dmc,
            dc,
            isi,
            temp,
            rh,
            wind,
            rain
        ]],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0]

    st.subheader("========== RESULT ==========")

    if prediction == 1:

        st.error("🔥 Prediction: FIRE LIKELY")

    else:

        st.success("✅ Prediction: NO FIRE")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "No Fire Probability",
            f"{probability[0] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Fire Probability",
            f"{probability[1] * 100:.2f}%"
        )