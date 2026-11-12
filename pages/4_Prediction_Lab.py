
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from PIL import Image


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Prediction Lab | AutoNAS-RL",
    page_icon="🔬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "rl_searched_cnn.keras"

CLASS_NAMES = [
    "T-shirt / Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]


# =========================================================
# 2. CUSTOM STYLING
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #0B1020;
}

.block-container {
    max-width: 1500px;
    padding: 2rem 3rem 3rem 3rem;
}

.prediction-hero {
    background: linear-gradient(
        115deg,
        #172554,
        #172033,
        #211C46
    );
    border: 1px solid #34446B;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 28px;
}

.prediction-hero h1 {
    color: #F8FAFC;
    font-size: 34px;
    font-weight: 800;
}

.prediction-hero p {
    color: #B6C4DD;
    font-size: 14px;
    line-height: 1.8;
}

.panel {
    background: #141D30;
    border: 1px solid #29364D;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 15px;
}

.panel-title {
    color: #F1F5F9;
    font-size: 18px;
    font-weight: 700;
}

.prediction-result {
    background: linear-gradient(
        120deg,
        #12352D,
        #142C35
    );
    border: 1px solid #21634F;
    border-radius: 15px;
    padding: 25px;
    margin-top: 15px;
}

.result-label {
    color: #86EFAC;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.result-value {
    color: #F0FDF4;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

div[data-testid="stMetric"] {
    background: #141D30;
    border: 1px solid #29364D;
    border-radius: 13px;
    padding: 17px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_prediction_model(model_path):

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


# =========================================================
# 4. HERO SECTION
# =========================================================

st.html("""
<div class="prediction-hero">

    <div style="
        color:#93C5FD;
        font-size:11px;
        font-weight:700;
        letter-spacing:2px;
    ">
        INTERACTIVE MODEL INFERENCE
    </div>

    <h1 style="
        color:#F8FAFC;
        font-size:36px;
        font-weight:800;
        margin-top:15px;
    ">
        🔬 Prediction Lab
    </h1>

    <p style="
        color:#B6C4DD;
        font-size:14px;
        line-height:1.9;
        max-width:850px;
    ">
        Run real-time Fashion-MNIST image classification
        using the Q-learning-selected convolutional neural
        network.

        Upload an image, inspect its processed representation
        and explore the model's predicted class probabilities.
    </p>

    <div style="
        display:inline-block;
        background:#12352D;
        color:#6EE7B7;
        border:1px solid #21634F;
        border-radius:20px;
        padding:7px 13px;
        font-size:10px;
        font-weight:700;
        letter-spacing:1px;
    ">
        MODEL READY · INFERENCE ENABLED
    </div>

</div>
""")



# =========================================================
# 5. MODEL STATUS
# =========================================================

if not MODEL_PATH.exists():

    st.error(
        f"Trained model not found: {MODEL_PATH}"
    )

    st.stop()


try:

    model = load_prediction_model(str(MODEL_PATH))

except Exception as error:

    st.error(f"Could not load the trained model: {error}")
    st.stop()


status_col1, status_col2, status_col3 = st.columns(3)

status_col1.metric(
    "Model Status",
    "Loaded"
)

status_col2.metric(
    "Architecture",
    "RL-Searched CNN"
)

status_col3.metric(
    "Input Shape",
    "28 × 28 × 1"
)

st.divider()


# =========================================================
# 6. IMAGE UPLOAD
# =========================================================

st.markdown("## Upload an Image")

st.caption(
    "Upload a single clothing item image in PNG, JPG or JPEG format."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
    help="For best results, use a centered clothing image similar to Fashion-MNIST."
)


if uploaded_file is None:

    st.info(
        "Upload an image to start classification."
    )

    st.markdown("""
    **Supported classes**

    T-shirt/Top · Trouser · Pullover · Dress · Coat ·
    Sandal · Shirt · Sneaker · Bag · Ankle Boot
    """)

    st.stop()


# =========================================================
# 7. IMAGE PREPROCESSING
# =========================================================

try:

    original_image = Image.open(uploaded_file).convert("RGB")

except Exception:

    st.error("Unable to read the uploaded image.")
    st.stop()


st.markdown("### Image Preprocessing")

invert_image = st.checkbox(
    "Invert grayscale image",
    value=False,
    help="Use this if the clothing item appears dark on a light background."
)


gray_image = original_image.convert("L")

resized_image = gray_image.resize(
    (28, 28),
    Image.Resampling.LANCZOS
)

image_array = np.asarray(
    resized_image,
    dtype=np.float32
)

if invert_image:

    image_array = 255.0 - image_array


processed_image = image_array / 255.0

model_input = processed_image[np.newaxis, ..., np.newaxis]


# =========================================================
# 8. DISPLAY IMAGE PREVIEW
# =========================================================

image_col, processed_col = st.columns(2, gap="large")

with image_col:

    st.markdown("#### Original Image")

    st.image(
        original_image,
        use_container_width=True
    )

with processed_col:

    st.markdown("#### Model Input · 28 × 28")

    st.image(
        processed_image,
        clamp=True,
        use_container_width=True
    )


st.caption(
    "Preprocessing: RGB → Grayscale → 28×28 Resize → "
    "Normalization [0, 1] → CNN Input"
)


# =========================================================
# 9. RUN PREDICTION
# =========================================================

st.divider()

st.markdown("## Model Inference")

predict_button = st.button(
    "🚀 Run RL-CNN Prediction",
    type="primary",
    use_container_width=True
)


if predict_button:

    try:

        with st.spinner("Running model inference..."):

            probabilities = model.predict(
                model_input,
                verbose=0
            )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(
            probabilities[predicted_index] * 100
        )


        # =============================================
        # PREDICTION RESULT
        # =============================================

        st.markdown("## Prediction Result")

        result_col1, result_col2 = st.columns(
            [1.4, 1],
            gap="large"
        )

        with result_col1:

            st.markdown(f"""
            <div class="prediction-result">

                <div class="result-label">
                    PREDICTED CLOTHING CATEGORY
                </div>

                <div class="result-value">
                    {predicted_class}
                </div>

            </div>
            """, unsafe_allow_html=True)

        with result_col2:

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.metric(
                "Predicted Class ID",
                predicted_index
            )


        # =============================================
        # CLASS PROBABILITIES
        # =============================================

        st.markdown("## Class Probability Distribution")

        probability_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probabilities
        })

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        )

        chart_col, top_col = st.columns(
            [1.5, 1],
            gap="large"
        )

        with chart_col:

            st.markdown("#### All 10 Class Probabilities")

            chart_data = probability_df.set_index("Class")

            st.bar_chart(
                chart_data,
                horizontal=True,
                x_label="Class Probability",
                y_label="Clothing Category"
            )

        with top_col:

            st.markdown("#### Top 3 Predictions")

            for rank, (_, row) in enumerate(
                probability_df.head(3).iterrows(),
                start=1
            ):

                st.metric(
                    f"#{rank} {row['Class']}",
                    f"{row['Probability'] * 100:.2f}%"
                )


        # =============================================
        # FULL PROBABILITY TABLE
        # =============================================

        with st.expander("View all class probabilities"):

            display_df = probability_df.copy()

            display_df["Probability"] = (
                display_df["Probability"] * 100
            ).round(3)

            display_df = display_df.rename(
                columns={
                    "Probability": "Probability (%)"
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


        st.success("Inference completed successfully.")


    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# =========================================================
# 10. MODEL INFORMATION
# =========================================================

with st.expander("About the RL-Searched CNN"):

    st.markdown("""
    **Model:** Q-learning-selected CNN

    **Dataset:** Fashion-MNIST

    **Input:** 28 × 28 grayscale image

    **Output:** 10 clothing categories

    **Architecture:**

    - Conv2D: 16 filters
    - Batch Normalization + ReLU
    - Max Pooling
    - Conv2D: 32 filters
    - Batch Normalization + ReLU
    - Average Pooling
    - Flatten
    - Dropout: 0.2
    - Dense: 64 units
    - Dropout: 0.2
    - Softmax: 10 classes

    **Parameters:** 106,058
    """)


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.caption(
    "AutoNAS-RL | Prediction Lab | TensorFlow Inference"
)
