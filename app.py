import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Eco Recycling Assistant",
    page_icon="♻",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "eco_points" not in st.session_state:
    st.session_state.eco_points = 0

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_text_model():
    model = pickle.load(open("recycling_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

@st.cache_resource
def load_image_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(6, activation="softmax")
    ])

    model.load_weights("image_model.weights.h5")
    return model

text_model, vectorizer = load_text_model()
image_model = load_image_model()

# ---------------- CLEAN MODERN CSS ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #134e4a, #065f46);
    color: white;
}

.eco-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.eco-score {
    background: rgba(255,255,255,0.1);
    padding: 10px 20px;
    border-radius: 15px;
    font-weight: bold;
}

.result-badge {
    padding: 20px;
    border-radius: 15px;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌍 Eco Control Panel")

st.sidebar.markdown("## ℹ About")
st.sidebar.write("""
The Eco Recycling Assistant is an AI-based waste classification system.

• Classifies waste using text (Machine Learning)  
• Identifies waste from images (Deep Learning - CNN)  
• Provides country-specific disposal rules  
• Encourages sustainable behavior using eco-points  
""")

st.sidebar.markdown("---")

mode = st.sidebar.radio("Choose Input Type", ["Text", "Image"])

country = st.sidebar.selectbox(
    "Select Your Country",
    ["India 🇮🇳", "Singapore 🇸🇬", "United States 🇺🇸"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌱 Your Eco Points")
st.sidebar.markdown(f"### {st.session_state.eco_points} Points")

# ---------------- HEADER ----------------
st.markdown(f"""
<div class="eco-header">
    <h1>♻ Eco Recycling Assistant</h1>
    <div class="eco-score">🌱 Total Points: {st.session_state.eco_points}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- COUNTRY RULES ----------------
rules = {
    "India 🇮🇳": {
        "recyclable": "Place in Dry Waste Bin.",
        "organic": "Place in Wet Waste Bin.",
        "trash": "Dispose in General Waste."
    },
    "Singapore 🇸🇬": {
        "recyclable": "Place in Blue Recycling Bin.",
        "organic": "Dispose via food waste collection.",
        "trash": "Place in General Waste Bin."
    },
    "United States 🇺🇸": {
        "recyclable": "Place in Recycling Cart.",
        "organic": "Place in Compost Bin (if available).",
        "trash": "Place in Trash Bin."
    }
}

# ---------------- MAIN CONTENT ----------------
st.markdown("## ♻ Waste Classification")

# ================= TEXT MODE =================
if mode == "Text":

    user_input = st.text_input("Enter waste item")

    if user_input:
        input_vec = vectorizer.transform([user_input])
        probabilities = text_model.predict_proba(input_vec)
        max_prob = np.max(probabilities)
        prediction = text_model.predict(input_vec)[0]

        if max_prob < 0.30:
            st.error("Item not recognized.")
        else:
            color_map = {
                "recyclable": "#22c55e",
                "organic": "#16a34a",
                "trash": "#dc2626"
            }

            st.markdown(f"""
            <div class="result-badge" style="background:{color_map.get(prediction,'#22c55e')};">
                Category: {prediction.upper()}
            </div>
            """, unsafe_allow_html=True)

            st.write("🌍 Disposal Guide:")
            st.write(rules[country].get(prediction, "Not available"))

            st.progress(int(max_prob * 100))
            st.write(f"Confidence: {round(max_prob,2)}")

            st.session_state.eco_points += 10
            st.session_state.history.append(prediction)
            st.success("✔ 10 Eco Points Added")

# ================= IMAGE MODE =================
if mode == "Image":

    uploaded_file = st.file_uploader("Upload waste image", type=["jpg","png","jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_container_width=True)

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = image_model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        image_classes = ["cardboard","glass","metal","paper","plastic","trash"]
        predicted_class = image_classes[class_index]

        if predicted_class in ["cardboard","paper","plastic","metal","glass"]:
            final_category = "recyclable"
        else:
            final_category = "trash"

        color_map = {
            "recyclable": "#22c55e",
            "trash": "#dc2626"
        }

        st.markdown(f"""
        <div class="result-badge" style="background:{color_map.get(final_category)};">
            Category: {final_category.upper()}
        </div>
        """, unsafe_allow_html=True)

        st.write("🌍 Disposal Guide:")
        st.write(rules[country].get(final_category, "Not available"))

        st.progress(int(confidence * 100))
        st.write(f"Confidence: {round(confidence,2)}")

        st.session_state.eco_points += 10
        st.session_state.history.append(final_category)
        st.success("✔ 10 Eco Points Added")

# ---------------- HISTORY ----------------
st.markdown("---")
st.markdown("### 📊 Classification History")
st.write(st.session_state.history)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>© 2026 Eco Recycling Assistant</center>", unsafe_allow_html=True)
