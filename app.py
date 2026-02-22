import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Eco Recycling Chatbot",
    page_icon="♻",
    layout="centered"
)

# ---------------- SESSION STATE ----------------
if "eco_points" not in st.session_state:
    st.session_state.eco_points = 0

# ---------------- LOAD TEXT MODEL ----------------
@st.cache_resource
def load_text_model():
    model = pickle.load(open("recycling_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

# ---------------- LOAD IMAGE MODEL (WEIGHTS ONLY) ----------------
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

# ---------------- CUSTOM CSS ----------------
# ---------------- MODERN CSS ----------------
# ---------------- MODERN CSS ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #134e4a, #065f46);
    color: white;
}

.eco-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0px;
}

.eco-score {
    background: rgba(255,255,255,0.1);
    padding: 12px 20px;
    border-radius: 15px;
    font-weight: bold;
}

.result-badge {
    padding: 20px;
    border-radius: 20px;
    font-size: 26px;
    font-weight: bold;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
st.sidebar.title("🌍 Eco Control Panel")

mode = st.sidebar.radio("Choose Input Type", ["Text", "Image"])

country = st.sidebar.selectbox(
    "Select Your Country",
    ["India 🇮🇳", "Singapore 🇸🇬", "United States 🇺🇸"]
)

st.sidebar.markdown("---")
st.sidebar.write("Earn points by classifying waste correctly!")

# ---------------- TOP HEADER WITH LIVE POINTS ----------------
st.markdown(f"""
<div class="eco-header">
    <h1>♻ Eco Recycling Assistant</h1>
    <div class="eco-score">🌱 Total Points: {st.session_state.eco_points}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- CENTER CONTENT ----------------
st.markdown("## ♻ Waste Classification")

mode = st.sidebar.radio("Choose Input Type", ["Text", "Image"])

country = st.sidebar.selectbox(
    "Select Your Country",
    ["India 🇮🇳", "Singapore 🇸🇬", "United States 🇺🇸"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌱 Your Eco Points")
st.sidebar.markdown(f"### {st.session_state.eco_points} Points")

if st.session_state.eco_points >= 100:
    st.sidebar.success("🏆 Recycling Champion")
elif st.session_state.eco_points >= 50:
    st.sidebar.info("🌍 Green Warrior")
elif st.session_state.eco_points >= 20:
    st.sidebar.warning("🌿 Eco Beginner")


# ---------------- MAIN HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>♻ Eco Recycling Assistant</h1>
<p style='text-align:center; opacity:0.8;'>Smart AI-Powered Waste Classification System</p>
""", unsafe_allow_html=True)


# ---------------- ECO DASHBOARD RIGHT PANEL ----------------
col1, col2 = st.columns([3,1])

with col2:
    st.markdown("""
    <div class="eco-card">
        <h3>🌿 Eco Score</h3>
        <h2>{}</h2>
        <p>Keep recycling to earn badges!</p>
    </div>
    """.format(st.session_state.eco_points), unsafe_allow_html=True)

    progress = min(st.session_state.eco_points, 100)
    st.progress(progress / 100)


with col1:
    st.markdown("### ♻ Waste Classification")


# ---------------- COUNTRY SELECTION ----------------
country = st.selectbox(
    "Select Your Country",
    ["India 🇮🇳", "Singapore 🇸🇬", "United States 🇺🇸"]
)

# ---------------- LOCALIZED RULES ----------------
rules = {
    "India 🇮🇳": {
        "recyclable": "Place in Dry Waste Bin.",
        "organic": "Place in Wet Waste Bin.",
        "trash": "Dispose in General Waste.",
        "hazardous": "Take to Authorized Hazardous Waste Facility.",
        "e-waste": "Dispose at E-Waste Collection Center."
    },
    "Singapore 🇸🇬": {
        "recyclable": "Place in Blue Recycling Bin.",
        "organic": "Dispose via food waste collection.",
        "trash": "Place in General Waste Bin.",
        "hazardous": "Bring to Toxic Industrial Waste Facility.",
        "e-waste": "Use E-Waste Recycling Points."
    },
    "United States 🇺🇸": {
        "recyclable": "Place in Recycling Cart.",
        "organic": "Place in Compost Bin (if available).",
        "trash": "Place in Trash Bin.",
        "hazardous": "Use Household Hazardous Waste Facility.",
        "e-waste": "Use Certified E-Waste Recycler."
    }
}

# ---------------- MODE SELECTOR ----------------
mode = st.radio("Choose Input Type", ["Text", "Image"])

# ================= TEXT CLASSIFICATION =================
if mode == "Text":
    user_input = st.text_input("Enter waste item")

    if user_input:
        input_vec = vectorizer.transform([user_input])
        probabilities = text_model.predict_proba(input_vec)
        max_prob = np.max(probabilities)
        prediction = text_model.predict(input_vec)[0]

        if max_prob < 0.30:
            st.error("Item not recognized. Please check spelling.")
        else:
            color_map = {
                "recyclable": "#2ecc71",
                "organic": "#27ae60",
                "trash": "#e74c3c",
                "hazardous": "#f39c12",
                "e-waste": "#3498db"
            }

            st.markdown(f"""
            <div class="result-badge" style="background:{color_map.get(prediction,'#2ecc71')};">
                Category: {prediction.upper()}
            </div>
            """, unsafe_allow_html=True)

            st.write("🌍 Disposal Guide:")
            st.write(rules[country].get(prediction, "Disposal guide not available."))

            st.progress(int(max_prob * 100))
            st.write(f"Confidence: {round(max_prob,2)}")

            st.session_state.eco_points += 10
            st.success("🎉 +10 Eco Points Earned!")

# ================= IMAGE CLASSIFICATION =================
# ================= IMAGE CLASSIFICATION =================
if mode == "Image":
    uploaded_file = st.file_uploader("Upload waste image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = image_model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        image_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        predicted_class = image_classes[class_index]

        # Debug (optional - remove later)
        st.write("Predicted TrashNet Class:", predicted_class)
        st.write("Confidence:", round(confidence, 2))

        # ---------- CLEAN BASIC MAPPING ----------
        if predicted_class in ["cardboard", "paper"]:
            final_category = "recyclable"

        elif predicted_class in ["plastic", "metal", "glass"]:
            final_category = "recyclable"

        elif predicted_class == "trash":
            final_category = "trash"

        else:
            final_category = "trash"

        color_map = {
            "recyclable": "#2ecc71",
            "trash": "#e74c3c"
        }

        st.markdown(f"""
        <div class="result-badge" style="background:{color_map.get(final_category,'#2ecc71')};">
            Category: {final_category.upper()}
        </div>
        """, unsafe_allow_html=True)

        st.write("🌍 Disposal Guide:")
        st.write(rules[country].get(final_category, "Disposal guide not available."))

        st.progress(int(confidence * 100))
        st.write(f"Confidence: {round(confidence,2)}")

        st.session_state.eco_points += 10
        st.success("🎉 +10 Eco Points Earned!")

# ---------------- ECO POINTS DISPLAY ----------------
st.markdown("---")
st.markdown(f"### 🌱 Your Eco Points: {st.session_state.eco_points}")

points = st.session_state.eco_points

if points >= 100:
    st.markdown("🏆 **Recycling Champion**")
elif points >= 50:
    st.markdown("🌍 **Green Warrior**")
elif points >= 20:
    st.markdown("🌿 **Eco Beginner**")

st.markdown("<br><hr><center>© 2026 Eco Recycling Assistant</center>", unsafe_allow_html=True)






