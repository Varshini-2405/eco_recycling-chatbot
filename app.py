import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Eco Recycling Chatbot",
    page_icon="♻",
    layout="centered"
)

# ---------- SESSION STATE FOR GAMIFICATION ----------
if "eco_points" not in st.session_state:
    st.session_state.eco_points = 0

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.stTextInput input {
    background-color: #1f2937 !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
.result-badge {
    padding: 18px;
    border-radius: 14px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = pickle.load(open("recycling_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center;'>♻ Eco Recycling Assistant</h1>
<p style='text-align:center; opacity:0.7;'>AI-powered smart waste classifier</p>
""", unsafe_allow_html=True)

# ---------- LOCATION DROPDOWN ----------
country = st.selectbox(
    "Select Your Country",
    ["India", "Singapore", "United States"]
)

# ---------- LOCALIZED RULES ----------
rules = {
    "India": {
        "recyclable": "Place in Dry Waste Bin.",
        "organic": "Place in Wet Waste Bin.",
        "trash": "Dispose in General Waste.",
        "hazardous": "Take to Authorized Hazardous Waste Facility.",
        "e-waste": "Dispose at E-Waste Collection Center."
    },
    "Singapore": {
        "recyclable": "Place in Blue Recycling Bin.",
        "organic": "Dispose via food waste collection.",
        "trash": "Place in General Waste Bin.",
        "hazardous": "Bring to Toxic Industrial Waste Facility.",
        "e-waste": "Use E-Waste Recycling Points."
    },
    "United States": {
        "recyclable": "Place in Recycling Cart.",
        "organic": "Place in Compost Bin (if available).",
        "trash": "Place in Trash Bin.",
        "hazardous": "Use Household Hazardous Waste Facility.",
        "e-waste": "Use Certified E-Waste Recycler."
    }
}

# ---------- INPUT ----------
user_input = st.text_input("Enter waste item")

if user_input:
    input_vec = vectorizer.transform([user_input])
    probabilities = model.predict_proba(input_vec)
    max_prob = np.max(probabilities)
    prediction = model.predict(input_vec)[0].lower()

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
        <div class="result-badge" style="background:{color_map.get(prediction, "#2ecc71")};">
            Category: {prediction.upper()}
        </div>
        """, unsafe_allow_html=True)

        # ---------- Disposal Guide ----------
        st.write("🌍 Disposal Guide:")

        if country in rules and prediction in rules[country]:
            st.write(rules[country][prediction])
        else:
            st.write("Disposal guide not available.")

        # ---------- Confidence ----------
        st.progress(int(max_prob * 100))
        st.write(f"Confidence: {round(max_prob,2)}")

        # ---------- GAMIFICATION ----------
        st.session_state.eco_points += 10
        st.success("🎉 +10 Eco Points Earned!")

# ---------- SCORE DISPLAY ----------
st.markdown("---")
st.markdown(f"### 🌱 Your Eco Points: {st.session_state.eco_points}")

# ---------- BADGE SYSTEM ----------
points = st.session_state.eco_points

if points >= 100:
    st.markdown("🏆 **Recycling Champion**")
elif points >= 50:
    st.markdown("🌍 **Green Warrior**")
elif points >= 20:
    st.markdown("🌿 **Eco Beginner**")

st.markdown("<br><hr><center>© 2026 Eco Recycling Assistant</center>", unsafe_allow_html=True)

