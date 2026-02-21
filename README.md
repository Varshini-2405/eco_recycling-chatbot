## Eco Recycling Chatbot
Eco Recycling Chatbot is an AI-powered waste classification web application that helps users identify how to dispose of waste responsibly.
This project combines Machine Learning, Web Development, and Environmental Awareness features into a single interactive application.

## Live Demo:
  https://ecorecycling-chatbot.streamlit.app/

## Project Overview:

The Eco Recycling Chatbot is a smart web application that:
* Classifies waste items using a trained Machine Learning model
* Provides country-specific recycling guidelines
* Includes gamification with eco-points and achievement badges
* Is deployed publicly using Streamlit Cloud


## Machine Learning Model:

The text classification model was built using:
* TF-IDF (Term Frequency–Inverse Document Frequency)
* Logistic Regression (Multiclass Classification)
* A custom balanced dataset
* Confidence threshold handling
* Input validation logic

The model predicts the following categories:
* Recyclable
* Organic
* Trash
* Hazardous
* E-Waste

The trained model files included in this repository are:
* recycling_model.pkl
* vectorizer.pkl

The full training process is documented in the file:
* model_training.ipynb


## Localized Recycling Guidelines:

The system supports country-specific disposal instructions for:
* India
* Singapore
* United States
* United Kingdom
Based on the selected location, the application provides accurate disposal guidance according to local waste management practices.


## Gamification Features:

To encourage environmental awareness, the application includes:
* +10 Eco Points for every successful classification
* Achievement badges based on total points:
  * Eco Beginner
  * Green Warrior
  * Recycling Champion
This makes the application interactive and promotes sustainable habits.


## User Interface:

The web interface was built using Streamlit with:
* Custom CSS styling
* Gradient backgrounds
* Animated result cards
* Responsive layout
* Interactive confidence progress bar
The goal was to make the project look and feel like a modern web application.


## Project Structure:

The repository includes:
* app.py (Streamlit application)
* recycling_model.pkl (trained ML model)
* vectorizer.pkl (TF-IDF vectorizer)
* requirements.txt (dependencies)
* training folder containing:
  * model_training.ipynb
  * recycling_dataset.csv


## How to Run Locally:
1. Clone the repository
2. Install dependencies using pip install -r requirements.txt
3. Run the application using streamlit run app.py


## Future Improvements:
* Persistent eco-score storage
* Leaderboard system
* Image-based waste classification
* Database integration
* Full frontend framework integration


## Technologies Used:
* Python
* Scikit-learn
* Streamlit
* NumPy
* HTML and CSS (custom styling)


## Author:
This project was developed as a complete end-to-end Machine Learning application, covering model training, UI development, and public deployment.


