## Eco Recycling Chatbot

Eco Recycling Chatbot is an AI-powered waste classification web application that helps users identify how to dispose of waste responsibly.
This project combines Natural Language Processing (NLP), Computer Vision (CV), Localization, and Gamification into a complete end-to-end Machine Learning solution deployed using Streamlit.

## Live Demo:

##  https://ecorecycling-chatbot.streamlit.app/

## Project Overview:

  The Eco Recycling Chatbot is a smart web application that:
  Classifies waste using text input (NLP model)
  Classifies waste using image upload (CNN model)
  Provides country-specific recycling guidelines
  Includes gamification with eco-points and achievement badges
  Is publicly deployed using Streamlit Cloud

## Machine Learning Models:
## 1️ Text Classification Model
  Built using:
  TF-IDF (Term Frequency–Inverse Document Frequency)
  Logistic Regression (Multiclass Classification)
  Custom balanced dataset
  Confidence threshold handling
  Input validation logic

# Categories Predicted:
  Recyclable
  Organic
  Trash
  Hazardous
  E-Waste

# Model Files:
  recycling_model.pkl
  vectorizer.pkl

# Training notebook:
   model_training.ipynb

## 2️ Image Classification Model
  Built using:
  TrashNet Dataset (6 classes)
  Transfer Learning with MobileNetV2 (Pre-trained on ImageNet)
  Frozen base layers
  Custom dense layers
  Softmax output layer (6 classes)

# Categories Predicted:
   Cardboard
   Glass
   Metal
   Paper
   Plastic
   Trash

# Image Preprocessing:
  Resize to 224 × 224 pixels
  Normalize pixel values
  Convert to NumPy array
  Expand dimensions before prediction

# Model Handling:
  Trained model weights saved after training
  Loaded into Streamlit app for real-time prediction

# Localized Recycling Guidelines:

  The system provides disposal instructions based on selected country:
  🇮🇳 India
  🇸🇬 Singapore
  🇺🇸 United States
  🇬🇧 United Kingdom
  The app adapts recycling guidance according to local waste management rules.

# Gamification Features:

  To promote environmental awareness:
  +10 Eco Points for every correct classification

# Achievement badges:

  Eco Beginner
  Green Warrior
  Recycling Champion
  This encourages responsible behavior and user engagement.

# User Interface:
  
  Built using Streamlit with:
  Custom CSS styling
  Gradient backgrounds
  Animated result cards
  Responsive layout
  Confidence score progress bar
  Sidebar country selector
  Text and image input modes
  The goal was to create a modern, interactive web application.

# Project Structure:
  Eco-Recycling-Chatbot/
  │
  ├── app.py
  ├── recycling_model.pkl
  ├── vectorizer.pkl
  ├── requirements.txt
  │
  ├── training/
  │   ├── model_training.ipynb
  │   ├── recycling_dataset.csv
  │
  ├── image_model/
  │   ├── mobilenet_model.h5 (or saved weights)
  │
  └── README.md


# How to Run Locally:

  Clone the repository
  Install dependencies:
  pip install -r requirements.txt
  Run the application:
  streamlit run app.py

  
# Future Improvements:

  Persistent eco-score storage
  User authentication system
  Leaderboard with database integration
  Cloud database (Firebase / MongoDB)
  Full frontend framework integration
  Mobile application version

# Technologies Used:

  Python
  Scikit-learn
  TensorFlow / Keras
  MobileNetV2
  Streamlit
  NumPy
  HTML & CSS (custom styling)

# Author:

  This project was developed as a complete end-to-end Machine Learning application covering:
  Data preprocessing
  Model training (NLP + CV)
  Model evaluation

Web application development

Cloud deployment

It demonstrates the practical application of AI in promoting sustainable waste management and responsible environmental behavior.
