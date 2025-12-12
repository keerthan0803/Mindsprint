My Smart Farm - Crop Recommendation System
A comprehensive agricultural decision support platform providing AI-powered crop recommendations, disease detection, fertilizer planning, and yield prediction.

Features
Crop Recommendation: AI model suggests top crops based on soil parameters (N, P, K, pH, temperature, humidity, rainfall) with irrigation guidance crop.html:118-158
Leaf Disease Detection: Upload plant leaf images for AI-powered disease identification with treatment recommendations disease predictor.html:234-253
Fertilizer Prediction: Get fertilizer recommendations based on crop type and soil values with weather integration fertilizer.html:318-340
Yield Prediction: Estimate crop yield based on farm metrics and conditions
Market Prices: View real-time crop market prices from Data.gov.in API
Multi-language Support: English, Hindi, and Telugu translations index.html:458-464
Responsive Design: Works on mobile, tablet, and desktop devices
Theme System: Light, dark, and forest themes index.html:575-594
Architecture
The application follows a three-tier architecture:

Client Tier: HTML pages with embedded JavaScript
Server Tier: Flask application (app.py) with REST API endpoints
Model Tier: Machine learning models and rule-based systems
API Endpoints
Endpoint	Method	Description
/api/predict	POST	Disease prediction from image upload
/api/crop-predict	POST	Crop recommendation based on soil parameters app.py:239-295
/api/fertilizer-predict	POST	Fertilizer recommendation with weather integration
/api/yield-predict	POST	Yield estimation
/api/weather	POST	Weather data from OpenWeatherMap API app.py:68-83
/api/model-status	GET	Health check for model availability
Installation
Prerequisites
Python 3.7+
Flask
TensorFlow/Keras
NumPy
scikit-learn
PIL (Pillow)
Requests
Model Files
The application requires the following model files (with fallback support):

ensemble_crop_model.pkl or crop_recommendation_model.pkl (crop recommendation) app.py:27-40
crop_disease_model.h5 (disease detection)
class_names.pkl or classes.pkl (disease class labels)
label_encoders.pkl (yield prediction)
disease_info.json (optional, generated fallback if missing)
Setup
Clone the repository
Install dependencies: pip install -r requirements.txt
Place model files in the root directory
Set up OpenWeatherMap API key in app.py app.py:65-67
Run the application: python app.py
Usage
Open the application in your browser at http://localhost:5000
Navigate to desired features from the main dashboard index.html:472-536
For crop recommendations:
Enter soil parameters (N, P, K values)
Provide environmental data (temperature, humidity, pH, rainfall)
Click "Predict Best Crop" for AI recommendations
System includes client-side fallback if model is unavailable crop.html:350-411
Security Features
File upload validation with extension whitelist (png, jpg, jpeg) app.py:13-21
Filename sanitization using secure_filename()
16MB file size limit for uploads
Input validation on all API endpoints
External Integrations
OpenWeatherMap API: For weather data integration
Data.gov.in API: For real-time crop market prices
Notes
The application implements graceful fallbacks when models are not available
All client-side features include offline capabilities
Theme and language preferences are persisted in localStorage
The system is designed for Indian agricultural contexts with regional language support
Notes
The README content is based on the actual implementation in the codebase, including the three-tier architecture, API endpoints, model loading with fallback mechanisms, and the multi-language responsive frontend design. The application is specifically designed for agricultural decision support with features like crop recommendation, disease detection, and fertilizer planning.

Wiki pages you might want to explore:

System Architecture (keerthan0803/Mindsprint)
