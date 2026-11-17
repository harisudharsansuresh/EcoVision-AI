# EcoVision AI – Smart Waste Segregation and Dumping Detection

## Project Overview
EcoVision AI is a deep learning-based project designed to support smart waste management through image-based waste classification.

## Features
- AI-based waste classification.
- Classifies waste into six categories:
  - Cardboard
  - Glass
  - Metal
  - Paper
  - Plastic
  - Trash
- Web-based interface for uploading waste images.
- Displays predicted waste category and confidence score.
- Dumping detection is a planned feature and is not yet implemented.

## Technologies Used
- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- Flask
- HTML, CSS and JavaScript

## Project Structure
- `backend/` – Flask API
- `frontend/` – Website interface
- `models/` – Trained CNN model and class names
- `datasets/` – Training dataset (not included in GitHub)

## How to Run
1. Install the required Python packages.
2. Run the Flask backend:
   `python backend/app.py`
3. Open the frontend `index.html` file in your browser.
4. Upload a waste image to test the classifier.

## Project Status
The waste classification module is implemented as a prototype. Dumping detection is planned for future development.