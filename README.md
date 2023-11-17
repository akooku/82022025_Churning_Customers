# Customer Churn Prediction Web App

This web application predicts customer churn using a machine learning model. It is built with Flask and uses a trained model to make predictions based on user input.

## Getting Started

1. **Installation:**
   - Clone the repository to your local machine.
   - Make sure you have Python installed (version 3.6 or higher).
   - Install the required dependencies using the following command:

     ```bash
     pip install -r requirements.txt
     ```

2. **Running the App:**
   - Open a terminal and navigate to the project directory.
   - Run the Flask app using the following command:

     ```bash
     python app.py
     ```

   - The app will be accessible at [http://localhost:5000/](http://localhost:5000/).

## File Structure

- **app.py:** The main Flask application file.
- **templates/index.html:** HTML template for the home page.
- **templates/result.html:** HTML template for displaying predictions and model evaluation results.
- **styles.css:** CSS file for styling the web pages.
- **best_model.joblib:** Pickle file containing the trained machine learning model.
- **label_encoder.pkl:** Pickle file containing label encoders for categorical features.
- **scalers.pkl:** Pickle file containing scalers for numerical features.
- **training_data.csv:** CSV file containing the training data used for model training.
- **validation_data.csv:** CSV file containing validation data for model evaluation.

## Usage

1. Open the web browser and navigate to [http://localhost:5000/](http://localhost:5000/).
2. Fill in the input form with the required information.
3. Click the "Predict" button to get the churn prediction.
4. The result page will display the churn prediction and the confidence level.

## Video Demonstration

Watch the video demonstration on [YouTube](https://youtu.be/z5VMZ0u8j7A).

## Acknowledgments

This project uses Flask, a lightweight WSGI web application framework, and scikit-learn, a machine learning library for Python.