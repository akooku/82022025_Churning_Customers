from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

app = Flask(__name__)

# Specify the path to your pickle files
model_filename = 'best_model.joblib'
label_encoder_filename = 'label_encoder.pkl'
scaler_filename = 'scalers.pkl'

# Load the model from the file
with open(model_filename, 'rb') as file:
    model = joblib.load(file)


# Load standard scaler
with open(scaler_filename, 'rb') as file:
    scalers = pickle.load(file)

# Load the data used during training (or use a similar dataset)
training_data = pd.read_csv('training_data.csv') 

# Define the selected features
selected_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender',
                     'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'Contract',
                     'PaymentMethod']

categorical_features = ['gender', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'Contract', 'PaymentMethod']

label_encoders = {feature: LabelEncoder() for feature in categorical_features}
scalers = {feature: StandardScaler() for feature in selected_features}

# Fit the label encoders
for feature in categorical_features:
    label_encoders[feature].fit(training_data[feature])

with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Fit the scalers
for feature in selected_features:
    # Ensure the feature is present in the training data
    if feature in training_data.columns:
        # Extract the feature column and reshape it to a 2D array
        feature_data = training_data[feature].values.reshape(-1, 1)
        
        # Fit the scaler on the reshaped data
        scalers[feature].fit(feature_data)

with open('scalers.pkl', 'wb') as file:
    pickle.dump(scalers, file)

with open('scalers.pkl', 'rb') as file:
    scalers = pickle.load(file)

print(scalers.keys())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    input_df = pd.DataFrame.from_dict([input_data])

    # Convert categorical values to numerical before using label encoder
    for feature in categorical_features:
        # Add unique values from the form to the label encoder
        unique_values = set(input_df[feature].tolist())
        unique_values = unique_values - set(label_encoders[feature].classes_)
        
        # Update label encoder with new values
        label_encoders[feature].classes_ = np.concatenate([label_encoders[feature].classes_, list(unique_values)])

        # Transform the categorical values in the input data
        input_df[feature] = label_encoders[feature].transform(input_df[feature])

        # Fit the label encoder on the updated training data
        label_encoders[feature].fit(training_data[feature])

        if feature in input_df.columns:
            try:
                # Print statements for debugging
                print(f"Original value for {feature}: {input_df[feature].iloc[0]}")

                # Convert categorical value to numeric using label encoder
                input_df[feature] = label_encoders[feature].transform([input_df[feature].iloc[0]])[0]

                # Print statements for debugging
                print(f"Transformed value for {feature}: {input_df[feature].iloc[0]}")
            except Exception as e:
                print(f"Error converting {feature}: {e}")

    # Scale numerical variables
    for feature in scalers:
        scaler = scalers[feature]

        # Check if the feature is present in the input data
        if feature in input_df.columns:
            # Reshape the feature data to a 2D array and transform
            input_df[feature] = scaler.transform(input_df[[feature]].values.reshape(-1, 1))

    # Select only the required features
    input_df = input_df[selected_features]

    # Make prediction
    prediction = model.predict(input_df)

    # Assuming 'prediction' is your numerical churn prediction
    numerical_prediction = prediction[0][0]
    threshold = 0.5

    # Convert to categorical value
    categorical_prediction = "Yes" if numerical_prediction >= threshold else "No"

    # Load the validation dataset
    validation_data = pd.read_csv('validation_data.csv')

    # Extract features and labels from the validation dataset
    X_val = validation_data[selected_features]
    y_val = validation_data['Churn']  

    # Transform categorical features using label encoders and scale numerical features using scalers
    for feature in categorical_features:
        X_val[feature] = label_encoders[feature].transform(X_val[feature])

    for feature in scalers:
        scaler = scalers[feature]
        X_val[feature] = scaler.transform(X_val[[feature]].values.reshape(-1, 1))

    # Make predictions on the validation set
    predictions_val = model.predict(X_val)

    # Evaluate accuracy and AUC score
    y_val_pred = (predictions_val >= threshold).astype(int)
    #accuracy = accuracy_score(y_val, y_val_pred)
    auc_score = roc_auc_score(y_val, predictions_val)
    confidence_level = round(roc_auc_score(y_val, predictions_val) * 100)


    print(f'AUC Score on validation set: {auc_score}')

    # Return the prediction, accuracy, and AUC score
    return render_template('result.html', prediction_text=f'Churn Prediction: \n{categorical_prediction}',
                           auc_score_text=f'Confidence Level: {confidence_level}%')


if __name__ == '__main__':
    app.run(debug=True)