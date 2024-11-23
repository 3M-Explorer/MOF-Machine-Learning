import pandas as pd
# Load the Excel file
df = pd.read_excel(r'D:\M.Sc Data\4th Semester -Thesis\ML-work\flask_app\data\atomcamp dataset 88 points.xlsx')

# Replace all spaces with underscores in all DataFrame values
df = df.applymap(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)

# Display the updated DataFrame
print(df.head())

from sklearn.preprocessing import LabelEncoder

# Define the columns to label encode
columns_to_encode = ['Synthesis technique', 'Cell', 'Electrolyte', 'Electrode/Substrate', 'Morphology']

# Initialize LabelEncoder
encoder = LabelEncoder()

# Apply Label Encoding to each specified column
for column in columns_to_encode:
    if column in df.columns:  # Check if the column exists in the DataFrame
        # Fit the encoder and transform the column
        df[column] = encoder.fit_transform(df[column])

        # Display the mapping of original values to encoded numbers
        print(f"Mapping for {column}:")
        for index, label in enumerate(encoder.classes_):
            print(f"{index} for {label}")
        print()

# Display the updated DataFrame
print(df.head())

# Drop the 'Material' column permanently
df.drop(columns=['Catalyst', 'pH.1', 'pH'], inplace=True)

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize Flask app
app = Flask(__name__)

# Sample data loading for the model (you can replace it with actual data)
# Here, you would load the df DataFrame from your data source
# df = pd.read_csv('your_data.csv')  # For example

# Define features and target
# Assuming you already have df as your dataset
# X = df[['Synthesis technique', 'Catalyst loading (mg/cm2)', 'Electrode/Substrate', 'Morphology', 'Potential']]
# y = df['FE (CO)']

# Preprocessing and model training (similar to your existing code)
X = df[['Synthesis technique', 'Catalyst loading (mg/cm2)', 'Electrode/Substrate', 'Morphology', 'Potential']]
y = df['FE (CO)']

# Split the data into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardize the data (scale the features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the AdaBoost Regressor
ada_model = AdaBoostRegressor(learning_rate=1.0, loss='linear', n_estimators=100, random_state=42)

# Fit the model on the training data
ada_model.fit(X_train_scaled, y_train)

# Create a prediction function
def predict_target(synthesis_technique, catalyst_loading, electrode_substrate, morphology, potential):
    # Prepare user input into a DataFrame
    user_input = pd.DataFrame([[synthesis_technique, catalyst_loading, electrode_substrate, morphology, potential]],
                              columns=['Synthesis technique', 'Catalyst loading (mg/cm2)', 'Electrode/Substrate', 'Morphology', 'Potential'])

    # Standardize the user input
    user_input_scaled = scaler.transform(user_input)

    # Make the prediction using the trained model
    prediction = ada_model.predict(user_input_scaled)

    return prediction[0]

# Flask route to render the form
@app.route('/')
def home():
    return render_template('index.html')  # You'll need an index.html file with a form for inputs

# Flask route to handle form submission and show prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        synthesis_technique = int(request.form['Synthesis technique'])
        catalyst_loading = float(request.form['Catalyst loading'])
        electrode_substrate = int(request.form['Electrode/Substrate'])
        morphology = int(request.form['Morphology'])
        potential = float(request.form['Potential'])
        
        # Get the prediction from the model
        prediction = predict_target(synthesis_technique, catalyst_loading, electrode_substrate, morphology, potential)
        
        return render_template('index.html', prediction=prediction)
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
