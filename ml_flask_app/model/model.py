import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
import pickle

# Load pre-trained model (assume model is saved as 'model.pkl')
with open('model.pkl', 'rb') as f:
    ada_model = pickle.load(f)

# Load scaler (assume it's saved as 'scaler.pkl')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_target(synthesis_technique, catalyst_loading, electrode_substrate, morphology, potential):
    # Prepare the input features as a DataFrame
    user_input = pd.DataFrame([[synthesis_technique, catalyst_loading, electrode_substrate, morphology, potential]],
                              columns=['Synthesis technique', 'Catalyst loading (mg/cm2)', 'Electrode/Substrate', 'Morphology', 'Potential'])

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Make the prediction using the trained model
    prediction = ada_model.predict(user_input_scaled)
    
    return prediction[0]
