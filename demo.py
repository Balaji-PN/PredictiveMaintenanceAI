#!/usr/bin/env python3
"""
Demo script for Predictive Maintenance AI System
This script demonstrates how to use the trained model programmatically
"""

import joblib
import pandas as pd
import numpy as np

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('./src/model_rf.pkl')
        scaler = joblib.load('./src/scaler.pkl')
        print("✅ Model and scaler loaded successfully!")
        return model, scaler
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def create_sample_input():
    """Create a sample input for prediction"""
    sample_data = {
        'Type': 'M',  # Medium type machine
        'Air temperature [K]': 298.5,
        'Process temperature [K]': 308.8,
        'Rotational speed [rpm]': 1550,
        'Torque [Nm]': 45.2,
        'Tool wear [min]': 120,
        'TWF': 0,  # No tool wear failure
        'HDF': 0,  # No heat dissipation failure
        'PWF': 0,  # No power failure
        'OSF': 0,  # No overstrain failure
        'RNF': 0   # No random failure
    }
    return sample_data

def prepare_features(input_data):
    """Prepare features for the model"""
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Create one-hot encoded Type columns (Type_H is reference category)
    df['Type_L'] = (df['Type'] == 'L').astype(int)
    df['Type_M'] = (df['Type'] == 'M').astype(int)
    
    # Drop the original Type column
    df = df.drop('Type', axis=1)
    
    # Add dummy UDI and Product ID
    df['UDI'] = 99999
    df['Product ID'] = 'DEMO'
    
    # Reorder columns to match training data
    expected_columns = ['UDI', 'Product ID', 'Air temperature [K]', 
                       'Process temperature [K]', 'Rotational speed [rpm]', 
                       'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
                       'Type_L', 'Type_M']
    df = df[expected_columns]
    
    # Remove non-feature columns
    features = df.drop(['UDI', 'Product ID'], axis=1)
    
    return features

def make_prediction(model, features):
    """Make prediction using the loaded model"""
    try:
        # Get prediction and probability
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        return prediction, prediction_proba
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None, None

def main():
    """Main demo function"""
    print("🔧 Predictive Maintenance AI System - Demo")
    print("=" * 50)
    
    # Load model
    model, scaler = load_model_and_scaler()
    if model is None:
        return
    
    # Create sample input
    print("\n📝 Sample Input Data:")
    sample_input = create_sample_input()
    for key, value in sample_input.items():
        print(f"  {key}: {value}")
    
    # Prepare features
    print("\n⚙️  Preparing features...")
    features = prepare_features(sample_input)
    print(f"  Feature shape: {features.shape}")
    
    # Make prediction
    print("\n🔮 Making prediction...")
    prediction, probability = make_prediction(model, features)
    
    if prediction is not None:
        print(f"  Prediction: {'FAILURE' if prediction == 1 else 'NO FAILURE'}")
        print(f"  No Failure Probability: {probability[0]*100:.2f}%")
        print(f"  Failure Probability: {probability[1]*100:.2f}%")
        
        # Interpretation
        if prediction == 0:
            print("  ✅ Machine is operating normally")
        else:
            print("  🚨 Machine failure predicted - maintenance required!")
    
    print("\n🎯 Demo completed!")
    print("💡 Run 'streamlit run app.py' to use the interactive web interface")

if __name__ == "__main__":
    main() 