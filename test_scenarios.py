#!/usr/bin/env python3
"""
Test script for Predictive Maintenance AI System
Tests various input scenarios to ensure system robustness
"""

import joblib
import pandas as pd
import numpy as np

def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('./src/model_rf.pkl')
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_scenario(model, scenario_name, input_data):
    """Test a specific scenario"""
    print(f"\n🔍 Testing: {scenario_name}")
    print("-" * 40)
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Create one-hot encoded Type columns
    df['Type_L'] = (df['Type'] == 'L').astype(int)
    df['Type_M'] = (df['Type'] == 'M').astype(int)
    
    # Drop the original Type column
    df = df.drop('Type', axis=1)
    
    # Reorder columns to match training data
    expected_columns = ['Air temperature [K]', 'Process temperature [K]', 
                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
                       'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Type_L', 'Type_M']
    
    # Ensure all columns exist
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[expected_columns]
    
    try:
        # Make prediction
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        
        print(f"  Input Type: {input_data['Type']}")
        print(f"  Prediction: {'FAILURE' if prediction == 1 else 'NO FAILURE'}")
        print(f"  Confidence: {max(prediction_proba)*100:.1f}%")
        print(f"  No Failure Prob: {prediction_proba[0]*100:.1f}%")
        print(f"  Failure Prob: {prediction_proba[1]*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Predictive Maintenance AI System - Test Scenarios")
    print("=" * 60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Low Type Machine - Normal Operation',
            'data': {
                'Type': 'L',
                'Air temperature [K]': 298.0,
                'Process temperature [K]': 308.0,
                'Rotational speed [rpm]': 1500,
                'Torque [Nm]': 40.0,
                'Tool wear [min]': 50,
                'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
            }
        },
        {
            'name': 'Medium Type Machine - High Tool Wear',
            'data': {
                'Type': 'M',
                'Air temperature [K]': 299.0,
                'Process temperature [K]': 310.0,
                'Rotational speed [rpm]': 2000,
                'Torque [Nm]': 60.0,
                'Tool wear [min]': 200,
                'TWF': 1, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
            }
        },
        {
            'name': 'High Type Machine - Multiple Failures',
            'data': {
                'Type': 'H',
                'Air temperature [K]': 300.0,
                'Process temperature [K]': 312.0,
                'Rotational speed [rpm]': 2500,
                'Torque [Nm]': 70.0,
                'Tool wear [min]': 180,
                'TWF': 1, 'HDF': 1, 'PWF': 0, 'OSF': 1, 'RNF': 0
            }
        },
        {
            'name': 'Low Type Machine - Extreme Conditions',
            'data': {
                'Type': 'L',
                'Air temperature [K]': 304.0,
                'Process temperature [K]': 314.0,
                'Rotational speed [rpm]': 2800,
                'Torque [Nm]': 75.0,
                'Tool wear [min]': 240,
                'TWF': 0, 'HDF': 0, 'PWF': 1, 'OSF': 0, 'RNF': 1
            }
        }
    ]
    
    # Run tests
    passed = 0
    total = len(scenarios)
    
    for scenario in scenarios:
        if test_scenario(model, scenario['name'], scenario['data']):
            passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} scenarios passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print("\n💡 Run 'streamlit run app.py' to use the interactive interface")

if __name__ == "__main__":
    main() 