import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('./src/model_rf.pkl')
        scaler = joblib.load('./src/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load sample data for visualization
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv('./data/ai4i2020.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.title("🔧 Predictive Maintenance AI System")
    st.markdown("---")
    
    # Load model and data
    model, scaler = load_model()
    df = load_sample_data()
    
    if model is None or scaler is None:
        st.error("Failed to load the model. Please check if the model files exist in the src folder.")
        return
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Analysis", "🔮 Prediction", "📈 Model Performance", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "📊 Data Analysis":
        show_data_analysis(df)
    elif page == "🔮 Prediction":
        show_prediction_page(model, scaler)
    elif page == "📈 Model Performance":
        show_model_performance(df, model)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(df):
    st.header("Welcome to Predictive Maintenance AI")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This system uses machine learning to predict potential machine failures based on various sensor readings and operational parameters.
        
        **Key Features:**
        - 🔮 **Real-time Predictions**: Get instant failure predictions
        - 📊 **Data Visualization**: Explore patterns in your data
        - 📈 **Performance Metrics**: Understand model accuracy
        - 🎯 **User-friendly Interface**: Easy input and interpretation
        
        **How it works:**
        1. Input machine parameters (temperature, speed, torque, etc.)
        2. Our AI model analyzes the data
        3. Get instant predictions with confidence scores
        4. Take preventive actions based on insights
        """)
    
    with col2:
        if df is not None:
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Features", f"{df.shape[1]-1}")
            st.metric("Failure Rate", f"{(df['Machine failure'].sum()/len(df)*100):.1f}%")
    
    st.markdown("---")
    
    # Quick stats
    if df is not None:
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Air Temperature", f"{df['Air temperature [K]'].mean():.1f} K")
        with col2:
            st.metric("Process Temperature", f"{df['Process temperature [K]'].mean():.1f} K")
        with col3:
            st.metric("Rotational Speed", f"{df['Rotational speed [rpm]'].mean():.0f} rpm")
        with col4:
            st.metric("Torque", f"{df['Torque [Nm]'].mean():.1f} Nm")

def show_data_analysis(df):
    st.header("📊 Data Analysis & Visualization")
    
    if df is None:
        st.error("No data available for analysis")
        return
    
    # Data overview
    st.subheader("Dataset Information")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Features:**", list(df.columns[:-1]))
        st.write("**Target:**", "Machine failure")
    
    with col2:
        st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Select feature to visualize
    feature_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    selected_feature = st.selectbox("Select feature to visualize:", feature_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x=selected_feature, color='Machine failure', 
                          title=f"Distribution of {selected_feature}",
                          color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, x='Machine failure', y=selected_feature,
                    title=f"{selected_feature} vs Machine Failure",
                    color='Machine failure',
                    color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(correlation_matrix,
                    title="Feature Correlation Heatmap",
                    color_continuous_scale='RdBu',
                    aspect='auto')
    st.plotly_chart(fig, use_container_width=True)
    
    # Failure analysis
    st.markdown("---")
    st.subheader("Failure Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        failure_counts = df['Machine failure'].value_counts()
        fig = px.pie(values=failure_counts.values, 
                    names=['No Failure', 'Failure'],
                    title="Machine Failure Distribution",
                    color_discrete_sequence=['green', 'red'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        failure_by_type = df.groupby('Type')['Machine failure'].sum()
        fig = px.bar(x=failure_by_type.index, y=failure_by_type.values,
                    title="Failures by Machine Type",
                    labels={'x': 'Machine Type', 'y': 'Number of Failures'})
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(model, scaler):
    st.header("🔮 Machine Failure Prediction")
    
    st.markdown("Enter the machine parameters below to get a failure prediction:")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Machine Parameters")
            
            # Machine type
            machine_type = st.selectbox(
                "Machine Type",
                options=['L', 'M', 'H'],
                help="L: Low, M: Medium, H: High"
            )
            
            # Temperature inputs
            air_temp = st.number_input(
                "Air Temperature [K]",
                min_value=295.0,
                max_value=305.0,
                value=298.0,
                step=0.1,
                help="Ambient air temperature in Kelvin"
            )
            
            process_temp = st.number_input(
                "Process Temperature [K]",
                min_value=305.0,
                max_value=315.0,
                value=308.0,
                step=0.1,
                help="Process temperature in Kelvin"
            )
            
            # Speed and torque
            rotational_speed = st.number_input(
                "Rotational Speed [rpm]",
                min_value=1000,
                max_value=3000,
                value=1500,
                step=10,
                help="Machine rotational speed in RPM"
            )
            
            torque = st.number_input(
                "Torque [Nm]",
                min_value=10.0,
                max_value=80.0,
                value=40.0,
                step=0.1,
                help="Machine torque in Newton-meters"
            )
            
            tool_wear = st.number_input(
                "Tool Wear [min]",
                min_value=0,
                max_value=250,
                value=100,
                step=1,
                help="Tool wear time in minutes"
            )
        
        with col2:
            st.subheader("Failure Indicators")
            
            # Binary failure indicators
            twf = st.checkbox("Tool Wear Failure (TWF)")
            hdf = st.checkbox("Heat Dissipation Failure (HDF)")
            pwf = st.checkbox("Power Failure (PWF)")
            osf = st.checkbox("Overstrain Failure (OSF)")
            rnf = st.checkbox("Random Failure (RNF)")
            
            st.markdown("---")
            st.markdown("**Note:** These are binary indicators (0 or 1)")
        
        submitted = st.form_submit_button("🔮 Predict Failure", type="primary")
        
        if submitted:
            # Prepare input data
            input_data = {
                'Type': machine_type,
                'Air temperature [K]': air_temp,
                'Process temperature [K]': process_temp,
                'Rotational speed [rpm]': rotational_speed,
                'Torque [Nm]': torque,
                'Tool wear [min]': tool_wear,
                'TWF': 1 if twf else 0,
                'HDF': 1 if hdf else 0,
                'PWF': 1 if pwf else 0,
                'OSF': 1 if osf else 0,
                'RNF': 1 if rnf else 0
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Create one-hot encoded Type columns (Type_H is reference category)
            input_df['Type_L'] = (input_df['Type'] == 'L').astype(int)
            input_df['Type_M'] = (input_df['Type'] == 'M').astype(int)
            
            # Drop the original Type column
            input_df = input_df.drop('Type', axis=1)
            
            # Add UDI and Product ID (dummy values)
            input_df['UDI'] = 99999
            input_df['Product ID'] = 'PREDICT'
            
            # Reorder columns to match training data
            expected_columns = ['UDI', 'Product ID', 'Air temperature [K]', 
                              'Process temperature [K]', 'Rotational speed [rpm]', 
                              'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
                              'Type_L', 'Type_M']
            input_df = input_df[expected_columns]
            
            # Make prediction
            try:
                # Remove non-feature columns
                features = input_df.drop(['UDI', 'Product ID'], axis=1)
                
                # Make prediction
                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 0:
                        st.success("✅ **NO FAILURE PREDICTED**")
                        st.metric("Confidence", f"{prediction_proba[0]*100:.1f}%")
                    else:
                        st.error("🚨 **FAILURE PREDICTED**")
                        st.metric("Confidence", f"{prediction_proba[1]*100:.1f}%")
                
                with col2:
                    st.metric("No Failure Probability", f"{prediction_proba[0]*100:.1f}%")
                    st.metric("Failure Probability", f"{prediction_proba[1]*100:.1f}%")
                
                with col3:
                    # Create gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction_proba[1]*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Failure Risk %"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("---")
                st.subheader("💡 Recommendations")
                
                if prediction == 0:
                    st.success("""
                    **Machine is operating normally!** 
                    - Continue regular maintenance schedule
                    - Monitor parameters for any significant changes
                    - Keep tracking tool wear and other indicators
                    """)
                else:
                    st.warning("""
                    **Immediate attention required!**
                    - Schedule maintenance as soon as possible
                    - Check for any unusual vibrations or sounds
                    - Review operating parameters and adjust if needed
                    - Consider reducing load or speed temporarily
                    """)
                
                # Show input summary
                st.markdown("---")
                st.subheader("📋 Input Summary")
                st.json(input_data)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

def show_model_performance(df, model):
    st.header("📈 Model Performance Analysis")
    
    if df is None:
        st.error("No data available for analysis")
        return
    
    # Prepare data for performance analysis
    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]
    
    # Remove non-feature columns
    feature_cols = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
                   'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    # Encode machine type
    X_encoded = X.copy()
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    X_encoded['Type'] = X_encoded['Type'].map(type_mapping)
    
    # Select only feature columns
    X_features = X_encoded[feature_cols]
    
    # Make predictions
    y_pred = model.predict(X_features)
    y_pred_proba = model.predict_proba(X_features)
    
    # Performance metrics
    st.subheader("Model Accuracy Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    with col1:
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
    with col2:
        st.metric("Precision", f"{precision*100:.2f}%")
    with col3:
        st.metric("Recall", f"{recall*100:.2f}%")
    with col4:
        st.metric("F1-Score", f"{f1*100:.2f}%")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    cm = confusion_matrix(y, y_pred)
    
    fig = px.imshow(cm,
                    text_auto=True,
                    aspect="auto",
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    color_continuous_scale='Blues')
    
    fig.update_xaxes(side="bottom")
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader("Detailed Classification Report")
    
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    st.dataframe(report_df, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature',
                    orientation='h',
                    title="Random Forest Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Distribution
    st.markdown("---")
    st.subheader("Prediction Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        actual_counts = pd.Series(y).value_counts()
        fig = px.pie(values=actual_counts.values, 
                    names=['No Failure', 'Failure'],
                    title="Actual vs Predicted (Actual)",
                    color_discrete_sequence=['green', 'red'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        pred_counts = pd.Series(y_pred).value_counts()
        fig = px.pie(values=pred_counts.values, 
                    names=['No Failure', 'Failure'],
                    title="Actual vs Predicted (Predicted)",
                    color_discrete_sequence=['green', 'red'])
        st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## Predictive Maintenance AI System
    
    This application uses machine learning to predict potential machine failures based on various sensor readings and operational parameters.
    
    ### 🎯 Purpose
    The goal is to help maintenance teams identify potential issues before they cause unplanned downtime, reducing costs and improving operational efficiency.
    
    ### 🔧 Features Used
    - **Machine Type**: L (Low), M (Medium), H (High)
    - **Temperature Sensors**: Air and Process temperatures
    - **Operational Parameters**: Rotational speed, Torque, Tool wear
    - **Failure Indicators**: TWF, HDF, PWF, OSF, RNF
    
    ### 🤖 Model Details
    - **Algorithm**: Random Forest Classifier
    - **Training Data**: 10,000+ machine records
    - **Accuracy**: High performance on failure prediction
    - **Output**: Binary classification (Failure/No Failure)
    
    ### 📊 How to Use
    1. Navigate to the **Prediction** page
    2. Input your machine parameters
    3. Get instant failure predictions
    4. Review confidence scores and recommendations
    
    ### 🚀 Benefits
    - **Preventive Maintenance**: Catch issues before they escalate
    - **Cost Reduction**: Avoid unplanned downtime
    - **Data-Driven Decisions**: Make informed maintenance choices
    - **Real-time Monitoring**: Get instant insights
    
    ### 📈 Performance
    The model has been trained on comprehensive machine data and provides reliable predictions for maintenance planning.
    """)
    
    st.markdown("---")
    
    st.subheader("🔍 Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Sources:**
        - Sensor readings
        - Operational logs
        - Maintenance records
        - Historical failure data
        
        **Model Features:**
        - 12 input parameters
        - Standardized scaling
        - Ensemble learning
        - Cross-validation
        """)
    
    with col2:
        st.markdown("""
        **Technology Stack:**
        - Python 3.x
        - Scikit-learn
        - Streamlit
        - Plotly
        - Pandas/NumPy
        
        **Deployment:**
        - Web-based interface
        - Real-time predictions
        - Responsive design
        - Easy integration
        """)

if __name__ == "__main__":
    main() 