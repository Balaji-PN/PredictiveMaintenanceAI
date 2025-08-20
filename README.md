# Predictive Maintenance AI System

A machine learning-based system for predicting machine failures using Random Forest classification.

## 🚀 Features

- **Real-time Predictions**: Get instant machine failure predictions
- **Interactive Dashboard**: Beautiful Streamlit interface with data visualizations
- **Data Analysis**: Explore patterns and correlations in your data
- **Model Performance**: View detailed accuracy metrics and feature importance
- **User-friendly Input**: Easy parameter input with validation

## 📋 Prerequisites

- Python 3.8 or higher
- Required packages (see requirements.txt)

## 🛠️ Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd predictive_maintenance_ai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure your model files are in the `src/` folder:
   - `model_rf.pkl` - Trained Random Forest model
   - `scaler.pkl` - Data scaler for preprocessing

## 🚀 Running the Application

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Open your browser and navigate to the displayed URL (usually http://localhost:8501)

## 📊 Application Pages

### 🏠 Home

- Overview of the system
- Quick statistics and dataset information

### 📊 Data Analysis

- Feature distributions and correlations
- Interactive visualizations
- Failure analysis by machine type

### 🔮 Prediction

- Input machine parameters
- Get instant failure predictions
- View confidence scores and recommendations

### 📈 Model Performance

- Accuracy metrics and confusion matrix
- Feature importance analysis
- Detailed classification report

### ℹ️ About

- System information and technical details
- Usage instructions and benefits

## 🔧 Input Parameters

The model requires the following parameters:

- **Machine Type**: L (Low), M (Medium), H (High)
- **Air Temperature [K]**: Ambient temperature in Kelvin
- **Process Temperature [K]**: Process temperature in Kelvin
- **Rotational Speed [rpm]**: Machine speed in RPM
- **Torque [Nm]**: Machine torque in Newton-meters
- **Tool Wear [min]**: Tool wear time in minutes
- **Failure Indicators**: TWF, HDF, PWF, OSF, RNF (binary)

## 📈 Model Details

- **Algorithm**: Random Forest Classifier
- **Training Data**: 10,000+ machine records
- **Features**: 12 input parameters
- **Output**: Binary classification (Failure/No Failure)
- **Performance**: High accuracy on failure prediction

## 🎯 Use Cases

- **Manufacturing**: Predict machine failures in production lines
- **Maintenance Planning**: Schedule preventive maintenance
- **Cost Reduction**: Avoid unplanned downtime
- **Quality Assurance**: Monitor machine health parameters

## 🔍 Data Sources

The model was trained on the AI4I 2020 Predictive Maintenance Dataset, which includes:

- Sensor readings from various machine types
- Operational parameters and environmental conditions
- Historical failure records and maintenance data

## 🛡️ Security Notes

- The application runs locally and doesn't send data to external servers
- Model files should be kept secure and not shared publicly
- Input data is processed locally for predictions

## 🤝 Contributing

Feel free to contribute to this project by:

- Reporting bugs
- Suggesting new features
- Improving the user interface
- Enhancing the machine learning model

## 📄 License

This project is for educational and research purposes.

## 📞 Support

For questions or issues, please check the documentation or create an issue in the repository.
