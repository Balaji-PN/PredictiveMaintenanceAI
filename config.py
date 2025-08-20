# Configuration file for Predictive Maintenance AI System

# Model paths
MODEL_PATH = "./src/model_rf.pkl"
SCALER_PATH = "./src/scaler.pkl"
DATA_PATH = "./data/ai4i2020.csv"

# Feature columns (excluding UDI and Product ID)
FEATURE_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF',
    'Type_L',
    'Type_M'
]

# Machine type mapping
TYPE_MAPPING = {
    'L': 0,  # Low
    'M': 1,  # Medium
    'H': 2   # High
}

# Input validation ranges
INPUT_RANGES = {
    'air_temperature': {'min': 295.0, 'max': 305.0, 'default': 298.0},
    'process_temperature': {'min': 305.0, 'max': 315.0, 'default': 308.0},
    'rotational_speed': {'min': 1000, 'max': 3000, 'default': 1500},
    'torque': {'min': 10.0, 'max': 80.0, 'default': 40.0},
    'tool_wear': {'min': 0, 'max': 250, 'default': 100}
}

# App configuration
APP_CONFIG = {
    'page_title': "Predictive Maintenance AI",
    'page_icon': "🔧",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Color schemes
COLORS = {
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'primary': '#007bff'
} 