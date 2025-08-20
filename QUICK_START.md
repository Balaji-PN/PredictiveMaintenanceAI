# 🚀 Quick Start Guide - Predictive Maintenance AI

## ⚡ Get Running in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Open Your Browser

Navigate to: `http://localhost:8501`

## 🔧 What You'll Get

- **🏠 Home Page**: System overview and quick stats
- **📊 Data Analysis**: Interactive visualizations of your data
- **🔮 Prediction**: Input machine parameters and get instant failure predictions
- **📈 Model Performance**: View model accuracy and feature importance
- **ℹ️ About**: System information and usage guide

## 🎯 Quick Test

Run the demo script to test the system:

```bash
python demo.py
```

## 📋 Input Parameters

| Parameter           | Description                   | Range         |
| ------------------- | ----------------------------- | ------------- |
| Machine Type        | L (Low), M (Medium), H (High) | L/M/H         |
| Air Temperature     | Ambient temperature           | 295-305 K     |
| Process Temperature | Process temperature           | 305-315 K     |
| Rotational Speed    | Machine speed                 | 1000-3000 rpm |
| Torque              | Machine torque                | 10-80 Nm      |
| Tool Wear           | Tool wear time                | 0-250 min     |
| TWF/HDF/PWF/OSF/RNF | Failure indicators            | 0 or 1        |

## 🚨 Troubleshooting

- **Model not found**: Ensure `src/model_rf.pkl` and `src/scaler.pkl` exist
- **Port already in use**: Change port with `streamlit run app.py --server.port 8502`
- **Dependencies missing**: Run `pip install -r requirements.txt`

## 💡 Pro Tips

- Use the sidebar to navigate between pages
- Input realistic values within the specified ranges
- Check the confidence scores for prediction reliability
- Use the data analysis page to understand your data patterns

## 🆘 Need Help?

- Check the full README.md for detailed documentation
- Run the demo script to verify system functionality
- Ensure all model files are in the correct locations
