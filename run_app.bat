@echo off
echo 🚀 Starting Predictive Maintenance AI System...
echo 📦 Installing dependencies...
pip install -r requirements.txt

echo 🔧 Starting Streamlit application...
echo 🌐 The app will open in your browser at http://localhost:8501
echo ⏹️  Press Ctrl+C to stop the application

streamlit run app.py
pause 