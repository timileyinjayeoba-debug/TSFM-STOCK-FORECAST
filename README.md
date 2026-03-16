🚀 **TSFM Crypto Price Forecasting System**

📌 **Overview**

This project is an AI-powered cryptocurrency forecasting system designed to predict the future prices of major digital assets using both modern transformer-based time-series models and traditional deep learning approaches.

The system combines:

1. A pretrained Amazon Chronos Time-Series Transformer from Amazon

2. A Long Short-Term Memory (LSTM) neural network

3. An interactive Streamlit web application for visualization and prediction

The application generates 14-day probabilistic forecasts, confidence intervals, and interactive charts for real-time analysis.

🎯 **Project Objectives**

1. Build an end-to-end AI forecasting pipeline

2. Compare transformer-based vs recurrent neural network models

3. Provide interpretable probabilistic forecasts

4. Deploy an interactive web dashboard for users

5. Demonstrate practical skills in AI Engineering & MLOps

📊 **Forecasted Assets**

Bitcoin

Ethereum

🧠 Models Used

1️⃣ **Amazon Chronos Time-Series Transformer**
. Pretrained foundation model for time-series forecasting
. Captures complex temporal dependencies
. Produces probabilistic forecasts
. Supports uncertainty estimation

2️⃣ **LSTM Neural Network**
. Traditional sequential deep-learning model
. Learns long-term dependencies in financial time series
. Useful baseline for model comparison

🌐 **Streamlit Web Application**
The project includes a fully interactive dashboard built with
Streamlit

**Users can**:

✅ Run price predictions
✅ Visualize historical trends
✅ View forecast confidence intervals
✅ Compare model outputs
✅ Interact with dynamic charts

🏗️ **Project Architecture**
TSFM-STOCK-FORECAST
│
├── app.py                # Streamlit web application
├── configs/              # Model & experiment configurations
├── data/                 # Input datasets
├── outputs/              # Forecast results
├── scripts/              # Training & forecasting scripts
├── src/                  # Core ML pipeline modules
├── requirements.txt      # Project dependencies
└── runtime.txt           # Deployment runtime settings

⚙️ **Key Features**

. Transformer-based forecasting pipeline

. LSTM baseline model

. Feature engineering for financial time-series

. Probabilistic multi-step forecasting

. Interactive visualization dashboard

. Modular and scalable code structure

. Ready for cloud deployment

🚀 Running the Project
1️⃣ **Install dependencies**
pip install -r requirements.txt
2️⃣ **Run Streamlit App**
streamlit run app.py

📈 **Future Improvements**

. Add more crypto assets

. Integrate real-time data APIs

. Deploy on cloud platforms

. Implement model performance tracking

. Add hyperparameter optimization

. Introduce automated retraining pipelines

👨‍💻 **Author**

AI Engineer | Data Scientist | Time-Series Forecasting Enthusiast
