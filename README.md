# Cold Chain Predictive Maintenance System

## Project Directory: cold-chain-prediction
## Clone this repository
git clone https//......

## Installation
1. cd cold-chain-prediction
2. python -m venv venv
3. venv\Scripts\activate
4. pip install -r requirements.txt

## Run orderly
Terminal 1: python scripts/01_generate_data.py, python scripts/02_preprocess_data.py , python scripts/03_train_xgboost.py, and then  python scripts/04_train_lstm.py

Terminal 2: python scripts/05_api.py

Terminal 3: streamlit run scripts/06_dashboard.py
