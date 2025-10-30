

🌧️ AI-Based Rainfall Prediction Using LSTM
Author: Timothy Kiprop
Context: Bomet University College Research Conference 2025
Keywords: LSTM, AI, Rainfall Prediction, Climate Change, Kericho, Bomet, Kenya

Overview
This project applies a Long Short-Term Memory (LSTM) neural network to predict rainfall patterns using historical climate data.
The model leverages temporal dependencies in rainfall records to forecast future precipitation levels, supporting climate-smart agriculture and environmental decision-making in Kericho and Bomet Counties, Kenya.

By accurately modeling rainfall variability, farmers and policymakers can better plan irrigation schedules, planting seasons, and soil conservation interventions to mitigate vegetation degradation and deforestation risks.

Features

Preprocesses rainfall data from ClimateEngine (CHIRPS dataset or similar sources)

Builds and trains an LSTM neural network using TensorFlow/Keras

Evaluates model performance using RMSE and MAE metrics

Generates rainfall forecasts and plots actual vs predicted rainfall trends

Exports predictions to a CSV file for further analysis

Methodology

Data Input
Load historical rainfall data (ClimateEngine.csv) containing daily or monthly totals.

Data Preprocessing

Resampling and scaling with MinMaxScaler

Sequence creation for time-series learning

Model Building

Sequential LSTM architecture

Dropout layer for regularization

Adam optimizer with MSE loss

Evaluation and Visualization

Metrics: RMSE, MAE

Predicted vs Actual rainfall plot

CSV output: lstm_predictions.csv

How to Run

Requirements
Create a virtual environment (optional but recommended):
python -m venv venv
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Run the model
Place your rainfall CSV file (e.g., ClimateEngine.csv) in the project folder, then run:
python ltsm_fixed.py

Output Files

lstm_predictions.csv → Actual vs Predicted rainfall data

best_lstm_model.h5 → Saved trained model

rainfall_plot.png → Visualization of rainfall prediction

Application in Kenya
This project aligns with climate adaptation efforts by enabling AI-driven rainfall forecasting for agricultural resilience.
In Kericho and Bomet, where rainfall variability affects tea and maize production, this model can help:

Predict dry or wet spells in advance

Improve irrigation planning

Support reforestation and soil conservation through better land management timing

Folder Structure
LSTM-Rainfall-Prediction/
│
├── ltsm_fixed.py → Main Python script
├── ClimateEngine.csv → Rainfall dataset (from Climate Engine or CHIRPS)
├── lstm_predictions.csv → Model output
├── requirements.txt → Project dependencies
├── README.md → Project documentation
└── .gitignore → Ignore large or temporary files

Example Output
Evaluation Results:
RMSE = 4.218
MAE = 3.142
Saved predictions to lstm_predictions.csv

Visualization Example:
A plotted line graph showing actual rainfall (blue) vs predicted rainfall (orange).

Citation / Conference Context
This project draws insights from AI applications in environmental prediction, aligning with Bomet University’s 2025 conference theme on "Harnessing Artificial Intelligence for Sustainable Development in Kenya."

License
This project is released under the MIT License.
You are free to use, modify, and share it for academic and research purposes.

Author
Timothy Kiprop
Researcher, Computer Science
Year: 2025


