import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import streamlit as st
import datetime
import time
import random

# Sample hourly data for one day
data = {
    "time": pd.date_range("2024-06-09", periods=24, freq="H"),
    "mining_input1": [1000]*24,
    "mining_input2": [500]*24,
    "mining_output1": [950]*24,
    "mining_ghg": [50]*24,
    "coke_input1": [800]*24,
    "coke_input2": [300]*24,
    "coke_output1": [640]*24,
    "coke_ghg": [70]*24,
    # Add other subprocesses similarly
}

df = pd.DataFrame(data)

# Add upstream outputs as features
df["mining_output1_lag"] = df["mining_output1"].shift(1).fillna(df["mining_output1"].mean())
df["coke_input1"] = df["mining_output1_lag"]  # Assuming coke production input depends on mining output

# Function to model each subprocess with multiple inputs and outputs, considering upstream effects
def model_subprocess(data, input_cols, output_cols):
    models = {}
    for output_col in output_cols:
        model = LinearRegression()
        X = data[input_cols]
        y = data[output_col]
        model.fit(X, y)
        models[output_col] = model
    return models

# Modeling each subprocess
mining_models = model_subprocess(df, ["mining_input1", "mining_input2"], ["mining_output1", "mining_ghg"])
coke_models = model_subprocess(df, ["coke_input1", "coke_input2", "mining_output1_lag"], ["coke_output1", "coke_ghg"])

# Add models for other subprocesses similarly

# Function to detect deviations
def detect_deviation(predicted_output, actual_output, threshold=0.05):
    deviation = (actual_output - predicted_output) / predicted_output
    if abs(deviation) > threshold:
        return True, deviation
    else:
        return False, deviation

# Function to generate alerts for each subprocess
def generate_alerts(data, models, threshold=0.05):
    alerts = []
    for i, row in data.iterrows():
        mining_alerts = {}
        for output_col, model in models["mining"].items():
            predicted_output = model.predict([[row["mining_input1"], row["mining_input2"]]])[0]
            actual_output = row[output_col]
            alert, dev = detect_deviation(predicted_output, actual_output, threshold)
            mining_alerts[output_col] = dev if alert else None

        coke_alerts = {}
        for output_col, model in models["coke"].items():
            predicted_output = model.predict([[row["coke_input1"], row["coke_input2"], row["mining_output1_lag"]]])[0]
            actual_output = row[output_col]
            alert, dev = detect_deviation(predicted_output, actual_output, threshold)
            coke_alerts[output_col] = dev if alert else None
        
        # Check other subprocesses similarly

        if any(mining_alerts.values()) or any(coke_alerts.values()):  # Add other subprocesses conditions
            alerts.append({
                "time": row["time"],
                "mining_alerts": mining_alerts,
                "coke_alerts": coke_alerts,
                # Add other subprocesses alerts
            })
    return alerts

# Generate models for the subprocesses
models = {
    "mining": mining_models,
    "coke": coke_models,
    # Add other models
}

# Streamlit interface
st.title("Steel Production Early Warning System")

placeholder = st.empty()

def generate_random_value(mean, stddev, extreme_probability=0.5):
    if random.random() < extreme_probability:
        # Generate an extreme value
        return mean + random.choice([-1, 1]) * stddev * random.uniform(2, 3)
    else:
        # Generate a normal value
        return mean + random.gauss(0, stddev)

while True:
    hourly_data = {
        "time": pd.Timestamp.now(),
        "mining_input1": generate_random_value(1000, 100),
        "mining_input2": generate_random_value(500, 50),
        "coke_input1": generate_random_value(800, 100),
        "coke_input2": generate_random_value(300, 50)
        # Add random inputs for other subprocesses
    }
    
    hourly_df = pd.DataFrame([hourly_data])
    # Add lagged output from mining for cascading effect
    hourly_df["mining_output1_lag"] = hourly_df["mining_input1"] * 0.95  # Assuming a simple relationship for illustration

    # Predict the outputs using the models
    mining_predictions = {col: model.predict([[hourly_df["mining_input1"][0], hourly_df["mining_input2"][0]]])[0] for col, model in mining_models.items()}
    coke_predictions = {col: model.predict([[hourly_df["coke_input1"][0], hourly_df["coke_input2"][0], hourly_df["mining_output1_lag"][0]]])[0] for col, model in coke_models.items()}
    
    # Add predictions to the dataframe for generating alerts
    hourly_df["mining_output1"] = mining_predictions["mining_output1"]
    hourly_df["mining_ghg"] = mining_predictions["mining_ghg"]
    hourly_df["coke_output1"] = coke_predictions["coke_output1"]
    hourly_df["coke_ghg"] = coke_predictions["coke_ghg"]

    new_alerts = generate_alerts(hourly_df, models)
    
    with placeholder.container():
        st.subheader(f"Time: {hourly_df['time'][0]}")
        st.write(f"Mining Input 1: {hourly_df['mining_input1'][0]}")
        st.write(f"Mining Input 2: {hourly_df['mining_input2'][0]}")
        st.write(f"Predicted Mining Output 1: {hourly_df['mining_output1'][0]}")
        st.write(f"Predicted Mining GHG Emissions: {hourly_df['mining_ghg'][0]}")
        st.write(f"Coke Production Input 1: {hourly_df['coke_input1'][0]}")
        st.write(f"Coke Production Input 2: {hourly_df['coke_input2'][0]}")
        st.write(f"Predicted Coke Production Output 1: {hourly_df['coke_output1'][0]}")
        st.write(f"Predicted Coke Production GHG Emissions: {hourly_df['coke_ghg'][0]}")

        if new_alerts:
            st.warning("Alert: Significant deviation detected!")
            st.write(new_alerts)
        else:
            st.success("All subprocesses are within normal range.")
    
    time.sleep(60)  # Wait for 1 minute before generating the next set of random data
