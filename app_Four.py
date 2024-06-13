import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
import plotly.express as px

from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_card import card as st_card
from streamlit_extras.metric_cards import style_metric_cards

# Set page configuration
st.set_page_config(layout="wide")

# gif_path = "assets/download.gif"

# # Display the GIF
# st.image(gif_path, use_column_width=False, width = 650)

# Load pre-trained models
@st.cache_resource
def load_models():
    mining_output1_model, mining_output1_poly = joblib.load('models/mining_output1_model.pkl')
    mining_ghg_model, mining_ghg_poly = joblib.load('models/mining_ghg_model.pkl')
    coke_output1_model, coke_output1_poly = joblib.load('models/coke_output1_model.pkl')
    coke_ghg_model, coke_ghg_poly = joblib.load('models/coke_ghg_model.pkl')
    return mining_output1_model, mining_output1_poly, mining_ghg_model, mining_ghg_poly, coke_output1_model, coke_output1_poly, coke_ghg_model, coke_ghg_poly

mining_output1_model, mining_output1_poly, mining_ghg_model, mining_ghg_poly, coke_output1_model, coke_output1_poly, coke_ghg_model, coke_ghg_poly = load_models()

# Function to generate random input values
def generate_inputs():
    np.random.seed(int(time.time()) % (2**32 - 1))
    mining_input1 = np.random.uniform(900, 1100, 1).astype(int)[0]
    mining_input2 = np.random.uniform(400, 600, 1).astype(int)[0]
    coke_input1 = np.random.uniform(700, 900, 1).astype(int)[0]
    coke_input2 = np.random.uniform(200, 400, 1).astype(int)[0]
    return mining_input1, mining_input2, coke_input1, coke_input2

# Function to make predictions
def make_predictions(mining_input1, mining_input2, coke_input1, coke_input2):
    X_mining = pd.DataFrame({"mining_input1": [mining_input1], "mining_input2": [mining_input2]})
    X_coke = pd.DataFrame({"coke_input1": [coke_input1], "coke_input2": [coke_input2]})

    mining_output1 = round(mining_output1_model.predict(mining_output1_poly.transform(X_mining))[0], 0)
    mining_ghg = round(mining_ghg_model.predict(mining_ghg_poly.transform(X_mining))[0], 0)
    coke_output1 = round(coke_output1_model.predict(coke_output1_poly.transform(X_coke))[0], 0)
    coke_ghg = round(coke_ghg_model.predict(coke_ghg_poly.transform(X_coke))[0], 0)

    return mining_output1, mining_ghg, coke_output1, coke_ghg

# Function to set the conditional alarms
def set_alarms(mining_output1, mining_ghg, coke_output1, coke_ghg):
    mining_output1_status = "Desired" if mining_output1 >= 950 else "Low"
    mining_ghg_status = "Red Alert" if mining_ghg >= 47.5 else "Green"
    coke_output1_status = "Desired" if coke_output1 >= 675 else "Low"
    coke_ghg_status = "Red Alert" if coke_ghg >= 67.5 else "Green"
    return mining_output1_status, mining_ghg_status, coke_output1_status, coke_ghg_status

# Streamlit dashboard
# st.title("Production Monitoring Dashboard")

# Create placeholders for dynamic content
input_placeholder = st.empty()
output_placeholder = st.empty()
status_placeholder = st.empty()
graph_placeholder = st.empty()

# DataFrame for storing streaming data
streaming_data = pd.DataFrame(columns=['time', 'mining_ghg'])

def update_dashboard():
    global streaming_data

    mining_input1, mining_input2, coke_input1, coke_input2 = generate_inputs()
    mining_output1, mining_ghg, coke_output1, coke_ghg = make_predictions(mining_input1, mining_input2, coke_input1, coke_input2)
    mining_output1_status, mining_ghg_status, coke_output1_status, coke_ghg_status = set_alarms(mining_output1, mining_ghg, coke_output1, coke_ghg)

    current_time = datetime.now().strftime('%H:%M:%S')
    new_data = pd.DataFrame({'time': [current_time], 'mining_ghg': [mining_ghg]})
    streaming_data = pd.concat([streaming_data, new_data], ignore_index=True)

    with input_placeholder.container():
        col00, col0, col1, col2, col3, col4 = st.columns([3.5,1,1,1,1,1])
        with col00:
            st.image('assets/process_1.png')
        with col0:
            st.metric(label="Production Processes", value="Inputs")
        with col1:
            st.metric(label="Process 1: Iron ore", value=mining_input1)
        with col2:
            st.metric(label="Process 1: Energy", value=mining_input2)
        with col3:
            st.metric(label="Process 2: Molten iron", value=coke_input1)
        with col4:
            st.metric(label="Process 2: Energy", value=coke_input2)

    st.write('<div class="row-spacing"></div>', unsafe_allow_html=True)

    with output_placeholder.container():
        col44, col45, col5, col6, col7, col8 = st.columns([3.5,1,1,1,1,1])
        with col44:
            st.image('assets/process_2.png')
        with col45:
            st.metric(label="Production Processes", value="Outputs")
        with col5:
            st.metric(label="Process 1: Molten iron", value=mining_output1)
        with col6:
            st.metric(label="Process 1: GHG", value=mining_ghg)
        with col7:
            st.metric(label="Process 2: Refined iron", value=coke_output1)
        with col8:
            st.metric(label="Process 2: GHG", value=coke_ghg)

    with status_placeholder.container():
        col84, col85, col9, col10, col11, col12 = st.columns([0.5,1,1,1,1,1])
        with col85:
            st.metric(label="alert", value="States")
        with col9:
            st.metric(label="Process 1: Production", value=mining_output1_status)
        with col10:
            st.metric(label="Process 1: Env.", value=mining_ghg_status)
        with col11:
            st.metric(label="Process 2: Production", value=coke_output1_status)
        with col12:
            st.metric(label="Process 2: Env.", value=coke_ghg_status)

            style_metric_cards()

    with graph_placeholder.container():
        streaming_data['color'] = np.where(streaming_data['mining_ghg'] > 47.5, 'lightcoral', 'lightgreen')
        fig = px.bar(streaming_data, x='time', y='mining_ghg', color='color', color_discrete_map={'lightcoral':'lightcoral', 'lightgreen':'lightgreen'}, title='Mining GHG Over Time')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

while True:
    update_dashboard()
    time.sleep(2)
