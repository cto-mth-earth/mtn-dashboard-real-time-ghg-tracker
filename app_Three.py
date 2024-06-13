import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from streamlit_card import card as st_card
from streamlit_card import card

import altair as alt

from streamlit_extras.add_vertical_space import add_vertical_space


# Set page configuration
st.set_page_config(layout="wide")

# Inject custom CSS to reduce gaps
st.markdown("""
    <style>
    .stApp {
        margin: 0;
        padding: 0;
    }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    .card {
        margin: 0px;
    }
    .stColumn > div {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    .stCard {
        margin-top: -10px !important;
        margin-bottom: -10px !important;
    }
    .row-spacing {
        margin-top: -10px !important;
        margin-bottom: -10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

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

def update_dashboard():
    mining_input1, mining_input2, coke_input1, coke_input2 = generate_inputs()
    mining_output1, mining_ghg, coke_output1, coke_ghg = make_predictions(mining_input1, mining_input2, coke_input1, coke_input2)
    mining_output1_status, mining_ghg_status, coke_output1_status, coke_ghg_status = set_alarms(mining_output1, mining_ghg, coke_output1, coke_ghg)

    with input_placeholder.container():
        col0, col1, col2, col3, col4 = st.columns(5)
        with col0:
            st_card(title="Inputs", 
                    text="in Production Processes", 
                    key=f"mining_input1_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "45px",
                        }},
                    image='https://cdn-icons-png.flaticon.com/256/3488/3488782.png')
        with col1:
            st_card(title=str(mining_input1), 
                    text="Process 1: Iron ore", 
                    key=f"mining_input1_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    image='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTiNBzQ-xP5uMYnxQu_Unm_Oe5fmqhL6vsNEL8WXhoPMUfvx1OFxsQdRXit2C4Q8pMm60s&usqp=CAU')
        
        with col2:
            st_card(title=str(mining_input2), 
                    text="Process 1: Energy", 
                    key=f"mining_input2_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    image='https://static3.bigstockphoto.com/3/2/1/large1500/1230652.jpg')
        with col3:
            st_card(title=str(coke_input1), 
                    text="Process 2: Molten iron", 
                    key=f"coke_input1_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    image='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTTlO7xiUZkSOwLlh7wxC0-ch3wNhLYPdZYiA&s')
        with col4:
            st_card(title=str(coke_input2), 
                    text="Process 2: Energy", 
                    key=f"coke_input2_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    image='https://www.electricrate.com/wp-content/uploads/2021/07/How-Electricity-Works.jpg.png')

    st.write('<div class="row-spacing"></div>', unsafe_allow_html=True)

    with output_placeholder.container():
        col45, col5, col6, col7, col8 = st.columns(5)
        with col45:
            st_card(title="Outputs", 
                    text="in Production Processes", 
                    key=f"mining_output1_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "45px",
                        }},
                    image='https://cdn-icons-png.flaticon.com/256/3488/3488822.png')

        with col5:
            st_card(title=str(mining_output1), 
                    text="Process 1: Molten iron", 
                    key=f"mining_output1_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    image='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTTlO7xiUZkSOwLlh7wxC0-ch3wNhLYPdZYiA&s')
        with col6:
            st_card(title=str(mining_ghg), 
                    text="Process 1: GHG", 
                    key=f"mining_ghg_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    image='https://as1.ftcdn.net/v2/jpg/00/08/04/10/1000_F_8041046_xFDqfjuKkJggTf7BcKMVcD2KN0A3EIB1.jpg')
        with col7:
            st_card(title=str(coke_output1), 
                    text="Process 2: Refined iron", 
                    key=f"coke_output1_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    image='https://miro.medium.com/v2/resize:fit:4572/1*mcGfv0dguWEAaKAdKik0iA.jpeg')
        with col8:
            st_card(title=str(coke_ghg), 
                    text="Process 2: GHG", 
                    key=f"coke_ghg_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    image='https://as1.ftcdn.net/v2/jpg/00/08/04/10/1000_F_8041046_xFDqfjuKkJggTf7BcKMVcD2KN0A3EIB1.jpg')

    st.write('<div class="row-spacing"></div>', unsafe_allow_html=True)

    with status_placeholder.container():
        col85, col9, col10, col11, col12 = st.columns(5)
        with col85:
            st_card(title="Alerts", 
                    text="states", 
                    key=f"mining_input0_status_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "45px",
                        }},
                    image='https://www.pngall.com/wp-content/uploads/2017/05/Alert-PNG-Clipart.png')

        with col9:
            image_url_mining_output1_color = 'https://cdn-icons-png.flaticon.com/512/9420/9420626.png' if mining_output1_status == 'Low' else 'https://cdn-icons-png.flaticon.com/256/7077/7077243.png'
            st_card(title=mining_output1_status, 
                    text="Process 1: Production", 
                    key=f"mining_output1_status_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    image=image_url_mining_output1_color)
        with col10:
            st_card(title=mining_ghg_status, 
                    text="Process 1: Env.", 
                    key=f"mining_ghg_status_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    )
        with col11:
            st_card(title=coke_output1_status, 
                    text="Process 2: Production", 
                    key=f"coke_output1_status_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    )
        with col12:
            st_card(title=coke_ghg_status, 
                    text="Process 2: Env.", 
                    key=f"coke_ghg_{time.time()}",
                    styles={
                        "card": {
                            "width": "200px",
                            "height": "200px",
                            "border-radius": "15px",
                        }},
                    )

while True:
    update_dashboard()
    time.sleep(10)
