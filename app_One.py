import pandas as pd
from semopy import Model, Optimizer
from semopy.inspector import inspect
import streamlit as st

# Sample data for each sub-process of aluminum production (replace with actual data)
initial_data = {
    'Energy_Drilling': [100, 120, 110, 115, 105],
    'Material_Drilling': [200, 220, 210, 215, 205],
    'GHG_Drilling': [300, 320, 310, 315, 305],
    'Energy_Blasting': [150, 170, 160, 165, 155],
    'Material_Blasting': [250, 270, 260, 265, 255],
    'GHG_Blasting': [350, 370, 360, 365, 355],
    'Energy_Hauling': [200, 220, 210, 215, 205],
    'Material_Hauling': [300, 320, 310, 315, 305],
    'GHG_Hauling': [400, 420, 410, 415, 405],
    'Energy_Crushing': [120, 140, 130, 135, 125],
    'Material_Crushing': [220, 240, 230, 235, 225],
    'GHG_Crushing': [320, 340, 330, 335, 325],
    # Add additional data for refining, smelting, casting, and recycling sub-processes
}

# Calculate GHG_Total as the sum of GHG emissions from each subprocess
initial_data['GHG_Total'] = [
    initial_data['GHG_Drilling'][i] + initial_data['GHG_Blasting'][i] + 
    initial_data['GHG_Hauling'][i] + initial_data['GHG_Crushing'][i]
    for i in range(len(initial_data['GHG_Drilling']))
]

# Convert data to pandas DataFrame
df = pd.DataFrame(initial_data)

# Define the SEM model
model_desc = """
Drilling =~ Energy_Drilling + Material_Drilling
Blasting =~ Energy_Blasting + Material_Blasting
Hauling =~ Energy_Hauling + Material_Hauling
Crushing =~ Energy_Crushing + Material_Crushing

Mining =~ Drilling + Blasting + Hauling + Crushing

GHG_Drilling ~ Drilling
GHG_Blasting ~ Blasting
GHG_Hauling ~ Hauling
GHG_Crushing ~ Crushing
GHG_Total ~ Mining
"""

# Streamlit app
st.title('GHG Emissions in Aluminum Production: SEM Analysis')

st.sidebar.header('Input New Data Point')

# Input fields for new data
new_data = {
    'Energy_Drilling': st.sidebar.number_input('Energy Drilling', value=100),
    'Material_Drilling': st.sidebar.number_input('Material Drilling', value=200),
    'GHG_Drilling': st.sidebar.number_input('GHG Drilling', value=300),
    'Energy_Blasting': st.sidebar.number_input('Energy Blasting', value=150),
    'Material_Blasting': st.sidebar.number_input('Material Blasting', value=250),
    'GHG_Blasting': st.sidebar.number_input('GHG Blasting', value=350),
    'Energy_Hauling': st.sidebar.number_input('Energy Hauling', value=200),
    'Material_Hauling': st.sidebar.number_input('Material Hauling', value=300),
    'GHG_Hauling': st.sidebar.number_input('GHG Hauling', value=400),
    'Energy_Crushing': st.sidebar.number_input('Energy Crushing', value=120),
    'Material_Crushing': st.sidebar.number_input('Material Crushing', value=220),
    'GHG_Crushing': st.sidebar.number_input('GHG Crushing', value=320),
}

# Calculate GHG_Total for new data point
new_data['GHG_Total'] = (
    new_data['GHG_Drilling'] + new_data['GHG_Blasting'] +
    new_data['GHG_Hauling'] + new_data['GHG_Crushing']
)

# Convert new data point to DataFrame
new_df = pd.DataFrame([new_data])

# Append new data point to the initial DataFrame
df = df.append(new_df, ignore_index=True)

# Create and fit the model
model = Model(model_desc)
model.load_dataset(df)
opt = Optimizer(model)
opt.optimize()

# Inspect the results
params = inspect(model)

# Display the columns of the params DataFrame and a sample of the data
st.subheader('Parameters DataFrame Columns')
st.write(params.columns)
st.write(params.head())

# Display the updated SEM results
st.subheader('Updated SEM Results')
st.dataframe(params)

# Extract coefficients for granular recommendations
drilling_coeff = params.loc[params['lval'] == 'Drilling', 'Estimate'].values[0]
blasting_coeff = params.loc[params['lval'] == 'Blasting', 'Estimate'].values[0]
hauling_coeff = params.loc[params['lval'] == 'Hauling', 'Estimate'].values[0]
crushing_coeff = params.loc[params['lval'] == 'Crushing', 'Estimate'].values[0]

# Display the params DataFrame to understand its structure
st.write(params)

# Predict GHG emissions for each subprocess based on new data
pred_ghg_drilling = drilling_coeff * new_data['Energy_Drilling'] + drilling_coeff * new_data['Material_Drilling']
pred_ghg_blasting = blasting_coeff * new_data['Energy_Blasting'] + blasting_coeff * new_data['Material_Blasting']
pred_ghg_hauling = hauling_coeff * new_data['Energy_Hauling'] + hauling_coeff * new_data['Material_Hauling']
pred_ghg_crushing = crushing_coeff * new_data['Energy_Crushing'] + crushing_coeff * new_data['Material_Crushing']

# Display predictions
st.subheader('Predicted GHG Emissions')
st.write(f"Predicted GHG Emissions for Drilling: {pred_ghg_drilling}")
st.write(f"Predicted GHG Emissions for Blasting: {pred_ghg_blasting}")
st.write(f"Predicted GHG Emissions for Hauling: {pred_ghg_hauling}")
st.write(f"Predicted GHG Emissions for Crushing: {pred_ghg_crushing}")

# Sum the predictions for total GHG emissions
pred_ghg_total = pred_ghg_drilling + pred_ghg_blasting + pred_ghg_hauling + pred_ghg_crushing

# Display total GHG emissions
st.write(f"Predicted Total GHG Emissions: {pred_ghg_total}")

# Provide recommendations based on coefficients and p-values
st.subheader('Granular Recommendations to Reduce GHG Emissions in Mining')
pvalue_threshold = 0.05

# Extracting the p-values correctly by inspecting the structure of the DataFrame
st.write("Params DataFrame Columns:", params.columns)

if 'p_value' in params.columns:
    params['p_value'] = params['p_value'].astype(float)
    if params.loc[params['lval'] == 'Drilling', 'p_value'].values[0] < pvalue_threshold:
        st.write(f"Focus on improving energy efficiency and material usage in Drilling (Coefficient: {drilling_coeff}).")
    if params.loc[params['lval'] == 'Blasting', 'p_value'].values[0] < pvalue_threshold:
        st.write(f"Focus on improving energy efficiency and material usage in Blasting (Coefficient: {blasting_coeff}).")
    if params.loc[params['lval'] == 'Hauling', 'p_value'].values[0] < pvalue_threshold:
        st.write(f"Focus on improving energy efficiency and material usage in Hauling (Coefficient: {hauling_coeff}).")
    if params.loc[params['lval'] == 'Crushing', 'p_value'].values[0] < pvalue_threshold:
        st.write(f"Focus on improving energy efficiency and material usage in Crushing (Coefficient: {crushing_coeff}).")
else:
    st.write("p_value column not found in the params DataFrame.")
