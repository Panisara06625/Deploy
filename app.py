import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rdkit import Chem
import requests
import random

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_epoch_94.h5')

model = load_model()

# Set max lengths
max_len_smiles = 100
max_len_protein = 1000

# Define encoding
char_to_int_smiles = {
    '#': 1, '%': 2, '(': 3, ')': 4, '+': 5, '-': 6, '.': 7, '/': 8, '0': 9, '1': 10, 
    '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18, '=': 19, 
    '@': 20, 'B': 21, 'C': 22, 'F': 23, 'H': 24, 'I': 25, 'N': 26, 'O': 27, 'P': 28, 
    'S': 29, '[': 30, '\\': 31, ']': 32, 'a': 33, 'c': 34, 'e': 35, 'i': 36, 
    'l': 37, 'n': 38, 'o': 39, 'r': 40, 's': 41
}

# Sample lists for random picking
sample_smiles_list = [
    "Br[Se]c1ccccc1",                         
    "BrCCOc1ccc2nc3ccc(=O)cc3oc2c1",   
    "Brc1c(Br)c(Br)c2[nH]cnc2c1Br",                  
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" 
]

sample_chembl_ids = [
    "CHEMBL3289657",
    "CHEMBL3798866",
    "CHEMBL1230177",
    "CHEMBL1908364",
    "CHEMBL3126333"
]

# Functions 
def encode_smiles(smiles):
    return [char_to_int_smiles.get(c, 0) for c in smiles]

def fetch_smiles_from_chembl(chembl_id):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['molecule_structures']['canonical_smiles']
    except:
        return None
    return None
    
def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None


@st.cache_data
def load_protein_data():
    df = pd.read_csv('encoded_protien.csv')
    return df

# Custom CSS for teal + coral styling
st.markdown(
    """
    <style>
    .main {max-width: 720px; margin: auto;}
    h1, h2, h3 {
        color: #008080; /* Teal */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #FF6F61; /* Coral */
        color: white;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 6px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF8A75;
        color: white;
    }
    .stTextInput>div>input {
        padding: 10px;
        font-size: 16px;
        border: 2px solid #008080;
        border-radius: 6px;
    }
    .stRadio>div {
        margin-bottom: 20px;
    }
    .stDownloadButton>button {
        background-color: #008080;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
    }
    .stDownloadButton>button:hover {
        background-color: #00A0A0;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("OFF-Target Binding Prediction")

smiles_input_type = st.radio("SMILES Input Type", ["SMILES Sequence", "ChEMBL ID"])

# Initialize session state for inputs
if 'smiles' not in st.session_state:
    st.session_state.smiles = ""
if 'chembl_id' not in st.session_state:
    st.session_state.chembl_id = ""

if smiles_input_type == "SMILES Sequence":
    selected_smiles = st.selectbox("Choose a sample SMILES:", [""] + sample_smiles_list)
    if selected_smiles:
        st.session_state.smiles = selected_smiles
else:
    selected_chembl = st.selectbox("Choose a sample ChEMBL ID:", [""] + sample_chembl_ids)
    if selected_chembl:
        st.session_state.chembl_id = selected_chembl

# Show input fields depending on input type
if smiles_input_type == "SMILES Sequence":
    smiles_input = st.text_input("Enter SMILES:", value=st.session_state.smiles)
    st.session_state.smiles = smiles_input  # update session state
    smiles_input_final = smiles_input
else:
    chembl_id_input = st.text_input("Enter ChEMBL ID (e.g., CHEMBL25):", value=st.session_state.chembl_id)
    st.session_state.chembl_id = chembl_id_input  # update session state
    smiles_input_final = ""
    if chembl_id_input:
        smiles_input_final = fetch_smiles_from_chembl(chembl_id_input)
        if smiles_input_final:
            st.success(f"SMILES retrieved: {smiles_input_final}")
        else:
            st.error("Failed to retrieve SMILES from ChEMBL.")
            smiles_input_final = ""

# Prediction triggered by button
if st.button("Enter"):
    if not smiles_input_final:
        st.warning("Please input valid SMILES or ChEMBL ID before pressing Enter.")
    elif not is_valid_smiles(smiles_input_final):
        st.error("Invalid SMILES format. Please enter a correct SMILES sequence.")
    else:
        encoded_smiles = encode_smiles(smiles_input_final)
        padded_smiles = pad_sequences([encoded_smiles], maxlen=max_len_smiles, padding='post')

        df = load_protein_data()
        protein_array = pad_sequences(df['protein_encoded'].apply(eval), maxlen=max_len_protein)

        smiles_array = np.repeat(padded_smiles, len(protein_array), axis=0)

        preds = model.predict([smiles_array, protein_array])
        kd_values = 10 ** (9 - preds.flatten())  # Undo log transform

        df['predicted_Kd(-log(Kd/1e9))'] = kd_values
        top50 = df.sort_values('predicted_Kd(-log(Kd/1e9))').head(50)

        st.subheader("Listed Predictions")
        st.dataframe(top50[['target_id', 'predicted_Kd(-log(Kd/1e9))']])

        csv = top50.to_csv(index=False)
        st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
