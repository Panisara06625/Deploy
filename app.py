import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# Load model
model = tf.keras.models.load_model('model_epoch_94.h5')

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

aa_to_int = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
              'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 
              'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

# === Functions ===
def encode_smiles(smiles):
    return [char_to_int_smiles.get(c, 0) for c in smiles]

def encode_protein(seq):
    return [aa_to_int.get(aa, 0) for aa in seq]

def fetch_fasta(uniprot_id):
        if pd.isna(uniprot_id):
            return None
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                fasta = response.text
                sequence = ''.join(fasta.split('\n')[1:]).strip()
                return sequence
        except:
            return None
        return None

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

# Load encoded protein CSV (already encoded)
@st.cache_data
def load_protein_data():
    df = pd.read_csv('encoded_protien.csv')
    return df

st.title("Drug-Target Binding Prediction")


st.subheader("Choose SMILES input type")

smiles_input_type = st.radio("SMILES Input Type", ["SMILES Sequence", "ChEMBL ID"])

smiles_input = ""
if smiles_input_type == "SMILES Sequence":
    smiles_input = st.text_input("Enter SMILES:")
else:
    chembl_id_input = st.text_input("Enter ChEMBL ID (e.g., CHEMBL25):")
    if chembl_id_input:
        smiles_input = fetch_smiles_from_chembl(chembl_id_input)
        if smiles_input:
            st.success(f"SMILES retrieved: {smiles_input}")
        else:
            st.error("Failed to retrieve SMILES from ChEMBL.")
            smiles_input = ""

if smiles_input:
    encoded_smiles = encode_smiles(smiles_input)
    padded_smiles = pad_sequences([encoded_smiles], maxlen=max_len_smiles, padding='post')

    # Load protein data from CSV (already encoded sequences)
    df = load_protein_data()  
    protein_array = pad_sequences(df['protein_encoded'].apply(eval), maxlen=max_len_protein)

    # Match lengths
    smiles_array = np.repeat(padded_smiles, len(protein_array), axis=0)

    # Predict
    preds = model.predict([smiles_array, protein_array])
    # kd_values = preds.flatten()
    kd_values = 10 ** (9 - preds.flatten())  # Undo log transform

    df['predicted_Kd'] = kd_values
    top50 = df.sort_values('predicted_Kd').head(50)

    st.subheader("Listed Predictions")
    st.dataframe(top50[['target_id', 'predicted_Kd']])

    csv = top50.to_csv(index=False)
    st.download_button("Download CSV", csv, "predictions.csv", "text/csv")

