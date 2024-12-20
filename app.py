import streamlit as st
import torch
import nltk
import pickle
import random
import numpy as np
import pandas as pd

from nnet import NeuralNet
from nltk_utils import bag_of_words

# Random seed for reproducibility
random.seed()

# Device setup
device = torch.device('cpu')
FILE = "models/data.pth"
model_data = torch.load(FILE)

# Model parameters
input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data['model_state']

# Load the trained model
nlp_model = NeuralNet(input_size, hidden_size, output_size).to(device)
nlp_model.load_state_dict(model_state)
nlp_model.eval()

# Load datasets
diseases_description = pd.read_csv("data/symptom_Description.csv")
diseases_description['Disease'] = diseases_description['Disease'].str.lower().str.strip()

disease_precaution = pd.read_csv("data/symptom_precaution.csv")
disease_precaution['Disease'] = disease_precaution['Disease'].str.lower().str.strip()

symptom_severity = pd.read_csv("data/Symptom-severity.csv")
symptom_severity = symptom_severity.applymap(
    lambda s: s.lower().strip().replace(" ", "") if isinstance(s, str) else s
)

# Load symptom list and prediction model
with open('data/list_of_symptoms.pickle', 'rb') as data_file:
    symptoms_list = pickle.load(data_file)

with open('models/fitted_model.pickle2', 'rb') as model_file:
    prediction_model = pickle.load(model_file)

# Initialize Streamlit app
st.title("Symptom Checker")
st.write("A symptom-based disease prediction system. Enter symptoms one by one.")

# Initialize session state for symptoms
if "user_symptoms" not in st.session_state:
    st.session_state["user_symptoms"] = set()


def get_symptom(sentence):
    """Predicts the most probable symptom from the sentence."""
    sentence = nltk.word_tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = nlp_model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()].item()

    return tag, prob


# Input and prediction
sentence = st.text_input("Enter a symptom (type 'done' when finished):")

if sentence:
    if sentence.lower().strip() == "done":
        if not st.session_state["user_symptoms"]:
            st.warning("Please enter some symptoms before concluding.")
        else:
            x_test = [1 if symptom in st.session_state["user_symptoms"] else 0 for symptom in symptoms_list]
            x_test = np.asarray(x_test)

            disease = prediction_model.predict(x_test.reshape(1, -1))[0].strip().lower()
            description = diseases_description.loc[diseases_description['Disease'] == disease, 'Description'].iloc[0]
            precaution = disease_precaution[disease_precaution['Disease'] == disease]

            precautions = ", ".join(
                precaution[f'Precaution_{i}'].iloc[0] for i in range(1, 5)
            )

            st.success(f"Prediction: {disease.capitalize()}")
            st.write(f"**Description:** {description}")
            st.write(f"**Precautions:** {precautions}")

            severity = [
                symptom_severity.loc[
                    symptom_severity['Symptom'] == symptom.lower().replace(" ", ""), 'weight'
                ].iloc[0]
                for symptom in st.session_state["user_symptoms"]
            ]

            if np.mean(severity) > 4 or np.max(severity) > 5:
                st.warning("Symptoms indicate high severity. Please consult a doctor.")

            st.session_state["user_symptoms"].clear()
    else:
        symptom, prob = get_symptom(sentence)
        if prob > 0.5:
            st.session_state["user_symptoms"].add(symptom)
            st.success(f"Identified symptom: {symptom} ({prob * 100:.2f}%)")
        else:
            st.error("Sorry, I couldn't recognize the symptom.")

# Display current symptoms
if st.session_state["user_symptoms"]:
    st.write("Current symptoms:")
    st.write(", ".join(st.session_state["user_symptoms"]))
