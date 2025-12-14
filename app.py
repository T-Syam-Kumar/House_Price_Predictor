import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Absolute path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "house_data.csv")

# Debug: show files (remove later)
st.write("Files in directory:", os.listdir(BASE_DIR))

# Load model safely
if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl not found. Please upload it to the GitHub repo.")
    st.stop()

with open(MODEL_PATH, "rb") as file:
    saved_data = pickle.load(file)

model = saved_data["model"]
accuracy = saved_data["accuracy"]

# Load dataset
data = pd.read_csv(DATA_PATH)
